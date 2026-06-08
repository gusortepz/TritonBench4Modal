#!/usr/bin/env bash
set -euo pipefail

PROMPT_FILE="${PROMPT_FILE:-prompt-12-general-router.txt}"
DATASET="${DATASET:-simp}"
MAX_TOKENS="${MAX_TOKENS:-16000}"
CONCURRENCY="${CONCURRENCY:-1}"
VOLUME_NAME="${VOLUME_NAME:-tritonbench-t-data}"
BATCH_STAMP="${BATCH_STAMP:-$(date +%Y%m%d-%H%M%S)}"
BATCH_DIR="${BATCH_DIR:-experiments/prompt12_claude_full_${BATCH_STAMP}}"

HAIKU_MODEL="${HAIKU_MODEL:-claude-haiku-4-5}"
SONNET_MODEL="${SONNET_MODEL:-claude-sonnet-4-6}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found. Run: source ~/venvs/modal/bin/activate" >&2
  exit 1
fi

if [ ! -f "${PROMPT_FILE}" ]; then
  echo "Prompt file not found: ${PROMPT_FILE}" >&2
  exit 1
fi

mkdir -p "${BATCH_DIR}"

model_tag() {
  local value="$1"
  value="${value//\//_}"
  value="${value//:/_}"
  printf '%s' "${value}"
}

validate_summary() {
  local expected_subdir="$1"
  python3 - "${expected_subdir}" <<'PY'
import json
import sys
from pathlib import Path

expected = sys.argv[1]
path = Path("latest-summary.json")
if not path.exists():
    print("latest-summary.json was not written; refusing to save stale results", file=sys.stderr)
    sys.exit(1)

summary = json.loads(path.read_text())
actual = summary.get("artifacts_subdir")
if actual != expected:
    print(
        "latest-summary.json belongs to a different run: "
        f"artifacts_subdir={actual!r}, expected={expected!r}",
        file=sys.stderr,
    )
    sys.exit(1)

required = ("total_predictions", "phase1_call_acc", "phase2_exec_acc", "phase3_efficiency")
missing = [key for key in required if key not in summary]
if missing:
    print(f"latest-summary.json is incomplete; missing {missing}", file=sys.stderr)
    sys.exit(1)
PY
}

write_combined_summary() {
  python3 - "${BATCH_DIR}" <<'PY'
import json
import sys
from pathlib import Path

batch_dir = Path(sys.argv[1])
rows = []
for summary_path in sorted(batch_dir.glob("*/latest-summary.json")):
    summary = json.loads(summary_path.read_text())
    phase3 = summary.get("phase3_efficiency", {})
    rows.append(
        {
            "run": summary_path.parent.name,
            "artifacts_subdir": summary.get("artifacts_subdir"),
            "total_predictions": summary.get("total_predictions"),
            "phase1_passed": summary.get("phase1_call_acc", {}).get("passed"),
            "phase1_rate": summary.get("phase1_call_acc", {}).get("rate"),
            "phase2_passed": summary.get("phase2_exec_acc", {}).get("passed"),
            "phase2_rate": summary.get("phase2_exec_acc", {}).get("rate"),
            "official_speedup_vs_upstream_golden": phase3.get(
                "official_speedup_vs_upstream_golden",
                phase3.get("speedup_vs_pytorch"),
            ),
            "local_speedup_vs_same_gpu_pytorch": phase3.get(
                "local_speedup_vs_same_gpu_pytorch"
            ),
        }
    )

(batch_dir / "combined-summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY
}

run_model() {
  local label="$1"
  local model="$2"
  local tag
  tag="$(model_tag "${model}")"
  local run_id="prompt12_${label}_full_${BATCH_STAMP}"
  local output_subdir="results/${run_id}"
  local archive_dir="${BATCH_DIR}/${label}"
  local prediction_remote="predictions/anthropic_${tag}_${DATASET}.jsonl"
  local args=(
    modal run modal_app.py::main
    --provider anthropic
    --model "${model}"
    --dataset "${DATASET}"
    --prompt-file "${PROMPT_FILE}"
    --output-subdir "${output_subdir}"
    --max-tokens "${MAX_TOKENS}"
    --concurrency "${CONCURRENCY}"
  )

  if [ -n "${ANTHROPIC_THINKING:-}" ]; then
    args+=(--anthropic-thinking "${ANTHROPIC_THINKING}")
  fi
  if [ -n "${ANTHROPIC_EFFORT:-}" ]; then
    args+=(--anthropic-effort "${ANTHROPIC_EFFORT}")
  fi

  echo
  echo "====================================================================="
  echo "Running prompt 12 full test: ${label} (${model})"
  echo "  prompt:        ${PROMPT_FILE}"
  echo "  dataset:       ${DATASET}"
  echo "  output_subdir: ${output_subdir}"
  echo "  archive_dir:   ${archive_dir}"
  echo "====================================================================="

  rm -f latest-summary.json latest-run.log
  "${args[@]}" 2>&1 | tee latest-run.log

  validate_summary "${output_subdir}"
  COPY_LOCAL_PREDICTIONS=0 bash save_results.sh "${output_subdir}" "${archive_dir}"

  mkdir -p "${archive_dir}/predictions"
  if modal volume get "${VOLUME_NAME}" "${prediction_remote}" "${archive_dir}/predictions"; then
    echo "Copied generated predictions: volume://${prediction_remote}"
  else
    echo "Warning: could not copy generated predictions from volume://${prediction_remote}" >&2
  fi

  cat > "${archive_dir}/run-metadata.json" <<EOF
{
  "label": "${label}",
  "model": "${model}",
  "provider": "anthropic",
  "prompt_file": "${PROMPT_FILE}",
  "dataset": "${DATASET}",
  "max_tokens": ${MAX_TOKENS},
  "concurrency": ${CONCURRENCY},
  "output_subdir": "${output_subdir}",
  "prediction_remote": "${prediction_remote}"
}
EOF
}

run_model "haiku45" "${HAIKU_MODEL}"
run_model "sonnet46" "${SONNET_MODEL}"

echo
echo "Combined summary:"
write_combined_summary
echo
echo "Saved batch to ${BATCH_DIR}"
