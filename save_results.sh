#!/usr/bin/env bash
set -euo pipefail

VOLUME_NAME="${VOLUME_NAME:-tritonbench-t-data}"
OUTPUT_SUBDIR="${1:-results/lmstudio}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
DEST_DIR="${2:-experiments/lmstudio_${TIMESTAMP}}"

if ! command -v modal >/dev/null 2>&1; then
  echo "modal CLI not found. Run: source ~/venvs/modal/bin/activate" >&2
  exit 1
fi

mkdir -p "${DEST_DIR}"

echo "Saving Modal volume artifacts..."
echo "  volume: ${VOLUME_NAME}"
echo "  remote: ${OUTPUT_SUBDIR}"
echo "  local:  ${DEST_DIR}"
modal volume get "${VOLUME_NAME}" "${OUTPUT_SUBDIR}" "${DEST_DIR}"

if compgen -G "local-predictions/*.jsonl" >/dev/null; then
  LATEST_PREDICTIONS="$(ls -t local-predictions/*.jsonl | head -n 1)"
  cp "${LATEST_PREDICTIONS}" "${DEST_DIR}/"
  echo "Copied latest local predictions: ${LATEST_PREDICTIONS}"
else
  echo "No local-predictions/*.jsonl file found to copy."
fi

if [ -f "latest-run.log" ]; then
  cp latest-run.log "${DEST_DIR}/"
  echo "Copied run log: latest-run.log"
fi

if [ -f "latest-summary.json" ]; then
  cp latest-summary.json "${DEST_DIR}/"
  echo "Copied final summary: latest-summary.json"
  python3 - <<'PY' | tee "${DEST_DIR}/speedup-summary.json"
import json

with open("latest-summary.json", encoding="utf-8") as f:
    phase3 = json.load(f).get("phase3_efficiency", {})

summary = {
    "official_speedup_vs_upstream_golden": phase3.get(
        "official_speedup_vs_upstream_golden",
        phase3.get("speedup_vs_pytorch"),
    ),
    "local_speedup_vs_same_gpu_pytorch": phase3.get(
        "local_speedup_vs_same_gpu_pytorch"
    ),
}
print(json.dumps(summary, indent=2))
PY
  echo "Wrote speedup summary: speedup-summary.json"
fi

modal volume ls "${VOLUME_NAME}" "${OUTPUT_SUBDIR}" > "${DEST_DIR}/modal-volume-listing.txt"

cat > "${DEST_DIR}/README.md" <<EOF
# TritonBench LM Studio Run

- Saved at: ${TIMESTAMP}
- Modal volume: ${VOLUME_NAME}
- Modal output subdir: ${OUTPUT_SUBDIR}
- Downloaded artifacts: ${OUTPUT_SUBDIR}
- Volume listing: modal-volume-listing.txt
- Run log: latest-run.log, when present
- Final summary: latest-summary.json, when present
- Speedup summary: speedup-summary.json, when present
EOF

echo "Saved run artifacts to ${DEST_DIR}"
