# TritonBench-T on Modal

Run the **TritonBench-T** benchmark (translate PyTorch ops → Triton kernels)
end-to-end on a single, **cheap** GPU rented from [Modal](https://modal.com).
Designed to be cloned, configured in ~5 minutes, and shared with students.

- **GPU:** NVIDIA **T4** by default — Modal's cheapest tier at **$0.59 / hour**
  (per Modal's [pricing page](https://modal.com/pricing)). T4 has compute
  capability 7.5, which Triton supports.
- **Benchmark:** [TritonBench](https://github.com/thunlp/TritonBench), track **T**
  (166 PyTorch operators with Alpaca-formatted instructions). Cloned and patched
  inside the container — you don't have to install or fix anything locally.
- **What you get back:** call-accuracy %, execution-accuracy %, the official
  TritonBench speedup vs. shipped golden PyTorch timings, and a same-GPU
  speedup vs. PyTorch remeasured on the current Modal GPU.

## Cost expectation

Order-of-magnitude budget for a *full* 166-op run on a single T4: a few dollars.
Quick smoke tests (`--limit 5`) cost cents. Modal bills per second; the GPU is
released as soon as `evaluate` returns.

LLM costs are separate and depend on your provider. With `--limit 5` on
Anthropic Claude Sonnet, generation is well under $0.05.

---

## 1. One-time setup

```bash
# 1. Modal client + auth
python3 -m venv ~/venvs/modal
source ~/venvs/modal/bin/activate
python -m pip install -r requirements-local.txt
modal setup                       # opens a browser to link your account

# 2. Pick ONE provider and add its key as a Modal secret named "tritonbench-llm".
#    The function reads whichever key is present.
modal secret create tritonbench-llm ANTHROPIC_API_KEY=sk-ant-...
# or
modal secret create tritonbench-llm OPENAI_API_KEY=sk-...
```

**Already have a secret with a different name?** Point the app at it once with
an environment variable instead of editing code:

```bash
export TRITONBENCH_LLM_SECRET=openai-secret    # or whatever yours is called
```

You only need the secret if you want Modal to **generate** the Triton predictions
through Anthropic/OpenAI for you. If you generate locally with LM Studio or
already have a `predictions.jsonl`, skip it and use the local/bring-your-own
flows below.

---

## 2. Run it

### End-to-end (generate + evaluate)

```bash
# Smoke test — first 5 ops, costs pennies, finishes in a few minutes
modal run modal_app.py::main --limit 5

# Full run — all 166 ops, defaults to Anthropic Claude Sonnet 4.6
modal run modal_app.py::main

# Use OpenAI instead
modal run modal_app.py::main --provider openai --model gpt-4o-mini

# Use the "complex" instruction variant
modal run modal_app.py::main --dataset comp
```

### Local LM Studio generation + Modal evaluation

Use this when the LLM is running on your laptop and only the benchmark test runs
on Modal. Start LM Studio's local server first, load your Qwen model, then run:

```bash
# Smoke test: generate 5 predictions locally, upload JSONL, evaluate on Modal
modal run modal_app_lmstudio.py::main --limit 5

# Full run. If --model is omitted, the script uses the first model reported by
# LM Studio's /v1/models endpoint.
modal run modal_app_lmstudio.py::main

# If LM Studio exposes a different model id or port:
modal run modal_app_lmstudio.py::main \
  --model qwen3.6-35b-a3b \
  --api native \
  --prompt-file prompt-5.txt \
  --base-url http://localhost:1234/v1
```

Local predictions are written under `local-predictions/` before upload. The
default `--api native` uses LM Studio's `/api/v1/chat` endpoint, which returns
the final message for reasoning models such as Qwen. The default `--concurrency
1` is intentional for desktop LLM serving; increase it only if LM Studio and
your hardware handle parallel requests well.

Before the local JSONL is uploaded to Modal, the pipeline now runs a local
Lex/Yacc preflight:

```bash
python3 parser/validate_predictions.py local-predictions/your_run.jsonl
```

The preflight extracts every `predict` module, builds the Flex/Bison frontend,
runs `lexer/triton_lexer` and `parser/triton_parser`, and stops the Modal upload
if any record has a lexer/parser failure. Results are written under
`parser/results/predictions_preflight_<timestamp>/`.

If you need to bypass this gate temporarily because Flex/Bison is not installed
locally, pass `--skip-preflight`.

### Bring your own predictions

```bash
modal run modal_app.py::evaluate_only --predictions ./my_predictions.jsonl
```

Bring-your-own predictions use the same local Lex/Yacc preflight before upload.
Use `--skip-preflight` only when you intentionally want to send an unchecked
JSONL to Modal.

`my_predictions.jsonl` must have one JSON object per line. Each object needs:

- the **instruction** text from the Alpaca dataset, *exactly as given* — the
  evaluator parses it to find the matching reference operator (it greps for
  the substring between `"Functional Description: "` and
  `"Wrapper Entry Information:"`)
- a `"predict"` field with the model's full reply, ideally wrapped in a
  ```` ```python ... ``` ```` fence

Example line (truncated):

```json
{"instruction": "You are an expert in Trion programming...\nFunctional Description: Computes the absolute value...\nWrapper Entry Information: abs(input_tensor, out=None) -> Tensor...", "predict": "```python\nimport torch\nimport triton\nimport triton.language as tl\n\n@triton.jit\ndef _abs_kernel(...):\n    ...\n\ndef abs(input_tensor, out=None):\n    ...\n```"}
```

The Alpaca source files live inside the container at
`/opt/TritonBench/data/TritonBench_T_simp_alpac_v1.json` and
`/opt/TritonBench/data/TritonBench_T_comp_alpac_v1.json`. The simplest way to
match instructions exactly is to call `generate_only` once with `--limit 0`
and reuse those instruction strings.

### Generate only

```bash
modal run modal_app.py::generate_only --provider anthropic --model claude-sonnet-4-6
```

Predictions land in the persistent volume `tritonbench-t-data`.

---

## 3. Inspect / download artifacts

Each run writes to the `tritonbench-t-data` volume under `results/`:

```
results/
├── call_acc/             # one .py per operator that passed phase 1 (then pruned by phase 2)
├── perf_results/         # generated-code timings consumed by phase 3
├── local_ref_ops/        # copied PyTorch reference modules for surviving ops
└── local_ref_results/    # same-GPU PyTorch reference timings
```

Browse / download from your laptop:

```bash
modal volume ls   tritonbench-t-data results/
modal volume get  tritonbench-t-data results/ ./local-results/
```

---

## 4. Switch GPU tier (optional)

T4 is the cheapest GPU but has no bf16 tensor cores, so a handful of operators
that rely on bf16 will fail at phase 1. To rerun on something beefier, edit
`DEFAULT_GPU` near the top of `modal_app.py` to e.g. `"L4"` ($0.80/hr) or
`"A10"` ($1.10/hr) and rerun.

---

## What the pipeline does (under the hood)

1. **Image build** — clones the upstream
   [TritonBench](https://github.com/thunlp/TritonBench) repo and patches three
   small things so the eval scripts run unattended:
   - `EVAL/eval_T/0_call_acc.py` — points `statis_path` at `*.jsonl` (upstream
     has `.json`), `py_folder` at `data/TritonBench_T_v1/` (upstream points to
     the G dataset), and `py_interpreter` at `sys.executable`.
   - `EVAL/eval_T/1_exe_acc.py` — same `py_interpreter` fix.
   - `performance_metrics/perf_T/run_bench/multiprocess_gpu_run.py` — sets
     `gpu_count = 1` (upstream assumes 8).
2. **Generate** — runs in a CPU container, calls the configured LLM in parallel
   threads, writes one JSON line per operator.
3. **Evaluate** — single GPU container, runs the three TritonBench-T phases
   sequentially:
   - Phase 1 (`0_call_acc.py::call_4file`) — concatenates the predicted module
     with the golden test driver, executes it; the `.py` files of the operators
     that **ran** end up in `results/call_acc/`.
   - Phase 2 (`1_exe_acc.py::execute_4folder`) — re-runs each survivor and the
     reference side-by-side, deletes any whose `stdout` differs.
   - Phase 3 (`perf_T/run_bench/*` + `2_efficiency.py`) — benchmarks each
     remaining op, reports the official speedup against TritonBench's shipped
     golden PyTorch timings, then remeasures the PyTorch reference modules on
     the same Modal GPU and reports a local fair speedup.

Final summary is printed as JSON, e.g.

```json
{
  "total_predictions": 166,
  "phase1_call_acc": { "passed": 88, "rate": 53.01 },
  "phase2_exec_acc": { "passed": 71, "rate": 42.77 },
  "phase3_efficiency": {
    "official_speedup_vs_upstream_golden": 0.83,
    "local_speedup_vs_same_gpu_pytorch": 1.02
  }
}
```

---

## References

- TritonBench paper: <https://arxiv.org/pdf/2502.14752>
- TritonBench repo: <https://github.com/thunlp/TritonBench>
- Modal docs: <https://modal.com/docs/guide>
- Modal GPU pricing: <https://modal.com/pricing>
