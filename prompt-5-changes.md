# Prompt 5 Changes

Baseline: `prompt-4.txt`.

Reason for the change: `prompt-4` achieved 100% call accuracy and 100% execution accuracy on the 5-op smoke test, but the geometric speedup was only `0.32x`, meaning the generated code was slower than PyTorch. `prompt-5` tries to recover speed without giving up the correctness gains.

Observed `prompt-4` smoke result:

- `total_predictions`: 5
- `phase1_call_acc`: 5 / 5, 100%
- `phase2_exec_acc`: 5 / 5, 100%
- `phase3_efficiency.speedup_vs_pytorch`: 0.32
- `sigmoid_conv2d.json` failed during perf measurement but passed call/execution accuracy.

## What Changed

- Replaced the broad "PyTorch fallback by default" stance with "speed-aware survival mode".
- Kept PyTorch fallback mandatory for hard semantic areas: `div`, convolution as a structural operation, dropout, normalization, pooling, linear algebra, reductions with tricky dim semantics, and stochastic ops.
- Explicitly encouraged Triton for safe unary floating point CUDA elementwise ops such as `tanh`, `sigmoid`, `relu`, `gelu`, `silu`, `sqrt`, `rsqrt`, `exp`, `log`, `log1p`, `abs`, and `neg`.
- Explicitly allowed Triton for simple same-shape floating point binary ops such as `add`, `sub`, `mul`, `maximum`, and `minimum`.
- Added a reusable safe unary Triton template.
- Added safe Triton expressions for common activations and math functions.
- Added guidance for simple fused elementwise post-processing after PyTorch structural ops, for example applying a Triton activation after `F.conv2d`.
- Preserved the run-specific fixes from `prompt-4`: `normalized_shape` int handling, safe RMS norm, no `pivot=` on `torch.linalg.lu_solve`, no fragile Triton `div`, tensors passed to kernels instead of `.data_ptr()`, and `BLOCK_SIZE` as `tl.constexpr`.

## Expected Effect

`prompt-5.txt` should keep high call/execution accuracy while improving speed on simple elementwise-heavy operators. It may be riskier than `prompt-4`, so start with the same 5-op smoke test before trying a larger limit.

## Suggested Test

```bash
modal run modal_app_lmstudio.py::main --limit 5 --model qwen/qwen3.6-35b-a3b --api native --prompt-file prompt-5.txt 2>&1 | tee latest-run.log
bash save_results.sh
```
