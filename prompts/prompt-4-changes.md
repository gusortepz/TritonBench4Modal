# Prompt 4 Changes

Baseline: `prompt-3.txt`.

Reason for the change: the first valid LM Studio/Qwen smoke run completed the full Modal evaluation, but only 2 of 5 generated operators survived call and execution accuracy. The failures were not infrastructure problems; they were generated-code issues.

Observed smoke result:

- `total_predictions`: 5
- `phase1_call_acc`: 2 / 5, 40%
- `phase2_exec_acc`: 2 / 5, 40%
- `phase3_efficiency.speedup_vs_pytorch`: 0.14
- Passing operators: `sigmoid_conv2d.py`, `tanh.py`
- Failing operators: `fused_bmm_rmsnorm_gelu_dropout_sub.py`, `div.py`, `solve_multiple_lu.py`

## What Changed

- Added a stronger "survival mode" policy for small local models: prefer exact PyTorch wrappers by default, and only use Triton for very simple cases.
- Made `div` explicitly PyTorch-only because the generated Triton version failed on Triton kernel metadata and is risky for broadcasting, dtype promotion, scalar divisors, complex inputs, `rounding_mode`, and `out`.
- Added a specific `normalized_shape` rule because Qwen generated `tuple(normalized_shape)` for an `int`, causing `TypeError: 'int' object is not iterable`.
- Added a safe RMS normalization pattern using either `F.rms_norm` or manual RMS normalization with an `int`-aware shape branch.
- Added a specific LU solve rule because Qwen generated `torch.linalg.lu_solve(..., pivot=pivot)`, but `lu_solve` does not accept `pivot`.
- Added Triton API guardrails for the exact failure mode seen in `div`: every meta argument such as `BLOCK_SIZE` must appear in the `@triton.jit` signature as `tl.constexpr`.
- Added a rule to pass tensors to Triton kernels, not `.data_ptr()`.
- Reweighted the implementation strategy toward PyTorch fallbacks for BMM, convolution, normalization, dropout, and linear algebra.
- Expanded the final checklist with the newly observed failure classes: invalid PyTorch keyword arguments, `tuple(int)`/`len(int)` shape bugs, and Triton constexpr mismatches.

## Expected Effect

`prompt-4.txt` should improve call accuracy more than speed. The main goal is to turn the three failing smoke-test operators into correct PyTorch fallbacks instead of fragile custom Triton implementations.

If the smoke run improves, use `prompt-4.txt` for a larger `--limit` run before trying the full 166-operator run.
