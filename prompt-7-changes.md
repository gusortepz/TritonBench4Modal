# Prompt 7 Changes

Baseline: `prompt-4.txt`, informed by the bad speed results from `prompt-6a`,
`prompt-6b`, and Opus 4.7 max with `prompt-4`.

Observed issue:

- Opus 4.7 max reached 5/5 call accuracy and 5/5 execution accuracy, but only
  `0.31x` speedup versus PyTorch.
- The generated code was mostly correct PyTorch, while generated or suggested
  custom kernels for single elementwise/post-processing work did not beat
  PyTorch.
- The prompt variants that encouraged fusion regressed correctness or produced
  fragile/stale results.

## What Changed

- Reframed the prompt as "PyTorch-first" rather than "fusion-first".
- Added a strict Triton allowlist: only pure deterministic same-shape floating
  CUDA elementwise chains with at least three simple steps may use Triton.
- Explicitly forbids Triton by default for standalone elementwise ops, div,
  conv, bmm/matmul/linear, normalization, dropout, solvers, broadcasting, dtype
  promotion, and `out=` complexity.
- Keeps `prompt-4` correctness guardrails for:
  - exact wrapper/signature;
  - `normalized_shape` int handling;
  - `torch.linalg.lu_factor` / `torch.linalg.lu_solve` safe pattern;
  - no Triton `div`;
  - no invalid Triton meta args or `.data_ptr()`.
- Adds stronger wording that names like `fused_*`, `*_conv2d`, and `*_bmm` do
  not automatically justify writing a fused Triton kernel.

## Expected Effect

`prompt-7.txt` should preserve high call/execution accuracy and reduce the
model's tendency to generate slow handmade Triton kernels. It is not expected
to produce a large speedup on the 5-op smoke set; the goal is a stable,
correct PyTorch baseline that avoids unnecessary regressions.

If this gets 5/5 correctness, compare speed against `prompt-4`. If speed is
still near `0.31x`, the next improvement likely needs benchmark-aware operator
selection or hand-authored templates rather than more generic prompt pressure.
