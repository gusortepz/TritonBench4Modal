# Prompt 9 Ultraspeed Changes

Baseline: `prompt-9-speed.txt`.

## Goal

This is a diagnostic speed-first prompt. It intentionally relaxes exact
edge-case semantics, CPU behavior, dtype promotion, and some `out=` precision in
order to see whether a more aggressive CUDA-oriented strategy can move the
efficiency score.

It does **not** instruct the model to return dummy tensors, constants,
uninitialized outputs, identity outputs, or omit the main computation just to
game timing.

## What Changed

- Adds module-level speed flags:
  - `torch.backends.cudnn.benchmark = True`
  - TF32 allowances for CUDA matmul/cuDNN
  - `torch.set_float32_matmul_precision("high")`
- Makes `torch.compile` the default strategy for structural multi-op chains.
- Allows more aggressive Triton for CUDA floating elementwise chains.
- Reduces fallback and edge-case emphasis.
- Keeps known anti-crash rules for exact names/signatures, `normalized_shape`,
  LU APIs, Triton constexpr args, and tensor pointers.

## Caveat

If the benchmark compares Modal T4 timings against upstream golden timings from
faster hardware, a prompt alone may still not push the first-5 smoke score above
`1.0x`. This prompt is for testing whether the generation strategy was the
limiter, not proving the hardware comparison is fair.
