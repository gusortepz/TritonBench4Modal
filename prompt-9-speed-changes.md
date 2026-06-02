# Prompt 9 Speed Changes

Baseline: created from scratch after `prompt-8` showed that a safe PyTorch-first
prompt reproduces prompt-4/Opus-style correctness but stays around `0.32x` on
the 5-op smoke test.

## Goal

`prompt-9-speed.txt` is a speed-seeking prompt, separate from the safe prompt
family. It tries to choose the fastest honest implementation path by operator
class instead of simply avoiding Triton.

## Key Ideas

- Triton is already parallel, but PyTorch CUDA kernels, cuDNN, cuBLAS, and
  cuSOLVER are also already massively parallel. Triton wins only by doing less
  memory traffic, fewer launches, or better fused work.
- For multi-op PyTorch chains that contain structural/library ops, use
  `torch.compile` on a pure helper first, with a plain PyTorch fallback.
- For pure same-shape elementwise chains, allow a real Triton fused fast path.
- For single ops, solvers, decompositions, conv/matmul/bmm without a meaningful
  epilogue, and special semantics, keep direct PyTorch.
- Keep the first-5 known-safe snippets for `div`, `tanh`, `sigmoid_conv2d`,
  `solve_multiple_lu`, and `fused_bmm_rmsnorm_gelu_dropout_sub`, but make the
  multi-op ones compile-friendly.

## Expected Effect

This prompt may improve multi-op chain performance if `torch.compile` fuses the
PyTorch graph during benchmark warmup. It is still unlikely to make the first
5-op smoke test exceed `1.0x` on Modal T4 if the benchmark compares against
upstream golden timings generated on faster hardware.

For a fair speed read, run both:

- `--limit 5` to confirm call accuracy;
- `--limit 20` to include more elementwise-chain candidates.
