# Prompt 8 Changes

Baseline: `prompt-7.txt`, informed by the Triton/TritonBench investigation.

## What We Learned

- Triton wins only with a concrete performance reason: less global-memory
  traffic, fewer intermediate tensors, fewer kernel launches, or row/tile data
  kept in on-chip memory.
- Standalone elementwise Triton kernels are usually slower than PyTorch native
  CUDA kernels. This matched the `div` and `tanh` smoke-test results.
- Structural operators such as conv2d, bmm, matmul, linear algebra, dropout,
  and normalization are not good generic LLM-written Triton targets.
- The first 5 smoke-test tasks are not a good speed sample: most are PyTorch-only
  categories, and the real fused task contains bmm, RMSNorm, GELU, and dropout.
- `2_efficiency.py` compares generated timings against upstream golden result
  JSONs and drops ratios outside `(0.1, 10)`, so same-hardware timing should be
  investigated separately before over-interpreting absolute speedup.

## What Changed

- Added an explicit Triton performance model: reduce DRAM traffic or kernel
  launches, otherwise prefer PyTorch.
- Replaced the broad "strict allowlist" with three clearer allowlisted cases:
  1. pure same-shape deterministic elementwise chains with at least two steps;
  2. simple last-dimension row reductions where one row fits in one block;
  3. post-structural deterministic pointwise epilogues only when saving at
     least two pointwise passes.
- Strengthened the PyTorch-only denylist for single ops, div/pow family,
  structural ops, solvers/decompositions, dropout/randomness, indexing, special
  functions, and dtype/broadcast-sensitive semantics.
- Added hot-path PyTorch guidance: avoid unnecessary `.contiguous()`, device
  transfers, defensive try/except, and speculative helper logic.
- Kept known correctness guardrails from earlier prompts:
  - exact wrapper/signature;
  - `normalized_shape` int handling;
  - safe `torch.linalg.lu_factor` / `torch.linalg.lu_solve` pattern;
  - no Triton `div`;
  - no standalone Triton `tanh`;
  - no invalid Triton meta args or `.data_ptr()`.

## Expected Effect

`prompt-8.txt` should keep prompt-7-level correctness while giving the model a
more useful performance triage. It may produce Triton for simple multi-step
elementwise chains, but should avoid slow standalone kernels and risky
structural rewrites.

Use `prompt-8` first on the 5-op smoke test for correctness, then on a larger
sample such as `--limit 20` because the first 5 tasks underrepresent the
elementwise-chain allowlist.
