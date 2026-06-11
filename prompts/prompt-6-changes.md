# Prompt 6 Variants

Baseline: `prompt-4.txt` (5/5 correctness, 0.32x speedup).
`prompt-5.txt` regressed correctness (4/5) without improving speed (0.35x), because broad "allow Triton when safe" guidance gives small local models too much rope on operations that PyTorch already executes via cuBLAS/cuDNN/cuSOLVER.

Prompt 6 splits the speed effort into two narrow, mutually exclusive bets so we can A/B them on the same 5-op smoke set.

## Why isolated Triton kernels lose

- `torch.tanh`, `torch.div`, `F.conv2d`, `torch.bmm`, `torch.linalg.lu_solve` already dispatch hand-tuned CUDA kernels. A single Triton kernel doing the same memory pass cannot beat them; it usually loses to kernel launch overhead.
- Triton wins where PyTorch issues MULTIPLE kernels for one logical operation. Two patterns dominate:
  1. Fusing chains of elementwise ops (and post-structural epilogues) into one kernel.
  2. Tiled matmul with the bias/activation epilogue inlined.

## prompt-6a — Fusion-Focused

- Keeps prompt-4's correctness fixes (normalized_shape int handling, lu_solve pivot rule, no Triton div, BLOCK_SIZE as tl.constexpr, tensors not data_ptr to kernels).
- Adds an explicit FUSION-FIRST policy: only write Triton when fusing multiple elementwise steps into one kernel.
- Adds a "fused elementwise template" with a same-shape contiguous fast path and PyTorch fallback.
- Adds a "fused activation-after-structural template" for ops like `fused_bmm_rmsnorm_gelu_dropout_sub` and `sigmoid_conv2d`: structural op in PyTorch, full elementwise tail in one Triton kernel.
- Forbids isolated single-op Triton kernels for `tanh`, `sigmoid`, `relu`, `div`, etc.

Expected effect: large jump on `fused_*` operators and `*_conv2d` / `*_bmm` operators with elementwise tails; small loss or break-even on pure elementwise ops; correctness comparable to prompt-4.

## prompt-6b — Tiled Matmul + Epilogue

- Same prompt-4 correctness base.
- Adds a single canonical tiled matmul kernel template with a fused epilogue switch: bias add and one of relu/gelu/sigmoid/tanh/silu.
- Forces "use PyTorch only" for everything that is NOT matmul/bmm/linear plus epilogue, including pure elementwise ops, decompositions, conv, normalization, etc.

Expected effect: large jump on linear+bias+activation, matmul+bias+gelu, and similar ops; no change on non-matmul ops; correctness comparable to prompt-4.

## Suggested A/B run

```bash
# Variant A: fusion
modal run modal_app_lmstudio.py::main \
  --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-6a.txt \
  --output-subdir results/lmstudio_6a 2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_6a experiments/lmstudio_6a_$(date +%Y%m%d-%H%M%S)

# Variant B: tiled matmul
modal run modal_app_lmstudio.py::main \
  --limit 5 --model qwen/qwen3.6-35b-a3b --api native \
  --prompt-file prompt-6b.txt \
  --output-subdir results/lmstudio_6b 2>&1 | tee latest-run.log
bash save_results.sh results/lmstudio_6b experiments/lmstudio_6b_$(date +%Y%m%d-%H%M%S)
```

Compare the resulting `latest-summary.json` between the two saved experiment folders. The winner becomes the prompt-7 base.

## Caveats

- Both variants intentionally narrow the speed strategy, so a 5-op smoke set may not show the win clearly if the chosen ops do not match the variant's strength. For a fair comparison run the same set under both prompts. For a sharper signal, use `--limit 20` after the smoke run.
- Local Qwen 35B may still get the more complex matmul-tile indexing wrong on some operators. If 6b regresses correctness too far on the smoke set, that is a signal to drop it.
