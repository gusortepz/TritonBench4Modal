# Prompt 10: Selective Speed

Prompt 10 is a response to the full run evidence:

| Prompt/run | Exec acc | Official upstream-golden speed | Local same-GPU speed |
| --- | ---: | ---: | ---: |
| prompt-8 limit 5 | 100% | 0.32 | 0.99 |
| prompt-9-speed limit 20 | 85% | 0.29 | 1.22 |
| prompt-9-speed full 166, saved under prompt9_ultraspeed | 74.7% | 0.47 | 1.24 |
| prompt-9-ultraspeed qwen3-coder limit 10 | 90% | 0.27 | 1.17 |

The important proof is that prompt 9 already exceeded 1.0 on the local same-GPU PyTorch reference. The failure is survival: the full run lost too many tasks because the prompt encouraged broad `torch.compile` and speculative Triton.

## What Changed

- Keeps prompt-9's speed-oriented mindset, but makes the strategy selective.
- Makes local same-GPU speed the primary optimization target and treats upstream-golden speed as secondary.
- Adds a strict direct-PyTorch list for fragile categories: solvers, decompositions, special functions, factories, random ops, indexing, shape/layout ops, bitwise/sign ops, and arbitrary reductions.
- Restricts `torch.compile` to profitable structural chains such as conv/matmul/bmm/linear plus activation/norm/dropout.
- Adds runtime fallback around compiled helpers, because many full-run failures were Dynamo/fake-tensor/runtime compile failures.
- Restricts Triton to measured-good CUDA floating elementwise chains and two standalone candidates: `rsqrt` and `selu`.
- Adds known API corrections: `torch.cholesky_solve`, `torch.roll(..., dims=...)`, `F.embedding(..., sparse=...)`, no `torch.relu(..., inplace=...)`, no `tl.erfc`, no `tl.signbit`, no `tl.pow`.
- Strengthens signature discipline: do not invent `out`; import `Tensor` from torch when bare Tensor annotations are copied.

## Hypothesis

Prompt 10 should score lower than prompt-9-ultraspeed on the most cherry-picked speed tasks, but should improve full-run execution accuracy while preserving a local same-GPU speedup above 1.0.
