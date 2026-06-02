# Prompt 11: Router

Prompt 11 turns the prompt-10 lessons into an explicit route selector.

## Why

Prompt 10 improved the limit-20 result:

| Prompt/run | Exec acc | Official upstream-golden speed | Local same-GPU speed |
| --- | ---: | ---: | ---: |
| prompt-9-speed limit 20 | 85% | 0.29 | 1.22 |
| prompt-10 limit 20 | 90% | 0.39 | 1.36 |

The two real call failures in prompt 10 were:

- `svd`: generated `def linalg_svd(...)`, while tests called `svd(...)`.
- `i0`: generated speculative Triton for a special function and used invalid `tl.ones_like`.

The main speed loss was:

- `fused_mv_logsoftmax_dropout`: called `torch.compile(...)` inside the public wrapper, compiling on the hot path and scoring only `0.206` locally.

## Main Changes

- Adds a first-class public-name rule for dotted targets: `linalg.svd` must become top-level `svd`, not `linalg_svd`.
- Adds exact recipes for the first-20 task types that matter most, including `i0`, `svd`, `fused_mv_logsoftmax_dropout`, `fused_lu_solve`, and `fused_index_select_eq`.
- Makes special functions direct PyTorch only.
- Makes `torch.compile` illegal inside the public wrapper or any hot path.
- Routes `fused_mv_logsoftmax_dropout` to direct `torch.mv + F.log_softmax + F.dropout`, not compile.
- Keeps Triton only for the measured-good elementwise whitelist.
- Adds `tl.ones_like` to the invalid Triton API list and tells the model to use `tl.full_like(x, 1.0)` if needed.

## Expected Outcome

On the same limit-20 smoke, prompt 11 should target:

- exec accuracy: `20/20`;
- local same-GPU speed: still above `1.25`;
- fewer full-run failures from wrong names, invalid Triton APIs, and hot-path compile.
