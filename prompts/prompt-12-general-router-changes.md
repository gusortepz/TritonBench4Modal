# Prompt 12: General Router

Prompt 12 intentionally moves away from task-by-task recipes.

## Why

Prompt 11 improved the limit-20 run to:

| Prompt/run | Exec acc | Official upstream-golden speed | Local same-GPU speed |
| --- | ---: | ---: | ---: |
| prompt-10 limit 20 | 90% | 0.39 | 1.36 |
| prompt-11 limit 20 | 95% | 0.43 | 1.40 |

The remaining failure was not a missing recipe for one task. It was a general semantic bug:

- an optional tensor was used without checking for `None`;
- shape was inferred from an optional affine weight instead of the computed tensor;
- compile fallback did not help because the reference helper had the same semantic bug.

## Main Concept Shift

Prompt 12 asks the model to act like a compiler engineer:

1. identify the public function name and exact signature;
2. write the PyTorch reference path first;
3. classify the operator family;
4. choose one route:
   - direct PyTorch for fragile/library families;
   - module-level `torch.compile` for stable structural chains;
   - Triton only for clear CUDA floating elementwise fusion;
5. apply optional-parameter and shape-safety rules before optimizing.

## What Was Generalized

- Dotted target names become the final component, not a hand-written per-task `svd` fix.
- Special functions are a category, not just `i0`.
- Optional tensor safety is a category, not just `weight=None`.
- Normalization shape inference is a general principle: prefer explicit normalized shape; otherwise infer from the computed tensor, and only use affine weights that exactly match.
- `torch.compile` placement is a general rule: module-level only, never in the public wrapper.
- Triton usage is a category gate: clear deterministic elementwise fusion only.

## Expected Outcome

On limit-20, prompt 12 may be slightly less aggressive than prompt 11, but should aim for:

- exec accuracy: `20/20`;
- local same-GPU speed: above `1.25`;
- fewer full-run failures caused by optional parameter misuse.

The real test is the full 166 run, where generalization matters more than isolated task recipes.
