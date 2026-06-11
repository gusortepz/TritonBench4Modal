# Corpus test report - newparser (flex + yacc recognizer)

- **Date:** 2026-06-10
- **Corpus:** `/Users/gustavoortiz/Documents/tec/8/reto-dev/TritonBench4Modal/experiments/prompt12_claude_full_20260607-201510/haiku45/prompt12_haiku45_full_20260607-201510/call_acc` (135 files, LLM-generated Triton programs)
- **Parser:** `newlexparser/newparser` (layout-free grammar, see grammar.md; 'gm' below)

## Results by stage

| Stage | files | ACCEPT | REJECT | accept rate |
| --- | --- | --- | --- | --- |
| 1. full files (host + kernels) | 135 | 0 | 135 | 0.0% |
| 2. extracted @triton.jit kernels | 55 | 52 | 3 | 94.5% |
| 3. kernels, docstrings stripped | 55 | 52 | 3 | 94.5% |

80 of the 135 files contain **no `@triton.jit` kernel at all** (the generator implemented those ops in pure PyTorch). They are host-only programs, outside the scope of a Triton-kernel grammar, so stages 2-3 cover the 55 files that define kernels.

## Stage 1 - full files: why they fail

| First-error cause | files | example |
| --- | --- | --- |
| deferred stmt: try/except/with/class (gm 14) | 135 | `abs.py` line 10 |

## Stage 2 - kernels only: why they fail

| First-error cause | files | example |
| --- | --- | --- |
| return-guard greediness (gm 13.2) | 2 | `fused_cross_entropy_log_softmax.py` line 22 |
| conditional expression (deferred, gm 13.4) | 1 | `fused_bmm_rmsnorm_gelu_dropout_sub.py` line 53 |

## Stage 3 - kernels without docstrings: why they fail

| First-error cause | files | example |
| --- | --- | --- |
| return-guard greediness (gm 13.2) | 2 | `fused_cross_entropy_log_softmax.py` line 19 |
| conditional expression (deferred, gm 13.4) | 1 | `fused_bmm_rmsnorm_gelu_dropout_sub.py` line 50 |

### Stage 3 failures, with the offending source line

- `fused_bmm_rmsnorm_gelu_dropout_sub.py` line 50 - conditional expression (deferred, gm 13.4)
  ```python
  scale = 1.0 / (1.0 - dropout_p) if dropout_p < 1.0 else 1.0
  ```
- `fused_cross_entropy_log_softmax.py` line 19 - return-guard greediness (gm 13.2)
  ```python
  offset = idx * num_classes
  ```
- `fused_layer_norm_relu_linear.py` line 24 - return-guard greediness (gm 13.2)
  ```python
  offsets = row_idx * M + col_start
  ```

## Interpretation

- **Stage 1 (0%) measures scope, not quality**: every generated file opens with the same `try:/except:` host boilerplate, and `try` is a deliberately deferred statement (gm 14). The parser correctly consumes the imports and `torch.backends...` assignments before stopping there.
- **Triple-quoted docstrings are handled by the lexer** (flex start conditions, one STRING token per docstring), so stage 2 and the docstring-stripped stage 3 now agree - stage 3 is kept as a control to confirm docstrings no longer cause failures.
- **The residue matches the documented limitations exactly**: the bare-`return` guard greediness (gm 13.2) and one real conditional expression inside a kernel (gm 13.4). No failure fell outside the causes already predicted in grammar.md.
- **Ranked next steps by measured impact**: (1) `try:`/`except:` headers (or a statement terminator) if full host files must parse - all 135 files; (2) `return`-guard fix (needs a terminator) - 2 files; (3) conditional expressions (needs a terminator) - 1 file.

## Per-file detail (stage 3)

| file | result | where | cause |
| --- | --- | --- | --- |
| `abs.py` | ACCEPT |  |  |
| `add.py` | ACCEPT |  |  |
| `add_gelu.py` | ACCEPT |  |  |
| `bitwise_and.py` | ACCEPT |  |  |
| `combined_activation.py` | ACCEPT |  |  |
| `cos.py` | ACCEPT |  |  |
| `cos_avg_pool1d.py` | ACCEPT |  |  |
| `cos_signbit.py` | ACCEPT |  |  |
| `dropout_sigmoid_linear.py` | ACCEPT |  |  |
| `elu_linear.py` | ACCEPT |  |  |
| `erf.py` | ACCEPT |  |  |
| `erfc_sqrt.py` | ACCEPT |  |  |
| `exp_mean.py` | ACCEPT |  |  |
| `exp_sqrt.py` | ACCEPT |  |  |
| `floor.py` | ACCEPT |  |  |
| `fused_bmm_rmsnorm_gelu_dropout.py` | ACCEPT |  |  |
| `fused_bmm_rmsnorm_gelu_dropout_sub.py` | REJECT | line 50 | conditional expression (deferred, gm 13.4) |
| `fused_cross_entropy_log_softmax.py` | REJECT | line 19 | return-guard greediness (gm 13.2) |
| `fused_embedding_add_tanh.py` | ACCEPT |  |  |
| `fused_hardshrink_dropout.py` | ACCEPT |  |  |
| `fused_hardsigmoid_batch_norm.py` | ACCEPT |  |  |
| `fused_layer_norm_relu_linear.py` | REJECT | line 24 | return-guard greediness (gm 13.2) |
| `fused_mul_add_logsoftmax_dropout_bmm.py` | ACCEPT |  |  |
| `fused_mv_sigmoid_sub.py` | ACCEPT |  |  |
| `gelu_conv2d.py` | ACCEPT |  |  |
| `gelu_std.py` | ACCEPT |  |  |
| `leaky_relu.py` | ACCEPT |  |  |
| `log.py` | ACCEPT |  |  |
| `log1p.py` | ACCEPT |  |  |
| `log_tanh.py` | ACCEPT |  |  |
| `matrix_multiply_and_row_dot.py` | ACCEPT |  |  |
| `min_gelu.py` | ACCEPT |  |  |
| `normalized_cosine_similarity.py` | ACCEPT |  |  |
| `rad2deg_sqrt.py` | ACCEPT |  |  |
| `reciprocal.py` | ACCEPT |  |  |
| `relu.py` | ACCEPT |  |  |
| `relu_conv2d.py` | ACCEPT |  |  |
| `relu_sqrt.py` | ACCEPT |  |  |
| `scaled_add_dot.py` | ACCEPT |  |  |
| `scaled_add_norm.py` | ACCEPT |  |  |
| `selu.py` | ACCEPT |  |  |
| `sigmoid_adaptive_avg_pool2d.py` | ACCEPT |  |  |
| `sigmoid_argmax.py` | ACCEPT |  |  |
| `sigmoid_conv2d.py` | ACCEPT |  |  |
| `signbit.py` | ACCEPT |  |  |
| `softmax_mul.py` | ACCEPT |  |  |
| `softplus_linear.py` | ACCEPT |  |  |
| `sqrt.py` | ACCEPT |  |  |
| `sqrt_tanh.py` | ACCEPT |  |  |
| `sub_gelu.py` | ACCEPT |  |  |
| `symmetric_mm_and_abs_sum.py` | ACCEPT |  |  |
| `tanh.py` | ACCEPT |  |  |
| `tanh_linear.py` | ACCEPT |  |  |
| `tensordot_rsqrt.py` | ACCEPT |  |  |
| `trunc.py` | ACCEPT |  |  |
