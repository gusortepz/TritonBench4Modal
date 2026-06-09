# Qwen Prompt Improvement Plot Data

Model: `qwen/qwen3.6-35b-a3b`

| Prompt | Run type | n | Exec accuracy | Same-GPU speedup | Note |
|---|---:|---:|---:|---:|---|
| P0 | smoke | 5 | 0.00% | n/a | baseline prompt failed |
| P4 | smoke | 5 | 100.00% | n/a | early correctness-focused smoke |
| P5 | smoke | 5 | 80.00% | n/a | smoke |
| P6a | smoke | 5 | 60.00% | n/a | smoke |
| P6b | smoke | 5 | 60.00% | n/a | smoke |
| P7 | smoke | 5 | 100.00% | n/a | smoke |
| P8 | smoke | 5 | 100.00% | 0.99x | safe PyTorch-first prompt |
| P9 | smoke | 20 | 85.00% | 1.22x | speed-seeking prompt |
| P10 | smoke | 20 | 90.00% | 1.36x | selective speed |
| P11 | smoke | 20 | 95.00% | 1.40x | router prompt |
| P12 | smoke/partial | 20 | 65.00% | n/a | speed phase repeatedly OOM-killed |
| P9 | full | 166 | 74.70% | 1.24x | full validation |
| P11 | full | 166 | 83.73% | 1.13x | full validation |

Poster endpoint for local Qwen: P11 router, with `19/20` correct at `1.40x`
in the smoke run and `139/166` correct at `1.13x` in the full run.
