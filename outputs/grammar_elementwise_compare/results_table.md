# Grammar Elementwise Compare

Model: `qwen/qwen3.6-35b-a3b`

Functions: `add`, `sub`, `sqrt`, `rsqrt`, `tanh`, `relu_sqrt`

## Aggregate

| condition | call acc | exec acc | official speedup | same-GPU speedup |
|---|---:|---:|---:|---:|
| Prompt 0 | 0/6 | 0/6 | n/a | n/a |
| Prompt 11 | 6/6 | 6/6 | 0.26x | 1.73x |
| Prompt 0 + grammar | 6/6 | 6/6 | 0.27x | 1.82x |
| Prompt 11 + grammar | 6/6 | 6/6 | 0.26x | 1.71x |

## Per Function, Same-GPU Speedup

| function | Prompt 0 | Prompt 11 | Prompt 0 + grammar | Prompt 11 + grammar |
|---|---:|---:|---:|---:|
| add | call fail | pass, 0.9990x | pass, 1.0544x | pass, 0.9807x |
| sub | call fail | pass, 1.6907x | pass, 1.7843x | pass, 1.6597x |
| sqrt | call fail | pass, 0.9993x | pass, 1.0441x | pass, 0.9826x |
| rsqrt | call fail | pass, 3.7330x | pass, 3.9009x | pass, 3.6728x |
| tanh | call fail | pass, 0.9997x | pass, 1.0446x | pass, 0.9805x |
| relu_sqrt | call fail | pass, 1.9564x | pass, 2.0824x | pass, 1.9601x |

## Per Function, Official Upstream-Golden Speedup

| function | Prompt 0 | Prompt 11 | Prompt 0 + grammar | Prompt 11 + grammar |
|---|---:|---:|---:|---:|
| add | call fail | 0.1465x | 0.1544x | 0.1436x |
| sub | call fail | 0.2418x | 0.2548x | 0.2369x |
| sqrt | call fail | 0.1469x | 0.1531x | 0.1439x |
| rsqrt | call fail | 0.6063x | 0.6337x | 0.5954x |
| tanh | call fail | 0.1469x | 0.1531x | 0.1436x |
| relu_sqrt | call fail | 0.2780x | 0.2954x | 0.2777x |
