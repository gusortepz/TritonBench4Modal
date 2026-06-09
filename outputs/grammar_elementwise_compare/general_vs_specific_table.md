# General vs Function-Specific Grammar

Model: `qwen/qwen3.6-35b-a3b`

Functions: `add`, `sub`, `sqrt`, `rsqrt`, `tanh`, `relu_sqrt`

General grammar mode uses the shared `root` rule:

```ebnf
root ::= add_module | sub_module | sqrt_module | rsqrt_module | tanh_module | relu_sqrt_module
```

Function-specific mode uses the matching start rule for each requested function,
for example `add_root`, `sqrt_root`, or `relu_sqrt_root`.

## Aggregate

| condition | call acc | exec acc | same-GPU speedup | official speedup |
|---|---:|---:|---:|---:|
| Prompt 0 | 0/6 | 0/6 | n/a | n/a |
| Prompt 11 | 6/6 | 6/6 | 1.73x | 0.26x |
| Prompt 0 + function-specific grammar | 6/6 | 6/6 | 1.82x | 0.27x |
| Prompt 0 + general grammar | 6/6 | 6/6 | 1.81x | 0.27x |
| Prompt 11 + function-specific grammar | 6/6 | 6/6 | 1.71x | 0.26x |
| Prompt 11 + general grammar | 6/6 | 6/6 | 1.71x | 0.26x |

## Per Function, Same-GPU Speedup

| function | Prompt 0 | Prompt 11 | Prompt 0 + specific | Prompt 0 + general | Prompt 11 + specific | Prompt 11 + general |
|---|---:|---:|---:|---:|---:|---:|
| add | call fail | 0.9990x | 1.0544x | 1.0543x | 0.9807x | 0.9809x |
| sub | call fail | 1.6907x | 1.7843x | 1.7503x | 1.6597x | 1.6605x |
| sqrt | call fail | 0.9993x | 1.0441x | 1.0446x | 0.9826x | 0.9820x |
| rsqrt | call fail | 3.7330x | 3.9009x | 3.8969x | 3.6728x | 3.6681x |
| tanh | call fail | 0.9997x | 1.0446x | 1.0443x | 0.9805x | 0.9800x |
| relu_sqrt | call fail | 1.9564x | 2.0824x | 2.0835x | 1.9601x | 1.9594x |
