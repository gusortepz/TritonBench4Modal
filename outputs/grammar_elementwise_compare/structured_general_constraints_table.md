# Structured Output With General Constraints

Model: `qwen/qwen3.6-35b-a3b`

Functions: `add`, `sub`, `sqrt`, `rsqrt`, `tanh`, `relu_sqrt`

This run uses LM Studio structured output with a general JSON schema:

```json
{
  "type": "object",
  "properties": {
    "code": { "type": "string" }
  },
  "required": ["code"],
  "additionalProperties": false
}
```

It does **not** enumerate valid function-specific code variants. The generated
code is then checked by the general lexer/parser preflight and Modal evaluation.

## Aggregate

| condition | call acc | exec acc | same-GPU speedup | official speedup |
|---|---:|---:|---:|---:|
| Prompt 0 | 0/6 | 0/6 | n/a | n/a |
| Prompt 11 | 6/6 | 6/6 | 1.73x | 0.26x |
| Prompt 0 + structured-general constraints | 2/6 | 2/6 | 1.47x | 0.21x |
| Prompt 11 + structured-general constraints | 6/6 | 6/6 | 1.73x | 0.26x |
| Prompt 0 + root/template grammar | 6/6 | 6/6 | 1.81x | 0.27x |
| Prompt 11 + root/template grammar | 6/6 | 6/6 | 1.71x | 0.26x |

## Per Function, Same-GPU Speedup

| function | Prompt 0 | Prompt 11 | Prompt 0 + structured-general | Prompt 11 + structured-general | Prompt 0 + root/template | Prompt 11 + root/template |
|---|---:|---:|---:|---:|---:|---:|
| add | call fail | 0.9990x | 0.9810x | 0.9988x | 1.0543x | 0.9809x |
| sub | call fail | 1.6907x | call fail | 1.6895x | 1.7503x | 1.6605x |
| sqrt | call fail | 0.9993x | call fail | 0.9995x | 1.0446x | 0.9820x |
| rsqrt | call fail | 3.7330x | call fail | 3.7420x | 3.8969x | 3.6681x |
| tanh | call fail | 0.9997x | call fail | 0.9998x | 1.0443x | 0.9800x |
| relu_sqrt | call fail | 1.9564x | 1.9586x | 1.9565x | 2.0835x | 1.9594x |

## Interpretation

Structured output with general constraints improves weak Prompt 0 formatting and
module shape, but it does not prevent invalid Triton semantics such as bad
pointer typing in `tl.load` or unsupported `tl.tanh`. The root/template grammar
is stronger because it constrains the generated code structure, not just the
transport format.
