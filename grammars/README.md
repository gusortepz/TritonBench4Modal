# Triton Elementwise Grammar

`triton_elementwise.ebnf` is a narrow XGrammar-style EBNF grammar for simple
TritonBench-T elementwise kernels. It intentionally does not try to parse or
generate arbitrary Python or arbitrary Triton.

Supported operations:

- `add`
- `sub`
- `sqrt`
- `rsqrt`
- `tanh`
- `relu_sqrt`

Use `triton_elementwise.py` to detect the requested operation and select the
matching start rule:

```python
from grammars import build_generation_payload

payload = build_generation_payload(instruction_text)
qwen_generate(
    prompt=payload["prompt"],
    grammar=payload["grammar"],
    start_rule=payload["start_rule"],
)
```

To add another operation:

1. Add `<op>_module`, `<op>_kernel`, and `<op>_wrapper` rules to the grammar.
2. Add `<op>_root ::= <op>_module`.
3. Add the operation to `root`.
4. Add the operation to `START_RULES`.
5. Add a detector case in `detect_operation`.
