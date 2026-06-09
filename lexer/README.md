# Triton Kernel Lexer

This directory contains the Lex/Flex scanner for a small Python-hosted Triton
frontend. The scanner is intentionally general: it tokenizes Python structure
and expressions, while Triton-specific meaning is left for the parser and later
semantic passes.

## Token Design

- Layout: `NEWLINE`, `INDENT`, `DEDENT`
- Names and literals: `IDENTIFIER`, `INTEGER`, `FLOAT`, `COMPLEX`, `STRING`
- Python syntax: imports, function definitions, flow keywords, operators, and
  delimiters
- Dotted APIs: `triton.jit`, `tl.load`, `torch.empty_like`, and similar names
  are emitted as `IDENTIFIER DOT IDENTIFIER ...`
- Kernel launches: `_kernel[grid](...)` is emitted with ordinary bracket and
  call tokens so the grammar can parse it as a normal expression pattern

Comments are skipped. Physical newlines inside `()`, `[]`, or `{}` are skipped,
matching Python's implicit continuation behavior. Explicit backslash-newline
continuations are also skipped. Triple-quoted strings are emitted as one
`STRING` token.

## Build

```bash
make -C lexer
```

## Grammar Contract

Every token in `triton_tokens.h` must be mentioned in both
`grammar/triton_kernel_cfg.md` and `parser/triton_parser.y`. Check that
contract with:

```bash
python3 grammar/check_token_coverage.py
```

## Try It

```bash
lexer/triton_lexer experiments/lmstudio_20260526-200014/lmstudio/call_acc/sigmoid_conv2d.py
```

The standalone binary prints `line:column`, token name, and token text. A yacc
parser can include `triton_tokens.h` and call `yylex()` directly.

The starter CFG for the future parser is in `grammar/triton_kernel_cfg.md`.

To scan all generated experiment code and write a dedicated result folder:

```bash
python3 lexer/run_experiment_scan.py
```

To parse the same generated code with the starter yacc parser:

```bash
make -C parser
python3 parser/run_smoke_tests.py
python3 parser/run_experiment_parse.py
```
