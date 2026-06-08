# Triton Kernel Lexer

This directory contains the first Lex/Flex pass for tokenizing Python-hosted
Triton kernels before a yacc grammar is added.

## Main Token Groups

- Python layout: `NEWLINE`, `INDENT`, `DEDENT`
- Python syntax: keywords, identifiers, literals, delimiters, and operators
- Triton decorators: `TRITON_JIT`, `TRITON_AUTOTUNE`, `TRITON_HEURISTICS`
- Triton language calls: `TL_LOAD`, `TL_STORE`, `TL_ARANGE`,
  `TL_PROGRAM_ID`, math/reduction tokens, `TL_CONSTEXPR`, and `TL_DTYPE`
- Host-framework symbols: `TORCH_SYMBOL` and `FUNCTIONAL_SYMBOL`

The scanner keeps comments out of the token stream, treats strings as atomic
tokens, and emits indentation tokens so yacc can parse Python-like blocks
without having to recalculate whitespace structure.

## Build

```bash
make -C lexer
```

## Try It

```bash
lexer/triton_lexer experiments/lmstudio_20260526-200014/lmstudio/call_acc/sigmoid_conv2d.py
```

The standalone binary prints `line:column`, token name, and token text. A yacc
parser can include `triton_tokens.h` and call `yylex()` directly.
