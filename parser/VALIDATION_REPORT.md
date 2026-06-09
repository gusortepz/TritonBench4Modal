# Triton Mini Frontend Validation Report

This report summarizes the current validation results for the Flex/Bison mini
frontend for Python-hosted Triton kernels. The goal of this frontend is not full
Python correctness yet. The current goal is an incremental compiler front end
that provides reliable tokens, structural syntax checks, and a clear base for
future top-down grammar and semantic passes.

The result artifacts referenced here are from runs timestamped `20260608`.

## Executive Summary

| Area | Result |
| --- | --- |
| Lexer token coverage contract | Passed: all 91 lexer tokens are mentioned in the CFG and yacc parser |
| Lexer scan over generated experiment code | Passed: 1258 Python files, 558614 tokens, 0 lexer failures, 0 `ERROR` tokens |
| Parser pass over generated experiment code | Passed: 1258 Python files, 0 rejected files |
| Parser smoke tests | Passed: 5 valid examples accepted, 5 invalid examples rejected |
| Dedicated invalid Triton-shaped tests | Passed: 6 invalid snippets detected, 0 missed |

The current frontend detects structural problems that are useful early in the
kernel-generation pipeline:

- invalid indentation and bad dedents
- indentation blocks that appear after a non-block line
- missing top-level colon before an indented suite
- unbalanced or crossed `()`, `[]`, and `{}` delimiters
- extra closing delimiters
- unterminated strings or other lexer `ERROR` tokens
- explicit backslash-newline line joining, so valid continued Triton statements
  do not create false indentation errors

It does not yet enforce full Python or Triton semantics. For example, it does
not yet check undefined variables, Triton API signatures, tensor shape
compatibility, decorator alias resolution, or whether a `tl.*` call is legal in
a given context.

## Frontend Under Test

The frontend currently has three layers:

1. `lexer/triton_lexer.l`
   - Flex lexer for Python-like Triton kernel files.
   - Emits general tokens such as identifiers, literals, operators, Python
     keywords, delimiters, `NEWLINE`, `INDENT`, and `DEDENT`.
   - Keeps dotted APIs generic: `tl.load` becomes
     `IDENTIFIER DOT IDENTIFIER`, not a special hard-coded token.
   - Skips comments, regular whitespace, implicit continuation newlines inside
     delimiters, and explicit backslash-newline continuation.

2. `grammar/triton_kernel_cfg.md`
   - Human-readable starter CFG.
   - Broader than the current executable parser.
   - Defines the intended direction for imports, decorators, definitions,
     control flow, assignments, expressions, calls, attributes, indexing, and
     kernel-launch shapes.

3. `parser/triton_parser.y`
   - Bison/yacc parser for the first executable syntax iteration.
   - Deliberately simple and structural.
   - Validates logical lines, block structure, block-header colons, final
     newline behavior, and balanced delimiter groups.

## Token And Grammar Contract

The lexer defines 91 tokens in `lexer/triton_tokens.h`.

The current contract is:

- every lexer token must appear in `grammar/triton_kernel_cfg.md`
- every lexer token must appear in `parser/triton_parser.y`
- `ERROR` and standalone `BACKSLASH` are documented as recovery or diagnostic
  tokens, not normal Triton/Python syntax
- backslash followed by a physical newline is consumed by the lexer as explicit
  line joining and does not reach the parser as `BACKSLASH`

Validation command:

```bash
python3 grammar/check_token_coverage.py
```

Observed result:

```text
All 91 lexer tokens are mentioned in grammar/triton_kernel_cfg.md and parser/triton_parser.y.
```

## Methodology

The validation used four complementary checks.

### 1. Smoke Tests

Command:

```bash
make -C parser
python3 parser/run_smoke_tests.py
```

Purpose:

- quickly test both accepted and rejected small examples
- catch regressions in delimiter balancing
- catch regressions in indentation and block-header handling
- verify final files may omit a trailing newline
- verify explicit backslash-newline line joining works

Current valid smoke cases:

| Case | What It Exercises |
| --- | --- |
| `simple_assignment.py` | expression line with calls, indexing, dict, tuple |
| `missing_final_newline.py` | file ending without final `NEWLINE` |
| `explicit_line_join.py` | Python explicit `\` line continuation |
| `indented_block.py` | colon block with `INDENT`/`DEDENT` |
| `decorated_kernel.py` | decorator line plus function definition |

Current invalid smoke cases:

| Case | What It Should Detect |
| --- | --- |
| `unclosed_paren.py` | missing closing delimiter |
| `extra_close.py` | extra closing delimiter |
| `crossed_delimiters.py` | crossed delimiter types |
| `assignment_block.py` | indented block after a non-header assignment |
| `missing_header_colon.py` | block header missing top-level colon |

Observed result:

```text
Parser smoke tests passed: 5 valid, 5 invalid.
```

### 2. Lexer Scan Over Generated Code

Command:

```bash
make -C lexer
python3 lexer/run_experiment_scan.py
```

Input corpus:

- all `*.py` files under `experiments/`
- generated TritonBench experiment outputs and support files

Pass criteria:

- lexer exits successfully for each file
- no `ERROR` tokens are emitted
- token streams can be produced for real generated Triton/Python files

Result directory:

```text
lexer/results/experiments_scan_20260608_221213
```

Observed result:

| Metric | Value |
| --- | ---: |
| Python files scanned | 1258 |
| Total tokens emitted | 558614 |
| Files with lexer errors | 0 |
| Total `ERROR` tokens | 0 |
| Token samples saved | 5 |

Most frequent emitted tokens in the valid corpus:

| Token | Count |
| --- | ---: |
| `IDENTIFIER` | 166505 |
| `COMMA` | 53189 |
| `NEWLINE` | 49442 |
| `ASSIGN` | 44560 |
| `DOT` | 31910 |
| `LPAREN` | 29431 |
| `RPAREN` | 29431 |
| `FLOAT` | 17551 |
| `LBRACKET` | 16622 |
| `RBRACKET` | 16622 |
| `STRING` | 14756 |
| `INTEGER` | 14139 |
| `COLON` | 11904 |
| `INDENT` | 8409 |
| `DEDENT` | 8409 |

Interpretation:

- the lexer is exercising real generated code, including calls, attributes,
  indexing, literals, indentation, and block structure
- balanced `INDENT` and `DEDENT` aggregate counts are a useful sanity signal
- zero `ERROR` tokens means the token set is broad enough for the current corpus

### 3. Parser Pass Over Generated Code

Command:

```bash
make -C parser
python3 parser/run_experiment_parse.py
```

Input corpus:

- the same `experiments/**/*.py` generated-code corpus

Pass criteria:

- parser exits successfully for each file
- no syntax errors are reported by the current structural parser

Result directory:

```text
parser/results/experiments_parse_20260608_221213
```

Observed result:

| Metric | Value |
| --- | ---: |
| Python files parsed | 1258 |
| Failed files | 0 |

Interpretation:

- the executable yacc parser accepts all generated experiment files at the
  current structural-validation level
- this is important because the parser should not reject existing generated code
  while we are still at the early structural stage
- earlier false failures around explicit `\` line continuation were resolved by
  treating backslash-newline as a lexer-level join

### 4. Dedicated Invalid Triton-Shaped Suite

Command:

```bash
make -C parser
python3 parser/run_negative_tests.py
```

Purpose:

- feed the frontend intentionally incorrect Triton-shaped code
- test issues that the current lexer/parser is expected to detect
- save the invalid snippets, stderr, and a result table in a timestamped folder

Result directory:

```text
parser/results/negative_triton_20260608_222304
```

Observed result:

| Metric | Value |
| --- | ---: |
| Invalid snippets tested | 6 |
| Detected by parser | 6 |
| Missed by parser | 0 |

Invalid-case matrix:

| Case | Intentional Error | Detected | Diagnostic |
| --- | --- | --- | --- |
| `missing_function_colon.py` | function definition opens an indented suite without `:` | yes | `syntax error at 5:28 near newline` |
| `unclosed_tl_load.py` | `tl.load(` call leaves `(` open | yes | `syntax error at 9:1 near <DEDENT>` |
| `crossed_delimiters.py` | delimiter types cross with `(...]` | yes | `syntax error at 7:30 near ]` |
| `indent_after_assignment.py` | assignment incorrectly introduces an indented block | yes | `syntax error at 5:1 near <INDENT>` |
| `bad_dedent_width.py` | dedent returns to a width never opened | yes | `lexer error at 8:1 near <BAD_DEDENT>` |
| `extra_close_paren.py` | statement has an extra `)` | yes | `syntax error at 7:52 near )` |

Interpretation:

- the invalid suite confirms the frontend catches the exact structural mistakes
  it currently claims to catch
- the parser returns nonzero for all six invalid snippets
- lexer-origin layout errors now produce a useful diagnostic instead of a silent
  parser failure

## What The Current Frontend Detects

### Lexical And Layout Detection

The lexer currently detects or normalizes:

- identifiers, literals, Python keywords, operators, and delimiters
- dotted names as generic token sequences
- `NEWLINE`, `INDENT`, and `DEDENT`
- indentation stack overflows
- inconsistent dedents as `ERROR` with text `<BAD_DEDENT>`
- unterminated single-line and triple-quoted strings
- unknown characters as `ERROR`
- physical newlines inside `()`, `[]`, and `{}` as implicit continuation
- explicit backslash-newline continuation

### Parser Detection

The yacc parser currently detects:

- statement boundaries by logical lines
- files with or without a final newline
- balanced parenthesis groups
- balanced bracket groups
- balanced brace groups
- crossed delimiter types
- extra closing delimiters
- unclosed delimiter groups at line/block/file boundaries
- indentation blocks introduced only by block headers
- block headers requiring a top-level colon
- bad lexer tokens through parser error accounting

The parser's current block-header starts include:

- `def`
- `if`, `elif`, `else`
- `for`, `while`
- `try`, `except`, `finally`
- `with`
- `class`
- `async`

## What It Does Not Detect Yet

The current parser is intentionally permissive inside logical lines. It is not a
full Python or Triton parser yet.

Known out-of-scope checks:

- undefined variable reads
- duplicate definitions
- import alias resolution
- whether `triton.jit` was imported under an alias
- whether a function is actually a Triton kernel
- whether `_kernel[grid](...)` launches a known kernel
- validity of `tl.load`, `tl.store`, `tl.arange`, `tl.dot`, or other Triton API
  signatures
- `tl.constexpr` meta-parameter correctness
- tensor/device/dtype/shape compatibility
- Python expression precedence and associativity beyond delimiter structure
- full function parameter grammar enforcement
- full decorator grammar enforcement
- semantic restrictions inside `@triton.jit` functions

This is expected for the current stage. The richer CFG in
`grammar/triton_kernel_cfg.md` documents the intended direction for replacing
generic line items with more specific syntactic productions.

## Reproduction Commands

Recommended full validation sequence:

```bash
make -C parser
python3 parser/run_smoke_tests.py
python3 grammar/check_token_coverage.py
python3 parser/run_experiment_parse.py

make -C lexer
python3 lexer/run_experiment_scan.py

python3 parser/run_negative_tests.py

make -C parser clean
make -C lexer clean
```

Notes:

- `make -C parser` regenerates the parser and shared lexer C source.
- `make -C lexer` builds the standalone token-printer binary.
- result directories are timestamped under `parser/results/` and
  `lexer/results/`.
- build artifacts are safe to clean after validation.

## Conclusion

The mini frontend is useful as a first compiler front end for generated Triton
code. It already provides:

- a general lexer with 91 covered tokens
- stable indentation and logical-line handling
- real-corpus validation across 1258 generated files
- negative validation for structural syntax errors
- dedicated result folders for repeatable evidence

The next useful iteration is to move from structural parsing toward the CFG's
specific productions: imports, decorators, function signatures, assignments,
target lists, expressions, calls, attributes, indexing, and kernel-launch
shapes. After that, a semantic pass can begin checking definitions, aliases,
Triton kernel identification, and generated-kernel correctness rules.
