# Grammar Design — Triton-Python Parser (Top-Down LL(1), implemented in Unix yacc)

Companion design document for `newlexer.l`. This defines the context-free grammar
**before** writing any parser code, in **top-down form**: no left recursion,
left-factored, one token of lookahead per decision. The implementation tool is
**Unix yacc** (§4, §15): every nonterminal below maps 1:1 to one rule of the
future `.y` file.

**Design decision (final): the parser is layout-free.** No `INDENT`/`DEDENT`
tokens, and no `NEWLINE` token either — the lexer treats line breaks as plain
whitespace. The consequences of this choice are designed-in throughout (§6, §7,
§11.3) and its limits are documented honestly in §13.

---

## 1. Goal and scope

We are parsing **Triton kernel source files**: a subset of Python that covers
what real kernels in TritonBench actually use. A representative input:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

The grammar validates: imports, decorators (`@triton.jit`,
`@triton.autotune(...)`), function headers with annotated/default parameters
(`BLOCK_SIZE: tl.constexpr`), assignments, attribute chains (`tl.load`), calls
with keyword arguments (`mask=mask`), subscripts with slices (`offs[:, None]`),
the kernel-launch syntax (`kernel[grid](args)`), full operator precedence, and
the headers of `if`/`elif`/`else`, `for`, `while`.

**What it deliberately does NOT validate: block structure.** Python delimits
blocks by indentation; since we erase all layout in the lexer, the information
"which statements are inside the `if`" simply does not exist in the token
stream. The honest grammar therefore treats a program as a **flat sequence**
of statements and block *headers* (§7). Pairing (`else` ↔ `if`, decorator ↔
`def`, body ↔ header) becomes a later semantic pass, outside the grammar.

Python features kernels do not use (`class`, `try`, `lambda`, `yield`,
comprehensions, conditional expressions, …) are deferred — see §14.

---

## 2. How to read the notation

```
S -> E $          one production: "S produces E followed by end-of-input"
E -> T E'         first alternative for E
  -> ...          further alternative for the same E ("|" implied)
E' -> + T E'      a "tail" nonterminal created by the top-down transformation
   -> ε           epsilon: E' may produce nothing
```

| Symbol style       | Meaning                                                                                              |
| ------------------ | ---------------------------------------------------------------------------------------------------- |
| `lower_snake_case` | **Nonterminal** (defined by productions in this document)                                            |
| `UPPERCASE`        | **Terminal** = token returned by the lexer (`NAME`, `IF`, …)                                         |
| `'+' '(' ','`      | Terminal written as its source text. §5 maps each one to its lexer token name (`'+'` = `PLUS`, etc.) |
| `ε`                | The empty string — the rule may match nothing                                                        |
| `$`                | End of input (the lexer's `return 0` at `<<EOF>>`)                                                   |
| `x_tail`           | Same role as the primed names (`E'`) from class — `_tail` suffix keeps the name a legal C identifier |

---

## 3. Going top-down: the two transformations

A top-down parser expands the **leftmost** nonterminal at every step, choosing
the production by looking at the next input token. That imposes two famous
restrictions, and fixing them is exactly what reshapes the classic grammar.

### 3.1 No left recursion

`E -> E + T` makes a top-down parser call `parse_E()` as the first action of
`parse_E()` — infinite recursion, before consuming a single token. The
mechanical fix:

```
A -> A α                          A  -> β A'
  -> β            becomes         A' -> α A'
                                     -> ε
```

Applied to the example from the brief:

```
Bottom-up (left recursive)        Top-down (LL(1))
--------------------------        ----------------------
E -> E + T                        E  -> T E'
  -> T                            E' -> + T E'
                                     -> ε
T -> T * F                        T  -> F T'
  -> F                            T' -> * F T'
                                     -> ε
F -> ( E )                        F  -> ( E )
  -> id                             -> id
```

Reading of `E'`: *"after one term, either a `+` continues the sum, or the sum is
over (ε)."* The decision needs exactly one lookahead token.

### 3.2 No common prefixes (left factoring)

If two alternatives start the same way, one lookahead token can't choose between
them. The fix is to share the prefix and postpone the decision:

```
A -> α β                          A  -> α A'
  -> α γ          becomes         A' -> β
                                     -> γ
```

We use this everywhere Python makes the parser "wait and see" — e.g. a line that
starts with an expression may turn out to be a bare expression, an assignment,
or an augmented assignment (§8.1).

### 3.3 The associativity caveat (important, and subtle)

`E -> E + T` produced left-leaning parse trees: `a + b + c` = `(a + b) + c`.
After the transformation the *string language is identical*, but the tree under
`E'` leans right. This does **not** change what programs are accepted — and when
an AST is built later, each `_tail` is implemented as a **loop** that folds
operands left-to-right, restoring left associativity:

```
parse_arith_expr():                      # E  -> T E'
    node = parse_term()
    while lookahead in {'+', '-'}:       # E' -> + T E' | - T E' | ε
        op = consume()
        node = BinOp(node, op, parse_term())   # left fold
    return node
```

So: grammar shape right-recursive, implementation iterative, semantics
left-associative. The genuinely right-associative operator (`**`) keeps true
recursion instead of a loop — it comes out naturally.

(With yacc this concern is deferred entirely: the first iteration is a pure
recognizer with no actions, and when AST construction arrives the `_tail`
rules can return op/operand lists that the head rule folds left — or the
expression core can mechanically revert to the left-recursive originals of
§3.1, which yacc also accepts. See §15.)

---

## 4. Formal definition

The grammar is the 4-tuple **G = (N, Σ, P, S)**:

| Component            | Value                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **N** (nonterminals) | ~65 symbols: `program`, `element`, `header`, `stmt`, `test`, `term`, `atom`, … plus `_tail` helpers |
| **Σ** (terminals)    | ~74 tokens of `newlexer.l` — **no layout tokens at all** (§5, §6)                           |
| **P** (productions)  | The ~115 rules in §7–§10                                                                    |
| **S** (start symbol) | `program`; the parser stops by matching `$`                                                 |

Target parser class: **LL(1) within each construct** — leftmost derivation,
1 token lookahead. A grammar is LL(1) iff for every nonterminal `A` with
alternatives `A -> α₁ | α₂ | …`:

1. **FIRST(αᵢ) ∩ FIRST(αⱼ) = ∅** for i ≠ j — lookahead picks one alternative; and
2. if some `αᵢ ⇒* ε`, then **FIRST(αⱼ) ∩ FOLLOW(A) = ∅** for j ≠ i — lookahead
   distinguishes "expand the other alternative" from "produce ε and return".

Inside every construct (an expression, a header, an import, a parameter list)
the rules below satisfy both conditions. **At statement boundaries they cannot
be satisfied** — with no terminator token, "the statement ends here" and "the
expression continues" can collide on the same lookahead. That residue is
handled by a deliberate policy, **maximal munch**: prefer to continue the
current construct. §11.3 enumerates every such point and what it implies.

**Relation to yacc.** yacc builds *bottom-up* LALR(1) parsers — strictly more
powerful (a classical result, Beatty 1982: every LL(1) grammar is LALR(1)).
yacc accepts this grammar as written, and its conflict report becomes our
checking tool: `yacc -v` must show **0 reduce/reduce conflicts**, and **every
shift/reduce conflict must be one of the statement-boundary cases listed in
§11.3** (yacc's default — shift — implements maximal munch for free, the same
mechanism it classically uses for the dangling `else` of C). Any conflict not
in that list is a design error: fix the grammar, not the parser.

---

## 5. Terminal alphabet Σ (mapping to `newlexer.l`)

Shorthand used in productions → token enum in the lexer:

**Literals and names**

| Grammar symbol | Lexer token | Example lexeme            |
| -------------- | ----------- | ------------------------- |
| `NAME`         | `NAME`      | `pid`, `tl`, `BLOCK_SIZE` |
| `INTEGER`      | `INTEGER`   | `1024`                    |
| `FLOAT`        | `FLOAT`     | `1e-6`, `0.5`             |
| `STRING`       | `STRING`    | `'mask'`, `"x"`           |
| `$`            | `0` (EOF)   | end of file               |

There are **no layout terminals**: no `NEWLINE`, no `INDENT`, no `DEDENT`.
Line breaks never reach the parser (§6).

**Keywords** (token name = spelled keyword): `DEF RETURN IF ELIF ELSE FOR WHILE
IN AND OR NOT IS TRUE FALSE NONE IMPORT FROM AS PASS RAISE ASSERT BREAK CONTINUE
GLOBAL NONLOCAL DEL` — all used in the core grammar.
Reserved but deferred to §14: `LAMBDA TRY EXCEPT FINALLY WITH CLASS YIELD ASYNC
AWAIT`, and the delimiter `';'`.

**Operators and delimiters**

| Grammar | Lexer token     |     | Grammar | Lexer token            |
| ------- | --------------- | --- | ------- | ---------------------- |
| `'+'`   | `PLUS`          |     | `'='`   | `ASSIGN`               |
| `'-'`   | `MINUS`         |     | `'+='`  | `PLUS_ASSIGN`          |
| `'*'`   | `STAR`          |     | `'-='`  | `MINUS_ASSIGN`         |
| `'**'`  | `DOUBLE_STAR`   |     | `'*='`  | `STAR_ASSIGN`          |
| `'/'`   | `SLASH`         |     | `'/='`  | `SLASH_ASSIGN`         |
| `'//'`  | `DOUBLE_SLASH`  |     | `'//='` | `DOUBLE_SLASH_ASSIGN`  |
| `'%'`   | `PERCENT`       |     | `'%='`  | `PERCENT_ASSIGN`       |
| `'@'`   | `AT`            |     | `'**='` | `DOUBLE_STAR_ASSIGN`   |
| `'<'`   | `LESS`          |     | `'&='`  | `AMPERSAND_ASSIGN`     |
| `'<='`  | `LESS_EQUAL`    |     | `'\|='` | `PIPE_ASSIGN`          |
| `'>'`   | `GREATER`       |     | `'^='`  | `CARET_ASSIGN`         |
| `'>='`  | `GREATER_EQUAL` |     | `'<<='` | `LEFT_SHIFT_ASSIGN`    |
| `'=='`  | `EQUAL_EQUAL`   |     | `'>>='` | `RIGHT_SHIFT_ASSIGN`   |
| `'!='`  | `NOT_EQUAL`     |     | `'('` `')'` | `OPEN/CLOSE_PARENTHESIS` |
| `'&'`   | `AMPERSAND`     |     | `'['` `']'` | `OPEN/CLOSE_BRACKET` |
| `'\|'`  | `PIPE`          |     | `'{'` `'}'` | `OPEN/CLOSE_BRACE`   |
| `'^'`   | `CARET`         |     | `','`   | `COMMA`                |
| `'~'`   | `TILDE`         |     | `':'`   | `COLON`                |
| `'<<'`  | `LEFT_SHIFT`    |     | `'.'`   | `DOT`                  |
| `'>>'`  | `RIGHT_SHIFT`   |     | `'->'`  | `ARROW`                |
|         |                 |     | `'...'` | `ELLIPSIS`             |

---

## 6. Lexer: one change, nothing added

The layout-free design reduces the lexer work to a **single edit** in
`newlexer.l`:

```
\n    { return NEWLINE; }      →      \n    { /* skip, like spaces */ }
```

(and `NEWLINE` disappears from the token set). Everything that the earlier
layout-aware design would have required is now unnecessary:

- **No indentation stack** — there are no INDENT/DEDENT tokens to compute.
- **No bracket-depth counter** — multi-line calls and decorators like
  `@triton.autotune(\n configs=[...],\n key=["n"],\n)` work automatically,
  because the line breaks inside them were never visible to the parser anyway.
- **No blank-line/comment-line handling** — `#...` is already skipped; blank
  lines now produce nothing by construction.

**Triple-quoted strings are handled** (added after corpus testing, see
CORPUS_REPORT.md): two exclusive flex start conditions (`%x TRIPLE_D
TRIPLE_S`) consume `"""..."""` / `'''...'''` of any length — newlines
included — and emit a single `STRING` token; an unterminated docstring at EOF
emits `ERROR`. This needs no grammar change: a docstring is just an
expression statement whose expression is one `STRING`.

*Remaining optional gap fixes (independent of the grammar, single tokens
only):* hex/binary integers (`0xFF` lexes as `INTEGER(0) NAME(xFF)`),
underscores in numbers (`1_000`), string prefixes (`f"..."`, `r"..."`),
missing `@=`.

---

## 7. The grammar — flat program layer

### 7.1 Start symbol and elements

```
S -> program $

program -> element program
        -> ε

element -> header
        -> stmt
```

*Explanation.* A module is a **flat sequence of elements**. An element is
either a *header* — the part of a compound statement up to and including its
`:` — or an ordinary statement. There is no `suite`/block nonterminal: with
layout erased, nesting is not recoverable, so the grammar does not pretend to
recover it (§1, §13). Dispatch is one token: lookahead ∈ { `IF`, `ELIF`,
`ELSE`, `WHILE`, `FOR`, `DEF`, `'@'` } → `header`; any other
statement-starter → `stmt`. The list is right-recursive per §3.1; `program`'s
ε is taken on lookahead `$`.

### 7.2 Headers

```
header -> IF test ':'
       -> ELIF test ':'
       -> ELSE ':'
       -> WHILE test ':'
       -> FOR exprlist IN testlist ':'
       -> funcheader
       -> decorator

funcheader -> DEF NAME parameters ret_annot ':'

ret_annot -> '->' test
          -> ε

decorator -> '@' dotted_name deco_args

deco_args -> '(' call_args ')'
          -> ε
```

*Explanation.* Each header fully validates its own syntax — `if` must be
followed by a well-formed expression and a `:`, `def` by a name and a legal
parameter list, a decorator by a dotted name with optional call arguments.
What headers no longer do is *own* the statements after them: in

```python
if pid >= n: return
```

the parser sees two elements (`IF test ':'`, then `return`) — the same two
elements it would see if the `return` were on the next line, indented. `ELSE
':'` and `ELIF test ':'` are standalone headers for the same reason: nothing
in a layout-free stream ties them to their `if`, so the grammar accepts them
wherever they appear and a later semantic pass checks pairing.

*Why `for` uses `exprlist`, not `testlist`* (unchanged from the layout-aware
design, and still essential): `IN` is also a comparison operator (`x in lst`,
§10.2). If the loop target were a full expression, the comparison layer would
greedily consume the `IN` of `for i in range(n):` as a binary operator and the
header would never find its `IN`. Loop targets therefore use **`exprlist`**,
whose ladder tops out at the bitwise-or level — *below* comparisons — so every
`_tail` inside the target takes ε when `IN` appears. Targets like `i`, `i, j`,
`a[k]`, `a.b` all still work.

### 7.3 Parameters

```
parameters -> '(' params ')'

params -> paramlist
       -> ε

paramlist      -> param paramlist_tail
paramlist_tail -> ',' param_after_comma
               -> ε

param_after_comma -> param paramlist_tail
                  -> ε                        (trailing comma: (a, b,))

param -> NAME param_tail

param_tail -> ':' test param_default
           -> '=' test
           -> ε

param_default -> '=' test
              -> ε
```

*Explanation.* Matches Triton signatures exactly:

```python
def kernel(x_ptr,                  # param_tail → ε
           n,
           eps=1e-6,               # param_tail → '=' test
           BLOCK: tl.constexpr,    # param_tail → ':' test, default ε
           N2: tl.constexpr = 128) # param_tail → ':' test '=' test
```

`param_tail` is left factoring (§3.2): every parameter starts with `NAME`; one
lookahead (`:`, `=`, or `,`/`)` → ε) picks the variant. Inside the
parentheses all decisions are strictly LL(1) — boundaries are delimited by
`,` and `)`, so none of the §11.3 greediness applies here.

---

## 8. Statements

```
stmt -> testlist assign_tail          (expression statement / assignment)
     -> return_stmt
     -> PASS
     -> BREAK
     -> CONTINUE
     -> import_stmt
     -> assert_stmt
     -> raise_stmt
     -> global_stmt
     -> nonlocal_stmt
     -> del_stmt
```

Every alternative except the first begins with its own keyword — trivially
disjoint FIRST sets. The first alternative (the *expression statement*)
catches everything that starts like an expression (`NAME`, literals, `(`,
`[`, `{`, `+`, `-`, `~`, `not`).

Two inlining simplifications are applied here, both meaning-preserving:
`PASS`/`BREAK`/`CONTINUE` sit directly in the alternation (a unit production
like `pass_stmt -> PASS` adds a nonterminal and a reduction step without
changing the language), and the former `expr_stmt -> testlist assign_tail`
is substituted into its only use site (inlining a single-production,
single-use nonterminal is pure substitution).

### 8.1 Expression statements and assignments — left factoring at work

```
assign_tail -> '=' testlist eq_chain_tail
            -> augassign testlist
            -> ε

eq_chain_tail -> '=' testlist eq_chain_tail
              -> ε

augassign -> '+='  | '-='  | '*='  | '/='  | '//=' | '%='  | '**='
          | '&='  | '|='  | '^='  | '<<=' | '>>='
```

*Explanation — two ideas packed together.*

**(a) The targets-as-expressions trick.** The parser cannot know whether

```python
x[i + 1]            # expression statement?
x[i + 1] = v        # ...or assignment target?
```

until it reaches the `=` — long after `x[i + 1]` has been consumed. So there
is no separate `target` nonterminal: **both sides parse as `testlist`**,
sharing the prefix, and `assign_tail` decides afterwards with one lookahead
token. Whether the left side is a *valid* target (`NAME`, `a.b`, `a[i]`,
tuples of those) is a semantic check. CPython and the previous
`parser/triton_parser.y` do exactly this.

**(b) Chains.** `eq_chain_tail` allows `a = b = c` (right recursion matching
Python's right-to-left assignment). Augmented assignments deliberately do not
chain — `a = b += c` is rejected, as in Python.

How does an expression statement *end*, with no terminator token? By inability to
continue: after the right-hand side, the next token is the first token of the
next element (a `NAME`, a keyword, `'@'`, `$`, …), and since no expression
rule can consume it, every `_tail` collapses by ε and control returns to
`program`. Two adjacent NAMEs never join (Python has no juxtaposition
operator), which is what makes this work for real kernel code. The cases
where the next statement's first token *could* continue the expression are
the maximal-munch points of §11.3.

### 8.2 Control-flow one-liners

```
return_stmt -> RETURN return_tail
return_tail -> testlist
            -> ε

assert_stmt -> ASSERT test assert_tail
assert_tail -> ',' test
            -> ε

raise_stmt -> RAISE raise_tail
raise_tail -> test
           -> ε

global_stmt   -> GLOBAL name_list
nonlocal_stmt -> NONLOCAL name_list
del_stmt      -> DEL exprlist

name_list      -> NAME name_list_tail
name_list_tail -> ',' NAME name_list_tail
               -> ε
```

*Explanation.* `return_tail`/`raise_tail` are the **one genuinely lossy spot**
of the layout-free design: with no terminator, `RETURN` followed by a token
that can start an expression is always read as `return <value>` (maximal
munch). `return acc` and bare `return` before a keyword both parse fine; the
failing pattern is bare `return` directly followed by an expression-starting
statement — see §13 for the concrete example and the mitigation.

### 8.3 Imports

```
import_stmt -> IMPORT dotted_as_names
            -> FROM dotted_name IMPORT import_target

import_target -> '*'
              -> name_as_names

dotted_as_names      -> dotted_as_name dotted_as_names_tail
dotted_as_names_tail -> ',' dotted_as_name dotted_as_names_tail
                     -> ε

dotted_as_name -> dotted_name as_opt

name_as_names      -> name_as_name name_as_names_tail
name_as_names_tail -> ',' name_as_name name_as_names_tail
                   -> ε

name_as_name -> NAME as_opt

as_opt -> AS NAME
       -> ε

dotted_name      -> NAME dotted_name_tail
dotted_name_tail -> '.' NAME dotted_name_tail
                 -> ε
```

*Explanation.* Covers the forms that open every Triton file:

```python
import triton
import triton.language as tl
from triton import language
```

Adjacency works without a terminator: after `import triton`, the lookahead is
the next `IMPORT` keyword, which no import-internal tail can consume — every ε
fires and the next element begins.

---

## 9. (reserved)

Section intentionally collapsed into §7 by the flat design: there are no
suites, blocks, or compound statements as grammar objects anymore. Kept as a
placeholder so §10–§15 numbering stays stable across revisions.

---

## 10. Expressions — the precedence ladder, top-down

The same idea as `E -> T E'`, stacked level by level. **One nonterminal per
precedence level**; each level parses one operand of the next-tighter level,
then its `_tail` loops while the lookahead is one of *its own* operators and
takes ε otherwise. Looser operators sit higher, so they split last. From
**loosest to tightest**:

| #  | Level        | Operators                              | Assoc. | Tail behaviour   |
| -- | ------------ | -------------------------------------- | ------ | ---------------- |
| 1  | `test`       | *(alias of `or_test`; see note)*       | —      | —                |
| 2  | `or_test`    | `or`                                   | left   | loop             |
| 3  | `and_test`   | `and`                                  | left   | loop             |
| 4  | `not_test`   | unary `not`                            | —      | prefix recursion |
| 5  | `comparison` | `< <= > >= == != in not in is is not`  | chains | loop             |
| 6  | `expr`       | `\|`                                   | left   | loop             |
| 7  | `xor_expr`   | `^`                                    | left   | loop             |
| 8  | `and_expr`   | `&`                                    | left   | loop             |
| 9  | `shift_expr` | `<< >>`                                | left   | loop             |
| 10 | `arith_expr` | `+ -`                                  | left   | loop             |
| 11 | `term`       | `* / // % @`                           | left   | loop             |
| 12 | `factor`     | unary `+ - ~`                          | right  | prefix recursion |
| 13 | `power`      | `**`                                   | right  | true recursion   |
| 14 | `atom_expr`  | trailers `f(x) a[i] a.b`               | left   | loop             |
| 15 | `atom`       | literals, names, `()` `[]` `{}`        | —      | dispatch         |

**Note on `test`.** In the layout-aware design `test` carried the conditional
expression (`x if c else y`). That production is now **deferred** (§14): with
flat `IF` headers and no terminator, `x = a` followed by the header `if b:`
would be indistinguishable from a conditional expression's beginning, and the
maximal-munch policy would mis-read it and then fail on the `:`. Real kernels
use `tl.where(cond, a, b)` for selection, so the loss is minimal. `test`
survives as a name (the official "full expression" entry point used by
parameters, arguments, slices, dicts) so that re-adding `lambda`/conditionals
later is purely additive:

```
test -> or_test
```

### 10.1 Expression lists (tuples without parentheses)

```
testlist      -> test testlist_tail
testlist_tail -> ',' testlist_after_comma
              -> ε
testlist_after_comma -> test testlist_tail
                     -> ε                      (trailing comma: (x,) / a, = f())

exprlist      -> expr exprlist_tail
exprlist_tail -> ',' exprlist_after_comma
              -> ε
exprlist_after_comma -> expr exprlist_tail
                     -> ε
```

*Explanation.* `a, b = b, a` and `return x, y` are unparenthesized tuples. The
`after_comma` split (left factoring) supports the trailing comma. `exprlist`
is the comparison-free twin used by `for` targets and `del` (§7.2).

### 10.2 The ladder itself

```
or_test -> and_test or_tail
or_tail -> OR and_test or_tail
        -> ε

and_test -> not_test and_tail
and_tail -> AND not_test and_tail
         -> ε

not_test -> NOT not_test
         -> comparison

comparison -> expr comp_tail
comp_tail  -> comp_op expr comp_tail
           -> ε

comp_op -> '<' | '<=' | '>' | '>=' | '==' | '!='
        -> IN
        -> NOT IN
        -> IS is_not_opt

is_not_opt -> NOT
           -> ε

expr       -> xor_expr bitor_tail
bitor_tail -> '|' xor_expr bitor_tail
           -> ε

xor_expr -> and_expr xor_tail
xor_tail -> '^' and_expr xor_tail
         -> ε

and_expr    -> shift_expr bitand_tail
bitand_tail -> '&' shift_expr bitand_tail
            -> ε

shift_expr -> arith_expr shift_tail
shift_tail -> '<<' arith_expr shift_tail
           -> '>>' arith_expr shift_tail
           -> ε

arith_expr -> term arith_tail
arith_tail -> '+' term arith_tail
           -> '-' term arith_tail
           -> ε

term      -> factor term_tail
term_tail -> '*'  factor term_tail
          -> '/'  factor term_tail
          -> '//' factor term_tail
          -> '%'  factor term_tail
          -> '@'  factor term_tail
          -> ε

factor -> '+' factor
       -> '-' factor
       -> '~' factor
       -> power

power      -> atom_expr power_tail
power_tail -> '**' factor
           -> ε
```

*Explanations, level by level:*

- **Comparison chains.** The `comp_tail` loop accepts `0 <= i < n`. Python's
  "pairwise `and`" meaning is a semantic-phase concern; recognition is just
  the chain. The two-word operators are two adjacent tokens. `NOT IN` causes
  no clash with unary `not` *inside an expression*: in `comp_tail` position we
  are after a complete operand, where unary `not` is impossible. `IS NOT` is
  left-factored into `is_not_opt`, whose ε is safe because no operand at this
  level (`expr`) can begin with `NOT`.
- **`@` as matmul** lives in `term_tail` (same precedence as `*`), used in
  attention kernels (`p @ v`). Its statement-boundary interaction with
  decorators is a §11.3 maximal-munch point.
- **`factor` / `power` asymmetry — Python's odd `**` rule, encoded
  structurally.** `**` binds tighter than unary minus on its *left* but looser
  on its *right*: `-x**2 == -(x**2)` and `2**-1 == 0.5`. Both fall out:
  `factor -> '-' factor` descends into `power`, whose `power_tail` grabs `**`
  first; and `power_tail -> '**' factor` lets the right operand start with
  `-`. Right associativity of `**` (`2**3**2 == 2**512`) also follows: the
  right side recurses at `factor` level, the left side is pinned to
  `atom_expr`.

### 10.3 Atoms and trailers

```
atom_expr -> atom trailer_list

trailer_list -> trailer trailer_list
             -> ε

trailer -> '(' call_args ')'
        -> '[' subscriptlist ']'
        -> '.' NAME

call_args -> arglist
          -> ε

atom -> NAME
     -> INTEGER
     -> FLOAT
     -> strings
     -> TRUE
     -> FALSE
     -> NONE
     -> '...'
     -> '(' paren_body ')'
     -> '[' list_body ']'
     -> '{' dict_body '}'

paren_body -> testlist
           -> ε

list_body -> testlist
          -> ε

dict_body -> dict_items
          -> ε

strings      -> STRING strings_tail
strings_tail -> STRING strings_tail
             -> ε

dict_items      -> dict_item dict_items_tail
dict_items_tail -> ',' dict_items_after_comma
                -> ε
dict_items_after_comma -> dict_item dict_items_tail
                       -> ε

dict_item -> test ':' test
```

*Explanation.* `trailer_list` is a loop on lookahead ∈ { `'('`, `'['`, `'.'` }
— this single loop parses all of:

```python
tl.program_id(0)                  # NAME .NAME (args)
x.to(tl.float32)                  # call result, then more trailers
add_kernel[grid](x, y, out, n)    # Triton launch: NAME [subscript] (args)
w[None, :].to(tl.float32)         # subscript, attribute, call — any order
```

`atom` is a pure one-token dispatch — every alternative starts with a distinct
token. `strings` chains adjacent literals (`"a" "b"` implicit concatenation).
`'(' paren_body ')'` is grouping *and* tuple display (with §10.1's trailing
comma, `(x,)` works). `{...}` covers dict displays, which appear in autotune
configs: `triton.Config({"BLOCK": 64}, num_warps=4)`.

### 10.4 Call arguments

```
arglist      -> argument arglist_tail
arglist_tail -> ',' arglist_after_comma
             -> ε
arglist_after_comma -> argument arglist_tail
                    -> ε

argument   -> test kwarg_tail
kwarg_tail -> '=' test
           -> ε
```

*Explanation.* Positional and keyword arguments:
`tl.load(p + offs, mask=m, other=0.0)`. Note it is **not**
`argument -> NAME '=' test | test` — both alternatives can start with `NAME`,
an unfixable FIRST/FIRST conflict for LL(1). Instead, the same trick as
assignments (§8.1): parse a `test`, then `kwarg_tail` peeks for `=` (inside
parentheses `=` can mean nothing else). "Keyword name must be a bare NAME" is
a semantic check. Starred args (`*a`, `**kw`) are deferred (§14).

### 10.5 Subscripts and slices

```
subscriptlist -> subscript subs_tail
subs_tail     -> ',' subs_after_comma
              -> ε
subs_after_comma -> subscript subs_tail
                 -> ε

subscript -> test subscript_tail
          -> ':' slice_upper

subscript_tail -> ':' slice_upper
               -> ε                      (plain index: x[i])

slice_upper -> test slice_step_opt       (upper bound present)
            -> ':' step_opt              (straight to step: x[a::c], x[::c])
            -> ε                         (open upper bound: x[a:], x[:])

slice_step_opt -> ':' step_opt
               -> ε

step_opt -> test
         -> ε
```

*Explanation.* Beyond plain indexing, Triton kernels lean hard on NumPy-style
slicing for broadcasting — all ten slice shapes reduce to this factored form:

```python
offs[:, None]        # subscript ':' → slice_upper ε ; then ',' ; then test None
a[1:2]               # test ':' test
b[::2]               # ':' ':' test
c[..., 0]            # '...' atom, then ',' test
```

Each decision is one token; inside brackets everything is delimited by `,`,
`:`, `]`, so this region is strictly LL(1) with no maximal-munch involvement.

---

## 11. FIRST/FOLLOW, the predictive table, and the boundary policy

### 11.1 Worked out fully on the toy grammar

```
E  -> T E'        E' -> + T E' | ε
T  -> F T'        T' -> * F T' | ε
F  -> ( E ) | id
```

FIRST sets:

| X             | FIRST(X)  |
| ------------- | --------- |
| `E`, `T`, `F` | `(`, `id` |
| `E'`          | `+`, ε    |
| `T'`          | `*`, ε    |

FOLLOW sets:

| X    | FOLLOW(X)          | why                                            |
| ---- | ------------------ | ---------------------------------------------- |
| `E`  | `)`, `$`           | start symbol; appears inside `( E )`           |
| `E'` | `)`, `$`           | tail of `E`                                    |
| `T`  | `+`, `)`, `$`      | followed by `E'`, which starts `+` or vanishes |
| `T'` | `+`, `)`, `$`      | tail of `T`                                    |
| `F`  | `*`, `+`, `)`, `$` | followed by `T'`, which starts `*` or vanishes |

Predictive table `M` (rows = nonterminals, columns = lookahead):

|      | `id`     | `(`      | `+`        | `*`        | `)`     | `$`     |
| ---- | -------- | -------- | ---------- | ---------- | ------- | ------- |
| `E`  | `E->TE'` | `E->TE'` | —          | —          | —       | —       |
| `E'` | —        | —        | `E'->+TE'` | —          | `E'->ε` | `E'->ε` |
| `T`  | `T->FT'` | `T->FT'` | —          | —          | —       | —       |
| `T'` | —        | —        | `T'->ε`    | `T'->*FT'` | `T'->ε` | `T'->ε` |
| `F`  | `F->id`  | `F->(E)` | —          | —          | —       | —       |

No cell holds two productions → the grammar is LL(1). **Precedence lives in the
`T'` row**: on `*` it expands (multiplication binds), on `+` it produces ε
(returns control to the addition level). Parse of `id + id * id` (stack machine,
top of stack at left):

```
stack            input            action
E $              id + id * id $   M[E,id]:  E -> T E'
T E' $           id + id * id $   M[T,id]:  T -> F T'
F T' E' $        id + id * id $   M[F,id]:  F -> id
id T' E' $       id + id * id $   match id
T' E' $          + id * id $      M[T',+]:  T' -> ε
E' $             + id * id $      M[E',+]:  E' -> + T E'
+ T E' $         + id * id $      match +
T E' $           id * id $        T -> F T' ;  F -> id ;  match id
T' E' $          * id $           M[T',*]:  T' -> * F T'     ← precedence!
* F T' E' $      * id $           match * ;  F -> id ;  match id
T' E' $          $                M[T',$]:  T' -> ε
E' $             $                M[E',$]:  E' -> ε
$                $                accept
```

The same trace *is* the recursive-descent call structure: every "expand" is a
function call, every ε is a function returning.

**And the same grammar, parsed bottom-up — what yacc will actually do.**
The grammar's *form* is top-down; the *machine* yacc builds from it is
shift/reduce: it never predicts, it consumes tokens onto a stack and reduces
when a complete right-hand side is on top (a rightmost derivation in reverse —
tree built leaves-first instead of root-first). Same input `id + id * id`:

```
stack            input            action
                 id + id * id $   shift id
id               + id * id $      reduce F -> id
F                + id * id $      lookahead '+': reduce T' -> ε
F T'             + id * id $      reduce T -> F T'
T                + id * id $      shift +
T +              id * id $        shift id ; reduce F -> id
T + F            * id $           lookahead '*': SHIFT *        ← precedence!
T + F *          id $             shift id ; reduce F -> id
T + F * F        $                lookahead '$': reduce T' -> ε
T + F * F T'     $                reduce T' -> * F T'
T + F T'         $                reduce T -> F T'
T + T            $                reduce E' -> ε
T + T E'         $                reduce E' -> + T E'
T E'             $                reduce E -> T E'
E                $                accept
```

Compare row by row with the predictive trace above: both machines spend their
one lookahead token at the **same decision points** — where the top-down parser
*predicts* `T' -> ε` on `+`, the bottom-up parser *reduces* `T' -> ε` on `+`;
where the predictive table says `T' -> * F T'` on `*`, yacc shifts the `*`.
That coincidence is the intuition behind Beatty's theorem (§4): an LL(1)
grammar hands LALR(1) exactly the information it needs. The parse tree at the
end is identical — only the order of its construction differs (top-down: `E`
created first; bottom-up: `F -> id` reduced first).

### 11.2 The decision table of the real grammar

Building the full table by hand is busywork — yacc re-derives it mechanically
and reports any violation as a conflict in `y.output` (§15). What matters at
design time is checking the LL(1) conditions at every decision point. The
load-bearing ones:

| Nonterminal               | lookahead → choice                                                                          |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| `element`                 | `IF ELIF ELSE WHILE FOR DEF '@'` → header; expression-starters & stmt keywords → stmt       |
| `program`                 | element-starter → another element · `$` → ε                                                 |
| `assign_tail`             | `'='` → assignment · augassign op → augmented · anything else → ε                           |
| `param_tail`              | `':'` → annotation · `'='` → default · `',' ')'` → ε                                        |
| `kwarg_tail`              | `'='` → keyword arg · `',' ')'` → ε                                                         |
| `subscript`               | `':'` → slice from empty lower · test-starter → index-or-slice, decided by `subscript_tail` |
| `as_opt` / `ret_annot`    | `AS` / `'->'` → take it · anything else → ε                                                 |
| every operator `_tail`    | own operator tokens → loop · anything else → ε                                              |

Inside delimited regions — parentheses, brackets, braces, header keywords up
to their `:` — both LL(1) conditions hold strictly: the closers `) ] } :` and
separators `, :` never start expressions, so every ε is unambiguous.

### 11.3 The boundary policy: maximal munch

At the **statement boundary** the FOLLOW set of an ending element is
FIRST(element) itself — and a handful of tokens live on both sides: they can
*continue* the current expression and *start* a new statement. One lookahead
token cannot tell those apart, so the design adopts a fixed policy — **always
continue (maximal munch)** — which in yacc is precisely the default
resolution of a shift/reduce conflict (shift), the same default that handles
C's dangling `else`. Every such point, enumerated:

| Decision point             | shared token(s)        | maximal munch reads…                  | impact on real kernels                                       |
| -------------------------- | ---------------------- | ------------------------------------- | ------------------------------------------------------------ |
| `trailer_list` ε           | `'('` `'['`            | call / subscript of previous expr     | none — statements don't start with `(`/`[` in the corpus     |
| `arith_tail` ε, `factor`   | `'+'` `'-'`            | binary plus/minus                     | none — nobody writes `-x` as a standalone statement          |
| `term_tail` ε              | `'@'`                  | matmul instead of next decorator      | stream still accepted; tree merges (recognizer-OK)           |
| `comp_tail` ε              | `NOT`                  | start of `not in`                     | none — `not x` as a statement doesn't occur                  |
| `strings_tail` ε           | `STRING`               | implicit concatenation                | none — adjacent string statements don't occur                |
| `deco_args` ε              | `'('`                  | decorator arguments                   | correct reading anyway                                       |
| `testlist_after_comma` / `exprlist_after_comma` ε | expression starters | another tuple member after a trailing comma | none — `x = a,` followed by an expression statement doesn't occur |
| `return_tail` / `raise_tail` | expression starters  | return/raise **with** a value         | **the one real limitation — see §13**                        |

Acceptance criteria for `yacc -v` follow directly (§4): 0 reduce/reduce
conflicts; every shift/reduce conflict must correspond to a row of this table;
anything else is a bug in the grammar.

*Verified against the built parser:* `yacc -d -v newparser.y` reports **0
reduce/reduce** and 80 shift/reduce conflicts spread over 20 automaton states
— every one of them traced in `y.output` to exactly the rows above
(`return_tail`/`raise_tail` and the two `*_after_comma` rules account for the
big counts, ~15 expression-starter tokens each; the rest are the single-token
rows: `( [ + - @ NOT STRING`, with `'@'` repeated once per `term_tail`
continuation state).

---

## 12. Worked example — leftmost derivation = call trace

For the kernel line pair

```python
mask = offs < n
x = tl.load(x_ptr + offs, mask=mask)
```

statement one, as a leftmost derivation (the lookahead that drives each
decision is noted at right):

```
element ⇒ stmt                                lookahead NAME — not a header keyword
        ⇒ testlist assign_tail                (the expression-statement alternative)
        ⇒* NAME(mask) assign_tail             every _tail under testlist → ε on '='
        ⇒ NAME '=' testlist eq_chain_tail     assign_tail saw '='
        ⇒* NAME '=' expr comp_tail …          descend the ladder to level 5
        ⇒* NAME '=' NAME(offs) comp_tail …    comp_tail sees '<' → comp_op expr
        ⇒* NAME '=' NAME '<' NAME(n) …        second operand
        ⇒* NAME '=' NAME '<' NAME             comp_tail → ε, eq_chain_tail → ε:
                                              lookahead is NAME(x) — the first
                                              token of the NEXT statement, which
                                              no expression rule can consume ✓
```

That last line is the layout-free boundary mechanism in action: the statement
ends not because a terminator token says so, but because nothing can extend it.

Statement two, viewed as a predictive parser's **call trace** (indentation =
call depth):

```
parse_element                     lookahead NAME → stmt (expression statement)
└─ parse_stmt
   ├─ parse_testlist → … → parse_atom            consumes: x
   ├─ assign_tail: lookahead '=' → consume '='
   │  └─ parse_testlist → … → parse_atom_expr
   │     ├─ parse_atom                           consumes: tl
   │     └─ trailer_list:
   │        ├─ sees '.'  → consumes: . load
   │        ├─ sees '('  → parse_call_args
   │        │   ├─ parse_argument
   │        │   │   └─ parse_test → … parse_arith_expr
   │        │   │       ├─ parse_term            consumes: x_ptr
   │        │   │       └─ arith_tail: sees '+'  consumes: + offs ; then ε on ','
   │        │   ├─ sees ',' → parse_argument
   │        │   │   ├─ parse_test → …            consumes: mask
   │        │   │   └─ kwarg_tail: sees '=' →    consumes: = mask
   │        │   └─ sees ')' → ε  ; consume ')'
   │        └─ trailer_list: ε                   (next token starts next element)
   └─ assign_tail/eq_chain_tail: ε → element done ✓
```

Precedence falls out exactly as in §11.1: `x_ptr + offs` is assembled inside
the argument because `arith_tail` loops on `'+'` and every looser level's tail
saw ε first.

---

## 13. Known limitations of the layout-free design

Stated plainly, so they are conscious trade-offs rather than surprises:

1. **Block structure is not verified.** `program` is flat; the parser checks
   that every header and statement is internally well-formed, but not that an
   `else:` has a matching `if:`, that a decorator precedes a `def`, or that a
   header has a body. Those become checks for a later semantic pass (which,
   given line/column info from `yylineno`, can even reconstruct nesting from
   the source indentation — *outside* the grammar).
2. **`return`/`raise` greediness can mis-reject one real pattern.** Maximal
   munch (§11.3) reads `RETURN x` as "return the value `x`". For the common
   guard idiom:

   ```python
   if pid >= n: return
   x = tl.load(...)          # stream: … ':' RETURN NAME '=' …
   ```

   the parser takes `x` as the return value and then fails on `=`. Patterns
   that are fine: `return acc` (real value), bare `return` followed by a
   keyword-starting line (`def`, `import`, `if`, `@…`, EOF). If corpus testing
   shows the guard idiom is frequent, the smallest fix is to reintroduce one
   terminator token — that is a one-line lexer revert plus FOLLOW-set changes,
   not a redesign.
3. **Some accepted programs get a "joined" tree.** A statement starting with
   `(`, `[`, unary `-`/`+`, or a decorator following an
   expression-statement, merges into the previous expression (§11.3 table; a
   leading `~` separates cleanly, since no binary operator uses it).
   The token stream is still *accepted* — only the tree shape is off — which
   is harmless for a recognizer and irrelevant for the corpus (those layouts
   don't occur in TritonBench kernels).
4. **Conditional expressions (`a if c else b`) are deferred** (§10 note): they
   collide with flat `IF` headers. Kernels use `tl.where(...)` instead.

---

## 14. Deferred extensions

Tokens the lexer already produces but the core grammar intentionally does not
use. Each comes with its top-down production sketch so adding it later is
mechanical:

| Feature                            | Sketch                                                                | Why deferred                                                          |
| ---------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------- |
| conditional expression             | `test -> or_test cond_tail`; `cond_tail -> IF or_test ELSE test \| ε` | collides with flat `IF` headers (§13.4); needs a terminator token     |
| `lambda`                           | `test -> lambdef`; `lambdef -> LAMBDA lambda_params ':' test`         | host-side launch code (`grid = lambda meta: ...`), not kernels        |
| `with`                             | `header -> WITH test with_as ':'`; `with_as -> AS expr \| ε`          | not used in kernels                                                   |
| `try/except/finally`               | three more flat headers                                               | not used in kernels                                                   |
| `class`                            | `header -> CLASS NAME class_bases ':'`                                | not used in kernels                                                   |
| `yield` / `await` / `async`        | expression + statement forms                                          | not used in kernels                                                   |
| comprehensions                     | `comp_for` tail inside list/dict displays                             | occasionally in autotune config lists                                 |
| star arguments                     | `argument -> '*' test \| '**' test`; same in params                   | rare in kernels                                                       |
| set display                        | `dict_body -> test …` left-factored against `dict_item`               | rare                                                                  |
| annotated assignment               | extra alternative in `assign_tail` (`':' test '=' …`)                 | rare at module level                                                  |
| `;` separator                      | `element -> ';'` (no-op)                                              | corpus never uses it                                                  |
| walrus `:=`                        | needs a new lexer token first                                         | not in the lexer                                                      |

---

## 15. Next steps (the coding phase)

Implementation tool (decided): **Unix yacc**, fed by the flex lexer.

1. **Lexer edit** (`newlexer.l`): change `\n { return NEWLINE; }` to skip, and
   remove `NEWLINE` from the token set. That is the only required change (§6);
   the remaining gap fixes (hex literals, f-strings) are optional work. The
   hand-written `enum token_type` must be **deleted** and replaced with
   `#include "y.tab.h"` (generated by `yacc -d`), so lexer and parser agree on
   token numbers — yacc assigns its own starting at 257 and reserves 256 for
   its built-in `error` token (the current enum's `ERROR = 256` collides).
2. **`newparser.y`**: transcribe §7–§10 **verbatim** — one yacc rule per
   production, ε as an empty alternative (`/* empty */`), `%token` declarations
   from §5, `%start program`. No `%left`/`%right`/`%prec` declarations: the
   ladder's shape already encodes precedence and associativity. First iteration
   is a pure recognizer — accept/reject plus `yylineno`-based error messages.
3. **Verify**: build with `yacc -d -v newparser.y && flex newlexer.l &&
   cc y.tab.c lex.yy.c -o newparser`, then read `y.output` against the §11.3
   acceptance criteria: **0 reduce/reduce**, and every shift/reduce conflict
   matching a row of the maximal-munch table (default shift = the intended
   policy). Run against the TritonBench corpus the way
   `parser/run_smoke_tests.py` drove the previous parser, plus negative tests
   (`run_negative_tests.py` pattern); watch specifically for the `return`
   guard idiom of §13.2 in corpus failures.
4. Later: `%union`/`yylval` to carry lexemes, then AST construction — at that
   point the `_tail` rules either return op/operand lists folded left by the
   head rule (§3.3), or the expression core mechanically reverts to the
   left-recursive originals of §3.1, which yacc also accepts and which make
   left-associative actions trivial.
