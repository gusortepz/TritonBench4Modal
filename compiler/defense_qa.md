# Defense Q&A — newlexparser (flex + yacc front end for Triton kernel code)

Answers grounded in the actual artifacts of this directory:
`newlexer.l` (flex lexer), `newparser.y` (yacc grammar), `grammar.md` (design
document), `Makefile`, `example_kernel.py`.

Key figures used throughout: **85 declared tokens · 214 grammar rules ·
317 LALR states · 80 shift/reduce conflicts (all intentional) · 0 reduce/reduce
conflicts**.

> **Naming note (read first).** The question sheet says "Tryton". **Tryton** is
> a Python ERP framework (models, fields, workflows); our project targets
> **Triton**, OpenAI's GPU-kernel language embedded in Python (`@triton.jit`,
> `tl.load`, `tl.store`). All answers below are about **Triton**-oriented code;
> where a question is clearly written for the ERP (e.g. Q45), we answer with
> the Triton equivalent and say so.

---

## 1. General Compiler Design

**1. What is the main goal of your compiler?**
To validate, cheaply and deterministically, that a source file is
syntactically well-formed Triton kernel code — without importing Python or
touching a GPU. It is the *front end* (lexical + syntax analysis) of the
TritonBench4Modal pipeline: given a file (e.g. an LLM-generated kernel), it
answers `ACCEPT` or `REJECT` with a line-numbered diagnostic.

**2. What does "Tryton/Triton-oriented code" mean in your project?**
Python written against the Triton GPU API: kernels decorated with
`@triton.jit` (optionally `@triton.autotune(...)`), parameters annotated
`tl.constexpr`, bodies built from `tl.`* calls (`tl.program_id`, `tl.arange`,
`tl.load`, `tl.store`), masks/offset arithmetic, and the launch syntax `kernel grid](args)`.

**3. What kind of input language does your compiler accept?**
A subset of Python — exactly the constructs that real TritonBench kernels use:
imports, decorators, function headers with annotated/default parameters,
assignments (plain, chained, augmented), full expression precedence,
attribute/call/subscript chains, slices, and `if/elif/else`, `for`, `while`
headers. Deliberately excluded: `class`, `try`, `lambda`, comprehensions,
conditional expressions (documented in grammar.md §14).

**4. What kind of output does your compiler generate?**
A verdict: `ACCEPT` (exit 0) or `REJECT` (exit 1), plus, on failure,
`syntax error at line N near 'TOKEN'` on stderr. It is a recognizer; it does
not emit translated code yet.

**5. Is your project a full compiler, an interpreter, or a direct syntax translator? Why?**
None of the three — it is the **front end of a compiler** (a recognizer):
scanner + parser only. We chose that scope because the pipeline's need is
*structural validation* of generated kernels; execution is done by the real
Triton toolchain. The grammar and `.y` skeleton are designed so translation
actions can be attached later (see §5 of this document).

**6. What are the main phases of your compiler?**
(1) **Lexical analysis** — flex turns characters into tokens
(`newlexer.l`). (2) **Syntax analysis** — yacc's LALR(1) parser checks the
token stream against the grammar (`newparser.y`). Planned next: (3) a semantic
pass (symbol table, decorator/header pairing) and (4) output generation
(AST/metadata).

**7. Which phase was the most difficult to implement and why?**
Syntax analysis — specifically deciding **where a statement ends** after we
removed all layout tokens (no NEWLINE/INDENT/DEDENT). Python normally
delimits statements by line structure; without it, "the expression continues"
vs. "a new statement starts" can collide on the same lookahead token. We
solved it with a documented *maximal munch* policy (grammar.md §11.3) that
maps exactly onto yacc's default shift.

**8. What assumptions did you make about the input program?**
That it stays inside the kernel subset (no classes/try/lambda); that block
structure does not need to be *verified* (layout is erased; pairing is a
future semantic pass); that strings are single-line and quoted simply (no
triple-quoted/f-strings yet); that integers are decimal (no hex yet); ASCII
identifiers.

---

## 2. Lexical Analysis

**9. What are the tokens of your language?**
85 declared in `newparser.y`, in groups: literals/names (`NAME INTEGER FLOAT STRING`); 26 active keywords (`DEF RETURN IF ELIF ELSE FOR WHILE IN AND OR NOT IS TRUE FALSE NONE IMPORT FROM AS PASS RAISE ASSERT BREAK CONTINUE GLOBAL NONLOCAL DEL`); 9 reserved-but-unused keywords (`LAMBDA TRY EXCEPT FINALLY WITH CLASS YIELD ASYNC AWAIT`); 8 arithmetic operators incl. `*`*, `//`, `@`;
6 comparisons; 6 bitwise; `=` plus 12 augmented assignments; 12 delimiters
(`( ) [ ] { } , : . ; -> ...`); and `ERROR` for unmatchable characters.

**10. Which tokens are specific to Triton-oriented code?**
Strictly speaking none — Triton is *embedded in Python*, so Triton-ness lives
in names (`triton`, `tl`), not in new token types. The tokens that *carry*
the Triton idioms are `AT` (decorators `@triton.jit`), `COLON` in parameter
position (`BLOCK: tl.constexpr`), `ARROW` (return annotations), and the
bracket tokens that form the launch syntax `kernel[grid](args)`.

**11. How do you distinguish between identifiers, keywords, and reserved words?**
By flex's two matching rules: **longest match first**, and **earliest rule
wins on ties**. Every keyword is written as a literal pattern *before* the
identifier rule `{LETTER}({LETTER}|{DIGIT})`*, so `def` matches the keyword
rule, while `define` (longer) matches NAME. All keywords are reserved: they
can never be returned as NAME. The nine deferred keywords are still lexed as
keywords, but no grammar rule uses them, so using one is a guaranteed syntax
error.

**12. What lexical errors can your compiler detect?**
Any character that no pattern matches — e.g. `$`, `?`, a lone `!`, a lone
backslash — is returned as the `ERROR` token. Malformed numbers like `3.` are
fine (FLOAT), but an unterminated string on one line fails the string pattern
and its quote character becomes `ERROR`.

**13. What happens if the input contains an unknown symbol?**
The lexer returns `ERROR`; since no grammar rule mentions `ERROR`, the parser
immediately reports `syntax error at line N near '$'` and rejects. We verified
this: `x = 3 $ 4` → REJECT at line 1 near `'$'`.

**14. How do you handle comments, spaces, and new lines?**
All invisible to the parser. `"#"[^\n]`* skips comments; `[ \t\r\n]+` skips
all whitespace **including newlines** — this is the project's signature
"layout-free" decision: the parser never sees line structure, which is why
multi-line `@triton.autotune( ... )` works with zero special handling.

**15. Can you show an example of input code and the tokens generated from it?**
`mask = offs < n` →
`NAME(mask) ASSIGN NAME(offs) LESS NAME(n)`.
And `x = tl.load(x_ptr + offs, mask=mask)` →
`NAME ASSIGN NAME DOT NAME OPEN_PARENTHESIS NAME PLUS NAME COMMA NAME ASSIGN NAME CLOSE_PARENTHESIS`.

**16. Which tool did you use for lexical analysis?**
**Flex** (the Unix lex successor), with `%option noyywrap yylineno noinput nounput`. Token numbers are not hand-defined: the lexer does
`#include "y.tab.h"`, generated by `yacc -d`, so scanner and parser always
agree.

---

## 3. Syntax Analysis and Grammar

**17. What grammar did you define for your language?**
A context-free grammar G = (N, Σ, P, S) written in **top-down (LL(1)) form**:
~65 nonterminals, 85 terminals, 214 rules, start symbol `program`. A program
is a *flat sequence of elements*; an element is either a block **header**
(`if test:`, `def name(params):`, `@decorator`, …) or a **statement**;
expressions form a 14-level precedence ladder from `or_test` down to `atom`.
The full grammar with explanations is `grammar.md` §7–§10; `newparser.y`
transcribes it rule-for-rule.

**18. Is your grammar ambiguous? How do you know?**
Inside every construct (expressions, headers, parameter lists, slices): no —
yacc builds the tables with **0 reduce/reduce conflicts**, and each decision
is driven by one distinct lookahead token. At **statement boundaries** the
underlying CFG *is* ambiguous, and we know precisely why: we removed the
statement terminator, so a string like `x = a (b)` derives both as one
statement (a call) and as two. The evidence is the **80 shift/reduce
conflicts** yacc reports; we made the parser deterministic by adopting the
*maximal munch* policy (always continue the expression), which is yacc's
default shift — the same mechanism textbooks use for C's dangling `else`.
Every conflict was audited against the table in grammar.md §11.3.

**19. Did you have left recursion in your grammar? How did you remove it?**
The natural grammar is left-recursive (`E -> E + T | T`). Because the course
methodology is top-down, we removed it mechanically with
`A -> Aα | β  ⇒  A -> β A' ; A' -> α A' | ε`, producing the `_tail`
nonterminals (`arith_expr -> term arith_tail`, `arith_tail -> PLUS term arith_tail | MINUS term arith_tail | ε`). Honest caveat we can defend: yacc
itself (LALR) handles left recursion fine — the removal demonstrates the
top-down design; the LALR machine accepts either form.

**20. What parsing strategy did you use?**
Two-layer answer: the grammar is **designed top-down** (no left recursion,
left-factored, one-token decisions — LL(1) within constructs), and it is
**executed bottom-up** by yacc's **LALR(1)** shift/reduce automaton (317
states). That mix is legal because every LL(1) grammar is also LALR(1)
(Beatty, 1982); grammar.md §11.1 shows the same toy grammar traced by both
machines.

**21. Did you use LL, LR, LALR, recursive descent, Yacc/Bison, or another method?**
**Unix yacc** (Berkeley yacc on macOS; `bison -y` builds identically), which
generates an LALR(1) parser. No `%left/%right/%prec` declarations were needed
— precedence and associativity are encoded in the grammar ladder itself.

**22. What syntax errors can your compiler detect?**
Malformed expressions (`x = = 3`), unbalanced delimiters (`x = a[1`), broken
headers (`def f(:`, `for in x:`), invalid parameter lists, misuse of reserved
keywords (`lambda`, `class`, …), stray operators, lexically illegal symbols,
truncated files.

**23. What happens if the user forgets a semicolon, parenthesis, or block delimiter?**
Semicolons don't exist in the language (statements end by structure), so
nothing to forget. A missing closing parenthesis/bracket is caught when the
parser hits the next token (or EOF) that cannot continue the open construct —
e.g. `x = a[1` → `syntax error at line 2 near ''` (EOF). Block delimiters
don't exist by design (layout-free): the parser cannot detect a missing
"end of block", which is the documented trade-off (grammar.md §13.1).

**24. Can you explain one production rule from your grammar?**
`for_stmt` header: `FOR exprlist IN testlist COLON`. The loop *target* is
`exprlist` — capped at the bitwise-or precedence level — instead of a full
expression, because `in` is *also* a comparison operator. If the target were
`testlist`, parsing `for i in range(n):` would greedily consume `in` as a
binary operator inside the target and the header's `IN` would never be found.
Capping the target below the comparison level removes the conflict — the same
solution CPython's grammar uses.

**25. Can you show how one input sentence is parsed using your grammar?**
`mask = offs < n` (leftmost derivation, grammar.md §12):
`element ⇒ stmt ⇒ testlist assign_tail ⇒* NAME assign_tail ⇒ NAME ASSIGN testlist eq_chain_tail ⇒* NAME ASSIGN comparison ⇒ NAME ASSIGN expr comp_tail ⇒* NAME ASSIGN NAME LESS NAME` — and every
`_tail` then takes ε because the next token (the first token of the next
statement) cannot continue an expression. That ε-collapse *is* the
statement-boundary mechanism.

**26. Did you have shift/reduce or reduce/reduce conflicts? How did you solve them?**
**80 shift/reduce, 0 reduce/reduce.** All 80 are statement-boundary cases and
all are *resolved by yacc's default shift, which is exactly the policy we
designed* (maximal munch). We verified by reading `y.output`: state 6 is
`RETURN . return_tail` (15 lookaheads — return greediness), state 62 is
`term . arith_tail` on `+/-`, state 94 is `AT dotted_name . deco_args` on
`(`. Acceptance criterion in grammar.md §11.3: any conflict *not* in that
table, or any reduce/reduce, is a grammar bug.

---

## 4. Semantic Analysis

**27. Does your compiler check semantic errors, or only syntax errors?**
Only lexical and syntax errors today. Semantics (declaration checking, header
pairing, types) is the designed next phase, deliberately out of scope for the
recognizer milestone.

**28. Do you use a symbol table?**
Two-level answer. **Lexical level: yes** — the standalone scanner deliverable
(`newscanner`, driver `scanner_main.c`) builds one while scanning: one entry
per distinct lexeme of the symbol-bearing tokens (`NAME`, `INTEGER`, `FLOAT`,
`STRING`) with id, token class, lexeme, first line, and occurrence count,
printed after the token sequence. Note the lexer proper (`yylex`) does not
build it — its output is only the token stream; the table lives in the driver
loop around it. **Semantic level: not yet** — the scoped table (kinds, types,
declarations) is the planned next phase, inserted at `funcheader`, `param`,
and assignment reductions.

**29. What information would you store in the symbol table?**
Name; kind (function, parameter, local, imported module/alias); whether the
parameter is `tl.constexpr`; the decorator state of the enclosing function
(`@triton.jit` or not); first-definition line for error messages.

**30. How do you detect undeclared variables?**
(Planned.) On every NAME reduced inside an expression, look it up through the
scope chain; not found → "undeclared name at line N". Needs `yylval` to carry
the lexeme (step 4 of grammar.md §15).

**31. How do you detect duplicated declarations?**
(Planned.) Insertion into the table fails if the name already exists in the
*same* scope — duplicate parameter names being the clearest kernel case.

**32. Do you check data types?**
No. Triton typing is mostly dynamic at this level; the realistic static checks
are constexpr-ness (`BLOCK: tl.constexpr` used where compile-time constants
are required) and dtype arguments (`tl.float32`), both semantic-pass work.

**33. What happens if the user writes a syntactically correct program that does not make sense semantically?**
It is ACCEPTed. Examples we can demo: `else:` with no `if` before it (legal
header in the flat grammar), using a variable never assigned, or
`x = undefined_fn(y)`. This is the textbook syntax/semantics boundary — the
recognizer checks *form*, not *meaning*.

**34. Which semantic rules are specific to Triton-oriented code?**
(For the future pass.) A `@triton.jit` decorator must be immediately followed
by a `def`; kernels should `return` bare (no value); `tl.constexpr`
parameters must receive compile-time constants at launch; `tl.load/tl.store`
keyword arguments are restricted (`mask=`, `other=`); the launch
`kernel[grid](...)` must reference a jit-decorated function.

---

## 5. Direct Syntax Translation

**35. What is a direct syntax translator?**
A program that produces its output *during parsing*, by executing semantic
actions attached to grammar productions (syntax-directed translation), with
no separate optimization or full intermediate representation.

**36. How is a direct syntax translator different from a traditional compiler?**
A traditional compiler builds intermediate structures (AST, IR), runs
analysis/optimization passes over them, then generates code. A direct
translator emits output the moment each production reduces — simpler and
faster, but it cannot use information that appears later in the input.

**37. Where do you attach translation actions in your grammar?**
In `newparser.y`, inside `{ ... }` at the end (or middle) of any rule. The
natural attachment points here: `funcheader` (a kernel was declared),
`decorator` (jit/autotune detected), `param_tail`'s `COLON test` alternative
(a constexpr annotation), and `trailer` (a `tl.`* call or a launch).

**38. Do you generate the output during parsing or after building an intermediate representation?**
Currently neither (recognizer). The architecture is staged for *during
parsing* first (printing/collecting metadata in actions), with an AST as the
later option once `%union`/`yylval` carry lexemes (grammar.md §15 step 4).

**39. What are your semantic actions?**
Today: none in the rules — the only outputs are `ACCEPT`/`REJECT` in `main`
and the message in `yyerror`. That is deliberate: a clean recognizer first,
actions second.

**40. Can you show one grammar rule and the translation action associated with it?**
The first action we would add:

```yacc
funcheader
    : DEF NAME parameters ret_annot COLON
          { printf("kernel header: %s (line %d)\n", $2, yylineno); }
    ;
```

(Requires `%union { char *str; }`, `%token <str> NAME`, and the lexer setting
`yylval.str = strdup(yytext)` for NAME.)

**41. How do you guarantee that the generated output preserves the meaning of the input?**
By construction plus testing: each production maps 1:1 to one source
construct, so its action sees exactly that construct; and we validate against
a corpus — positive files must round-trip/report correctly, negative files
must be rejected (the `run_smoke_tests.py` / `run_negative_tests.py` pattern
already used by the older `parser/` pipeline).

**42. What happens if the input is valid but cannot be translated?**
Then it is valid Python-subset but outside the Triton idiom (e.g. no
`@triton.jit` function in the file). The recognizer accepts it; the
translation/semantic layer is the right place to report "no Triton kernel
found" as a diagnostic rather than a syntax error.

---

## 6. Triton-Oriented Code Detection

**43. What features of Triton are you trying to detect?**
Kernel definitions (`@triton.jit` + `def`), autotuning configuration
(`@triton.autotune(configs=[...], key=[...])`), compile-time parameters
(`name: tl.constexpr [= default]`), the `tl.`* operation vocabulary, and the
launch syntax `kernel[grid](args, KW=...)`.

**44. Which parts of the input language map directly to Triton concepts?**
`decorator -> AT dotted_name deco_args` ↔ jit/autotune; `param_tail -> COLON test ...` ↔ constexpr annotation; `atom_expr` trailer chains ↔ `tl.load(...)`
etc.; `trailer '[' subscriptlist ']'` followed by `'(' call_args ')'` ↔ the
launch; slices `[:, None]` ↔ broadcasting.

**45. Are you detecting Tryton models, fields, methods, workflows…?**
No — that vocabulary belongs to the **Tryton ERP**, a different system. Our
target is **Triton (GPU kernels)**; the analogous structures we detect are
kernels (decorated functions), constexpr parameters, memory ops, and
launches. See the naming note at the top.

**46. How do you represent Triton-specific structures in your grammar?**
As ordinary grammar shapes, not special tokens: the decorator rule, the
annotated-parameter rule, and the trailer chain are general Python forms whose
*instances* (`triton.jit`, `tl.constexpr`) are NAMEs. This keeps the grammar
small and makes Triton-detection a semantic predicate over parsed structure.

**47. What makes a piece of code "Triton-oriented"?**
Operationally: at least one `def` preceded by `@triton.jit`, importing
`triton`/`triton.language`, using `tl.`* operations and constexpr parameters,
and being launched via `kernel[grid](...)`.

**48. Can your compiler reject code that is not Triton-oriented?**
Not today — any well-formed program in the Python subset is accepted (e.g. a
plain math script). Rejecting non-Triton code is a one-rule semantic check
("file must contain ≥1 jit kernel"), planned, and trivially attachable to the
`decorator`/`funcheader` reductions.

**49. Can your compiler detect incorrect Triton patterns?**
Syntactically malformed ones, yes (`@triton.jit(` unclosed, `BLOCK:` with no
annotation, broken launch brackets). Semantically incorrect ones (jit
decorator not followed by `def`, `tl.load` with an unknown keyword), not yet
— same future pass.

**50. What are the limitations of your Triton detection?**
It is structural, not semantic: no block pairing (layout-free), no symbol
table, lexer gaps (triple-quoted docstrings, hex literals, f-strings,
`1_000`, `@=`), conditional expressions deferred, and the documented
`return`-guard mis-rejection (grammar.md §13.2).

**51. Does your translator generate Triton-compatible code?**
No — the input already *is* Triton code; we recognize it. The natural
generation target is metadata (kernel names, parameters, constexprs as JSON)
rather than code.

**52. What would be needed to make your compiler useful in a real Triton development workflow?**
Carry lexemes (`%union`), build a small AST, add the semantic pass (pairing,
symbol table, Triton predicates), fix the lexer gaps, restore one terminator
token if corpus testing shows the `return`-guard idiom matters, and wire it
into the existing TritonBench validation scripts as a pre-execution filter —
e.g. as a fast structural check before grammar-constrained generation
(xgrammar) or GPU execution.

---

## 7. Code Generation / Output

**53. Can you show an example of input code and the generated output?**
`./newparser example_kernel.py` → `ACCEPT` (exit 0).
`printf 'x = = 3\n' | ./newparser` → `syntax error at line 1 near '='` +
`REJECT` (exit 1).

**54. Is the generated code executable, or is it only a translated representation?**
Neither — the output is a *verdict* (plus diagnostics). The recognizer's exit
code is what scripts consume.

**55. How do you validate that the generated output is correct?**
A test matrix: the canonical kernel and a hard kernel (multi-line autotune,
dicts, slices `[:, None]`, chained comparisons, `for k in range(...)`, flat
`if/elif/else`, `-acc ** 2`, `.to(...)`, launch syntax) must ACCEPT; five
malformed inputs must REJECT with correct line numbers; the two documented
edge behaviors must hold (bare `return` before a keyword ACCEPTs; the
return-guard idiom REJECTs as §13.2 predicts). Plus the table-level check:
`y.output` must show 0 reduce/reduce and only whitelisted shift/reduce.

**56. What limitations does your code generation phase have?**
It doesn't exist yet — by scope. The recognizer also discards lexemes
(no `yylval`), so nothing about *which* names were used survives parsing.

**57. Could your compiler generate different output targets in the future?**
Yes, cheaply, because targets attach as actions to the same reductions:
kernel-metadata JSON, a canonical pretty-printed form, an AST dump, or
statistics for the TritonBench validation reports.

---

## 8. Error Handling

**58. What kinds of errors can your compiler detect?**
Lexical (unknown characters via the `ERROR` token) and syntactic (any token
sequence outside the grammar). Not yet: semantic errors (§4).

**59. How clear are your error messages?**
One uniform shape: `syntax error at line N near 'TEXT'`, where TEXT is the
exact offending lexeme (`yytext`). Clear about *where*; not yet about *what
was expected*.

**60. Do your error messages include line number and column number?**
Line number yes (`%option yylineno`). Column no — adding it means tracking
`yycolumn` in a `YY_USER_ACTION` macro in the lexer; straightforward future
work.

**61. What happens after the first error is detected? / 62. Stop or recover?**
The parser stops immediately: `yyerror` prints the message, `yyparse` returns
non-zero, `main` prints `REJECT`. There are no `error`-token recovery rules —
one error per run, by design (a validator wants a crisp verdict).

**63. Can you show an example of an invalid input and the error reported?**
`def f(: pass` → `syntax error at line 1 near ':'` / REJECT.
`x = 3 $ 4` → `syntax error at line 1 near '$'` / REJECT.
`x = a[1` (then EOF) → `syntax error at line 2 near ''` / REJECT.

**64. What was the hardest error case to handle?**
The one we *cannot* fix with one token of lookahead: bare `return`
immediately followed by an expression-starting statement
(`if pid >= n: return` newline `x = tl.load(y)`). Maximal munch reads `x` as
the return value and errors at `=`. It is documented (grammar.md §13.2) with
its mitigation: reintroduce a single terminator token.

**65. How would you improve error recovery?**
Add panic-mode rules — `element : error` synchronizing on the next header
keyword (`DEF`, `IF`, `IMPORT`, `@`) so one run reports many errors; add
column tracking; extend `yyerror` to list expected tokens from the parser
state.

---

## 10. Implementation and Demo

**66. What programming language did you use?**
C — both generated components (flex → `lex.yy.c`, yacc → `y.tab.c`) and the
hand-written `main`/`yyerror` in the `.y` epilogue.

**67. Which tools or libraries did you use?**
Flex, Unix yacc (Berkeley yacc on macOS; `bison -y` is equivalent), `cc`, and
`make`. No external libraries.

**68. How is your project organized? / 69. Main files? / 70. Role of each?**
Everything in `newlexparser/`:

- `grammar.md` — the design document: notation, transformations, the full
grammar with explanations, FIRST/FOLLOW + predictive table, conflict
whitelist, limitations. The single source of truth.
- `newlexer.l` — flex spec: tokens only; includes `y.tab.h`; layout-free
whitespace rule.
- `newparser.y` — token declarations, the 214-rule grammar (transcribed 1:1
from grammar.md §7–§10 with matching section banners), `yyerror` and `main`.
- `scanner_main.c` — standalone driver for the lexical phase (`newscanner`):
prints the token sequence, builds/prints the symbol table of identifiers and
literals, and reports *all* lexical errors with line numbers (it keeps
scanning past them, unlike the parser).
- `Makefile` — build graph: `yacc -d -v` → `flex` → `cc`; `make test`.
- `example_kernel.py` — canonical ACCEPT input.
- Generated: `y.tab.c/.h` (parser + token numbers), `lex.yy.c` (scanner),
`y.output` (automaton + conflict report), `newparser` + `newscanner`
(binaries).

**71. How do you compile and run? / 72. Command to test?**

```sh
cd newlexparser
make            # yacc -d -v newparser.y && flex newlexer.l && cc ...
make test       # runs ./newparser example_kernel.py  -> ACCEPT
./newparser some_kernel.py        # or: ./newparser < some_kernel.py
printf 'x = = 1\n' | ./newparser  # -> syntax error ... REJECT
```

**73. What was the biggest implementation challenge?**
Making the layout-free decision *safe*: proving which statement-boundary
collisions exist, that all of them resolve to the intended reading under
shift-preference, and that none of them is reduce/reduce — then auditing
`y.output` (80 S/R, 0 R/R, all whitelisted) to confirm the analysis.

**74. What part would you redesign with more time?**
Reintroduce exactly one terminator token (NEWLINE) — it costs one lexer line
and a handful of FOLLOW changes, but it eliminates all 80 conflicts, fixes
the `return`-guard caveat, and re-enables block verification via
INDENT/DEDENT later. Then `yylval` + AST.

**75. During the demo, can you walk the complete flow from source code to output?**
Yes: (1) show `example_kernel.py`; (2) `make` — point at the yacc conflict
line and explain 80/0; (3) show 2–3 token examples conceptually
(Q15); (4) `make test` → ACCEPT; (5) a negative input → line-numbered REJECT;
(6) open `y.output` at state 6 to show a real shift/reduce decision; (7) open
`grammar.md` §12 for the derivation that matches what just happened.

---

## 11. Deeper Understanding Questions

**76. Why did you design your grammar in this way?**
Three forces: the course requires *top-down* design (so: left-recursion
elimination, left factoring, ε-tails, FIRST/FOLLOW discipline); the corpus
defines *coverage* (exactly the constructs TritonBench kernels use); and the
simplicity mandate removed layout tokens (so: flat headers + maximal munch).
Each force is traceable to a section of grammar.md (§3, §1, §6/§11.3).

**77. What would happen if we added nested blocks to your language?**
We would need block delimiters back: either INDENT/DEDENT tokens (an
indentation stack in the lexer — the previous `parser/triton_parser.y` did
exactly this) or braces. The grammar change is localized: headers regain a
`suite` nonterminal (`suite -> NEWLINE INDENT block DEDENT | stmt`), and the
dangling-`else` question reappears — solved structurally by the layout
tokens.

**78. What would happen if we added function calls?**
They are already in: `trailer -> '(' call_args ')'` chained by
`trailer_list`, including keyword arguments and the launch idiom
`kernel[grid](args)`. The question's real lesson: calls cost *nothing* extra
because the trailer chain is one loop, not a special case.

**79. How difficult would it be to add type checking?**
Moderate. Prerequisites: lexemes (`%union`), an AST or at least
per-reduction attributes, and a symbol table (§4). Triton narrows the
problem: the interesting checks are constexpr-ness and dtype consistency, not
full Hindley-Milner. Estimate: the infrastructure is more work than the rules.

**80. What part of your compiler depends the most on the grammar?**
The parser is *generated from* the grammar — `y.tab.c`'s 317-state automaton
is nothing but the grammar compiled. Everything downstream (future actions,
error positions) keys off rule reductions.

**81. What part of your compiler would break if the grammar changes?**
(1) The conflict whitelist — any rule change requires re-auditing `y.output`
against §11.3 (a new conflict may be a real bug, not maximal munch). (2)
Future semantic actions, which reference positional symbols (`$1`, `$2`) —
reordering a rule's right-hand side silently breaks them. (3) The lexer is
*almost* immune (it only shares token names via `y.tab.h`) — removing a token
from the `.y` breaks the lexer build, which is a feature: the mismatch is
caught at compile time.

**82. Why is ambiguity a problem in compiler design?**
An ambiguous grammar gives one input several parse trees, hence potentially
several meanings — and a deterministic parser must pick one without a stated
rule. Conflicts in yacc are the symptom. Our project shows the disciplined
version: we *chose* ambiguity at statement boundaries (by deleting
terminators), then neutralized it with an explicit, documented policy and an
audited conflict whitelist — ambiguity managed, not ignored.

**83. Why is left recursion a problem for some parsers?**
A top-down parser expands the leftmost nonterminal; `E -> E + T` makes it
re-enter `parse_E` before consuming any token — infinite recursion. Bottom-up
parsers don't predict, they reduce after seeing a whole right-hand side, so
left recursion is not only legal for yacc but *preferred* (constant stack).
Our grammar removes it anyway because the design methodology is top-down; the
LALR machine accepts both forms.

**84. What is the difference between syntax and semantics?**
Syntax is *form* — derivability from the grammar; semantics is *meaning* —
constraints and behavior beyond form. Concrete in our project: `else:` with
no preceding `if` is syntactically ACCEPTed (it matches the header rule) but
semantically wrong; `x = y` with undefined `y` likewise. The recognizer draws
exactly this line.

**85. What is the difference between recognizing code and translating code?**
Recognition answers a yes/no question — "is this string in the language?" —
which is what `newparser` does (ACCEPT/REJECT). Translation additionally
*produces an equivalent artifact* in another form (code, AST, metadata),
which requires attaching semantic actions and carrying values through the
parse. Our design keeps the recognizer pure and stages translation as the
next increment (grammar.md §15 step 4).