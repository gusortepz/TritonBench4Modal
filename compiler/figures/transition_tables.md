# Transition tables (generated from the same DFA definitions as the figures)

### Transition table - Identifiers and keywords

| state | `letter, _` | `letter, digit, _` | accepts |
|---|---|---|---|
| S0 | I1 | - | - |
| I1 | - | I1 | NAME / keyword |

Keywords use this same automaton; the lex rule order decides: keyword rules are listed before the NAME rule, so on an equal-length match the keyword wins (flex tie-break).

### Transition table - Numeric literals (INTEGER and FLOAT)

| state | `digit` | `.` | `e, E` | `+, -` | accepts |
|---|---|---|---|---|---|
| S0 | N1 | P1 | - | - | - |
| N1 | N1 | N2 | N3 | - | INTEGER |
| N2 | N2 | - | N3 | - | FLOAT |
| N3 | N7 | - | - | N6 | - |
| N5 | N5 | - | N3 | - | FLOAT |
| N6 | N7 | - | - | - | - |
| N7 | N7 | - | - | - | FLOAT |
| P1 | N5 | - | - | - | - |

P1, N3, N6 are not accepting: if the exponent or fraction never arrives, flex backtracks to the last accepting state (maximal munch), e.g. `1e` is INTEGER(1) NAME(e) and a lone `.` is handled by the operator automaton (DOT).

### Transition table - String literals (single-line and triple-quoted)

| state | `\"` | `not \", not \\n` | `not \"  (incl. \\n)` | `not \"` | accepts |
|---|---|---|---|---|---|
| S0 | Q1 | - | - | - | - |
| Q1 | Q2 | Q1 | - | - | - |
| Q2 | T0 | - | - | - | STRING (single-line) |
| T0 | T1 | - | T0 | - | - |
| T1 | T2 | - | - | T0 | - |
| T2 | T3 | - | - | T0 | - |
| T3 | - | - | - | - | STRING (triple-quoted) |

Q2 is accepting AND has an outgoing edge: maximal munch makes a third quote continue into the triple-quoted automaton instead of returning the empty string. The same shape applies to '...'. In the lex file the T-states are the start conditions TRIPLE_D / TRIPLE_S, with yymore() accumulating the lexeme; EOF inside a T-state returns ERROR (unterminated string).

### Transition table - Operators and delimiters (maximal munch chains)

| state | `<` | `=` | `*` | `!` | `-` | `>` | `.` | `+, &, |, ^, %` | `( ) [ ] { } , : ; ~ @` | accepts |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | L1 | E1 | M1 | X1 | H1 | - | P1 | O1 | K1 | - |
| E1 | - | E2 | - | - | - | - | - | - | - | ASSIGN |
| E2 | - | - | - | - | - | - | - | - | - | EQUAL_EQUAL |
| H1 | - | H2 | - | - | - | H3 | - | - | - | MINUS |
| H2 | - | - | - | - | - | - | - | - | - | MINUS_ASSIGN |
| H3 | - | - | - | - | - | - | - | - | - | ARROW |
| K1 | - | - | - | - | - | - | - | - | - | delimiter token |
| L1 | L3 | L2 | - | - | - | - | - | - | - | LESS |
| L2 | - | - | - | - | - | - | - | - | - | LESS_EQUAL |
| L3 | - | L4 | - | - | - | - | - | - | - | LEFT_SHIFT |
| L4 | - | - | - | - | - | - | - | - | - | LEFT_SHIFT_ASSIGN |
| M1 | - | M3 | M2 | - | - | - | - | - | - | STAR |
| M2 | - | M4 | - | - | - | - | - | - | - | DOUBLE_STAR |
| M3 | - | - | - | - | - | - | - | - | - | STAR_ASSIGN |
| M4 | - | - | - | - | - | - | - | - | - | DOUBLE_STAR_ASSIGN |
| O1 | - | O2 | - | - | - | - | - | - | - | PLUS, AMPERSAND, ... |
| O2 | - | - | - | - | - | - | - | - | - | ...same + _ASSIGN |
| P1 | - | - | - | - | - | - | P2 | - | - | DOT |
| P2 | - | - | - | - | - | - | P3 | - | - | - |
| P3 | - | - | - | - | - | - | - | - | - | ELLIPSIS |
| X1 | - | X2 | - | - | - | - | - | - | - | - |
| X2 | - | - | - | - | - | - | - | - | - | NOT_EQUAL |

The chains for > (GREATER family) and / (SLASH family) are identical in shape to < and *. X1 and P2 are not accepting: a lone `!` is a lexical ERROR, and `..` backtracks to DOT, DOT. Maximal munch is why `<<=` is one token and not three.

### Transition table - Comments, whitespace and the error catch-all

| state | `#` | `not \\n` | `space, tab, \\r, \\n` | `any other char` | accepts |
|---|---|---|---|---|---|
| S0 | C1 | - | W1 | R1 | - |
| C1 | - | C1 | - | - | (comment, skipped) |
| R1 | - | - | - | - | ERROR |
| W1 | - | - | W1 | - | (whitespace, skipped) |

Layout-free design: \\n is ordinary whitespace, the parser never sees layout. Anything no automaton accepts is returned as the ERROR token with its line number.
