#!/usr/bin/env python3
"""Generate the design artifacts for the lexical-phase report.

Every automaton is defined ONCE as data; from that single source this script
emits (a) Graphviz figures (figures/*.png) and (b) the matching transition
tables (figures/transition_tables.md). Figures and tables therefore cannot
disagree -- the design is traceable by construction.

Also extracts the Token-ID table from y.tab.h (figures/token_ids.tsv).

Usage: python3 make_figures.py     (requires graphviz `dot` in PATH)
"""
import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
DOT = "/opt/homebrew/bin/dot" if Path("/opt/homebrew/bin/dot").exists() else "dot"

# ---------------------------------------------------------------- DFA spec
# Each DFA: (name, title, accepting {state: token}, transitions
#            [(src, label, dst)], note). State S0 is always the start.

DFAS = [
    ("dfa_identifiers", "Identifiers and keywords",
     {"I1": "NAME / keyword"},
     [("S0", "letter, _", "I1"),
      ("I1", "letter, digit, _", "I1")],
     "Keywords use this same automaton; the lex rule order decides: keyword "
     "rules are listed before the NAME rule, so on an equal-length match the "
     "keyword wins (flex tie-break)."),

    ("dfa_numbers", "Numeric literals (INTEGER and FLOAT)",
     {"N1": "INTEGER", "N2": "FLOAT", "N5": "FLOAT", "N7": "FLOAT"},
     [("S0", "digit", "N1"),
      ("N1", "digit", "N1"),
      ("N1", ".", "N2"),
      ("N2", "digit", "N2"),
      ("S0", ".", "P1"),
      ("P1", "digit", "N5"),
      ("N5", "digit", "N5"),
      ("N1", "e, E", "N3"),
      ("N2", "e, E", "N3"),
      ("N5", "e, E", "N3"),
      ("N3", "+, -", "N6"),
      ("N3", "digit", "N7"),
      ("N6", "digit", "N7"),
      ("N7", "digit", "N7")],
     "P1, N3, N6 are not accepting: if the exponent or fraction never "
     "arrives, flex backtracks to the last accepting state (maximal munch), "
     "e.g. `1e` is INTEGER(1) NAME(e) and a lone `.` is handled by the "
     "operator automaton (DOT)."),

    ("dfa_strings", "String literals (single-line and triple-quoted)",
     {"Q2": "STRING (single-line)", "T3": "STRING (triple-quoted)"},
     [("S0", "\\\"", "Q1"),
      ("Q1", "not \\\", not \\\\n", "Q1"),
      ("Q1", "\\\"", "Q2"),
      ("Q2", "\\\"", "T0"),
      ("T0", "not \\\"  (incl. \\\\n)", "T0"),
      ("T0", "\\\"", "T1"),
      ("T1", "not \\\"", "T0"),
      ("T1", "\\\"", "T2"),
      ("T2", "not \\\"", "T0"),
      ("T2", "\\\"", "T3")],
     "Q2 is accepting AND has an outgoing edge: maximal munch makes a third "
     "quote continue into the triple-quoted automaton instead of returning "
     "the empty string. The same shape applies to '...'. In the lex file the "
     "T-states are the start conditions TRIPLE_D / TRIPLE_S, with yymore() "
     "accumulating the lexeme; EOF inside a T-state returns ERROR "
     "(unterminated string)."),

    ("dfa_operators", "Operators and delimiters (maximal munch chains)",
     {"L1": "LESS", "L2": "LESS_EQUAL", "L3": "LEFT_SHIFT",
      "L4": "LEFT_SHIFT_ASSIGN",
      "M1": "STAR", "M2": "DOUBLE_STAR", "M3": "STAR_ASSIGN",
      "M4": "DOUBLE_STAR_ASSIGN",
      "E1": "ASSIGN", "E2": "EQUAL_EQUAL",
      "X2": "NOT_EQUAL",
      "H1": "MINUS", "H2": "MINUS_ASSIGN", "H3": "ARROW",
      "P1": "DOT", "P3": "ELLIPSIS",
      "O1": "PLUS, AMPERSAND, ...", "O2": "...same + _ASSIGN",
      "K1": "delimiter token"},
     [("S0", "<", "L1"), ("L1", "=", "L2"), ("L1", "<", "L3"),
      ("L3", "=", "L4"),
      ("S0", "*", "M1"), ("M1", "*", "M2"), ("M1", "=", "M3"),
      ("M2", "=", "M4"),
      ("S0", "=", "E1"), ("E1", "=", "E2"),
      ("S0", "!", "X1"), ("X1", "=", "X2"),
      ("S0", "-", "H1"), ("H1", "=", "H2"), ("H1", ">", "H3"),
      ("S0", ".", "P1"), ("P1", ".", "P2"), ("P2", ".", "P3"),
      ("S0", "+, &, |, ^, %", "O1"), ("O1", "=", "O2"),
      ("S0", "( ) [ ] { } , : ; ~ @", "K1")],
     "The chains for > (GREATER family) and / (SLASH family) are identical "
     "in shape to < and *. X1 and P2 are not accepting: a lone `!` is a "
     "lexical ERROR, and `..` backtracks to DOT, DOT. Maximal munch is why "
     "`<<=` is one token and not three."),

    ("dfa_layout", "Comments, whitespace and the error catch-all",
     {"C1": "(comment, skipped)", "W1": "(whitespace, skipped)",
      "R1": "ERROR"},
     [("S0", "#", "C1"),
      ("C1", "not \\\\n", "C1"),
      ("S0", "space, tab, \\\\r, \\\\n", "W1"),
      ("W1", "space, tab, \\\\r, \\\\n", "W1"),
      ("S0", "any other char", "R1")],
     "Layout-free design: \\\\n is ordinary whitespace, the parser never "
     "sees layout. Anything no automaton accepts is returned as the ERROR "
     "token with its line number."),
]


def emit_dfa(name, title, accepting, transitions, note):
    lines = ["digraph G {",
             '  rankdir=LR; bgcolor="white";',
             '  node [fontname="Helvetica", fontsize=12];',
             '  edge [fontname="Helvetica", fontsize=11];',
             '  start [shape=point];']
    states = {s for t in transitions for s in (t[0], t[2])}
    for s in sorted(states):
        if s in accepting:
            lines.append(f'  {s} [shape=doublecircle, '
                         f'xlabel="{accepting[s]}"];')
        else:
            lines.append(f'  {s} [shape=circle];')
    lines.append("  start -> S0;")
    for src, label, dst in transitions:
        lines.append(f'  {src} -> {dst} [label="{label}"];')
    lines.append("}")
    dot_file = OUT / f"{name}.dot"
    dot_file.write_text("\n".join(lines))
    subprocess.run([DOT, "-Tpng", "-Gdpi=180", str(dot_file),
                    "-o", str(OUT / f"{name}.png")], check=True)


def transition_table(name, title, accepting, transitions, note):
    labels = []
    for _, lab, _ in transitions:
        if lab not in labels:
            labels.append(lab)
    states = ["S0"] + sorted({t[2] for t in transitions} |
                             {t[0] for t in transitions} - {"S0"})
    out = [f"### Transition table - {title}\n",
           "| state | " + " | ".join(f"`{l}`" for l in labels) +
           " | accepts |",
           "|" + "---|" * (len(labels) + 2)]
    for s in states:
        row = [s]
        for l in labels:
            dst = [d for src, lab, d in transitions
                   if src == s and lab == l]
            row.append(dst[0] if dst else "-")
        row.append(accepting.get(s, "-"))
        out.append("| " + " | ".join(row) + " |")
    out.append(f"\n{note}\n")
    return "\n".join(out)


def emit_pipeline():
    g = """digraph G {
  rankdir=LR; bgcolor="white";
  node [shape=box, style="rounded,filled", fillcolor="#eef2fa",
        fontname="Helvetica", fontsize=12];
  edge [fontname="Helvetica", fontsize=11];
  src   [label="kernel.py\\n(Triton source)", fillcolor="#ffffff"];
  lex   [label="Scanner\\nnewlexer.l -> flex -> lex.yy.c"];
  toks  [label="token sequence\\n+ symbol table", shape=note,
         fillcolor="#fdf6e3"];
  yacc  [label="Parser\\nnewparser.y -> yacc -> y.tab.c"];
  out   [label="ACCEPT / REJECT\\n+ error line", shape=note,
         fillcolor="#fdf6e3"];
  hdr   [label="y.tab.h\\n(shared token ids)", shape=component,
         fillcolor="#e8f5e9"];
  src -> lex; lex -> toks [label=" newscanner"];
  lex -> yacc [label=" yylex()"]; yacc -> out [label=" newparser"];
  hdr -> lex [style=dashed]; hdr -> yacc [style=dashed];
}"""
    (OUT / "pipeline.dot").write_text(g)
    subprocess.run([DOT, "-Tpng", "-Gdpi=180", str(OUT / "pipeline.dot"),
                    "-o", str(OUT / "pipeline.png")], check=True)


def emit_token_ids():
    hdr = (HERE / "y.tab.h").read_text()
    pairs = re.findall(r"#define\s+(\w+)\s+(\d+)", hdr)
    rows = [(n, v) for n, v in pairs
            if n not in ("YYERRCODE", "YYDEBUG", "YYPATCH")]
    with open(OUT / "token_ids.tsv", "w") as f:
        f.write("TOKEN\tID\n")
        for n, v in rows:
            f.write(f"{n}\t{v}\n")
    return len(rows)


def main():
    tables = ["# Transition tables (generated from the same DFA "
              "definitions as the figures)\n"]
    for spec in DFAS:
        emit_dfa(*spec)
        tables.append(transition_table(*spec))
    (OUT / "transition_tables.md").write_text("\n".join(tables))
    emit_pipeline()
    n = emit_token_ids()
    print(f"figures: {len(DFAS) + 1} png, transition_tables.md, "
          f"token_ids.tsv ({n} tokens)")


if __name__ == "__main__":
    main()
