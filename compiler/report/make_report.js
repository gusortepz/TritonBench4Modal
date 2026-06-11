// make_report.js - builds the Lexical Analysis Phase written report (DOCX)
// from the real project artifacts in compiler/ (lex file, scanner driver,
// generated DFA figures, transition tables, token ids, test outputs).
// Usage: node make_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const FIG = (f) => path.join(ROOT, "figures", f);
const read = (f) => fs.readFileSync(path.join(ROOT, f), "utf8");

// ---------------------------------------------------------------- helpers
const CONTENT_W = 9360; // US Letter, 1in margins
const MONO = "Courier New";
const INCLUDE_REFS = false; // set true to restore section 7 + [n] citations

const stripCites = (s) =>
  INCLUDE_REFS ? s : s.replace(/\s*\[\d+\](\[\d+\])*/g, "");

const p = (text, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED,
    spacing: { after: 160 },
    children: Array.isArray(text)
      ? text
      : [new TextRun({ text: stripCites(text), size: 22 })],
    ...opts,
  });

const runs = (...rs) => rs.map((r) => new TextRun({ size: 22, ...r }));

const h1 = (t) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true,
                  children: [new TextRun(t)] });
const h1nb = (t) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1,
                  children: [new TextRun(t)] });
const h2 = (t) =>
  new Paragraph({ heading: HeadingLevel.HEADING_2,
                  children: [new TextRun(t)] });

const codeBlock = (text, size = 15) =>
  text.split("\n").map((line) =>
    new Paragraph({
      spacing: { after: 0 },
      children: [new TextRun({ text: line.replace(/\t/g, "    ") || " ",
                               font: MONO, size })],
    }));

const caption = (t) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 240 },
    children: [new TextRun({ text: t, italics: true, size: 20 })],
  });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

function figure(file, cap, maxW = 600) {
  const { w, h } = pngSize(file);
  const scale = Math.min(maxW / w, 1 / 2.2);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 160 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(file),
        transformation: { width: Math.round(w * scale),
                          height: Math.round(h * scale) },
        altText: { title: cap, description: cap, name: cap },
      })],
    }),
    caption(cap),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 2, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };

function mkTable(headerCells, rows, colWidths, cellSize = 18) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, bold, fill) =>
    new TableCell({
      borders,
      width: { size: w, type: WidthType.DXA },
      shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        spacing: { after: 0 },
        children: [new TextRun({ text: String(txt), bold,
                                 size: cellSize,
                                 font: /[`]/.test(String(txt)) ? MONO : undefined })],
      })],
    });
  const mkRow = (cells, bold, fill) =>
    new TableRow({
      children: cells.map((c, i) => cell(c, colWidths[i], bold, fill)),
    });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [mkRow(headerCells, true, "DCE6F1"),
           ...rows.map((r) => mkRow(r, false))],
  });
}

// monospaced text inside narrative
const m = (t) => ({ text: t, font: MONO, size: 20 });

// ------------------------------------------------------------- load assets
const lexSrc = read("lexer.l");
const [lexDefs, lexRules, lexUser] = lexSrc.split(/^%%$/m);
const scannerSrc = (lexUser.match(/#ifdef SCANNER_MAIN[\s\S]*?#endif/) || [""])[0];
const makefileSrc = read("Makefile");

const tokenIds = read("figures/token_ids.tsv").trim().split("\n").slice(1)
  .map((l) => l.split("\t"));

const t1in = read("tests/t1_valid_kernel.py");
const t1out = read("tests/expected/t1_valid_kernel.scanner.out");
const t3out = read("tests/expected/t3_invalid_symbols.scanner.out");
const t3err = read("tests/expected/t3_invalid_symbols.scanner.err");
const t4out = read("tests/expected/t4_unterminated_string.scanner.out");
const t2out = read("tests/expected/t2_literals_operators.scanner.out");

// transition tables: parse the generated markdown (single source of truth)
function parseTransitionTables(md) {
  const sections = md.split(/^### /m).slice(1);
  return sections.map((sec) => {
    const lines = sec.split("\n");
    const title = lines[0].replace("Transition table - ", "").trim();
    const tbl = lines.filter((l) => l.startsWith("|"));
    const parse = (l) => l.split("|").slice(1, -1)
      .map((c) => c.trim().replace(/`/g, ""));
    const header = parse(tbl[0]);
    const rows = tbl.slice(2).map(parse);
    const note = lines.slice(lines.indexOf(tbl[tbl.length - 1]) + 1)
      .join(" ").trim();
    return { title, header, rows, note };
  });
}
const ttables = parseTransitionTables(read("figures/transition_tables.md"));

// ------------------------------------------------------------------ content
const children = [];

// ===== Title page =====
children.push(
  new Paragraph({ spacing: { before: 2400 } }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Instituto Tecnológico y de Estudios Superiores de Monterrey", size: 28, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new TextRun({ text: "Campus Guadalajara", size: 24 })] }),
  new Paragraph({ spacing: { before: 1600 } }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "TC3002B — Computer Science Advanced Applications Development", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 },
    children: [new TextRun({ text: "Module #3: Compilers — Step I", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 480 },
    children: [new TextRun({ text: "Lexical Analysis Phase for the Triton GPU Kernel Language", size: 40, bold: true })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240 },
    children: [new TextRun({ text: "A scanner built with UNIX lex, integrated with a yacc parser", size: 24, italics: true })] }),
  new Paragraph({ spacing: { before: 1600 } }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Student: Gustavo Ortiz", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Learning Unit Leader: Dr. Salvador Hinojosa", size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240 },
    children: [new TextRun({ text: "June 10, 2026", size: 26 })] }),
  new Paragraph({ children: [new PageBreak()] }),
);

// ===== TOC =====
children.push(
  h1nb("Table of Contents"),
  new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// ===== 1. Introduction =====
children.push(h1("1. Introduction"));
children.push(h2("1.1 Summary"));
children.push(p(
  "This report documents the analysis, design, implementation, and verification of the lexical " +
  "analysis phase of a compiler front end for the Triton GPU kernel language. Triton is a " +
  "Python-like domain-specific language developed by OpenAI for writing high-performance GPU " +
  "kernels [3][5]. The scanner is implemented with the UNIX lex tool (flex) as required, without " +
  "any regular-expression library or Python ast module, and it produces the two outputs required " +
  "for this phase: the sequence of tokens of the scanned source file and the symbol table for the " +
  "tokens that require it. The scanner is additionally integrated with a yacc parser through the " +
  "shared token-definition header y.tab.h, so the token identifiers documented in this report are " +
  "the exact integers exchanged between both phases. The implementation was validated against a " +
  "corpus of 135 LLM-generated Triton programs, reaching a 94.5% acceptance rate on the 55 files " +
  "that contain Triton kernels, with every remaining failure traced to a documented, deliberate " +
  "scope decision."));
children.push(p(
  "Section 1.2 introduces the notation and the development model. Section 2 (Analysis) states the " +
  "informal lexical specification and its formal counterpart as regular expressions, together with " +
  "the token identifiers and the error messages. Section 3 (Design) presents the automata, the " +
  "transition tables, and the symbol-table data structures and algorithms, all traceable to the " +
  "analysis. Section 4 (Implementation) provides the complete printout of the lex file and of the " +
  "scanner driver, explained section by section. Section 5 presents verification and validation, " +
  "including the test-case design and the snapshots of the scanner outputs. Section 6 contains the " +
  "work plan" + (INCLUDE_REFS ? " and Section 7 the references." : ".")));

children.push(h2("1.2 Notation and development model"));
children.push(p(
  "Regular expressions (RE). A regular expression is a finite formula that denotes a regular " +
  "language: the set of strings it matches. REs are built from single characters, concatenation, " +
  "alternation (|), and repetition (the Kleene star * and the positive closure +), plus the " +
  "abbreviations used by lex such as character classes [a-z] and optionality (?). REs are the " +
  "formal specification language of this phase: every kind of token in Section 2 is defined by " +
  "one RE [1]."));
children.push(p(
  "Finite state machines (FSM/DFA). A deterministic finite automaton is a 5-tuple " +
  "(Q, Σ, δ, q0, F): a finite set of states, an input alphabet, a transition function " +
  "δ: Q × Σ → Q, an initial state, and a set of accepting states. A DFA recognizes exactly the " +
  "regular languages, which is why every RE of the analysis can be implemented by a DFA of the " +
  "design. lex performs this construction mechanically (RE → NFA → DFA by subset construction, " +
  "then state minimization), which is the central justification for using it [1][4]."));
children.push(p(
  "Transition tables. A transition table is the tabular form of δ: one row per state, one column " +
  "per input symbol or character class, and the accepting states annotated. The tables in Section " +
  "3 are written over character classes (letter, digit, quote, operator characters) instead of the " +
  "256 raw input characters; this is the same equivalence-class compression that flex applies " +
  "internally and it is what makes the tables compact and efficient."));
children.push(p(
  "Development model. The project follows the classic phase model — Analysis, Design, " +
  "Implementation, Verification and Validation — executed as a short waterfall with one measured " +
  "feedback iteration. The model fits this problem because the deliverables form a strict chain: " +
  "the informal specification fixes the token set, the REs formalize it, the automata and tables " +
  "realize the REs, the lex file transcribes the automata, and the tests check the lex file " +
  "against the specification. One controlled iteration occurred when corpus-level validation " +
  "(Section 5.4) revealed that triple-quoted docstrings were mis-lexed; the analysis and design " +
  "were extended (states T0–T3 of Figure #4) and the implementation was updated accordingly. The " +
  "iteration is documented rather than hidden because it shows the V&V process doing exactly its " +
  "job [6]."));
children.push(p(
  "Why lex. The course restriction requires the scanner to be generated by UNIX lex, and the tool " +
  "is also the technically correct choice: lex takes the REs of the analysis directly as its rules " +
  "section, builds the minimized DFA automatically, applies the maximal-munch (longest match) " +
  "policy that Section 3 depends on, and resolves ties by rule order, which is how keywords win " +
  "over identifiers. No regular-expression library, API, or Python ast module is used anywhere in " +
  "the project [2][4]."));

// ===== 2. Analysis =====
children.push(h1("2. Analysis"));
children.push(p(
  "The goal of this phase is to recognize every lexeme that occurs in real Triton GPU kernels — " +
  "the decorated Python functions that TritonBench programs define with @triton.jit — and to " +
  "reject everything else with a precise error message. The requirements below follow the " +
  "deliverables required for the scanner: the informal lexical description (2.1), the formal " +
  "specification as regular expressions (2.2), the tokens and their identification (2.3), and the " +
  "error messages (2.4). Each requirement is labeled R1–R8 so that the design (Section 3) and the " +
  "implementation (Section 4) can be traced back to it."));

children.push(h2("2.1 Informal lexical specification"));
children.push(p(
  "A Triton kernel file is a sequence of Python-like logical lines containing imports, decorators, " +
  "function headers, assignments, and expressions over tensor operations. Reading real kernels " +
  "(the add/matmul/softmax families of TritonBench) yields the following lexeme categories. Only " +
  "the lexemes needed for the goal are included; everything outside the table is, by " +
  "specification, a lexical error. Table #1 is the informal lexical specification of the " +
  "language."));
children.push(mkTable(
  ["Req.", "Category", "Description", "Examples"],
  [
    ["R1", "Identifiers", "names of variables, functions, modules and attributes: a letter or underscore followed by letters, digits or underscores", "pid, tl, BLOCK_SIZE, _mask"],
    ["R2", "Keywords", "35 reserved words of the Python subset: 26 used by kernel constructs and 9 reserved for future phases", "def, return, if, for, in, not, import, lambda"],
    ["R3", "Integer literals", "decimal digit sequences", "0, 42, 1024"],
    ["R3", "Float literals", "digits with a decimal point and/or a scientific exponent; the leading-dot form is included", "3.14, .5, 1e6, 2.5e-3"],
    ["R4", "String literals", "single-line strings in double or single quotes, and triple-quoted strings (docstrings) that may span lines", "\"mask\", 'x', \"\"\"doc\"\"\""],
    ["R5", "Operators", "arithmetic, comparison, bitwise, and the 13 assignment operators, with multi-character forms lexed as one token", "+, **, //, <=, !=, <<=, ->, @"],
    ["R5", "Delimiters", "grouping and punctuation symbols", "( ) [ ] { } , : . ; ..."],
    ["R6", "Comments", "from # to the end of the line; consumed, no token", "# one block per program"],
    ["R6", "Whitespace", "spaces, tabs, carriage returns and newlines; consumed, no token (layout-free design)", ""],
    ["R7", "Invalid symbols", "any character not covered above produces the ERROR token and a message", "$, ?, a lone !"],
  ],
  [600, 1500, 4760, 2500]));
children.push(caption("Table #1: Informal lexical specification of the Triton kernel language."));
children.push(p([...runs(
  { text: "Two decisions in Table #1 deserve justification. First, the newline is consumed like " +
          "any whitespace (R6): the project adopts a layout-free token stream, so the parser " +
          "never receives layout tokens; this removes the need for an indentation stack in the " +
          "scanner and is documented at length in the grammar design (grammar.md). Second, nine " +
          "keywords (" },
  m("lambda, try, except, finally, with, class, yield, async, await"),
  { text: ") are recognized and reserved even though no grammar rule uses them yet: reserving " +
          "them now guarantees they can never be captured as identifiers by user code, which " +
          "would make adding those constructs later a breaking change." })]));

children.push(h2("2.2 Formal specification: regular expressions"));
children.push(p(
  "Table #2 gives the regular expression for each kind of token, in exactly the notation that " +
  "appears in the rules section of the lex file (Section 4), using the three named definitions " +
  "DIGIT = [0-9], LETTER = [a-zA-Z_], and EXPONENT = [eE][+-]?{DIGIT}+. Each expression is " +
  "explained and justified below the table."));
children.push(mkTable(
  ["Token kind", "Regular expression", "Token(s)"],
  [
    ["Identifier", "{LETTER}({LETTER}|{DIGIT})*", "NAME"],
    ["Keyword", "one literal RE per word: \"def\", \"return\", ...", "DEF, RETURN, ... (35)"],
    ["Integer", "{DIGIT}+", "INTEGER"],
    ["Float (forms 1-3)", "{DIGIT}+\\.{DIGIT}*{EXPONENT}?   |   \\.{DIGIT}+{EXPONENT}?   |   {DIGIT}+{EXPONENT}", "FLOAT"],
    ["String (single-line)", "\\\"[^\\\"\\n]*\\\"   |   '[^'\\n]*'", "STRING"],
    ["String (triple-quoted)", "\\\"\\\"\\\" ... \\\"\\\"\\\" via start conditions TRIPLE_D / TRIPLE_S", "STRING"],
    ["Operators / delimiters", "one literal RE per lexeme: \"+\", \"**\", \"<<=\", \"->\", \"...\", ...", "PLUS, DOUBLE_STAR, ... (45)"],
    ["Comment", "#[^\\n]*", "(consumed)"],
    ["Whitespace", "[ \\t\\r\\n]+", "(consumed)"],
    ["Anything else", ".", "ERROR"],
  ],
  [1900, 5060, 2400]));
children.push(caption("Table #2: Regular expressions of the formal lexical specification."));
children.push(p(
  "Identifiers (R1) start with a letter or underscore so that 1x cannot be an identifier, and the " +
  "underscore is a letter because Triton code uses names such as x_ptr and _mask. Keywords (R2) " +
  "are spelled as literal REs and placed before the identifier rule: lex prefers the longest " +
  "match and, on ties, the earliest rule, so def is the keyword DEF while define is one NAME — " +
  "both behaviors are required. The three float forms (R3) cover every shape found in real " +
  "kernels: 3.14 and 1. (form 1), .5 (form 2), and 1e6 (form 3); the EXPONENT abbreviation makes " +
  "the optional sign and mandatory exponent digits explicit, so a string like 1e is correctly " +
  "split into INTEGER(1) NAME(e) by backtracking. Single-line strings (R4) exclude the newline so " +
  "that an unterminated quote cannot silently swallow the rest of the file; the multi-line " +
  "docstring case is handled by a dedicated automaton (Section 3.2, Figure #4) because no single " +
  "lex pattern can both span newlines and fail cleanly at end of file. Every operator and " +
  "delimiter (R5) is its own literal RE so that maximal munch assembles <<= as one token instead " +
  "of three. The final rule, the dot, implements R7: any character that no other rule can start " +
  "is returned as the ERROR token, which keeps the scanner total — it never crashes on bad " +
  "input."));

children.push(h2("2.3 Tokens and their identification"));
children.push(p(
  "Every token kind is identified by an integer Token ID. The IDs are not invented by hand: they " +
  "are generated by yacc from the %token declarations of the companion grammar (parser.y) into " +
  "the header y.tab.h, and the lex file includes that header. This is a deliberate analysis " +
  "decision with two consequences. First, the scanner and the parser can never disagree about a " +
  "token number, eliminating an entire class of integration defects. Second, the numbering starts " +
  "at 257 because yacc reserves 0–255 for single characters, 256 for its internal error token — " +
  "which is also why the project's ERROR token receives its own number above 257 instead of " +
  "colliding with 256, a defect that existed in an early version of the lexer and was removed " +
  "during design. The complete list is given in Table #4 (Section 3.3) next to the transition " +
  "tables that produce each token."));

children.push(h2("2.4 Error messages"));
children.push(p(
  "The scanner reports two lexical error situations, both with the line number provided by lex's " +
  "yylineno facility. They are specified in Table #3 and exercised by test cases T3 and T4 of " +
  "Section 5. The recovery policy is panic-free: the offending character is consumed, the ERROR " +
  "token is emitted, and scanning continues, so a single run reports every lexical error in the " +
  "file instead of stopping at the first one. When the scanner runs under the parser, the ERROR " +
  "token has no grammar rule, so the parser rejects the file at exactly that position — the same " +
  "defect is caught in both configurations."));
children.push(mkTable(
  ["Situation", "Message (stderr)", "Justification"],
  [
    ["Invalid symbol (R7): a character no RE accepts, e.g. $ ? or a lone !",
     "lexical error at line N: invalid symbol 'c'",
     "Identifies the exact character and line; scanning continues so all errors are reported in one pass."],
    ["Unterminated triple-quoted string: EOF reached inside TRIPLE_D/TRIPLE_S",
     "ERROR token at EOF (parser: syntax error at the line of the opening quotes)",
     "An unclosed docstring would otherwise swallow the file silently; the explicit EOF rule makes the failure visible and locatable."],
  ],
  [3100, 3160, 3100]));
children.push(caption("Table #3: Lexical error messages and their justification."));

// ===== 3. Design =====
children.push(h1("3. Design"));
children.push(h2("3.1 Architecture"));
children.push(p(
  "Figure #1 shows the architecture. The lex specification lexer.l is compiled by flex into " +
  "lex.yy.c, a table-driven DFA scanner exposing yylex(). Two drivers consume it: scanner " +
  "(the SCANNER_MAIN driver embedded in lexer.l) produces this phase's deliverables — the token sequence and the symbol table " +
  "— and parser (the yacc grammar) consumes the same yylex() for syntax analysis. The shared " +
  "header y.tab.h (dashed arrows) carries the Token IDs to both, implementing the design decision " +
  "of Section 2.3. This architecture satisfies traceability by construction: there is exactly one " +
  "lexical definition for the whole front end."));
children.push(...figure(FIG("pipeline.png"), "Figure #1: Architecture of the lexical phase and its integration with the parser.", 620));

children.push(h2("3.2 Automata"));
children.push(p(
  "The scanner is the union of five class automata, one per requirement group. Splitting the " +
  "design by class keeps every automaton small and readable, while flex computes the single " +
  "combined DFA mechanically from their union — the construction seen in class, applied by the " +
  "tool. In every figure the initial state is S0, double circles are accepting states, and the " +
  "token produced is written next to them. Each figure is paired with its transition table in " +
  "Section 3.3; figures and tables are generated from one single machine-readable definition " +
  "(make_figures.py), so they cannot disagree with each other."));
children.push(...figure(FIG("dfa_identifiers.png"), "Figure #2: DFA for identifiers and keywords (R1, R2).", 360));
children.push(p(
  "Figure #2 recognizes identifiers. Keywords deliberately share this automaton: a keyword is " +
  "lexically an identifier, and the keyword/identifier decision is made by lex's tie-break rule " +
  "(earlier rule wins on equal length), not by a larger automaton. This is the most efficient " +
  "design: 2 states instead of a trie of 35 keywords with ~150 states."));
children.push(...figure(FIG("dfa_numbers.png"), "Figure #3: DFA for numeric literals (R3).", 620));
children.push(p(
  "Figure #3 recognizes INTEGER and the three FLOAT forms with seven states. States P1, N3 and N6 " +
  "are not accepting on purpose: they represent a lone dot, a dangling exponent marker, and a " +
  "dangling exponent sign. If the input ends there, maximal munch backtracks to the last " +
  "accepting state, which yields exactly Python's behavior (1e ⇒ INTEGER NAME, 1.e3 ⇒ FLOAT(1.) " +
  "NAME(e3) DOT-free, and a lone dot falls to the operator automaton as DOT)."));
children.push(...figure(FIG("dfa_strings.png"), "Figure #4: DFA for string literals, including triple-quoted docstrings (R4).", 620));
children.push(p(
  "Figure #4 is the automaton added in the documented design iteration (Section 1.2). Q1–Q2 " +
  "recognize single-line strings. Q2 is accepting and has an outgoing edge: on a third quote, " +
  "maximal munch continues into T0 instead of returning the empty string, which is precisely how " +
  "\"\"\"doc\"\"\" is one token while \"\" remains valid. The T-states accept any character " +
  "including newlines until three consecutive quotes close the string. In the implementation the " +
  "T-states are the lex start conditions TRIPLE_D and TRIPLE_S — a start condition is the lex " +
  "mechanism for giving the scanner explicit DFA states — and yymore() accumulates the full " +
  "lexeme. Reaching end of file inside a T-state triggers the unterminated-string error of Table " +
  "#3."));
children.push(...figure(FIG("dfa_operators.png"), "Figure #5: DFA for operators and delimiters, maximal-munch chains (R5).", 620));
children.push(p(
  "Figure #5 shows the operator chains. Multi-character operators are paths: < to LESS, <= to " +
  "LESS_EQUAL, << to LEFT_SHIFT, <<= to LEFT_SHIFT_ASSIGN; the GREATER and SLASH families are " +
  "shape-identical to LESS and STAR and are therefore drawn once (the transition table in Section " +
  "3.3 lists them all). Two states are intentionally non-accepting: X1, because a lone ! is not a " +
  "lexeme of the language (only != is), and P2, because .. is not one either — both fall through " +
  "to the error and backtracking rules. State K1 collects the ten single-character delimiters, " +
  "and O1/O2 generalize the simple op / op= pairs (+, &, |, ^, %)."));
children.push(...figure(FIG("dfa_layout.png"), "Figure #6: Comments, whitespace and the error catch-all (R6, R7).", 520));
children.push(p(
  "Figure #6 closes the union: comments are consumed to end of line without producing a token, " +
  "all whitespace including the newline is consumed (the layout-free decision of Section 2.1), " +
  "and the catch-all edge to R1 is the implementation of requirement R7 — every character not " +
  "claimed by Figures #2 to #5 becomes the ERROR token."));

children.push(h2("3.3 Transition tables and Token IDs"));
children.push(p(
  "Tables #5 to #9 are the transition tables of Figures #2 to #6, generated from the same " +
  "definitions. Efficiency argument: the columns are character classes rather than raw " +
  "characters, which compresses the alphabet from 256 columns to at most ten per table — the " +
  "same equivalence-class technique flex applies when it emits its internal tables — and the " +
  "per-class split keeps the whole design at 30 explicit states. A dash means 'no transition': " +
  "the scanner then either accepts (if the state is accepting), backtracks to the last accepting " +
  "configuration (maximal munch), or, only in S0, takes the error edge of Figure #6."));
let tno = 5;
for (const t of ttables) {
  const ncols = t.header.length;
  const stateW = 700, accW = 1860;
  const w = Math.floor((CONTENT_W - stateW - accW) / (ncols - 2));
  const widths = [stateW, ...Array(ncols - 2).fill(w), accW];
  widths[widths.length - 2] += CONTENT_W - widths.reduce((a, b) => a + b, 0);
  children.push(mkTable(t.header, t.rows, widths, 16));
  children.push(caption(`Table #${tno}: Transition table — ${t.title}.`));
  if (t.note) children.push(p(t.note));
  tno++;
}
children.push(p(
  "Table #4 lists the complete Token ID assignment extracted from the generated y.tab.h, " +
  "fulfilling the deliverable 'Tokens and their identification'. The ERROR token closes the " +
  "list; identifiers and literals additionally carry their lexeme into the symbol table of " +
  "Section 3.4."));
{
  const perCol = Math.ceil(tokenIds.length / 3);
  const rows = [];
  for (let i = 0; i < perCol; i++) {
    const row = [];
    for (let c = 0; c < 3; c++) {
      const e = tokenIds[c * perCol + i] || ["", ""];
      row.push(e[0], e[1]);
    }
    rows.push(row);
  }
  children.push(mkTable(
    ["Token", "ID", "Token", "ID", "Token", "ID"],
    rows, [2330, 790, 2330, 790, 2330, 790], 16));
  children.push(caption("Table #4: Token IDs as generated into y.tab.h (yacc numbering, starting at 257)."));
}

children.push(h2("3.4 Symbol table: data structure and algorithm"));
children.push(p(
  "The symbol table records every distinct lexeme of the four token kinds that carry a value — " +
  "NAME, INTEGER, FLOAT and STRING — because those are the tokens later phases must look up; " +
  "keywords and operators are fully identified by their Token ID and would only duplicate " +
  "information. Each entry is the record of Table #10."));
children.push(mkTable(
  ["Field", "Type", "Meaning"],
  [
    ["id", "int", "symbol number, assigned consecutively on first appearance"],
    ["token", "int", "Token ID from y.tab.h (NAME, INTEGER, FLOAT or STRING)"],
    ["lexeme", "char[64]", "the lexeme; lexemes longer than 63 characters (docstrings) are stored truncated with an ellipsis mark"],
    ["first_line", "int", "line of the first appearance (from yylineno)"],
    ["count", "int", "number of occurrences"],
  ],
  [1500, 1400, 6460]));
children.push(caption("Table #10: Symbol-table record (struct symbol in lexer.l's SCANNER_MAIN driver)."));
children.push(p(
  "The algorithm is insert-or-count: for every value-carrying token the driver searches the table " +
  "for an entry with the same token kind and lexeme; if found, its counter is incremented, " +
  "otherwise a new entry is appended with the next id and the current line. The pseudocode is:"));
children.push(...codeBlock(
`symtab_insert(token, lexeme, line):
    key = truncate(lexeme, 63)
    for each entry e in symtab:            # linear search
        if e.token == token and e.lexeme == key:
            e.count += 1
            return
    symtab.append( id=n+1, token, lexeme=key,
                   first_line=line, count=1 )`, 18));
children.push(p(
  "The lookup is a linear search over an array, an O(n) choice made deliberately: measured kernel " +
  "files declare a few dozen distinct symbols (test T1 yields 22), so a hash table would add " +
  "complexity without measurable benefit at this scale. The decision is encapsulated — " +
  "symtab_insert() is the only function that touches the array — so upgrading to hashing is a " +
  "one-function change, which is noted in the source. The table is printed after EOF, sorted by " +
  "construction in order of first appearance, which makes the output deterministic and easy to " +
  "diff in regression tests."));

children.push(h2("3.5 Traceability to the analysis"));
children.push(p(
  "Table #11 maps every requirement of Section 2 to its design artifact, to the exact rules of " +
  "the implementation, and to the test case that verifies it. This is the acid-test property " +
  "demanded of the design: a developer who never saw this project can go from any row to the " +
  "code and back."));
children.push(mkTable(
  ["Req.", "Analysis (RE)", "Design", "Implementation (lexer.l)", "Verified by"],
  [
    ["R1 identifiers", "Table #2 row 1", "Fig. #2 / Table #5", "IDENTIFIERS rule", "T1, T2"],
    ["R2 keywords", "Table #2 row 2", "Fig. #2 + rule order", "KEYWORDS block (35 rules)", "T1, T2"],
    ["R3 numbers", "Table #2 rows 3-4", "Fig. #3 / Table #6", "LITERALS block, 4 numeric rules", "T2"],
    ["R4 strings", "Table #2 rows 5-6", "Fig. #4 / Table #7", "LITERALS block + %x TRIPLE_D, TRIPLE_S", "T1, T2, T4"],
    ["R5 operators/delimiters", "Table #2 row 7", "Fig. #5 / Table #8", "OPERATORS .. DELIMITERS blocks (45 rules)", "T2"],
    ["R6 comments/whitespace", "Table #2 rows 8-9", "Fig. #6 / Table #9", "COMMENT and LAYOUT rules", "T1"],
    ["R7 invalid symbols", "Table #2 row 10, Table #3", "Fig. #6 / Table #9", "final '.' rule; ERROR token", "T3, T4"],
    ["R8 outputs (tokens + symbol table)", "Section 2.3", "Section 3.4 / Table #10", "SCANNER_MAIN driver", "T1-T4, corpus"],
  ],
  [1450, 1500, 1700, 3010, 1700], 16));
children.push(caption("Table #11: Traceability matrix from requirements to code and tests."));

// ===== 4. Implementation =====
children.push(h1("4. Implementation"));
children.push(p(
  "The scanner consists of the lex specification plus the build script: lexer.l includes both " +
  "the rules required for UNIX lex and the SCANNER_MAIN driver that turns the generated analyzer " +
  "into the deliverable program. The complete " +
  "printout of the lex file follows, explained by its three constituent sections; no regular " +
  "expression library or API is involved anywhere — flex generates the entire matching engine " +
  "from the rules below."));
children.push(h2("4.1 Definitions section"));
children.push(p([...runs(
  { text: "The definitions section configures the generated scanner and declares the named " +
          "sub-expressions used by the rules. " },
  m("#include \"y.tab.h\""),
  { text: " imports the Token IDs of Table #4 (the integration decision of Section 2.3). The " +
          "options remove the unused input/unput helpers, enable " },
  m("yylineno"),
  { text: " for error messages, and declare the two exclusive start conditions " },
  m("%x TRIPLE_D TRIPLE_S"),
  { text: " that realize the T-states of Figure #4. The three named definitions DIGIT, LETTER " +
          "and EXPONENT are the building blocks of Table #2." })]));
children.push(...codeBlock(lexDefs.trimEnd() + "\n%%"));
children.push(h2("4.2 Rules section"));
children.push(p(
  "The rules section is the formal specification of Section 2.2, transcribed rule by rule: each " +
  "line is one RE of Table #2 with the action that returns its Token ID. The order is " +
  "load-bearing and matches the design: keywords precede the identifier rule (tie-break of " +
  "Figure #2); the triple-quote openers precede the single-line string rules (longest match " +
  "prefers them anyway, the placement documents the intent); inside the start conditions, " +
  "yymore() accumulates the docstring so the token carries its complete lexeme; and the final " +
  "dot rule is the error edge of Figure #6 — it must be last because lex prefers earlier rules " +
  "on ties, and everything is a tie of length one against the dot."));
children.push(...codeBlock(lexRules.trimEnd() + "\n%%"));
children.push(h2("4.3 User code section and the driver"));
children.push(p(
  "The user code section of the lex file is intentionally a single comment: the project links the " +
  "same lex.yy.c under two different mains, so the driver lives outside the lex file. " +
  "the SCANNER_MAIN driver in lexer.l, printed below, contains the three components explained in the design: " +
  "token_name() (the inverse of Table #4, for readable output), the symbol table of Section 3.4 " +
  "(struct symbol, symtab_insert, symtab_print), and main(), the loop that calls yylex() until " +
  "EOF, prints each token of the sequence with its lexeme and line, feeds value-carrying tokens " +
  "to the symbol table, and implements the recover-and-continue error policy of Table #3. The " +
  "display() helper escapes newlines so that multi-line docstring lexemes do not break the " +
  "tabular output — tokens themselves are unaffected."));
children.push(...codeBlock(scannerSrc.trimEnd()));
children.push(h2("4.4 Build"));
children.push(p(
  "The Makefile builds both programs from the single lexical definition. yacc -d -v generates " +
  "y.tab.c, y.tab.h (the Token IDs) and y.output; flex generates lex.yy.c; cc links the scanner " +
  "driver and the parser driver separately:"));
children.push(...codeBlock(makefileSrc.trimEnd(), 16));

// ===== 5. V&V =====
children.push(h1("5. Verification and Validation"));
children.push(h2("5.1 Test model"));
children.push(p(
  "Verification asks whether the product was built right (does the scanner conform to the " +
  "specification of Section 2?); validation asks whether the right product was built (does it " +
  "scan real Triton kernels?). The test model therefore has two levels. Verification: five " +
  "designed test files (T1–T5), each targeting specific requirements, with expected outputs " +
  "recorded under tests/expected/ and compared on every build. Validation: the full corpus of " +
  "135 LLM-generated Triton programs from the TritonBench experiments, executed by the " +
  "run_corpus.py harness. The scanner must also pass the professor's test cases; the designed " +
  "set was chosen to cover every requirement so that any input exercising the specification " +
  "falls on an already-tested path."));
children.push(h2("5.2 Test case design"));
children.push(mkTable(
  ["Case", "Purpose (requirements)", "Input", "Expected output", "Result"],
  [
    ["T1", "happy path: a complete real kernel (R1-R6, R8)", "tests/t1_valid_kernel.py", "full token sequence, 22-entry symbol table, no errors; parser: ACCEPT", "PASS"],
    ["T2", "every literal form and all 45 operators (R3-R5)", "tests/t2_literals_operators.py", "all token kinds appear; no errors", "PASS"],
    ["T3", "invalid symbols (R7)", "tests/t3_invalid_symbols.py", "3 ERROR tokens with messages for $, ?, !; scanning continues; exit code 1", "PASS"],
    ["T4", "unterminated docstring (R4, R7)", "tests/t4_unterminated_string.py", "ERROR at EOF; exit code 1", "PASS"],
    ["T5", "lexically valid, syntactically invalid (integration)", "tests/t5_syntax_error.py", "scanner: clean token sequence; parser: REJECT with line", "PASS"],
    ["Corpus", "validation on real generated programs", "experiments/.../call_acc (135 files)", "all 55 kernel-bearing files scan; 52/55 parse (94.5%)", "PASS"],
  ],
  [800, 2600, 2300, 2860, 800], 16));
children.push(caption("Table #12: Test-case design."));

children.push(h2("5.3 Snapshots of the scanner outputs"));
children.push(p(
  "Snapshot 1 — T1, the deliverable outputs. The token sequence (excerpt) shows the decorator, " +
  "the function header, and the docstring carried as a single STRING token whose lexeme is the " +
  "whole string with escaped newlines; the symbol table follows, with the 22 distinct symbols of " +
  "the kernel, their Token kind, first line and occurrence count. The input file is the " +
  "add_kernel of Section 2."));
children.push(...codeBlock(
  t1out.split("\n").slice(0, 20).join("\n") +
  "\n        ...   (" + (t1out.split("\n").length - 20) + " more lines)\n\n" +
  t1out.split("\n").slice(-29).join("\n"), 15));
children.push(p(
  "Snapshot 2 — T3, error reporting. All three invalid symbols are reported with their line, " +
  "scanning continues after each one, and the process exit code is 1:"));
children.push(...codeBlock(t3out.trimEnd() + "\n\nstderr:\n" + t3err.trimEnd(), 15));
children.push(p(
  "Snapshot 3 — T4, unterminated docstring. The ERROR token carries the accumulated lexeme and " +
  "the exit code is 1, per Table #3:"));
children.push(...codeBlock(t4out.trimEnd(), 15));
children.push(p(
  "Snapshot 4 — T2 (excerpt). The literal forms of Table #2 each produce the designed token; the " +
  "full output is tests/expected/t2_literals_operators.scanner.out:"));
children.push(...codeBlock(t2out.split("\n").slice(0, 24).join("\n") + "\n        ...", 15));

children.push(h2("5.4 Validation on the TritonBench corpus"));
children.push(p(
  "The harness run_corpus.py runs the front end over the 135 generated programs of the " +
  "prompt12/haiku45 experiment in three stages and writes CORPUS_REPORT.md. Stage 1 measures " +
  "scope, not quality: every file starts with the same try/except host boilerplate, which is " +
  "outside the kernel language by specification (and 80 of the 135 files contain no Triton " +
  "kernel at all). Stage 2 is the design target — the extracted @triton.jit kernels — and stage " +
  "3 is a control with docstrings stripped. Table #13 summarizes the measurement, including the " +
  "documented iteration: before the docstring automaton of Figure #4 existed, stage 2 stood at " +
  "70.9%; after it, stages 2 and 3 coincide at 94.5%, proving the docstring failure class is " +
  "closed. The three remaining rejections are parser-level scope decisions documented in the " +
  "grammar design (the bare-return guard pattern, twice, and one conditional expression), not " +
  "lexical defects."));
children.push(mkTable(
  ["Stage", "Files", "Accepted", "Rate"],
  [
    ["1. Full files (host + kernel code)", "135", "0", "0% (out-of-scope host constructs)"],
    ["2. Extracted @triton.jit kernels — before docstring automaton", "55", "39", "70.9%"],
    ["2. Extracted @triton.jit kernels — final scanner", "55", "52", "94.5%"],
    ["3. Control: kernels with docstrings stripped", "55", "52", "94.5%"],
  ],
  [4760, 1300, 1300, 2000]));
children.push(caption("Table #13: Corpus validation results (135 generated TritonBench programs)."));

// ===== 6. Work plan =====
children.push(h1("6. Work Plan"));
children.push(p(
  "The plan follows the development model of Section 1.2; each phase ends at a verifiable " +
  "artifact, and the V&V phase feeds one controlled iteration back into design and " +
  "implementation, recorded in Table #14."));
children.push(mkTable(
  ["Phase", "Dates (2026)", "Activities", "Artifact"],
  [
    ["Analysis", "Jun 7 - 8", "informal lexical spec from real kernels; REs; token set; error policy", "Section 2; Table #1-#3"],
    ["Design", "Jun 8 - 9", "class automata; transition tables; token-id scheme via y.tab.h; symbol-table design; grammar design for integration", "Figures #1-#6; Tables #4-#11; grammar.md"],
    ["Implementation", "Jun 9 - 10", "lexer.l; parser.y; Makefile", "Section 4 sources"],
    ["Verification", "Jun 10", "test cases T1-T5 with expected outputs", "tests/, Table #12"],
    ["Validation + iteration", "Jun 10", "corpus run (70.9%); docstring automaton added (Fig. #4); corpus re-run (94.5%)", "run_corpus.py, CORPUS_REPORT.md, Table #13"],
    ["Report", "Jun 10", "this document, generated from the live artifacts", "make_report.js"],
  ],
  [1700, 1400, 4060, 2200], 16));
children.push(caption("Table #14: Work plan and produced artifacts."));

// ===== 7. References (disabled via INCLUDE_REFS) =====
if (INCLUDE_REFS) {
  children.push(h1("7. References"));
  [
    "[1] Alfred V. Aho, Monica S. Lam, Ravi Sethi, and Jeffrey D. Ullman, Compilers: Principles, Techniques, and Tools, 2nd. edition (Addison-Wesley, 2006), 109-189.",
    "[2] John R. Levine, Tony Mason, and Doug Brown, lex & yacc, 2nd. edition (O'Reilly Media, 1992), 1-83.",
    "[3] Philippe Tillet, Hsiang-Tsung Kung, and David Cox, “Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations”, Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (June 2019): 10-19.",
    "[4] Vern Paxson, Will Estes, and John Millaway, Lexical Analysis with Flex [manual on-line] (The Flex Project, 2012, accessed June 10 2026); available from https://westes.github.io/flex/manual/; Internet.",
    "[5] OpenAI, Triton Documentation [documentation on-line] (accessed June 10 2026); available from https://triton-lang.org/; Internet.",
    "[6] IEEE, IEEE Recommended Practice for Software Requirements Specifications, IEEE Std 830-1998 (IEEE, 1998).",
  ].forEach((r) => children.push(p(r, { alignment: AlignmentType.LEFT })));
}

// ------------------------------------------------------------- document
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true, run: { size: 32, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 240 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true, run: { size: 26, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 160 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: {
      page: { size: { width: 12240, height: 15840 },
              margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
      titlePage: true,
    },
    headers: {
      default: new Header({ children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "TC3002B - Compilers - Step I: Lexical Analysis (Triton)", size: 16, color: "666666" })] })] }),
      first: new Header({ children: [new Paragraph({ children: [] })] }),
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Page ", size: 18 }),
                   new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }),
      first: new Footer({ children: [new Paragraph({ children: [] })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "Lexical_Analysis_Report_Triton.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${Math.round(buf.length / 1024)} KB)`);
});
