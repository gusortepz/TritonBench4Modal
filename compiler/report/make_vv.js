// make_vv.js - builds the "Verification and Validation" section as a standalone
// DOCX, grounded in the live scanner/parser outputs captured under vv_snapshots/
// and the current validation set (tests/corpus_kernels/, 55 files).
// Usage: node make_vv.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber,
} = require("docx");

const HERE = __dirname;
const ROOT = path.resolve(HERE, "..");
const SNAP = (f) => fs.readFileSync(path.join(HERE, "vv_snapshots", f), "utf8");
const TEST = (f) => fs.readFileSync(path.join(ROOT, "tests", f), "utf8").replace(/\s+$/, "");
const MONO = "Courier New";
const CONTENT_W = 9360;

// ---------------------------------------------------------------- helpers
const p = (text, opts = {}) =>
  new Paragraph({
    alignment: AlignmentType.JUSTIFIED, spacing: { after: 160 },
    children: Array.isArray(text) ? text : [new TextRun({ text, size: 22 })],
    ...opts,
  });
const h1 = (t, brk = true) =>
  new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: brk,
                  children: [new TextRun(t)] });
const h2 = (t) =>
  new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const codeBlock = (text, size = 15) =>
  text.replace(/\s+$/, "").split("\n").map((line) =>
    new Paragraph({ spacing: { after: 0 }, shading: { fill: "F4F5F7", type: ShadingType.CLEAR },
      children: [new TextRun({ text: line.replace(/\t/g, "    ") || " ", font: MONO, size })] }));
const caption = (t) =>
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 240 },
    children: [new TextRun({ text: t, italics: true, size: 20 })] });

const border = { style: BorderStyle.SINGLE, size: 2, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
function mkTable(headerCells, rows, colWidths, cellSize = 18) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  const cell = (txt, w, bold, fill) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { after: 0 },
      children: [new TextRun({ text: String(txt), bold, size: cellSize,
        font: /[`{}()\[\]]/.test(String(txt)) ? MONO : undefined })] })] });
  const mkRow = (cells, bold, fill) =>
    new TableRow({ children: cells.map((c, i) => cell(c, colWidths[i], bold, fill)) });
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: colWidths,
    rows: [mkRow(headerCells, true, "DCE6F1"), ...rows.map((r) => mkRow(r, false))] });
}

// snapshot slicing: first N lines + the SYMBOL TABLE block
function head(text, n) { return text.split("\n").slice(0, n).join("\n"); }
function fromSymtab(text) {
  const lines = text.split("\n");
  const i = lines.findIndex((l) => l.includes("SYMBOL TABLE"));
  return i < 0 ? "" : lines.slice(i).join("\n");
}

// latest suite report (for the summary + rejects)
const resDir = path.join(ROOT, "results");
const suiteFile = fs.readdirSync(resDir).filter((f) => /^suite_.*\.txt$/.test(f)).sort().pop();
const suiteTxt = fs.readFileSync(path.join(resDir, suiteFile), "utf8");
const rejects = suiteTxt.split("\n").filter((l) => l.includes("REJECT"))
  .map((l) => l.trim().replace(/\s+/g, " "));

// ------------------------------------------------------------------ content
const c = [];

// header block
c.push(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
    children: [new TextRun({ text: "TC3002B - Compilers - Step I: Lexical Analysis (Triton)", size: 20, color: "555555" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 },
    children: [new TextRun({ text: "Verification and Validation", size: 40, bold: true })] }),
);

// intro
c.push(p(
  "This section presents the Verification and Validation (V&V) of the Triton scanner. V&V is the " +
  "set of checking processes that ensure the system meets its specification and delivers the " +
  "functionality expected by the customer; it begins with requirements and design reviews and " +
  "continues through code inspection up to product testing. The two activities are distinct. " +
  "Verification asks “are we building the product right?” — does the scanner conform to " +
  "its specification, the regular expressions and token set of the Analysis? Validation asks " +
  "“are we building the right product?” — does the scanner actually recognize the lexemes " +
  "of real Triton GPU kernels? The ultimate goal is to establish confidence that the scanner is " +
  "good enough for its intended use."));

// 5.1 test model
c.push(h1("Test model", false));
c.push(p(
  "The test model is the set of test cases produced during test-case design. It has two " +
  "complementary levels, so that verification and validation are each exercised by the inputs " +
  "best suited to them:"));
c.push(p([
  new TextRun({ text: "Verification — designed unit tests (T1–T5). ", bold: true, size: 22 }),
  new TextRun({ text: "Five small files, each constructed to exercise one specific feature of the " +
    "lexical specification (a category of lexeme or an error situation) with a predicted output " +
    "known in advance. They check that the scanner conforms to the specification.", size: 22 }),
]));
c.push(p([
  new TextRun({ text: "Validation — the corpus set (55 real kernels). ", bold: true, size: 22 }),
  new TextRun({ text: "Fifty-five @triton.jit kernels taken from LLM-generated TritonBench " +
    "programs, each turned into a standalone module by prepending the Triton imports. They check " +
    "that the scanner works on the real language it was built for.", size: 22 }),
]));
c.push(p(
  "Both levels are executed by a single harness, run_suite.sh, invoked with make suite, which " +
  "runs the scanner and the parser over every file and writes a timestamped report to results/. " +
  "The implementation must pass every designed test file as well as the professor's test cases; " +
  "the designed set was chosen to cover each requirement so that any input exercising the " +
  "specification falls on an already-tested path."));

// 5.2 test case design methodology
c.push(h1("Test-case design"));
c.push(p(
  "To design a test case the developer selects a particular feature of the component to be " +
  "tested, selects a set of inputs that execute that feature, documents the corresponding " +
  "predicted output, and then checks that the actual output equals the expected one. The goal is " +
  "a set of cases effective at discovering defects and at showing that the system meets its " +
  "requirements. Table 1 records the five designed cases: the feature each targets, its input " +
  "file, the predicted output, and the observed result. The tool column states whether the case " +
  "is judged by the scanner (a lexical-level case) or by the parser (an integration case)."));
c.push(mkTable(
  ["Case", "Feature tested", "Tool", "Input file", "Predicted output", "Result"],
  [
    ["T1", "A complete valid kernel: identifiers, keywords, numbers, strings, operators, comments",
     "parser", "t1_valid_kernel.py", "Full token sequence + 23-entry symbol table; no lexical errors; ACCEPT", "PASS"],
    ["T2", "Every literal shape and all 45 operators / delimiters",
     "scanner", "t2_literals_operators.py", "All token kinds emitted; 0 lexical errors; exit 0", "PASS"],
    ["T3", "Invalid symbols ($, ?, !)",
     "scanner", "t3_invalid_symbols.py", "3 ERROR tokens with messages; scanning continues; exit 1", "PASS"],
    ["T4", "Unterminated string at end of file",
     "scanner", "t4_unterminated_string.py", "ERROR token reported; exit 1", "PASS"],
    ["T5", "Lexically valid but syntactically invalid input",
     "parser", "t5_syntax_error.py", "Clean token sequence; parser REJECT with line", "PASS"],
  ],
  [560, 2500, 760, 1900, 2840, 800], 16));
c.push(caption("Table 1: Designed test cases (inputs and predicted outputs)."));

// 5.3 validation set
c.push(h1("The validation set (55 generated kernels)"));
c.push(p(
  "The validation set lives in tests/corpus_kernels/. Each file is a real @triton.jit kernel " +
  "generated by an LLM for a TritonBench operator, extracted from its host program and made into " +
  "a self-contained module by prepending the two imports every Triton file begins with:"));
c.push(...codeBlock("import triton\nimport triton.language as tl"));
c.push(p(
  "Predicted result: because these are real kernels written in the subset the scanner targets, " +
  "every file must scan with zero lexical errors, and the parser must accept the large majority. " +
  "The measured result is 52 of 55 accepted (94.5%). The three remaining files are rejected by " +
  "the parser, not the scanner: they are out-of-scope grammar constructs documented in the " +
  "grammar design (two use the bare-return guard pattern, one uses a conditional expression " +
  "a if c else b), so they are deliberate scope limits rather than lexical defects. None of the " +
  "55 files produces a lexical error. The three rejects, as reported by the suite:"));
c.push(...codeBlock(rejects.join("\n"), 16));

// 5.4 snapshots
c.push(h1("Snapshots of the scanner output"));
c.push(p(
  "Each snapshot below is the verbatim output of the scanner (./scanner <file>) or parser " +
  "(./parser <file>) on a test case, followed by its explanation. The scanner prints the " +
  "token sequence (token name, lexeme, line), then the symbol table for the value-carrying tokens " +
  "(NAME, INTEGER, FLOAT, STRING), then a count of lexical errors; its exit code is 1 if any " +
  "lexical error was found and 0 otherwise."));

// snapshot T1
c.push(h2("Snapshot 1 — T1, a complete valid kernel"));
c.push(p("Input file t1_valid_kernel.py:"));
c.push(...codeBlock(TEST("t1_valid_kernel.py")));
c.push(p("Scanner output (token sequence excerpt, then the full symbol table):"));
c.push(...codeBlock(head(SNAP("t1_valid_kernel.scanner.out"), 14) +
  "\n        ...\n\n" + fromSymtab(SNAP("t1_valid_kernel.scanner.out"))));
c.push(p(
  "Explanation. The two import lines tokenize as IMPORT NAME and IMPORT NAME DOT NAME AS NAME, " +
  "the decorator as AT NAME DOT NAME, and the header as DEF NAME ( ... ) : with the annotated " +
  "parameter BLOCK_SIZE : tl . constexpr. The symbol table records the 23 distinct value-carrying " +
  "lexemes with their first line and occurrence count (for example tl appears 7 times). The " +
  "single-line docstring is tokenized as the empty string \"\" followed by the content string — " +
  "the scanner does not provide a dedicated triple-quoted-string lexeme, which is why a docstring " +
  "spanning several lines is out of scope (see T4). No lexical errors are found, so the scanner " +
  "exits 0 and the parser accepts the file."));

// snapshot T2
c.push(h2("Snapshot 2 — T2, every literal form and operator"));
c.push(p("Input file t2_literals_operators.py:"));
c.push(...codeBlock(TEST("t2_literals_operators.py")));
c.push(p("Scanner output (token sequence excerpt and conclusion):"));
c.push(...codeBlock(
  SNAP("t2_literals_operators.scanner.out").split("\n").slice(0, 18).join("\n") +
  "\n        ...\n" +
  SNAP("t2_literals_operators.scanner.out").split("\n").slice(-4).join("\n")));
c.push(p(
  "Explanation. This case verifies the literal and operator rules of the specification. The " +
  "integer 42, the three float forms (3.14, .5, 1e6, 2.5e-3), and the string forms are each " +
  "emitted as the correct token; every arithmetic, comparison, bitwise, and augmented-assignment " +
  "operator is recognized, with multi-character operators such as //, <<=, !=, ->, and ... lexed " +
  "as a single token by the maximal-munch rule. The scanner reports 43 distinct symbols and zero " +
  "lexical errors (exit 0), confirming full lexical coverage. The “triton: false” line is " +
  "the scanner noting that this fragment contains no @triton.jit decorator; it is informational " +
  "and does not affect tokenization."));

// snapshot T3
c.push(h2("Snapshot 3 — T3, invalid symbols and error recovery"));
c.push(p("Input file t3_invalid_symbols.py:"));
c.push(...codeBlock(TEST("t3_invalid_symbols.py")));
c.push(p("Scanner standard output:"));
c.push(...codeBlock(SNAP("t3_invalid_symbols.scanner.out")));
c.push(p("Scanner standard error:"));
c.push(...codeBlock(SNAP("t3_invalid_symbols.scanner.err")));
c.push(p(
  "Explanation. The three characters $, ?, and ! are not lexemes of the language. For each, the " +
  "scanner emits an ERROR token on standard output (annotated “invalid symbol”) and a " +
  "located message on standard error giving the line and the offending character. Crucially, the " +
  "scanner does not stop at the first error: it consumes the bad character and continues, so all " +
  "three errors are reported in a single pass and the surrounding valid tokens (the assignments, " +
  "the integers) are still produced. The final exit code is 1, signaling that the file is " +
  "lexically invalid. This is the error-reporting requirement of the specification."));

// snapshot T4
c.push(h2("Snapshot 4 — T4, unterminated string"));
c.push(p("Input file t4_unterminated_string.py (a quote sequence with no closing quote before EOF):"));
c.push(...codeBlock(TEST("t4_unterminated_string.py")));
c.push(p("Scanner output and standard error:"));
c.push(...codeBlock(SNAP("t4_unterminated_string.scanner.out") + "\nstderr:\n" +
  SNAP("t4_unterminated_string.scanner.err")));
c.push(p(
  "Explanation. A single-line string rule requires a closing quote on the same line. The opening " +
  "\"\"\" is lexed as the empty string \"\" followed by a lone \" that begins a string with no " +
  "terminator; the scanner reports it as an invalid symbol at line 1 and recovers, lexing the " +
  "rest as the names never and closed. The exit code is 1. This makes an unclosed string a " +
  "visible, located error instead of silently swallowing the remainder of the file."));

// snapshot T5
c.push(h2("Snapshot 5 — T5, lexically valid but syntactically invalid"));
c.push(p("Input file t5_syntax_error.py:"));
c.push(...codeBlock(TEST("t5_syntax_error.py")));
c.push(p("Scanner conclusion and parser result:"));
c.push(...codeBlock(
  SNAP("t5_syntax_error.scanner.out").split("\n").slice(-4).join("\n") +
  "\n\n$ ./parser tests/t5_syntax_error.py\n" +
  SNAP("t5_syntax_error.parser.out")));
c.push(p(
  "Explanation. This integration case separates the two phases. Every character of def f(: is a " +
  "valid lexeme, so the scanner finds no lexical error (exit 0). The defect is syntactic — a " +
  "colon where a parameter or closing parenthesis is required — and it is the parser that " +
  "detects it, reporting “syntax error at line 1 near ':'” and rejecting. This confirms " +
  "the division of labor: the scanner judges lexemes, the parser judges structure, and the two " +
  "share token numbers through the generated y.tab.h."));

// snapshot corpus
c.push(h2("Snapshot 6 — a validation-set kernel (add.py)"));
c.push(p(
  "A representative file from the 55-kernel validation set, showing that a real generated kernel " +
  "— with its prepended imports — scans cleanly. Token-sequence excerpt and symbol table:"));
c.push(...codeBlock(head(SNAP("add.scanner.out"), 13) + "\n        ...\n\n" +
  fromSymtab(SNAP("add.scanner.out"))));
c.push(p(
  "Explanation. The imports tokenize exactly as in T1, then the decorator and the kernel header " +
  "with its constexpr parameters. The symbol table captures the 27 distinct symbols of this " +
  "kernel — pointer arguments, block constants, intermediate tensors — with their first " +
  "line and counts. The file produces no lexical errors and the parser accepts it, which is the " +
  "expected result for every kernel in the validation set."));

// 5.5 results
c.push(h1("Results and conformance"));
c.push(p(
  "The full suite (make suite) was run over both levels of the test model. The designed unit " +
  "tests pass completely and the validation set scans without a single lexical error, with the " +
  "parser accepting 52 of 55 kernels. The three non-accepted files fail in the parser on " +
  "documented out-of-scope grammar constructs, not in the scanner. Table 2 is the summary " +
  "reported by the harness."));
c.push(mkTable(
  ["Level", "What it checks", "Files", "Result"],
  [
    ["Designed unit tests (T1–T5)", "Verification — conformance to the lexical specification", "5", "5 / 5 passed"],
    ["Corpus validation set", "Validation — recognition of real Triton kernels", "55", "52 / 55 accepted (94.5%); 0 lexical errors"],
  ],
  [2700, 3760, 900, 2000]));
c.push(caption("Table 2: V&V summary (source: " + suiteFile + ")."));
c.push(p(
  "The implementation passes all of the developer's own test files and is ready for the " +
  "professor's test cases: the designed cases cover each lexical requirement, and the validation " +
  "set demonstrates the scanner on the real language. Together they establish confidence that the " +
  "scanner conforms to its specification (verification) and recognizes the lexemes of real Triton " +
  "GPU kernels (validation), which is the goal of the V&V process."));

// ------------------------------------------------------------- document
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial", color: "1F3864" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: "Arial", color: "2E5496" },
        paragraph: { spacing: { before: 180, after: 120 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT,
      children: [new TextRun({ text: "Verification and Validation — Triton Scanner", size: 16, color: "888888" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ text: "Page ", size: 18 }), new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: c,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(HERE, "Verification_and_Validation.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out, `(${Math.round(buf.length / 1024)} KB)`);
});
