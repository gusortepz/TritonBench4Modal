#!/bin/sh
# run_suite.sh - run the scanner + parser over the full test suite and write a
# timestamped report to results/. Invoked by `make suite`.
#
#   Part A: 55 generated Triton kernels (tests/corpus_kernels/) - parser ACCEPT/REJECT
#   Part B: 5 designed unit tests (tests/t*.py) - scanner/parser vs expected outcome
#
# A kernel "passes" when the parser accepts it (exit 0). The 3 known REJECTs are
# documented out-of-scope grammar cases (return-guard x2, conditional expr x1),
# not lexical defects - see grammar.md and CORPUS_REPORT.md.

cd "$(dirname "$0")" || exit 1

TS=$(date +%Y%m%d-%H%M%S)
OUT="results/suite_${TS}.txt"
mkdir -p results

{
  echo "================================================================"
  echo " Triton scanner + parser - full test suite"
  echo " $(date '+%Y-%m-%d %H:%M:%S')"
  echo "================================================================"
  echo

  # ---------- Part A: corpus kernels ----------
  echo "## Part A - corpus kernels (tests/corpus_kernels/, 55 files)"
  echo
  acc=0; rej=0
  for f in tests/corpus_kernels/*.py; do
    name=$(basename "$f")
    if ./parser "$f" >/dev/null 2>&1; then
      acc=$((acc + 1))
      printf "  %-46s ACCEPT\n" "$name"
    else
      rej=$((rej + 1))
      where=$(./parser "$f" 2>&1 | grep -o 'line [0-9]*' | head -1)
      printf "  %-46s REJECT (%s)\n" "$name" "$where"
    fi
  done
  total=$((acc + rej))
  rate=$(awk "BEGIN{printf \"%.1f\", 100*$acc/$total}")
  echo
  echo "  corpus result: $acc / $total accepted (${rate}%)"
  echo

  # ---------- Part B: designed unit tests ----------
  echo "## Part B - designed unit tests (expected vs actual)"
  echo
  printf "  %-26s %-6s %-22s %s\n" "case" "tool" "expected / actual" "result"

  # check  <case>  <tool: scan|parse>  <expected: ok|err>  <purpose>
  pass=0; fail=0
  check() {
    f="tests/$1.py"
    if [ "$2" = "scan" ]; then
      ./scanner "$f" >/dev/null 2>&1
    else
      ./parser "$f" >/dev/null 2>&1
    fi
    [ $? -eq 0 ] && act=ok || act=err
    if [ "$act" = "$3" ]; then r=PASS; pass=$((pass + 1)); else r=FAIL; fail=$((fail + 1)); fi
    printf "  %-26s %-6s exp=%-3s act=%-12s %s\n" "$1" "$2" "$3" "$act" "$r"
  }
  check t1_valid_kernel        parse ok    # full valid kernel -> ACCEPT
  check t2_literals_operators  scan  ok    # every literal/operator -> no lexical error
  check t3_invalid_symbols     scan  err   # $ ? ! -> lexical errors reported
  check t4_unterminated_string scan  err   # unterminated string -> error
  check t5_syntax_error        parse err   # lexically clean, syntactically invalid -> REJECT
  echo
  echo "  designed result: $pass / $((pass + fail)) passed"
  echo

  echo "================================================================"
  echo " SUMMARY: corpus $acc/$total (${rate}%) | designed $pass/$((pass + fail))"
  echo "================================================================"
} | tee "$OUT"

echo
echo "results written to $OUT"
