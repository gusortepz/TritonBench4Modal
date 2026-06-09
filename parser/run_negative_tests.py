#!/usr/bin/env python3
"""Run intentionally invalid Triton-shaped snippets through the parser."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class NegativeCase:
    name: str
    description: str
    source: str


BROKEN_CASES = [
    NegativeCase(
        name="missing_function_colon.py",
        description="function definition opens an indented suite without a colon",
        source="""\
import triton
import triton.language as tl

@triton.jit
def broken_kernel(ptr, out)
    pid = tl.program_id(0)
    tl.store(out + pid, tl.load(ptr + pid))
""",
    ),
    NegativeCase(
        name="unclosed_tl_load.py",
        description="call expression leaves a parenthesis open",
        source="""\
import triton
import triton.language as tl

@triton.jit
def broken_kernel(ptr, out):
    offsets = tl.arange(0, 16)
    x = tl.load(ptr + offsets
    tl.store(out + offsets, x)
""",
    ),
    NegativeCase(
        name="crossed_delimiters.py",
        description="bracket and parenthesis delimiters cross",
        source="""\
import triton
import triton.language as tl

@triton.jit
def broken_kernel(ptr, out):
    offsets = tl.arange(0, 16)
    x = tl.load(ptr + offsets])
    tl.store(out + offsets, x)
""",
    ),
    NegativeCase(
        name="indent_after_assignment.py",
        description="an assignment incorrectly introduces an indented block",
        source="""\
import triton
import triton.language as tl

x = 1
    y = 2
""",
    ),
    NegativeCase(
        name="bad_dedent_width.py",
        description="dedent returns to a width that was never opened",
        source="""\
import triton
import triton.language as tl

@triton.jit
def broken_kernel(ptr):
    if True:
        x = tl.load(ptr)
      y = x
""",
    ),
    NegativeCase(
        name="extra_close_paren.py",
        description="statement has an extra closing parenthesis",
        source="""\
import triton
import triton.language as tl

@triton.jit
def broken_kernel(ptr, out):
    offsets = tl.arange(0, 16)
    tl.store(out + offsets, tl.load(ptr + offsets)))
""",
    ),
]


def run_case(parser: Path, path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(parser), str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = repo_root / "parser" / "triton_parser"
    if not parser.exists():
        raise SystemExit(f"missing parser binary: {parser}. Run `make -C parser` first.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = repo_root / "parser" / "results" / f"negative_triton_{timestamp}"
    case_dir = result_dir / "cases"
    stderr_dir = result_dir / "stderr"
    case_dir.mkdir(parents=True)
    stderr_dir.mkdir()

    rows = []
    failed_expectations = []

    for case in BROKEN_CASES:
        case_path = case_dir / case.name
        case_path.write_text(case.source, encoding="utf-8")

        result = run_case(parser, case_path)
        detected = result.returncode != 0
        if not detected:
            failed_expectations.append(case.name)

        stderr_path = stderr_dir / f"{case.name}.stderr.txt"
        stderr_path.write_text(result.stderr, encoding="utf-8")

        rows.append(
            {
                "name": case.name,
                "description": case.description,
                "return_code": str(result.returncode),
                "detected": "yes" if detected else "no",
                "stderr": result.stderr.strip().replace("\t", " "),
            }
        )

    with (result_dir / "results.tsv").open("w", encoding="utf-8") as output:
        output.write("name\tdetected\treturn_code\tdescription\tstderr\n")
        for row in rows:
            output.write(
                "{name}\t{detected}\t{return_code}\t{description}\t{stderr}\n".format(
                    **row
                )
            )

    detected_count = sum(1 for row in rows if row["detected"] == "yes")
    with (result_dir / "summary.md").open("w", encoding="utf-8") as output:
        output.write("# Negative Triton Parser Test\n\n")
        output.write(f"- Output directory: `{result_dir.relative_to(repo_root)}`\n")
        output.write(f"- Invalid snippets tested: {len(BROKEN_CASES)}\n")
        output.write(f"- Detected by parser: {detected_count}\n")
        output.write(f"- Missed by parser: {len(failed_expectations)}\n\n")
        output.write("## Cases\n\n")
        for row in rows:
            output.write(
                f"- `{row['name']}`: detected={row['detected']}, "
                f"return_code={row['return_code']}, {row['description']}\n"
            )
        if failed_expectations:
            output.write("\n## Missed Cases\n\n")
            for name in failed_expectations:
                output.write(f"- `{name}`\n")

    print(result_dir)
    print(
        f"invalid={len(BROKEN_CASES)} detected={detected_count} "
        f"missed={len(failed_expectations)}"
    )
    return 1 if failed_expectations else 0


if __name__ == "__main__":
    raise SystemExit(main())
