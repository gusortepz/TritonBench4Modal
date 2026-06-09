#!/usr/bin/env python3
"""Small parser smoke tests for accepted and rejected syntax shapes."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


VALID_CASES = {
    "simple_assignment.py": "x = f(a[0], {'k': (1, 2)})\n",
    "missing_final_newline.py": "result = kernel[grid](x, y)",
    "explicit_line_join.py": "x = 1 + " + chr(92) + "\n    2\n",
    "indented_block.py": "if x:\n    y = (x + 1)\n",
    "decorated_kernel.py": "@triton.jit\ndef kernel(x):\n    return x\n",
}

INVALID_CASES = {
    "unclosed_paren.py": "x = f(a[0], {'k': (1, 2)}\n",
    "extra_close.py": "x = value)\n",
    "crossed_delimiters.py": "x = ([1, 2)]\n",
    "assignment_block.py": "x = 1\n    y = 2\n",
    "missing_header_colon.py": "if x\n    y = 2\n",
}


def run_case(parser: Path, path: Path) -> int:
    result = subprocess.run(
        [str(parser), str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = repo_root / "parser" / "triton_parser"
    if not parser.exists():
        raise SystemExit(f"missing parser binary: {parser}. Run `make -C parser` first.")

    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="triton_parser_smoke_") as tmp:
        tmpdir = Path(tmp)

        for name, source in VALID_CASES.items():
            path = tmpdir / name
            path.write_text(source, encoding="utf-8")
            if run_case(parser, path) != 0:
                failures.append(f"expected valid but rejected: {name}")

        for name, source in INVALID_CASES.items():
            path = tmpdir / name
            path.write_text(source, encoding="utf-8")
            if run_case(parser, path) == 0:
                failures.append(f"expected invalid but accepted: {name}")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"Parser smoke tests passed: {len(VALID_CASES)} valid, {len(INVALID_CASES)} invalid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
