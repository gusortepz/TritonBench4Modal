#!/usr/bin/env python3
"""Run the starter yacc parser against generated experiment Python files."""

from __future__ import annotations

import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse experiments/**/*.py with parser/triton_parser."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of parser/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for parse results. Defaults to parser/results/experiments_parse_<timestamp>.",
    )
    return parser.parse_args()


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    experiments_dir = repo_root / "experiments"
    parser_bin = repo_root / "parser" / "triton_parser"

    if not parser_bin.exists():
        raise SystemExit(f"missing parser binary: {parser_bin}. Run `make -C parser` first.")
    if not experiments_dir.exists():
        raise SystemExit(f"missing experiments directory: {experiments_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "parser" / "results" / f"experiments_parse_{timestamp}"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(experiments_dir.rglob("*.py"))
    file_rows = ["path\tstatus\treturncode\tstderr\n"]
    failed_rows: list[str] = []

    for source in files:
        rel = relative_to_root(source, repo_root)
        result = subprocess.run(
            [str(parser_bin), str(source)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        status = "ok" if result.returncode == 0 else "failed"
        stderr_text = result.stderr.strip().replace("\t", " ").replace("\n", " | ")
        file_rows.append(f"{rel}\t{status}\t{result.returncode}\t{stderr_text}\n")
        if status != "ok":
            failed_rows.append(f"{rel}\n")

    files_list = [relative_to_root(path, repo_root) + "\n" for path in files]
    write_lines(output_dir / "files.txt", files_list)
    write_lines(output_dir / "files.tsv", file_rows)
    write_lines(output_dir / "failed_files.txt", failed_rows)

    summary = [
        "# Parser Experiment Scan\n",
        "\n",
        f"- Output directory: `{relative_to_root(output_dir, repo_root)}`\n",
        "- Parser mode: structural lines, balanced delimiters, and colon block headers\n",
        f"- Python files parsed: {len(files)}\n",
        f"- Failed files: {len(failed_rows)}\n",
        "\n",
        "## Result Files\n",
        "\n",
        "- `files.txt`: parsed source files\n",
        "- `files.tsv`: per-file status, parser return code, stderr\n",
        "- `failed_files.txt`: files rejected by the parser\n",
    ]
    write_lines(output_dir / "summary.md", summary)

    print(output_dir)
    print(f"files={len(files)} failed={len(failed_rows)}")
    return 0 if not failed_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
