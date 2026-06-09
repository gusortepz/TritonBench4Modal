#!/usr/bin/env python3
"""Run the Triton lexer against generated experiment Python files."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path


TOKEN_LINE_RE = re.compile(r"^\s*(\d+):(\d+)\s+([A-Z_]+)\s?(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan experiments/**/*.py with lexer/triton_lexer."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root. Defaults to the parent of lexer/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for scan results. Defaults to lexer/results/experiments_scan_<timestamp>.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="How many token samples to save for files containing @triton.jit.",
    )
    return parser.parse_args()


def relative_to_root(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def run_lexer(lexer: Path, source: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(lexer), str(source)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    experiments_dir = repo_root / "experiments"
    lexer = repo_root / "lexer" / "triton_lexer"

    if not lexer.exists():
        raise SystemExit(f"missing lexer binary: {lexer}. Run `make -C lexer` first.")
    if not experiments_dir.exists():
        raise SystemExit(f"missing experiments directory: {experiments_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "lexer" / "results" / f"experiments_scan_{timestamp}"
    output_dir = output_dir.resolve()
    samples_dir = output_dir / "sample_tokens"
    samples_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(experiments_dir.rglob("*.py"))
    file_rows = ["path\tstatus\ttoken_count\terror_count\tstderr\n"]
    error_rows = ["path\tline\tcolumn\ttext\n"]
    failed_rows: list[str] = []
    token_counts: Counter[str] = Counter()
    sample_count = 0

    for source in files:
        rel = relative_to_root(source, repo_root)
        result = run_lexer(lexer, source)
        token_count = 0
        error_count = 0

        for line in result.stdout.splitlines():
            match = TOKEN_LINE_RE.match(line)
            if not match:
                continue
            token_count += 1
            token_name = match.group(3)
            token_text = match.group(4)
            token_counts[token_name] += 1
            if token_name == "ERROR":
                error_count += 1
                error_rows.append(
                    f"{rel}\t{match.group(1)}\t{match.group(2)}\t{token_text}\n"
                )

        status = "ok" if result.returncode == 0 and error_count == 0 else "failed"
        stderr_text = result.stderr.strip().replace("\t", " ").replace("\n", " | ")
        file_rows.append(f"{rel}\t{status}\t{token_count}\t{error_count}\t{stderr_text}\n")

        if status != "ok":
            failed_rows.append(f"{rel}\n")

        if sample_count < args.sample_limit and "@triton.jit" in source.read_text(
            encoding="utf-8", errors="replace"
        ):
            sample_name = rel.replace("/", "__") + ".tokens.txt"
            (samples_dir / sample_name).write_text(result.stdout, encoding="utf-8")
            sample_count += 1

    token_count_rows = ["token\tcount\n"]
    for token, count in sorted(token_counts.items()):
        token_count_rows.append(f"{token}\t{count}\n")

    files_list = [relative_to_root(path, repo_root) + "\n" for path in files]
    write_lines(output_dir / "files.txt", files_list)
    write_lines(output_dir / "files.tsv", file_rows)
    write_lines(output_dir / "errors.tsv", error_rows)
    write_lines(output_dir / "failed_files.txt", failed_rows)
    write_lines(output_dir / "token_counts.tsv", token_count_rows)

    total_errors = len(error_rows) - 1
    total_failed = len(failed_rows)
    total_tokens = sum(token_counts.values())
    summary = [
        "# Lexer Experiment Scan\n",
        "\n",
        f"- Output directory: `{relative_to_root(output_dir, repo_root)}`\n",
        f"- Python files scanned: {len(files)}\n",
        f"- Total tokens emitted: {total_tokens}\n",
        f"- Files with lexer errors: {total_failed}\n",
        f"- Total `ERROR` tokens: {total_errors}\n",
        f"- Token samples saved: {sample_count}\n",
        "\n",
        "## Result Files\n",
        "\n",
        "- `files.txt`: scanned source files\n",
        "- `files.tsv`: per-file status, token count, error count, stderr\n",
        "- `errors.tsv`: all lexer `ERROR` tokens with location\n",
        "- `failed_files.txt`: files with lexer errors or nonzero lexer exit\n",
        "- `token_counts.tsv`: aggregate token distribution\n",
        "- `sample_tokens/`: representative token streams from Triton kernels\n",
    ]
    write_lines(output_dir / "summary.md", summary)

    print(output_dir)
    print(f"files={len(files)} tokens={total_tokens} failed={total_failed} errors={total_errors}")
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
