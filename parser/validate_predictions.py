#!/usr/bin/env python3
"""Preflight predictions JSONL with the Triton Flex/Bison frontend."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path


TOKEN_LINE_RE = re.compile(r"^\s*(\d+):(\d+)\s+([A-Z_]+)\s?(.*)$")
FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)\n```", re.DOTALL)
ENTRY_RE = re.compile(r"Wrapper Entry Information:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract each prediction module from a JSONL file and validate it "
            "with lexer/triton_lexer and parser/triton_parser before Modal upload."
        )
    )
    parser.add_argument("predictions", type=Path, help="Local predictions JSONL path.")
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
        help=(
            "Directory for preflight results. Defaults to "
            "parser/results/predictions_preflight_<timestamp>."
        ),
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use existing parser/lexer binaries instead of running make first.",
    )
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow empty extracted modules to pass through to lexer/parser.",
    )
    return parser.parse_args()


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def extract_code(text: str) -> str:
    source = text.strip()
    match = FENCE_RE.search(source)
    if match:
        return match.group(1).strip() + "\n"
    source = re.sub(r"^```(?:python|py)?\s*\n?", "", source)
    source = re.sub(r"\n?```\s*$", "", source)
    return source.strip() + "\n"


def infer_entry_name(instruction: str, fallback: str) -> str:
    match = ENTRY_RE.search(instruction)
    if match:
        return match.group(1)
    return fallback


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return name.strip("._-") or "prediction"


def run_command(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def build_frontend(repo_root: Path) -> None:
    for command in (["make", "-C", "parser"], ["make", "-C", "lexer"]):
        result = run_command(command, cwd=repo_root)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if result.returncode != 0:
            raise SystemExit(f"build failed: {' '.join(command)}")


def count_lexer_tokens(stdout: str) -> tuple[int, int]:
    token_count = 0
    error_count = 0
    for line in stdout.splitlines():
        match = TOKEN_LINE_RE.match(line)
        if not match:
            continue
        token_count += 1
        if match.group(3) == "ERROR":
            error_count += 1
    return token_count, error_count


def compact(text: str) -> str:
    return text.strip().replace("\t", " ").replace("\n", " | ")


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record) + "\n")


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    predictions = args.predictions.resolve()
    if not predictions.exists():
        raise SystemExit(f"predictions file not found: {predictions}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = repo_root / "parser" / "results" / f"predictions_preflight_{timestamp}"
    output_dir = output_dir.resolve()
    cases_dir = output_dir / "cases"
    stderr_dir = output_dir / "stderr"
    cases_dir.mkdir(parents=True, exist_ok=True)
    stderr_dir.mkdir(exist_ok=True)

    if not args.skip_build:
        build_frontend(repo_root)

    lexer_bin = repo_root / "lexer" / "triton_lexer"
    parser_bin = repo_root / "parser" / "triton_parser"
    if not lexer_bin.exists():
        raise SystemExit(f"missing lexer binary: {lexer_bin}. Run `make -C lexer`.")
    if not parser_bin.exists():
        raise SystemExit(f"missing parser binary: {parser_bin}. Run `make -C parser`.")

    result_rows = [
        "line\tentry\tstatus\tstage\ttoken_count\tlexer_errors\tlexer_rc\tparser_rc\tmessage\tsource_path\n"
    ]
    accepted_records: list[dict] = []
    failed_records: list[dict] = []
    total = 0
    passed = 0
    failed = 0

    with predictions.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue

            total += 1
            entry_name = f"line_{line_number}"
            source_path = cases_dir / f"{line_number:04d}_{entry_name}.py"
            status = "failed"
            stage = "json"
            token_count = 0
            lexer_errors = 0
            lexer_rc = -1
            parser_rc = -1
            message = ""
            record: dict = {}

            try:
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError("record is not a JSON object")
                record = loaded
                instruction = record.get("instruction", "")
                if not isinstance(instruction, str):
                    instruction = ""
                entry_name = infer_entry_name(instruction, f"line_{line_number}")
                source_path = cases_dir / f"{line_number:04d}_{safe_name(entry_name)}.py"
                predict = record.get("predict", "")
                if not isinstance(predict, str):
                    raise ValueError("record field `predict` is not a string")

                source = extract_code(predict)
                source_path.write_text(source, encoding="utf-8")
                if not source.strip() and not args.allow_empty:
                    stage = "extract"
                    message = "empty extracted module"
                    raise ValueError(message)

                lexer_result = run_command([str(lexer_bin), str(source_path)], cwd=repo_root)
                lexer_rc = lexer_result.returncode
                token_count, lexer_errors = count_lexer_tokens(lexer_result.stdout)
                (stderr_dir / f"{line_number:04d}_{safe_name(entry_name)}.lexer.txt").write_text(
                    lexer_result.stderr,
                    encoding="utf-8",
                )
                if lexer_rc != 0 or lexer_errors:
                    stage = "lexer"
                    message = compact(lexer_result.stderr) or f"{lexer_errors} lexer ERROR tokens"
                    raise ValueError(message)

                parser_result = run_command([str(parser_bin), str(source_path)], cwd=repo_root)
                parser_rc = parser_result.returncode
                (stderr_dir / f"{line_number:04d}_{safe_name(entry_name)}.parser.txt").write_text(
                    parser_result.stderr,
                    encoding="utf-8",
                )
                if parser_rc != 0:
                    stage = "parser"
                    message = compact(parser_result.stderr) or "parser returned nonzero"
                    raise ValueError(message)

                status = "ok"
                stage = "passed"
                passed += 1
                accepted_records.append(record)
            except Exception as exc:  # noqa: BLE001 - preflight must preserve diagnostics.
                failed += 1
                status = "failed"
                if not message:
                    message = str(exc)
                if record:
                    failed_records.append(record)

            result_rows.append(
                "\t".join(
                    [
                        str(line_number),
                        entry_name,
                        status,
                        stage,
                        str(token_count),
                        str(lexer_errors),
                        str(lexer_rc),
                        str(parser_rc),
                        message.replace("\t", " "),
                        relative_to_root(source_path, repo_root),
                    ]
                )
                + "\n"
            )

    (output_dir / "results.tsv").write_text("".join(result_rows), encoding="utf-8")
    write_jsonl(output_dir / "accepted_predictions.jsonl", accepted_records)
    write_jsonl(output_dir / "failed_predictions.jsonl", failed_records)

    summary_lines = [
        "# Predictions Preflight\n",
        "\n",
        f"- Input predictions: `{relative_to_root(predictions, repo_root)}`\n",
        f"- Output directory: `{relative_to_root(output_dir, repo_root)}`\n",
        "- Validator: lexer/triton_lexer + parser/triton_parser\n",
        f"- Total records: {total}\n",
        f"- Passed: {passed}\n",
        f"- Failed: {failed}\n",
        "\n",
        "## Result Files\n",
        "\n",
        "- `results.tsv`: per-record validation status\n",
        "- `cases/`: extracted Python modules\n",
        "- `stderr/`: lexer and parser stderr per record\n",
        "- `accepted_predictions.jsonl`: records that passed preflight\n",
        "- `failed_predictions.jsonl`: records that failed preflight\n",
    ]
    (output_dir / "summary.md").write_text("".join(summary_lines), encoding="utf-8")

    print(output_dir)
    print(f"records={total} passed={passed} failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
