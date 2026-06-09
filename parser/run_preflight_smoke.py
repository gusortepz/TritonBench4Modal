#!/usr/bin/env python3
"""Smoke test the predictions preflight gate."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


VALID_RECORD = {
    "instruction": (
        "Functional Description: Adds two tensors.\n"
        "Wrapper Entry Information: add(input, other) -> Tensor."
    ),
    "predict": (
        "```python\n"
        "import torch\n\n"
        "def add(input, other):\n"
        "    return torch.add(input, other)\n"
        "```"
    ),
}

INVALID_RECORD = {
    "instruction": (
        "Functional Description: Adds two tensors.\n"
        "Wrapper Entry Information: add(input, other) -> Tensor."
    ),
    "predict": (
        "```python\n"
        "import torch\n\n"
        "def add(input, other)\n"
        "    return torch.add(input, other)\n"
        "```"
    ),
}


def write_jsonl(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def run_validator(repo_root: Path, predictions: Path, output_dir: Path) -> int:
    validator = repo_root / "parser" / "validate_predictions.py"
    result = subprocess.run(
        [
            "python3",
            str(validator),
            str(predictions),
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    return result.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="triton_preflight_smoke_") as tmp:
        tmpdir = Path(tmp)
        valid_path = tmpdir / "valid.jsonl"
        invalid_path = tmpdir / "invalid.jsonl"
        write_jsonl(valid_path, VALID_RECORD)
        write_jsonl(invalid_path, INVALID_RECORD)

        valid_rc = run_validator(repo_root, valid_path, tmpdir / "valid_results")
        invalid_rc = run_validator(repo_root, invalid_path, tmpdir / "invalid_results")

    if valid_rc != 0:
        print("expected valid predictions JSONL to pass preflight")
        return 1
    if invalid_rc == 0:
        print("expected invalid predictions JSONL to fail preflight")
        return 1

    print("Predictions preflight smoke passed: valid accepted, invalid rejected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
