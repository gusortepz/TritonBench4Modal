#!/usr/bin/env python3
"""Check that the CFG and yacc parser mention every lexer token."""

from __future__ import annotations

import re
from pathlib import Path


TOKEN_RE = re.compile(r"^\s*T_([A-Z][A-Z0-9_]*)(?:\s*=\s*\d+)?\s*,", re.MULTILINE)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    token_header = repo_root / "lexer" / "triton_tokens.h"
    grammar = repo_root / "grammar" / "triton_kernel_cfg.md"
    parser = repo_root / "parser" / "triton_parser.y"

    token_text = token_header.read_text(encoding="utf-8")
    grammar_text = grammar.read_text(encoding="utf-8")
    parser_text = parser.read_text(encoding="utf-8")

    tokens = TOKEN_RE.findall(token_text)
    grammar_missing = [
        token for token in tokens if not re.search(rf"\b{re.escape(token)}\b", grammar_text)
    ]
    parser_missing = [
        token for token in tokens if not re.search(rf"\bT_{re.escape(token)}\b", parser_text)
    ]

    if grammar_missing or parser_missing:
        if grammar_missing:
            print("Missing CFG mentions for lexer tokens:")
            for token in grammar_missing:
                print(f"- {token}")
        if parser_missing:
            print("Missing yacc parser mentions for lexer tokens:")
            for token in parser_missing:
                print(f"- T_{token}")
        return 1

    print(
        f"All {len(tokens)} lexer tokens are mentioned in "
        f"{grammar.relative_to(repo_root)} and {parser.relative_to(repo_root)}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
