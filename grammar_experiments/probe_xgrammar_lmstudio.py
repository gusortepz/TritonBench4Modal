from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import argparse
import json


DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
DEFAULT_GRAMMAR = Path("grammars/triton_python_xgrammar.ebnf")


VALID_SAMPLE = """import torch
import triton
import triton.language as tl

def add(input, other, *, alpha=1, out=None):
    return torch.add(input, other, alpha=alpha, out=out)
"""


INVALID_SAMPLE = "this is not python {{{\n"


def request_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
    )
    try:
        with urlopen(request, timeout=60 * 5) as response:
            return {
                "ok": True,
                "status": response.status,
                "body": json.loads(response.read().decode("utf-8")),
            }
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = body
        return {"ok": False, "status": exc.code, "body": parsed}


def native_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def compile_and_smoke_test(grammar_path: Path) -> dict:
    import xgrammar as xgr
    from xgrammar import testing

    grammar_text = grammar_path.read_text(encoding="utf-8")
    grammar = xgr.Grammar.from_ebnf(grammar_text, root_rule_name="root")
    return {
        "grammar_path": str(grammar_path),
        "compiled": True,
        "valid_sample_accepted": bool(testing._is_grammar_accept_string(grammar, VALID_SAMPLE)),
        "invalid_sample_accepted": bool(testing._is_grammar_accept_string(grammar, INVALID_SAMPLE)),
    }


def probe_lmstudio(base_url: str, model: str) -> dict:
    tiny_grammar = 'root ::= "OK"'
    openai_url = f"{base_url.rstrip('/')}/chat/completions"
    native_url = f"{native_base_url(base_url)}/api/v1/chat"
    messages = [{"role": "user", "content": "Output OK only."}]

    top_level_grammar = request_json(
        openai_url,
        {
            "model": model,
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "grammar": tiny_grammar,
        },
    )
    response_format_grammar = request_json(
        openai_url,
        {
            "model": model,
            "messages": messages,
            "max_tokens": 8,
            "temperature": 0,
            "stream": False,
            "response_format": {"type": "grammar", "grammar": tiny_grammar},
        },
    )
    native_grammar = request_json(
        native_url,
        {
            "model": model,
            "input": "Output OK only.",
            "temperature": 0,
            "grammar": tiny_grammar,
        },
    )
    return {
        "base_url": base_url,
        "model": model,
        "probes": {
            "openai_top_level_grammar": summarize_probe(top_level_grammar),
            "openai_response_format_grammar": summarize_probe(response_format_grammar),
            "native_top_level_grammar": summarize_probe(native_grammar),
        },
    }


def summarize_probe(result: dict) -> dict:
    body = result["body"]
    content = None
    error = None
    if isinstance(body, dict):
        error = body.get("error")
        try:
            message = body["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content")
        except (KeyError, IndexError, TypeError):
            pass
    return {
        "ok": result["ok"],
        "status": result["status"],
        "content_preview": content[:200] if isinstance(content, str) else content,
        "error": error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grammar", type=Path, default=DEFAULT_GRAMMAR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", type=Path, default=Path("outputs/xgrammar_lmstudio_probe.json"))
    parser.add_argument("--skip-lmstudio", action="store_true")
    args = parser.parse_args()

    result = {"xgrammar": compile_and_smoke_test(args.grammar)}
    if not args.skip_lmstudio:
        result["lmstudio"] = probe_lmstudio(args.base_url, args.model)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
