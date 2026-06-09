from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


DEFAULT_GRAMMAR_PATH = Path(__file__).with_name("triton_elementwise.ebnf")

START_RULES: dict[str, str] = {
    "add": "add_root",
    "sub": "sub_root",
    "sqrt": "sqrt_root",
    "rsqrt": "rsqrt_root",
    "tanh": "tanh_root",
    "relu_sqrt": "relu_sqrt_root",
}

SUPPORTED_OPERATIONS = tuple(START_RULES)


@dataclass(frozen=True)
class ElementwiseGrammarSelection:
    operation: str
    start_rule: str
    grammar_path: Path
    grammar: str
    prompt: str

    def as_generation_kwargs(self) -> dict[str, str]:
        """Payload shape expected by the local generation integration layer."""
        return {
            "prompt": self.prompt,
            "grammar": self.grammar,
            "start_rule": self.start_rule,
        }


def load_grammar(grammar_path: Path | str = DEFAULT_GRAMMAR_PATH) -> str:
    return Path(grammar_path).read_text()


def start_rule_for_operation(operation: str) -> str:
    try:
        return START_RULES[operation]
    except KeyError as exc:
        supported = ", ".join(SUPPORTED_OPERATIONS)
        raise ValueError(f"unsupported operation {operation!r}; expected one of: {supported}") from exc


def detect_operation(source: str) -> str:
    text = source.lower()

    declared = _declared_function_name(text)
    if declared in START_RULES:
        return declared

    if "relu_sqrt" in text or "relu-sqrt" in text or ("relu" in text and "sqrt" in text):
        return "relu_sqrt"
    if _mentions(text, "torch.rsqrt", "rsqrt"):
        return "rsqrt"
    if _mentions(text, "torch.add", "add"):
        return "add"
    if _mentions(text, "torch.sub", "sub"):
        return "sub"
    if _mentions(text, "torch.sqrt", "sqrt"):
        return "sqrt"
    if _mentions(text, "torch.tanh", "tanh"):
        return "tanh"

    supported = ", ".join(SUPPORTED_OPERATIONS)
    raise ValueError(f"unsupported operation; expected one of: {supported}")


def select_elementwise_grammar(
    source: str,
    *,
    prompt: str | None = None,
    grammar_path: Path | str = DEFAULT_GRAMMAR_PATH,
) -> ElementwiseGrammarSelection:
    operation = detect_operation(source)
    path = Path(grammar_path)
    return ElementwiseGrammarSelection(
        operation=operation,
        start_rule=start_rule_for_operation(operation),
        grammar_path=path,
        grammar=load_grammar(path),
        prompt=source if prompt is None else prompt,
    )


def build_generation_payload(
    source: str,
    *,
    prompt: str | None = None,
    grammar_path: Path | str = DEFAULT_GRAMMAR_PATH,
) -> dict[str, str]:
    """Return grammar content and selected start rule for a Qwen/XGrammar call."""
    return select_elementwise_grammar(
        source,
        prompt=prompt,
        grammar_path=grammar_path,
    ).as_generation_kwargs()


def generate_with_xgrammar(
    generate_fn,
    source: str,
    *,
    prompt: str | None = None,
    grammar_path: Path | str = DEFAULT_GRAMMAR_PATH,
    **generation_kwargs,
):
    """Integration point for the existing local Qwen call.

    `generate_fn` is the existing backend-specific generation function. It must
    accept `prompt`, `grammar`, and `start_rule` keyword arguments, or adapt
    those names before calling XGrammar in the inference engine.
    """
    payload = build_generation_payload(source, prompt=prompt, grammar_path=grammar_path)
    return generate_fn(**payload, **generation_kwargs)


def _declared_function_name(text: str) -> str | None:
    match = re.search(
        r"wrapper entry information:\s*(?:def\s+)?(?:torch\.)?([a-z_][\w.]*)",
        text,
    )
    if not match:
        match = re.search(r"\bdef\s+([a-z_][\w.]*)\s*\(", text)
    if not match:
        return None
    return match.group(1).split(".")[-1]


def _mentions(text: str, torch_name: str, bare_name: str) -> bool:
    return torch_name in text or re.search(rf"\b{re.escape(bare_name)}\s*\(", text) is not None
