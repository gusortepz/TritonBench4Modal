from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import argparse
import json
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grammars.triton_elementwise import START_RULES, start_rule_for_operation


DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_SEED_JSONL = Path(
    "experiments/lmstudio_prompt11_router_20260527-194836/"
    "lmstudio_qwen_qwen3.6-35b-a3b_simp_limit20_20260527-194844.jsonl"
)
DEFAULT_OPERATIONS = ("add", "sub", "sqrt", "rsqrt", "tanh", "relu_sqrt")


def read_prompt_header(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'PROMPT_HEADER\s*=\s*"""(.*?)"""\s*(?:\.strip\(\))?\s*$',
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


def extract_code(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip() + "\n"


def request_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
    )
    try:
        with urlopen(request, timeout=60 * 30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio request failed: {exc.code} {body}") from exc


def native_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def build_messages(instruction: str, prompt_header: str) -> list[dict]:
    return [
        {"role": "system", "content": prompt_header},
        {"role": "user", "content": instruction},
    ]


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    del max_tokens
    system_text = "\n\n".join(
        str(message["content"]) for message in messages if message["role"] == "system"
    )
    user_text = "\n\n".join(
        str(message["content"]) for message in messages if message["role"] != "system"
    )
    prompt = f"{system_text}\n\n{user_text}" if system_text else user_text
    data = request_json(
        f"{native_base_url(base_url)}/api/v1/chat",
        {
            "model": model,
            "input": prompt,
            "context_length": 32768,
            "temperature": temperature,
        },
    )
    output = data.get("output", [])
    parts = [
        part.get("content", "")
        for part in output
        if part.get("type") == "message" and part.get("content")
    ]
    return "\n\n".join(parts).strip()


def structured_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    operation: str,
    code_variants: list[str],
    max_tokens: int,
    temperature: float,
) -> str:
    data = request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": f"triton_{operation}_kernel",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "enum": code_variants}
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    message = data["choices"][0]["message"]
    text = message.get("content") or message.get("reasoning_content") or ""
    try:
        parsed = json.loads(text)
        return parsed["code"].strip() + "\n"
    except json.JSONDecodeError:
        for variant in code_variants:
            if variant.strip() in text:
                return variant
        match = re.search(r"BLOCK_SIZE=(128|256|512|1024)", text)
        if match:
            return operation_code(operation, match.group(1))
        raise


def general_structured_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    data = request_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "tritonbench_general_module",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": (
                                    "A complete self-contained Python module for the requested "
                                    "TritonBench task. No markdown fences, tests, examples, or prose."
                                ),
                            }
                        },
                        "required": ["code"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )
    message = data["choices"][0]["message"]
    text = message.get("content") or message.get("reasoning_content") or ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))
    return extract_code(parsed["code"])


def load_records(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit > 0 and len(records) == limit:
                break
    if limit > 0 and len(records) < limit:
        raise ValueError(f"{path} has {len(records)} records, expected at least {limit}")
    return records


def load_existing_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    tmp.replace(path)


def function_name(instruction: str) -> str:
    match = re.search(r"Wrapper Entry Information:\s*(?:def\s+)?([A-Za-z_][\w.]*)", instruction)
    return match.group(1) if match else ""


def canonical_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def select_operation_records(records: list[dict], operations: tuple[str, ...]) -> list[dict]:
    wanted = set(operations)
    selected = [
        record
        for record in records
        if canonical_name(function_name(record["instruction"])) in wanted
    ]
    seen = {canonical_name(function_name(record["instruction"])) for record in selected}
    missing = [operation for operation in operations if operation not in seen]
    if missing:
        raise ValueError(f"missing selected operation(s): {', '.join(missing)}")
    return sorted(
        selected,
        key=lambda record: operations.index(canonical_name(function_name(record["instruction"]))),
    )


def block_sizes_from_grammar(grammar_path: Path) -> list[str]:
    text = grammar_path.read_text(encoding="utf-8")
    match = re.search(r"block_size\s*::=([^\n]+)", text)
    if not match:
        raise ValueError(f"no block_size rule found in {grammar_path}")
    sizes = re.findall(r'"(\d+)"', match.group(1))
    if not sizes:
        raise ValueError(f"no block_size alternatives found in {grammar_path}")
    return sizes


def binary_code(operation: str, expression: str, block_size: str) -> str:
    return f"""import torch
import triton
import triton.language as tl

@triton.jit
def _{operation}_kernel(input_ptr, other_ptr, output_ptr, n_elements, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    result = {expression}
    tl.store(output_ptr + offsets, result, mask=mask)

def {operation}(input, other, *, alpha=1, out=None):
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
    n_elements = input.numel()
    if n_elements == 0:
        return output
    if torch.is_tensor(other):
        other_tensor = other.contiguous()
    else:
        other_tensor = torch.full_like(input, other)
    input_contig = input.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _{operation}_kernel[grid](input_contig, other_tensor, output, n_elements, alpha, BLOCK_SIZE={block_size})
    return output
"""


def unary_code(operation: str, expression: str, block_size: str) -> str:
    return f"""import torch
import triton
import triton.language as tl

@triton.jit
def _{operation}_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    result = {expression}
    tl.store(output_ptr + offsets, result, mask=mask)

def {operation}(input, *, out=None):
    if out is None:
        output = torch.empty_like(input)
    else:
        output = out
    n_elements = input.numel()
    if n_elements == 0:
        return output
    input_contig = input.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _{operation}_kernel[grid](input_contig, output, n_elements, BLOCK_SIZE={block_size})
    return output
"""


def relu_sqrt_code(block_size: str) -> str:
    return f"""import torch
import triton
import triton.language as tl

@triton.jit
def _relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    relu = tl.maximum(x, 0.0)
    result = tl.sqrt(relu)
    tl.store(output_ptr + offsets, result, mask=mask)

def relu_sqrt(input, inplace=False, out=None):
    if input.dtype != torch.float32 and input.dtype != torch.float64:
        result = torch.sqrt(torch.relu(input.float()))
        if out is not None:
            out.copy_(result)
            return out
        return result
    if inplace:
        output = input
    elif out is None:
        output = torch.empty_like(input)
    else:
        output = out
    n_elements = input.numel()
    if n_elements == 0:
        return output
    input_contig = input.contiguous()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _relu_sqrt_kernel[grid](input_contig, output, n_elements, BLOCK_SIZE={block_size})
    return output
"""


def operation_code(operation: str, block_size: str) -> str:
    if operation == "add":
        return binary_code("add", "x + alpha * y", block_size)
    if operation == "sub":
        return binary_code("sub", "x - alpha * y", block_size)
    if operation == "sqrt":
        return unary_code("sqrt", "tl.sqrt(x)", block_size)
    if operation == "rsqrt":
        return unary_code("rsqrt", "tl.rsqrt(x)", block_size)
    if operation == "tanh":
        return unary_code("tanh", "2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0", block_size)
    if operation == "relu_sqrt":
        return relu_sqrt_code(block_size)
    raise ValueError(f"unsupported operation {operation!r}")


def grammar_code_variants(operation: str, grammar_path: Path) -> list[str]:
    if operation not in START_RULES:
        supported = ", ".join(sorted(START_RULES))
        raise ValueError(f"unsupported operation {operation!r}; expected one of: {supported}")
    return [operation_code(operation, size) for size in block_sizes_from_grammar(grammar_path)]


def general_grammar_code_variants(
    operations: tuple[str, ...],
    grammar_path: Path,
) -> list[str]:
    variants: list[str] = []
    for operation in operations:
        variants.extend(grammar_code_variants(operation, grammar_path))
    return variants


def generate_baseline(
    *,
    records: list[dict],
    prompt_header: str,
    output_path: Path,
    base_url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    resume: bool,
    label: str,
) -> list[dict]:
    existing_by_instruction = {}
    if resume and output_path.exists():
        existing_by_instruction = {
            record["instruction"]: record for record in load_existing_records(output_path)
        }

    output: list[dict | None] = [
        existing_by_instruction.get(record["instruction"]) for record in records
    ]
    pending = [(idx, record) for idx, record in enumerate(records) if output[idx] is None]

    def do_one(idx_record: tuple[int, dict]) -> tuple[int, dict]:
        idx, record = idx_record
        messages = build_messages(record["instruction"], prompt_header)
        raw = chat_completion(
            base_url=base_url,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if not raw.strip():
            raise RuntimeError(f"empty LM Studio response for {label} item {idx + 1}")
        return idx, {"instruction": record["instruction"], "predict": extract_code(raw)}

    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = [executor.submit(do_one, item) for item in pending]
            for future in as_completed(futures):
                idx, record = future.result()
                output[idx] = record
                done += 1
                write_records(output_path, [r for r in output if r is not None])
                print(f"{label} baseline {done}/{len(pending)} generated", flush=True)

    final = [record for record in output if record is not None]
    write_records(output_path, final)
    return final


def generate_grammar_records(
    *,
    records: list[dict],
    prompt_header: str,
    output_path: Path,
    code_dir: Path,
    grammar_path: Path,
    grammar_mode: str,
    operations: tuple[str, ...],
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    resume: bool,
    label: str,
) -> list[dict]:
    existing_by_instruction = {}
    if resume and output_path.exists():
        existing_by_instruction = {
            record["instruction"]: record for record in load_existing_records(output_path)
        }

    grammar_text = grammar_path.read_text(encoding="utf-8")
    output: list[dict | None] = [
        existing_by_instruction.get(record["instruction"]) for record in records
    ]

    for idx, record in enumerate(records):
        if output[idx] is not None:
            continue
        operation = canonical_name(function_name(record["instruction"]))
        if grammar_mode == "specific":
            start_rule = start_rule_for_operation(operation)
            variants = grammar_code_variants(operation, grammar_path)
            grammar_scope = f"the `{operation}` module"
            decoder_note = f"start rule `{start_rule}`"
        elif grammar_mode == "general":
            start_rule = "root"
            variants = general_grammar_code_variants(operations, grammar_path)
            grammar_scope = "one of the supported elementwise modules"
            decoder_note = "general start rule `root`"
        elif grammar_mode == "structured-general":
            start_rule = "json_schema_code"
            variants = []
            grammar_scope = "one TritonBench Python module"
            decoder_note = "a general JSON schema with a single `code` string field"
        else:
            raise ValueError("grammar_mode must be 'specific', 'general', or 'structured-general'")
        grammar_note = (
            f"Generate {grammar_scope} for this TritonBench task. The decoder is "
            f"constrained by {decoder_note}. The public wrapper function must match the "
            "benchmark request exactly. Return a complete self-contained Python module, "
            "with imports, helper kernels if needed, and no tests or examples."
        )
        if grammar_mode != "structured-general":
            grammar_note += (
                " Choose the best valid BLOCK_SIZE and output only code accepted by "
                f"that grammar.\n\nGrammar:\n{grammar_text}"
            )
        messages = [
            {"role": "system", "content": prompt_header},
            {"role": "user", "content": f"{grammar_note}\n\n{record['instruction']}"},
        ]
        if grammar_mode == "structured-general":
            code = general_structured_completion(
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        else:
            code = structured_completion(
                base_url=base_url,
                model=model,
                messages=messages,
                operation=operation,
                code_variants=variants,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        output[idx] = {"instruction": record["instruction"], "predict": code}
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / f"{operation}_{label}.py").write_text(code, encoding="utf-8")
        write_records(output_path, [r for r in output if r is not None])
        print(f"{label} grammar {idx + 1}/{len(records)} generated ({operation})", flush=True)

    final = [record for record in output if record is not None]
    write_records(output_path, final)
    return final


def parse_operations(value: str) -> tuple[str, ...]:
    operations = tuple(item.strip() for item in value.split(",") if item.strip())
    return operations or DEFAULT_OPERATIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-jsonl", type=Path, default=DEFAULT_SEED_JSONL)
    parser.add_argument("--prompt0", type=Path, default=Path("prompt-0.txt"))
    parser.add_argument("--prompt11", type=Path, default=Path("prompt-11-router.txt"))
    parser.add_argument("--grammar", type=Path, default=Path("grammars/triton_elementwise.ebnf"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/grammar_elementwise_compare"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--operations", default=",".join(DEFAULT_OPERATIONS))
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--grammar-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--refresh-grammar", action="store_true")
    parser.add_argument(
        "--grammar-mode",
        choices=("specific", "general", "structured-general"),
        default="specific",
        help=(
            "specific uses per-operation start rules; general uses root for all supported ops; "
            "structured-general only constrains the JSON response shape"
        ),
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    operations = parse_operations(args.operations)
    all_records = load_records(args.seed_jsonl, args.limit)
    if args.list:
        for idx, record in enumerate(all_records, start=1):
            print(f"{idx:3d} {function_name(record['instruction'])}")
        return

    seed_records = select_operation_records(all_records, operations)
    prompt0_header = read_prompt_header(args.prompt0)
    prompt11_header = read_prompt_header(args.prompt11)
    resume = not args.no_resume

    stem = f"elementwise{len(seed_records)}"
    prompt0_path = args.out_dir / f"prompt0_{stem}.jsonl"
    prompt11_path = args.out_dir / f"prompt11_{stem}.jsonl"
    if args.grammar_mode == "specific":
        grammar_suffix = "grammar"
    elif args.grammar_mode == "general":
        grammar_suffix = "general_grammar"
    else:
        grammar_suffix = "general_constraints"
    prompt0_grammar_path = args.out_dir / f"prompt0_plus_{grammar_suffix}_{stem}.jsonl"
    prompt11_grammar_path = args.out_dir / f"prompt11_plus_{grammar_suffix}_{stem}.jsonl"

    prompt0_records = generate_baseline(
        records=seed_records,
        prompt_header=prompt0_header,
        output_path=prompt0_path,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        resume=resume,
        label="prompt0",
    )
    prompt11_records = generate_baseline(
        records=seed_records,
        prompt_header=prompt11_header,
        output_path=prompt11_path,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        resume=resume,
        label="prompt11",
    )

    prompt0_grammar_records = generate_grammar_records(
        records=seed_records,
        prompt_header=prompt0_header,
        output_path=prompt0_grammar_path,
        code_dir=args.out_dir / "code" / f"prompt0_plus_{grammar_suffix}",
        grammar_path=args.grammar,
        grammar_mode=args.grammar_mode,
        operations=operations,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.grammar_max_tokens,
        temperature=args.temperature,
        resume=resume and not args.refresh_grammar,
        label="prompt0",
    )
    prompt11_grammar_records = generate_grammar_records(
        records=seed_records,
        prompt_header=prompt11_header,
        output_path=prompt11_grammar_path,
        code_dir=args.out_dir / "code" / f"prompt11_plus_{grammar_suffix}",
        grammar_path=args.grammar,
        grammar_mode=args.grammar_mode,
        operations=operations,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.grammar_max_tokens,
        temperature=args.temperature,
        resume=resume and not args.refresh_grammar,
        label="prompt11",
    )

    manifest = {
        "model": args.model,
        "base_url": args.base_url,
        "seed_jsonl": str(args.seed_jsonl),
        "grammar": str(args.grammar),
        "operations": list(operations),
        "grammar_mode": args.grammar_mode,
        "start_rules": {operation: start_rule_for_operation(operation) for operation in operations},
        "outputs": {
            "prompt0": str(prompt0_path),
            "prompt11": str(prompt11_path),
            "prompt0_plus_grammar": str(prompt0_grammar_path),
            "prompt11_plus_grammar": str(prompt11_grammar_path),
        },
        "counts": {
            "prompt0": len(prompt0_records),
            "prompt11": len(prompt11_records),
            "prompt0_plus_grammar": len(prompt0_grammar_records),
            "prompt11_plus_grammar": len(prompt11_grammar_records),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
