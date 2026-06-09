from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import argparse
import json
import re
import time


DEFAULT_MODEL = "qwen/qwen3.6-35b-a3b"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
FAMILIES = {
    "add": ["add"],
    "first20_elementwise": ["tanh", "relu_sqrt", "sqrt", "sub", "rsqrt", "add"],
    "unary_elementwise": [
        "abs",
        "cos",
        "erf",
        "exp",
        "floor",
        "log",
        "reciprocal",
        "relu",
        "rsqrt",
        "sigmoid",
        "sqrt",
        "tanh",
        "trunc",
    ],
    "binary_elementwise": ["add", "sub", "mul", "div", "pow"],
    "prompt11_triton_whitelist": [
        "relu_sqrt",
        "sqrt_exp",
        "exp_sqrt",
        "log_tanh",
        "mul_sub",
        "add_gelu",
        "sub_gelu",
        "mul_relu",
        "combined_activation",
        "rad2deg_sqrt",
        "rsqrt",
        "selu",
    ],
}


def read_prompt_header(path: Path) -> str:
    text = path.read_text()
    match = re.search(
        r'PROMPT_HEADER\s*=\s*"""(.*?)"""\s*(?:\.strip\(\))?\s*$',
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


def extract_code(text: str) -> str:
    s = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", s, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    s = re.sub(r"^```(?:python|py)?\s*\n?", "", s)
    s = re.sub(r"\n?```\s*$", "", s)
    return s.strip() + "\n"


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
    message_parts = [
        part.get("content", "")
        for part in output
        if part.get("type") == "message" and part.get("content")
    ]
    return "\n\n".join(message_parts).strip()


def openai_chat_completion(
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
        },
    )
    message = data["choices"][0]["message"]
    return message.get("content") or message.get("reasoning_content") or ""


def structured_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
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
                    "name": "triton_add_kernel",
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
            return add_code(match.group(1))
        raise


def load_records(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
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
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    tmp.replace(path)


def function_name(instruction: str) -> str:
    match = re.search(r"Wrapper Entry Information:\s*(?:def\s+)?([A-Za-z_][\w.]*)", instruction)
    return match.group(1) if match else ""


def canonical_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_indices(value: str) -> list[int]:
    indices: list[int] = []
    for chunk in split_csv(value):
        if "-" in chunk:
            start_text, end_text = chunk.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                raise ValueError(f"bad descending index range: {chunk}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(chunk))
    return indices


def select_records(
    records: list[dict],
    *,
    names: str,
    indices: str,
    family: str,
) -> list[dict]:
    selected = records

    wanted_names = set(split_csv(names))
    if family:
        if family not in FAMILIES:
            choices = ", ".join(sorted(FAMILIES))
            raise ValueError(f"unknown family {family!r}; choices: {choices}")
        wanted_names.update(FAMILIES[family])

    if indices:
        picked = []
        for index in parse_indices(indices):
            if index < 1 or index > len(records):
                raise ValueError(f"index {index} out of range 1..{len(records)}")
            picked.append(records[index - 1])
        selected = picked

    if wanted_names:
        wanted_canonical = {canonical_name(name) for name in wanted_names}
        selected = [
            record
            for record in selected
            if canonical_name(function_name(record["instruction"])) in wanted_canonical
        ]

    if not selected:
        raise ValueError("selection produced no records")
    return selected


def add_code(block_size: str) -> str:
    return f"""import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(input_ptr, other_ptr, output_ptr, n_elements, alpha: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0)
    result = x + alpha * y
    tl.store(output_ptr + offsets, result, mask=mask)

def add(input, other, *, alpha=1, out=None):
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
    _add_kernel[grid](input_contig, other_tensor, output, n_elements, alpha, BLOCK_SIZE={block_size})
    return output
"""


def grammar_code_variants(grammar_path: Path) -> list[str]:
    text = grammar_path.read_text()
    sizes = re.findall(r'"(\d+)"', re.search(r"block_size\s*::=([^\n]+)", text).group(1))
    if not sizes:
        raise ValueError(f"no block_size alternatives found in {grammar_path}")
    return [add_code(size) for size in sizes]


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
            raise RuntimeError(f"empty LM Studio response for item {idx + 1}")
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
                print(f"baseline {done}/{len(pending)} generated", flush=True)

    final = [record for record in output if record is not None]
    write_records(output_path, final)
    return final


def generate_add_with_grammar(
    *,
    add_instruction: str,
    prompt_header: str,
    code_variants: list[str],
    output_path: Path,
    base_url: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    if output_path.exists():
        return output_path.read_text()

    grammar_note = (
        "Generate the TritonBench add module. The decoder is constrained to the "
        "provided grammar; choose the best valid block size."
    )
    messages = [
        {"role": "system", "content": prompt_header},
        {"role": "user", "content": f"{grammar_note}\n\n{add_instruction}"},
    ]
    code = structured_completion(
        base_url=base_url,
        model=model,
        messages=messages,
        code_variants=code_variants,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(code)
    return code


def replace_add(records: list[dict], code: str) -> list[dict]:
    replaced = []
    seen_add = False
    for record in records:
        new_record = dict(record)
        if function_name(record["instruction"]) == "add":
            new_record["predict"] = code
            seen_add = True
        replaced.append(new_record)
    if not seen_add:
        raise ValueError("no add item found in records")
    return replaced


def output_stem(records: list[dict]) -> str:
    if len(records) == 1:
        return canonical_name(function_name(records[0]["instruction"]))
    return f"selected{len(records)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-jsonl", type=Path, default=Path("experiments/lmstudio_prompt11_router_20260527-194836/lmstudio_qwen_qwen3.6-35b-a3b_simp_limit20_20260527-194844.jsonl"))
    parser.add_argument("--prompt0", type=Path, default=Path("prompt-0.txt"))
    parser.add_argument("--prompt11", type=Path, default=Path("prompt-11-router.txt"))
    parser.add_argument("--grammar", type=Path, default=Path("/Users/gustavoortiz/Downloads/grammar.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/grammar_add_compare"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--grammar-max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--only-add", action="store_true")
    parser.add_argument("--names", default="", help="Comma-separated wrapper names to keep, e.g. add,sub,rsqrt")
    parser.add_argument("--indices", default="", help="1-based item indices or ranges, e.g. 6,7,13,16 or 6-8,16")
    parser.add_argument("--family", default="", help=f"Named family selector: {', '.join(sorted(FAMILIES))}")
    parser.add_argument("--list", action="store_true", help="List functions from --seed-jsonl after --limit and exit")
    args = parser.parse_args()

    all_records = load_records(args.seed_jsonl, args.limit)
    if args.list:
        for idx, record in enumerate(all_records, start=1):
            print(f"{idx:3d} {function_name(record['instruction'])}")
        return

    seed_records = select_records(
        all_records,
        names=args.names,
        indices=args.indices,
        family=args.family,
    )
    code_variants = grammar_code_variants(args.grammar)
    prompt0_header = read_prompt_header(args.prompt0)
    prompt11_header = read_prompt_header(args.prompt11)

    add_record = next(
        (
            record
            for record in seed_records
            if canonical_name(function_name(record["instruction"])) == "add"
        ),
        None,
    )
    if args.only_add:
        if add_record is None:
            raise ValueError("the current grammar only targets add, but add is not selected")
        seed_records = [add_record]
    if add_record is None:
        selected = ", ".join(function_name(record["instruction"]) for record in seed_records)
        raise ValueError(
            "the current grammar only targets add; include add in the selected subset "
            f"or use a broader grammar. Selected: {selected}"
        )

    stem = output_stem(seed_records)
    prompt0_path = args.out_dir / f"prompt0_{stem}.jsonl"
    prompt11_path = args.out_dir / f"prompt11_{stem}.jsonl"
    prompt0_grammar_path = args.out_dir / f"prompt0_plus_add_grammar_{stem}.jsonl"
    prompt11_grammar_path = args.out_dir / f"prompt11_plus_add_grammar_{stem}.jsonl"
    prompt0_add_only_path = args.out_dir / "prompt0_add_only.jsonl"
    prompt0_add_grammar_only_path = args.out_dir / "prompt0_add_grammar_only.jsonl"
    prompt11_add_only_path = args.out_dir / "prompt11_add_only.jsonl"
    prompt11_add_grammar_only_path = args.out_dir / "prompt11_add_grammar_only.jsonl"

    prompt0_records = generate_baseline(
        records=seed_records,
        prompt_header=prompt0_header,
        output_path=prompt0_path,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        resume=not args.no_resume,
    )
    prompt11_records = seed_records
    write_records(prompt11_path, prompt11_records)

    prompt0_add_code = generate_add_with_grammar(
        add_instruction=add_record["instruction"],
        prompt_header=prompt0_header,
        code_variants=code_variants,
        output_path=args.out_dir / "add_from_prompt0_grammar.py",
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.grammar_max_tokens,
        temperature=args.temperature,
    )
    prompt11_add_code = generate_add_with_grammar(
        add_instruction=add_record["instruction"],
        prompt_header=prompt11_header,
        code_variants=code_variants,
        output_path=args.out_dir / "add_from_prompt11_grammar.py",
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.grammar_max_tokens,
        temperature=args.temperature,
    )

    write_records(prompt0_grammar_path, replace_add(prompt0_records, prompt0_add_code))
    write_records(prompt11_grammar_path, replace_add(prompt11_records, prompt11_add_code))

    prompt0_add_record = next(
        record for record in prompt0_records if function_name(record["instruction"]) == "add"
    )
    prompt11_add_record = next(
        record for record in prompt11_records if function_name(record["instruction"]) == "add"
    )
    write_records(prompt0_add_only_path, [prompt0_add_record])
    write_records(
        prompt0_add_grammar_only_path,
        [{"instruction": add_record["instruction"], "predict": prompt0_add_code}],
    )
    write_records(prompt11_add_only_path, [prompt11_add_record])
    write_records(
        prompt11_add_grammar_only_path,
        [{"instruction": add_record["instruction"], "predict": prompt11_add_code}],
    )

    manifest = {
        "model": args.model,
        "base_url": args.base_url,
        "limit": args.limit,
        "only_add": args.only_add,
        "names": args.names,
        "indices": args.indices,
        "family": args.family,
        "selected_functions": [
            function_name(record["instruction"]) for record in seed_records
        ],
        "grammar": str(args.grammar),
        "seed_jsonl": str(args.seed_jsonl),
        "outputs": {
            "prompt0": str(prompt0_path),
            "prompt0_plus_add_grammar": str(prompt0_grammar_path),
            "prompt11": str(prompt11_path),
            "prompt11_plus_add_grammar": str(prompt11_grammar_path),
            "prompt0_add_only": str(prompt0_add_only_path),
            "prompt0_add_grammar_only": str(prompt0_add_grammar_only_path),
            "prompt11_add_only": str(prompt11_add_only_path),
            "prompt11_add_grammar_only": str(prompt11_add_grammar_only_path),
            "prompt0_add_code": str(args.out_dir / "add_from_prompt0_grammar.py"),
            "prompt11_add_code": str(args.out_dir / "add_from_prompt11_grammar.py"),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
