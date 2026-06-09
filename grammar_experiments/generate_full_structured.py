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
DEFAULT_SEED_JSONL = Path(
    "experiments/lmstudio_20260529-191522/"
    "lmstudio_qwen_qwen3.6-35b-a3b_simp_all_20260528-212826.jsonl"
)


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


def request_json(url: str, payload: dict, *, timeout: int = 60 * 30) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer lm-studio"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio request failed: {exc.code} {body}") from exc


def native_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def build_messages(instruction: str, prompt_header: str, *, structured: bool) -> list[dict]:
    if structured:
        instruction = (
            "Generate one complete self-contained Python module for this TritonBench-T task. "
            "The public wrapper function must match the benchmark request exactly. "
            "Return imports, helper kernels/functions if needed, and the wrapper. "
            "Do not include markdown fences, tests, examples, TODOs, placeholders, or prose.\n\n"
            f"{instruction}"
        )
    return [
        {"role": "system", "content": prompt_header},
        {"role": "user", "content": instruction},
    ]


def native_chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
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
    parts = [
        part.get("content", "")
        for part in output
        if part.get("type") == "message" and part.get("content")
    ]
    return "\n\n".join(parts).strip()


def structured_chat_completion(
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
                                    "TritonBench-T task. No markdown fences, tests, examples, or prose."
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
        raise ValueError(f"{path} has {len(records)} records, expected {limit}")
    return records


def load_existing(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    existing: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            instruction = record.get("instruction")
            if isinstance(instruction, str):
                existing[instruction] = record
    return existing


def write_records(path: Path, records: list[dict | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            if record is not None:
                f.write(json.dumps(record) + "\n")
    tmp.replace(path)


def generate(
    *,
    seed_records: list[dict],
    prompt_header: str,
    output_path: Path,
    mode: str,
    base_url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    resume: bool,
) -> None:
    structured = mode == "structured"
    existing = load_existing(output_path) if resume else {}
    output: list[dict | None] = [
        existing.get(record["instruction"]) for record in seed_records
    ]
    pending = [
        (idx, record)
        for idx, record in enumerate(seed_records)
        if output[idx] is None
    ]
    print(
        f"generating {len(pending)} missing {mode} predictions with {model}",
        flush=True,
    )
    if existing:
        print(f"resuming with {len(existing)} existing records from {output_path}", flush=True)

    def do_one(item: tuple[int, dict]) -> tuple[int, dict]:
        idx, record = item
        messages = build_messages(
            record["instruction"],
            prompt_header,
            structured=structured,
        )
        error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                if structured:
                    code = structured_chat_completion(
                        base_url=base_url,
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    raw = native_chat_completion(
                        base_url=base_url,
                        model=model,
                        messages=messages,
                        temperature=temperature,
                    )
                    if not raw.strip():
                        raise RuntimeError("LM Studio returned an empty response")
                    code = extract_code(raw)
                return idx, {"instruction": record["instruction"], "predict": code}
            except Exception as exc:  # noqa: BLE001
                error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))
        return idx, {
            "instruction": record["instruction"],
            "predict": f"# generation failed: {error}\n",
        }

    if pending:
        done = 0
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            futures = [executor.submit(do_one, item) for item in pending]
            for future in as_completed(futures):
                idx, record = future.result()
                output[idx] = record
                done += 1
                write_records(output_path, output)
                if done % 5 == 0 or done == len(pending):
                    print(f"  {done}/{len(pending)} generated", flush=True)
    write_records(output_path, output)
    print(f"wrote {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-jsonl", type=Path, default=DEFAULT_SEED_JSONL)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "structured"), required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    seed_records = load_records(args.seed_jsonl, args.limit)
    prompt_header = read_prompt_header(args.prompt_file)
    generate(
        seed_records=seed_records,
        prompt_header=prompt_header,
        output_path=args.output_path,
        mode=args.mode,
        base_url=args.base_url,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        retries=args.retries,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
