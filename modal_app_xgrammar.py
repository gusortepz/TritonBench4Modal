"""Modal + SGLang real XGrammar generation for TritonBench-T.

This is the real constrained-decoding path: SGLang applies XGrammar token
masks during sampling through the EBNF structured-output parameter.

Example:
    XGRAMMAR_GPU=H100 python3 -m modal run modal_app_xgrammar.py::compare \
      --limit 20 \
      --operations add,sub,sqrt,rsqrt,tanh,relu_sqrt \
      --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
      --tp-size 1

Then evaluate the emitted JSONL files with modal_app.py::evaluate_only.
"""

from __future__ import annotations

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json
import os
import re
import shlex
import subprocess
import time

import modal


APP_NAME = "tritonbench-t-xgrammar"
SGLANG_PORT = 30000
GPU_CONFIG = os.environ.get("XGRAMMAR_GPU", "H100")
HF_CACHE_DIR = "/root/.cache/huggingface"

DEFAULT_MODEL_PATH = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_SEED_JSONL = (
    "experiments/lmstudio_prompt11_router_20260527-194836/"
    "lmstudio_qwen_qwen3.6-35b-a3b_simp_limit20_20260527-194844.jsonl"
)
DEFAULT_OPERATIONS = ("add", "sub", "sqrt", "rsqrt", "tanh", "relu_sqrt")


image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .entrypoint([])
    .pip_install("openai>=1.50")
)
app = modal.App(APP_NAME, image=image)
hf_cache_volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)


CONSTRAINED_OUTPUT_SUFFIX = """
XGRAMMAR CONSTRAINED OUTPUT MODE

For this run, the decoder is constrained by an EBNF grammar whose root is raw
Python source, not Markdown. Output raw Python module text only:
- do not use ``` fences;
- do not include prose, explanations, tests, or example calls;
- begin with a valid top-level Python statement such as import/from/def/try;
- stop after the generated module is complete.
""".strip()


def _read_prompt_header(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'PROMPT_HEADER\s*=\s*"""(.*?)"""\s*(?:\.strip\(\))?\s*$',
        text,
        re.DOTALL,
    )
    return match.group(1).strip() if match else text.strip()


def _extract_code(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n(.*?)\n```", text, re.DOTALL)
    if match:
        return match.group(1).strip() + "\n"
    text = re.sub(r"^```(?:python|py)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip() + "\n"


def _load_records(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    return records


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    tmp.replace(path)


def _function_name(instruction: str) -> str:
    match = re.search(r"Wrapper Entry Information:\s*(?:def\s+)?([A-Za-z_][\w.]*)", instruction)
    return match.group(1).rsplit(".", 1)[-1] if match else ""


def _parse_operations(value: str) -> tuple[str, ...]:
    operations = tuple(item.strip() for item in value.split(",") if item.strip())
    return operations or DEFAULT_OPERATIONS


def _select_operation_records(records: list[dict], operations: tuple[str, ...]) -> list[dict]:
    selected = [
        record
        for record in records
        if _function_name(record.get("instruction", "")) in set(operations)
    ]
    seen = {_function_name(record.get("instruction", "")) for record in selected}
    missing = [operation for operation in operations if operation not in seen]
    if missing:
        raise ValueError(f"missing selected operation(s): {', '.join(missing)}")
    return sorted(
        selected,
        key=lambda record: operations.index(_function_name(record["instruction"])),
    )


def _build_messages(instruction: str, prompt_header: str, grammar: str | None) -> list[dict]:
    if grammar:
        system = f"{prompt_header.strip()}\n\n{CONSTRAINED_OUTPUT_SUFFIX}"
        user = (
            "The following XGrammar EBNF is enforced by the decoder. Generate a "
            "single raw Python module that satisfies it and solves the task.\n\n"
            f"{grammar.strip()}\n\nTask:\n{instruction}"
        )
    else:
        system = prompt_header.strip()
        user = instruction
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _wait_for_sglang(port: int, process: subprocess.Popen, timeout_s: int = 900) -> None:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SGLang server exited with code {process.returncode}")
        try:
            with urlopen(Request(url), timeout=5) as response:
                if response.status == 200:
                    return
        except (HTTPError, URLError, TimeoutError):
            time.sleep(2)
    raise TimeoutError(f"SGLang server did not become ready within {timeout_s}s")


def _start_sglang_server(
    *,
    model_path: str,
    tp_size: int,
    context_length: int,
    reasoning_parser: str,
    extra_server_args: str,
) -> subprocess.Popen:
    cmd = [
        "python3",
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
        "--grammar-backend",
        "xgrammar",
        "--tp-size",
        str(tp_size),
        "--context-length",
        str(context_length),
    ]
    if reasoning_parser:
        cmd.extend(["--reasoning-parser", reasoning_parser])
    if extra_server_args.strip():
        cmd.extend(shlex.split(extra_server_args))

    print("Starting SGLang server:", " ".join(shlex.quote(part) for part in cmd), flush=True)
    process = subprocess.Popen(cmd)
    _wait_for_sglang(SGLANG_PORT, process)
    return process


def _chat_generate(
    *,
    client,
    model_path: str,
    messages: list[dict],
    grammar: str | None,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> str:
    kwargs: dict = {
        "model": model_path,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    extra_body: dict = {}
    if top_k > 0:
        extra_body["top_k"] = top_k
    if grammar:
        extra_body["ebnf"] = grammar
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    content = getattr(message, "content", None) or ""
    if not content.strip():
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        if reasoning:
            print("Warning: response had reasoning but empty content", flush=True)
    return content


@app.function(
    gpu=GPU_CONFIG,
    timeout=60 * 60 * 6,
    memory=131072,
    volumes={HF_CACHE_DIR: hf_cache_volume},
)
def generate_compare_remote(
    *,
    records: list[dict],
    prompt0_header: str,
    prompt11_header: str,
    grammar: str,
    model_path: str,
    tp_size: int,
    context_length: int,
    reasoning_parser: str,
    extra_server_args: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> dict[str, list[dict]]:
    from openai import OpenAI

    process = _start_sglang_server(
        model_path=model_path,
        tp_size=tp_size,
        context_length=context_length,
        reasoning_parser=reasoning_parser,
        extra_server_args=extra_server_args,
    )
    client = OpenAI(base_url=f"http://127.0.0.1:{SGLANG_PORT}/v1", api_key="EMPTY")

    conditions = [
        ("prompt0", prompt0_header, None),
        ("prompt11", prompt11_header, None),
        ("prompt0_plus_xgrammar", prompt0_header, grammar),
        ("prompt11_plus_xgrammar", prompt11_header, grammar),
    ]
    results: dict[str, list[dict]] = {name: [] for name, _, _ in conditions}

    try:
        for condition_name, prompt_header, maybe_grammar in conditions:
            print(f"Generating {condition_name}: {len(records)} records", flush=True)
            for idx, record in enumerate(records, start=1):
                messages = _build_messages(record["instruction"], prompt_header, maybe_grammar)
                raw = _chat_generate(
                    client=client,
                    model_path=model_path,
                    messages=messages,
                    grammar=maybe_grammar,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                code = _extract_code(raw)
                results[condition_name].append(
                    {"instruction": record["instruction"], "predict": code}
                )
                print(
                    f"  {condition_name} {idx}/{len(records)} "
                    f"({_function_name(record['instruction'])})",
                    flush=True,
                )
    finally:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)

    return results


@app.local_entrypoint()
def compare(
    seed_jsonl: str = DEFAULT_SEED_JSONL,
    prompt0: str = "prompt-0.txt",
    prompt11: str = "prompt-11-router.txt",
    grammar_path: str = "grammars/triton_python_xgrammar.ebnf",
    out_dir: str = "outputs/xgrammar_sglang_compare",
    limit: int = 20,
    operations: str = ",".join(DEFAULT_OPERATIONS),
    model_path: str = DEFAULT_MODEL_PATH,
    tp_size: int = 1,
    context_length: int = 32768,
    reasoning_parser: str = "qwen3",
    extra_server_args: str = "",
    max_tokens: int = 8192,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 20,
) -> None:
    """Generate prompt0/prompt11 with and without real XGrammar constraints."""
    operations_tuple = _parse_operations(operations)
    records = _select_operation_records(
        _load_records(Path(seed_jsonl), limit),
        operations_tuple,
    )
    grammar = Path(grammar_path).read_text(encoding="utf-8")
    prompt0_header = _read_prompt_header(Path(prompt0))
    prompt11_header = _read_prompt_header(Path(prompt11))

    results = generate_compare_remote.remote(
        records=records,
        prompt0_header=prompt0_header,
        prompt11_header=prompt11_header,
        grammar=grammar,
        model_path=model_path,
        tp_size=tp_size,
        context_length=context_length,
        reasoning_parser=reasoning_parser,
        extra_server_args=extra_server_args,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    output_root = Path(out_dir)
    stem = f"xgrammar_{len(records)}"
    paths: dict[str, str] = {}
    for condition_name, condition_records in results.items():
        path = output_root / f"{condition_name}_{stem}.jsonl"
        _write_records(path, condition_records)
        paths[condition_name] = str(path)

    manifest = {
        "engine": "sglang",
        "grammar_backend": "xgrammar",
        "model_path": model_path,
        "gpu_config": GPU_CONFIG,
        "tp_size": tp_size,
        "context_length": context_length,
        "grammar_path": grammar_path,
        "seed_jsonl": seed_jsonl,
        "operations": list(operations_tuple),
        "outputs": paths,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))
    print("\nEvaluate with:")
    for condition_name, path in paths.items():
        print(
            "python3 -m modal run modal_app.py::evaluate_only "
            f"--predictions {path} --output-subdir results/xgrammar_sglang_{condition_name}"
        )
