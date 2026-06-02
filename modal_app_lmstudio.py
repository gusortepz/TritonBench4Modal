"""
Local LM Studio generation + Modal GPU evaluation for TritonBench-T.

This file is intentionally parallel to modal_app.py:
1. fetch the exact Alpaca instructions from the Modal image;
2. generate predictions locally through LM Studio's OpenAI-compatible API;
3. upload the local JSONL to the Modal Volume;
4. run the existing GPU evaluator on Modal.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import modal

from modal_app import (
    DATA_DIR,
    DEFAULT_GPU,
    EVAL_MEMORY_MB,
    _build_messages,
    _evaluate_impl,
    _extract_code,
    _load_alpaca,
    _read_prompt_header,
    _upload_local_predictions,
    data_volume,
    image as base_image,
)

DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
DEFAULT_LOCAL_OUTPUT_DIR = "local-predictions"

# Remote containers import this module too, so include the shared modal_app.py
# source alongside this file instead of relying on Modal's single-file mount.
image = base_image.add_local_python_source("modal_app")
app = modal.App("tritonbench-t-lmstudio", image=image)


@app.function(timeout=60 * 10, cpu=1)
def fetch_alpaca_items(dataset: str = "simp", limit: int | None = None) -> list[dict]:
    """Return benchmark items from the patched TritonBench checkout in Modal."""
    items = _load_alpaca(dataset)
    if limit:
        items = items[:limit]
    return items


@app.function(
    gpu=DEFAULT_GPU,
    timeout=60 * 60 * 6,
    volumes={DATA_DIR: data_volume},
    memory=EVAL_MEMORY_MB,
)
def evaluate_predictions(
    predictions_path: str = "predictions.jsonl",
    output_subdir: str = "results/lmstudio",
) -> dict:
    """Run TritonBench-T eval phases in this app's GPU container."""
    return _evaluate_impl(predictions_path=predictions_path, output_subdir=output_subdir)


def _safe_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return tag.strip("._-") or "lmstudio"


def _default_output_path(model: str, dataset: str, limit: int) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    size = f"limit{limit}" if limit else "all"
    name = f"lmstudio_{_safe_tag(model)}_{dataset}_{size}_{timestamp}.jsonl"
    return Path(DEFAULT_LOCAL_OUTPUT_DIR) / name


def _request_json(url: str, *, api_key: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(url, data=data, headers=headers)
    try:
        with urlopen(request, timeout=60 * 30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LM Studio request failed: {exc.code} {body}") from exc


def _openai_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _native_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _models_url(base_url: str) -> str:
    base = _openai_base_url(base_url)
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


def _get_openai_client(base_url: str, api_key: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The local LM Studio flow needs the OpenAI Python client. "
            "Run `pip install -r requirements-local.txt`."
        ) from exc

    return OpenAI(base_url=_openai_base_url(base_url), api_key=api_key)


def _resolve_model(base_url: str, api_key: str, model: str) -> str:
    if model:
        return model

    models = _request_json(_models_url(base_url), api_key=api_key)
    model_ids = [m["id"] for m in models.get("data", []) if "id" in m]
    if not model_ids:
        raise RuntimeError(
            "LM Studio did not report any loaded models. Load Qwen in LM Studio "
            "or pass --model with the model id shown by LM Studio."
        )
    return model_ids[0]


def _chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def _native_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    del max_tokens  # LM Studio's native /api/v1/chat endpoint does not accept it.
    system_text = "\n\n".join(
        str(message["content"]) for message in messages if message["role"] == "system"
    )
    user_text = "\n\n".join(
        str(message["content"]) for message in messages if message["role"] != "system"
    )
    prompt = f"{system_text}\n\n{user_text}" if system_text else user_text
    response = _request_json(
        f"{_native_base_url(base_url)}/api/v1/chat",
        api_key=api_key,
        payload={
            "model": model,
            "input": prompt,
            "context_length": 32768,
            "temperature": temperature,
        },
    )
    output = response.get("output", [])
    message_parts = [
        part.get("content", "")
        for part in output
        if part.get("type") == "message" and part.get("content")
    ]
    return "\n\n".join(message_parts).strip()


def _generate_reply(
    *,
    api: str,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    temperature: float,
) -> str:
    if api == "native":
        return _native_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    if api == "openai":
        client = _get_openai_client(base_url, api_key)
        return _chat_completion(
            client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise ValueError("api must be 'native' or 'openai'")


def _read_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}

    existing: dict[str, dict] = {}
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            instruction = record.get("instruction")
            if isinstance(instruction, str):
                existing[instruction] = record
    return existing


def _write_predictions(path: Path, records: list[dict | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for record in records:
            if record is not None:
                f.write(json.dumps(record) + "\n")
    tmp_path.replace(path)


def _generate_local_predictions(
    *,
    items: list[dict],
    output_path: Path,
    prompt_header: str,
    api: str,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    retries: int,
    resume: bool,
) -> Path:
    resolved_model = _resolve_model(base_url, api_key, model)

    existing = _read_existing_predictions(output_path) if resume else {}
    records: list[dict | None] = [
        existing.get(item["instruction"]) for item in items
    ]
    pending = [
        (idx, item)
        for idx, item in enumerate(items)
        if records[idx] is None
    ]

    print(
        f"generating {len(pending)} missing predictions locally "
        f"with {resolved_model} via {api} at {base_url}",
        flush=True,
    )
    if existing:
        print(f"resuming with {len(existing)} existing records from {output_path}", flush=True)

    def do_one(idx_item: tuple[int, dict]) -> tuple[int, dict]:
        idx, item = idx_item
        messages = _build_messages(item, prompt_header=prompt_header)
        error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                raw = _generate_reply(
                    api=api,
                    base_url=base_url,
                    api_key=api_key,
                    model=resolved_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                if not raw.strip():
                    raise RuntimeError(
                        "LM Studio returned an empty final message. If using "
                        "OpenAI-compatible mode with a reasoning model, retry "
                        "with --api native."
                    )
                code = _extract_code(raw)
                return idx, {"instruction": item["instruction"], "predict": code}
            except Exception as exc:  # noqa: BLE001
                error = exc
                if attempt < retries:
                    time.sleep(min(2**attempt, 10))

        code = f"# generation failed: {error}\n"
        return idx, {"instruction": item["instruction"], "predict": code}

    if not pending:
        _write_predictions(output_path, records)
        return output_path

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(do_one, idx_item) for idx_item in pending]
        for future in as_completed(futures):
            idx, record = future.result()
            records[idx] = record
            done += 1
            _write_predictions(output_path, records)
            if done % 5 == 0 or done == len(pending):
                print(f"  {done}/{len(pending)} generated", flush=True)

    return output_path


@app.local_entrypoint()
def main(
    dataset: str = "simp",
    limit: int = 0,
    output_path: str = "",
    output_subdir: str = "results/lmstudio",
    prompt_file: str = "prompt-4.txt",
    api: str = "native",
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = "",
    concurrency: int = 1,
    max_tokens: int = 8192,
    temperature: float = 0.1,
    retries: int = 1,
    resume: bool = True,
):
    """Generate locally with LM Studio, upload predictions, then evaluate on Modal."""
    items = fetch_alpaca_items.remote(dataset=dataset, limit=limit or None)
    prompt_header = _read_prompt_header(prompt_file)
    print(f"using prompt header: {prompt_file or 'embedded default'}", flush=True)
    resolved_name = model or "auto"
    local_output = (
        Path(output_path)
        if output_path
        else _default_output_path(resolved_name, dataset, limit)
    )

    predictions_path = _generate_local_predictions(
        items=items,
        output_path=local_output,
        prompt_header=prompt_header,
        api=api,
        base_url=base_url,
        api_key=api_key,
        model=model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        resume=resume,
    )

    remote = _upload_local_predictions(predictions_path)
    print(f"\nevaluating: volume://{remote}\n", flush=True)
    summary = evaluate_predictions.remote(
        predictions_path=remote,
        output_subdir=output_subdir,
    )
    print("\n=== Final summary ===")
    print(json.dumps(summary, indent=2))
    Path("latest-summary.json").write_text(json.dumps(summary, indent=2) + "\n")


@app.local_entrypoint()
def generate_only(
    dataset: str = "simp",
    limit: int = 0,
    output_path: str = "",
    prompt_file: str = "prompt-4.txt",
    api: str = "native",
    base_url: str = DEFAULT_BASE_URL,
    api_key: str = DEFAULT_API_KEY,
    model: str = "",
    concurrency: int = 1,
    max_tokens: int = 8192,
    temperature: float = 0.1,
    retries: int = 1,
    resume: bool = True,
):
    """Generate a local predictions JSONL without uploading or evaluating."""
    items = fetch_alpaca_items.remote(dataset=dataset, limit=limit or None)
    prompt_header = _read_prompt_header(prompt_file)
    print(f"using prompt header: {prompt_file or 'embedded default'}", flush=True)
    resolved_name = model or "auto"
    local_output = (
        Path(output_path)
        if output_path
        else _default_output_path(resolved_name, dataset, limit)
    )
    predictions_path = _generate_local_predictions(
        items=items,
        output_path=local_output,
        prompt_header=prompt_header,
        api=api,
        base_url=base_url,
        api_key=api_key,
        model=model,
        concurrency=concurrency,
        max_tokens=max_tokens,
        temperature=temperature,
        retries=retries,
        resume=resume,
    )
    print(f"wrote {predictions_path}")
