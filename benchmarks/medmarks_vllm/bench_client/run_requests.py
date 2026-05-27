#!/usr/bin/env python3
"""Run MedMarks OpenAI-chat request JSONL benchmarks against a vLLM endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class BenchConfig:
    dataset_path: Path
    num_prompts: int
    max_tokens: int
    stream: bool
    request_rate: float
    max_concurrency: int
    warmup_requests: int
    timeout_seconds: float
    retries: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-config", type=Path, required=True)
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL, for example http://host:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> BenchConfig:
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    dataset_path = Path(str(data["dataset_path"]))
    if not dataset_path.is_absolute():
        dataset_path = path.parent / dataset_path
    return BenchConfig(
        dataset_path=dataset_path,
        num_prompts=int(data["num_prompts"]),
        max_tokens=int(data["max_tokens"]),
        stream=bool(data.get("stream", True)),
        request_rate=float(data.get("request_rate", 0)),
        max_concurrency=int(data.get("max_concurrency", 16)),
        warmup_requests=int(data.get("warmup_requests", 1)),
        timeout_seconds=float(data.get("timeout_seconds", 120)),
        retries=int(data.get("retries", 2)),
    )


def load_requests(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            validate_request_row(row, line_number)
            rows.append(row)
            if len(rows) >= limit:
                break
    if len(rows) < limit:
        raise ValueError(f"{path} contains {len(rows)} request rows, fewer than num_prompts={limit}")
    return rows


def validate_request_row(row: dict[str, Any], line_number: int) -> None:
    if "prompt" in row or "output_tokens" in row:
        raise ValueError(f"line {line_number}: requests JSONL must not contain prompt/output_tokens fields")
    sampling_args = row.get("sampling_args") or {}
    if not isinstance(sampling_args, dict):
        raise ValueError(f"line {line_number}: sampling_args must be an object")
    reserved = {"model", "messages", "prompt", "input", "max_tokens", "stream", "tools"}
    present_reserved = reserved.intersection(sampling_args)
    if present_reserved:
        raise ValueError(
            f"line {line_number}: sampling_args contains reserved request keys: {sorted(present_reserved)}"
        )
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"line {line_number}: missing non-empty messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"line {line_number}: invalid OpenAI chat message")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    raise ValueError(f"line {line_number}: non-text content parts are unsupported")


async def main_async() -> None:
    args = parse_args()
    config = load_config(args.bench_config)
    rows = load_requests(config.dataset_path, config.num_prompts)
    endpoint = args.base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Authorization": f"Bearer {args.api_key}"}
    timeout = httpx.Timeout(config.timeout_seconds)
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        warmup = rows[: config.warmup_requests]
        for row in warmup:
            await send_one(client, endpoint, args.model, row, config)

        started = time.perf_counter()
        semaphore = asyncio.Semaphore(config.max_concurrency)
        tasks = []
        for index, row in enumerate(rows):
            if config.request_rate > 0 and index > 0:
                await asyncio.sleep(1 / config.request_rate)
            tasks.append(asyncio.create_task(send_measured(semaphore, client, endpoint, args.model, row, config)))
        results = await asyncio.gather(*tasks)
        finished = time.perf_counter()

    output = build_result(config, args, results, started, finished)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def send_measured(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    config: BenchConfig,
) -> dict[str, Any]:
    async with semaphore:
        request_started = time.perf_counter()
        try:
            response = await send_one(client, endpoint, model, row, config)
            status = "ok"
            error = None
        except Exception as exc:  # noqa: BLE001
            response = {}
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
        request_finished = time.perf_counter()
        return {
            "status": status,
            "error": error,
            "latency_s": request_finished - request_started,
            "env_id": row.get("env_id"),
            "variant_id": row.get("variant_id"),
            "example_id": row.get("example_id"),
            "usage": response.get("usage"),
        }


async def send_one(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    config: BenchConfig,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": row["messages"],
        "stream": config.stream,
        **dict(row.get("sampling_args") or {}),
        "max_tokens": config.max_tokens,
    }
    last_error: Exception | None = None
    for _ in range(config.retries + 1):
        try:
            if config.stream:
                return await send_streaming(client, endpoint, payload)
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    assert last_error is not None
    raise last_error


async def send_streaming(client: httpx.AsyncClient, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    usage: dict[str, Any] | None = None
    chunk_count = 0
    async with client.stream("POST", endpoint, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ").strip()
            if data == "[DONE]":
                break
            chunk_count += 1
            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
    return {"usage": usage, "stream_chunks": chunk_count}


def build_result(
    config: BenchConfig,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    started: float,
    finished: float,
) -> dict[str, Any]:
    latencies = [result["latency_s"] for result in results if result["status"] == "ok"]
    duration = finished - started
    return {
        "schema": "medmarks-vllm-request-bench-v1",
        "dataset_path": str(config.dataset_path),
        "model": args.model,
        "base_url": args.base_url,
        "num_prompts": config.num_prompts,
        "max_tokens": config.max_tokens,
        "stream": config.stream,
        "request_rate": config.request_rate,
        "max_concurrency": config.max_concurrency,
        "duration_s": duration,
        "successful_requests": sum(1 for result in results if result["status"] == "ok"),
        "failed_requests": sum(1 for result in results if result["status"] != "ok"),
        "request_throughput_per_s": len(results) / duration if duration > 0 else None,
        "latency_s": summarize_float(latencies),
        "results": results,
    }


def summarize_float(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "p50": None, "p90": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def percentile(sorted_values: list[float], q: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = q * (len(sorted_values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo)


if __name__ == "__main__":
    asyncio.run(main_async())
