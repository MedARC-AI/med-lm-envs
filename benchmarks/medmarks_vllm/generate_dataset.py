#!/usr/bin/env python3
"""Generate faithful initial OpenAI chat requests for MedMarks vLLM benchmarks."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import math
import random
import re
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.utils.message_utils import normalize_messages


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "medmarks-verified.toml"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PREFIX = "medmarks_vllm"
DEFAULT_TARGET_SIZES = (500, 1000, 2000)
TARGET_ENVS = {
    "careqa",
    "longhealth",
    "medbullets",
    "medcalc_bench",
    "medhallu",
    "medxpertqa",
    "supergpqa_medicine",
}
LONGHEALTH_MIN_FRACTION = 0.12
OPENAI_CHAT_ROLES = {"system", "user", "assistant", "tool"}
FLATTENED_ROLE_RE = re.compile(r"^\s*System:\s*.+\n\s*\n\s*User:\s*.+", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class Variant:
    env_id: str
    variant_id: str
    variant_name: str
    env_args: dict[str, Any]
    source: str


@dataclass
class RenderedRow:
    record: dict[str, Any]
    prompt_chars: int
    input_tokens_approx: int
    input_tokens: int | None


class UnsupportedRowError(ValueError):
    """Raised when a dataset row cannot be exported faithfully for this benchmark."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-dir", type=Path, default=REPO_ROOT / "environments")
    parser.add_argument(
        "--target-size",
        type=int,
        action="append",
        dest="target_sizes",
        help=(
            "Target dataset size. Repeat to write multiple same-ratio datasets. "
            f"Defaults to {', '.join(str(size) for size in DEFAULT_TARGET_SIZES)}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--output", type=Path, help="Explicit requests JSONL output path. Requires one --target-size.")
    parser.add_argument("--stats-output", type=Path, help="Explicit stats JSON path. Requires one --target-size.")
    parser.add_argument("--bench-output", type=Path, help="Explicit bench TOML path. Requires one --target-size.")
    parser.add_argument("--seed", type=int, default=20260523)
    parser.add_argument("--tokenizer", default=None, help="Optional HF tokenizer path/name for audit token counts.")
    parser.add_argument("--max-tokens", type=int, default=512, help="Run-level generation limit written to bench TOML.")
    parser.add_argument(
        "--include-audit-metadata",
        action="store_true",
        help="Include question/answer/info metadata in JSONL rows. Off by default to keep request rows lean.",
    )
    parser.add_argument("--max-variants-per-env", type=int, default=0, help="0 keeps all variants from the config.")
    parser.add_argument("--dry-run", action="store_true", help="Load and summarize variants without writing files.")
    return parser.parse_args()


def validate_output_args(args: argparse.Namespace, target_sizes: tuple[int, ...]) -> None:
    if not target_sizes:
        raise SystemExit("at least one --target-size is required")
    if any(size <= 0 for size in target_sizes):
        raise SystemExit("--target-size values must be positive")
    if len(set(target_sizes)) != len(target_sizes):
        raise SystemExit("--target-size values must be unique")
    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be positive")
    if (args.output is not None or args.stats_output is not None or args.bench_output is not None) and len(
        target_sizes
    ) != 1:
        raise SystemExit("--output, --stats-output, and --bench-output require exactly one --target-size")


def output_path_for_size(args: argparse.Namespace, target_size: int) -> Path:
    if args.output is not None:
        return args.output
    return args.output_dir / f"{args.output_prefix}_{size_suffix(target_size)}.requests.jsonl"


def stats_output_path_for_size(args: argparse.Namespace, target_size: int) -> Path:
    if args.stats_output is not None:
        return args.stats_output
    return args.output_dir / f"{args.output_prefix}_{size_suffix(target_size)}.stats.json"


def bench_output_path_for_size(args: argparse.Namespace, target_size: int) -> Path:
    if args.bench_output is not None:
        return args.bench_output
    return args.output_dir / f"{args.output_prefix}_{size_suffix(target_size)}.bench.toml"


def size_suffix(target_size: int) -> str:
    if target_size >= 1000 and target_size % 1000 == 0:
        return f"{target_size // 1000}k"
    return str(target_size)


def main() -> None:
    args = parse_args()
    target_sizes = tuple(args.target_sizes or DEFAULT_TARGET_SIZES)
    validate_output_args(args, target_sizes)
    rng = random.Random(args.seed)
    add_environment_paths(args.env_dir)

    tokenizer = load_tokenizer(args.tokenizer)
    variants = load_variants(args.config)
    variants = limit_variants_per_env(variants, args.max_variants_per_env, rng)

    rows_by_variant: dict[str, list[RenderedRow]] = {}
    failures: list[dict[str, Any]] = []
    for variant in variants:
        key = variant_key(variant)
        if is_tool_enabled_variant(variant):
            failure = skipped_variant(
                variant, "tool-enabled variants are excluded from canonical non-agent input sampling"
            )
            failures.append(failure)
            rows_by_variant[key] = []
            print(f"skipped {key}: {failure['error']}", file=sys.stderr)
            continue
        try:
            rows_by_variant[key] = render_variant(variant, args.env_dir, args, tokenizer)
            print(f"loaded {key}: {len(rows_by_variant[key])} rows", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "env_id": variant.env_id,
                    "variant_id": variant.variant_id,
                    "env_args": variant.env_args,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            rows_by_variant[key] = []
            print(f"failed {key}: {type(exc).__name__}: {exc}", file=sys.stderr)

    all_stats: dict[str, Any] = {}
    for target_size in target_sizes:
        selected = stratified_sample(rows_by_variant, target_size, args.seed)
        selected_records = [row.record for row in selected]
        stats = build_stats(
            selected_rows=selected,
            rows_by_variant=rows_by_variant,
            failures=failures,
            args=args,
            target_size=target_size,
            tokenizer_exact=tokenizer is not None,
        )
        all_stats[str(target_size)] = stats

        if args.dry_run:
            continue

        output_path = output_path_for_size(args, target_size)
        stats_output_path = stats_output_path_for_size(args, target_size)
        bench_output_path = bench_output_path_for_size(args, target_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stats_output_path.parent.mkdir(parents=True, exist_ok=True)
        bench_output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_path, selected_records)
        write_json(stats_output_path, stats)
        write_bench_toml(bench_output_path, output_path, target_size, args.max_tokens)

        print(f"wrote {len(selected_records)} rows to {output_path}", file=sys.stderr)
        print(f"wrote stats to {stats_output_path}", file=sys.stderr)
        print(f"wrote bench config to {bench_output_path}", file=sys.stderr)

    if args.dry_run:
        payload: dict[str, Any] | Any = all_stats
        if len(target_sizes) == 1:
            payload = all_stats[str(target_sizes[0])]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if failures:
        print(f"completed with {len(failures)} skipped/failed variants or rows; see stats JSON", file=sys.stderr)


def add_environment_paths(env_dir: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    for child in sorted(env_dir.iterdir()):
        if child.is_dir():
            sys.path.insert(0, str(child))


def load_tokenizer(tokenizer_name: str | None) -> Any | None:
    if not tokenizer_name:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_name)


def load_variants(config_path: Path) -> list[Variant]:
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    variants: list[Variant] = []
    for entry in config.get("eval", []):
        env_id = entry.get("env_id")
        if env_id not in TARGET_ENVS:
            continue
        env_args = dict(entry.get("env_args") or {})
        name = render_name(str(entry.get("name") or "base"), env_args)
        variants.append(
            Variant(env_id=env_id, variant_id=slugify(name), variant_name=name, env_args=env_args, source="eval")
        )

    for entry in config.get("ablation", []):
        env_id = entry.get("env_id")
        if env_id not in TARGET_ENVS:
            continue
        base_args = dict(entry.get("env_args") or {})
        sweep_args = (entry.get("sweep") or {}).get("env_args") or {}
        for swept_args in expand_sweep(sweep_args):
            env_args = {**base_args, **swept_args}
            name = render_name(str(entry.get("name") or "base"), env_args)
            variants.append(
                Variant(
                    env_id=env_id, variant_id=slugify(name), variant_name=name, env_args=env_args, source="ablation"
                )
            )

    return dedupe_variants(variants)


def expand_sweep(sweep_args: dict[str, Any]) -> Iterable[dict[str, Any]]:
    if not sweep_args:
        yield {}
        return
    keys = sorted(sweep_args)
    values = [sweep_args[key] if isinstance(sweep_args[key], list) else [sweep_args[key]] for key in keys]
    for combo in product(*values):
        yield dict(zip(keys, combo, strict=True))


def render_name(template: str, env_args: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        expr = match.group(1)
        prefix = "env_args."
        if expr.startswith(prefix):
            return str(env_args.get(expr[len(prefix) :], ""))
        return match.group(0)

    return re.sub(r"\{([^{}]+)\}", replace, template)


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "base"


def dedupe_variants(variants: list[Variant]) -> list[Variant]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[Variant] = []
    for variant in variants:
        key = (variant.env_id, variant.variant_id, json.dumps(variant.env_args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(variant)
    return deduped


def limit_variants_per_env(variants: list[Variant], limit: int, rng: random.Random) -> list[Variant]:
    if limit <= 0:
        return variants
    grouped: dict[str, list[Variant]] = defaultdict(list)
    for variant in variants:
        grouped[variant.env_id].append(variant)
    limited: list[Variant] = []
    for env_id in sorted(grouped):
        env_variants = list(grouped[env_id])
        rng.shuffle(env_variants)
        limited.extend(sorted(env_variants[:limit], key=lambda item: item.variant_id))
    return limited


def is_tool_enabled_variant(variant: Variant) -> bool:
    return bool(
        variant.env_args.get("add_python_tool")
        or variant.env_args.get("add_calculator_tool")
        or variant.env_args.get("tools")
        or variant.variant_id == "tools"
    )


def skipped_variant(variant: Variant, reason: str) -> dict[str, Any]:
    return {"env_id": variant.env_id, "variant_id": variant.variant_id, "env_args": variant.env_args, "error": reason}


def render_variant(
    variant: Variant,
    env_dir: Path,
    args: argparse.Namespace,
    tokenizer: Any | None,
) -> list[RenderedRow]:
    load_environment = resolve_load_environment(variant.env_id, env_dir)
    env = load_environment(**variant.env_args)
    if getattr(env, "tool_defs", None):
        raise UnsupportedRowError("environment exposes tool definitions, which are excluded from canonical sampling")

    dataset = env.get_eval_dataset()
    if dataset is None:
        raise ValueError("environment get_eval_dataset() returned None")

    rows: list[RenderedRow] = []
    row_failures: Counter[str] = Counter()
    converter = OpenAIChatCompletionsClient(object())
    with asyncio.Runner() as runner:
        for idx, item in enumerate(dataset):
            item_dict = dict(item)
            try:
                messages = export_openai_messages(item_dict, env=env, converter=converter, runner=runner)
            except UnsupportedRowError as exc:
                row_failures[str(exc)] += 1
                continue

            prompt_chars = prompt_chars_from_messages(messages)
            audit_rendering = render_messages_for_token_audit(messages)
            input_tokens_approx = approximate_tokens(audit_rendering)
            exact_tokens = count_tokens(tokenizer, audit_rendering) if tokenizer is not None else None
            record = {
                "messages": messages,
                "sampling_args": {},
                "env_id": variant.env_id,
                "variant_id": variant.variant_id,
                "variant_name": variant.variant_name,
                "env_args": json_safe(variant.env_args),
                "item_index": idx,
                "example_id": stable_example_id(variant, idx, item_dict),
            }
            if args.include_audit_metadata:
                record.update(
                    {
                        "question": json_safe(item_dict.get("question")),
                        "answer": json_safe(item_dict.get("answer")),
                        "info": json_safe(item_dict.get("info", {})),
                    }
                )
            validate_record(record)
            rows.append(
                RenderedRow(
                    record=record,
                    prompt_chars=prompt_chars,
                    input_tokens_approx=input_tokens_approx,
                    input_tokens=exact_tokens,
                )
            )

    if row_failures:
        summary = ", ".join(f"{count} rows: {reason}" for reason, count in sorted(row_failures.items()))
        print(f"skipped rows for {variant_key(variant)}: {summary}", file=sys.stderr)
    return rows


def resolve_load_environment(env_id: str, env_dir: Path) -> Any:
    module_name = env_id.replace("-", "_")
    env_path = env_dir / module_name
    if str(env_path) not in sys.path:
        sys.path.insert(0, str(env_path))
    candidates = [
        module_name,
        f"{module_name}.{module_name}",
        f"environments.{module_name}",
        f"environments.{module_name}.{module_name}",
    ]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
            load_fn = getattr(module, "load_environment", None)
            if load_fn is not None:
                return load_fn
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise ImportError(f"Unable to resolve load_environment for {env_id}") from last_error


def export_openai_messages(
    item: Mapping[str, Any],
    *,
    env: Any,
    converter: OpenAIChatCompletionsClient,
    runner: asyncio.Runner,
) -> list[dict[str, Any]]:
    raw_prompt = item.get("prompt")
    if isinstance(raw_prompt, str):
        raw_prompt = prompt_from_string(item, env=env)
    if not isinstance(raw_prompt, (str, list)):
        raise UnsupportedRowError(f"prompt must be a string or list of messages, got {type(raw_prompt).__name__}")

    messages = normalize_messages(raw_prompt, field_name="input.prompt")
    native_messages, native_kwargs = runner.run(converter.to_native_prompt(messages))
    if native_kwargs:
        raise UnsupportedRowError(f"unsupported native prompt kwargs for first-request export: {sorted(native_kwargs)}")
    exported = [json_safe(dict(message)) for message in native_messages]
    validate_messages(exported)
    return exported


def prompt_from_string(item: Mapping[str, Any], *, env: Any) -> str | list[dict[str, Any]]:
    prompt = item.get("prompt")
    assert isinstance(prompt, str)
    if FLATTENED_ROLE_RE.search(prompt):
        raise UnsupportedRowError("flattened System/User prompt strings are not role-faithful inputs")

    system_prompt = getattr(env, "system_prompt", None)
    question = item.get("question")
    if system_prompt:
        if isinstance(question, str) and question == prompt:
            messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
            few_shot = getattr(env, "few_shot", None)
            if few_shot:
                messages.extend(json_safe(few_shot))
            messages.append({"role": "user", "content": question})
            return messages
        raise UnsupportedRowError(
            "string prompt with environment system_prompt cannot be exported without flattening roles"
        )
    return prompt


def validate_record(record: Mapping[str, Any]) -> None:
    if "output_tokens" in record:
        raise ValueError("canonical requests rows must not contain output_tokens")
    if "prompt" in record:
        raise ValueError("canonical requests rows must not contain flattened prompt strings")
    validate_messages(record.get("messages"))


def validate_messages(messages: Any) -> None:
    if not isinstance(messages, list) or not messages:
        raise UnsupportedRowError("messages must be a non-empty list")
    for message in messages:
        if not isinstance(message, Mapping):
            raise UnsupportedRowError("each message must be an object")
        role = message.get("role")
        if role not in OPENAI_CHAT_ROLES:
            raise UnsupportedRowError(f"unsupported OpenAI chat message role: {role!r}")
        if "content" not in message and role != "assistant":
            raise UnsupportedRowError(f"{role} message is missing content")
        if role == "tool" and not message.get("tool_call_id"):
            raise UnsupportedRowError("tool messages require tool_call_id")
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                    raise UnsupportedRowError("non-text message content parts are unsupported for this benchmark")
        elif content is not None and not isinstance(content, str):
            raise UnsupportedRowError("message content must be text, null assistant content, or text content parts")


def prompt_chars_from_messages(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(part["text"]) for part in content)
    return total


def render_messages_for_token_audit(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{message['role']}:\n{message_text(message)}" for message in messages)


def message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part["text"] for part in content if isinstance(part, Mapping) and part.get("type") == "text")
    return ""


def approximate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def count_tokens(tokenizer: Any, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def stable_example_id(variant: Variant, idx: int, item: Mapping[str, Any]) -> str:
    material = json.dumps(
        {
            "env_id": variant.env_id,
            "variant_id": variant.variant_id,
            "idx": idx,
            "question": item.get("question"),
            "info": item.get("info", {}),
        },
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]
    return f"{variant.env_id}:{variant.variant_id}:{idx}:{digest}"


def stratified_sample(rows_by_variant: dict[str, list[RenderedRow]], target_size: int, seed: int) -> list[RenderedRow]:
    rng = random.Random(seed)
    available = {key: list(rows) for key, rows in rows_by_variant.items() if rows}
    for rows in available.values():
        rng.shuffle(rows)
    if not available:
        return []

    env_to_keys: dict[str, list[str]] = defaultdict(list)
    for key, rows in available.items():
        env_to_keys[rows[0].record["env_id"]].append(key)

    per_env_target = max(1, target_size // len(env_to_keys))
    env_targets = {env_id: per_env_target for env_id in env_to_keys}
    remainder = target_size - per_env_target * len(env_to_keys)
    for env_id in sorted(env_to_keys)[:remainder]:
        env_targets[env_id] += 1

    longhealth_min = min(
        sum(len(available[key]) for key in env_to_keys.get("longhealth", [])),
        math.ceil(target_size * LONGHEALTH_MIN_FRACTION),
    )
    if longhealth_min > env_targets.get("longhealth", 0):
        delta = longhealth_min - env_targets["longhealth"]
        env_targets["longhealth"] += delta
        for env_id in sorted(env_targets, key=lambda item: env_targets[item], reverse=True):
            if env_id == "longhealth":
                continue
            take = min(delta, max(0, env_targets[env_id] - 1))
            env_targets[env_id] -= take
            delta -= take
            if delta == 0:
                break

    selected: list[RenderedRow] = []
    for env_id, env_target in env_targets.items():
        selected.extend(sample_env(env_to_keys[env_id], available, env_target, rng))

    remaining_pool = [row for rows in available.values() for row in rows]
    rng.shuffle(remaining_pool)
    needed = target_size - len(selected)
    if needed > 0:
        selected.extend(remaining_pool[:needed])
    elif needed < 0:
        selected = selected[:target_size]

    rng.shuffle(selected)
    return selected


def sample_env(
    variant_keys: list[str],
    available: dict[str, list[RenderedRow]],
    env_target: int,
    rng: random.Random,
) -> list[RenderedRow]:
    per_variant = max(1, env_target // len(variant_keys))
    remainder = env_target - per_variant * len(variant_keys)
    selected: list[RenderedRow] = []
    leftovers: list[RenderedRow] = []
    for offset, key in enumerate(sorted(variant_keys)):
        target = per_variant + (1 if offset < remainder else 0)
        rows = available[key]
        selected.extend(rows[:target])
        leftovers.extend(rows[target:])
    if len(selected) < env_target:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: env_target - len(selected)])
    return selected


def build_stats(
    selected_rows: list[RenderedRow],
    rows_by_variant: dict[str, list[RenderedRow]],
    failures: list[dict[str, Any]],
    args: argparse.Namespace,
    target_size: int,
    tokenizer_exact: bool,
) -> dict[str, Any]:
    selected_records = [row.record for row in selected_rows]
    by_env = Counter(record["env_id"] for record in selected_records)
    by_variant = Counter(f"{record['env_id']}::{record['variant_id']}" for record in selected_records)
    exact_token_values = [row.input_tokens for row in selected_rows if row.input_tokens is not None]
    approx_token_values = [row.input_tokens_approx for row in selected_rows]
    bucket_values = exact_token_values if tokenizer_exact else approx_token_values
    prompt_char_values = [row.prompt_chars for row in selected_rows]
    return {
        "artifact_schema": "openai-chat-requests-v1",
        "target_size": target_size,
        "selected_count": len(selected_records),
        "seed": args.seed,
        "tokenizer": args.tokenizer,
        "input_tokens_are_exact": tokenizer_exact,
        "input_tokens_definition": (
            "Exact tokenizer count over deterministic role/content audit rendering. "
            "Null summary values mean no tokenizer was provided."
        ),
        "input_tokens_approx_definition": (
            "Approximate ceil(chars/4) count over deterministic role/content audit rendering."
        ),
        "prompt_chars_definition": "Sum of text content lengths across exported OpenAI-native input messages.",
        "counts_by_env": dict(sorted(by_env.items())),
        "counts_by_variant": dict(sorted(by_variant.items())),
        "available_by_variant": {key: len(rows) for key, rows in sorted(rows_by_variant.items())},
        "token_buckets": bucket_counts([int(value) for value in bucket_values]),
        "token_buckets_source": "input_tokens" if tokenizer_exact else "input_tokens_approx",
        "prompt_chars": summarize(prompt_char_values),
        "input_tokens_approx": summarize(approx_token_values),
        "input_tokens": summarize([int(value) for value in exact_token_values]),
        "failures": failures,
    }


def bucket_counts(values: list[int]) -> dict[str, int]:
    buckets = [(0, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192), (8193, 16384)]
    counts: dict[str, int] = {}
    for low, high in buckets:
        counts[f"{low}-{high}"] = sum(1 for value in values if low <= value <= high)
    counts[">16384"] = sum(1 for value in values if value > 16384)
    return counts


def summarize(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p99": None, "max": None}
    sorted_values = sorted(values)
    return {
        "min": sorted_values[0],
        "p50": percentile(sorted_values, 0.50),
        "p90": percentile(sorted_values, 0.90),
        "p99": percentile(sorted_values, 0.99),
        "max": sorted_values[-1],
    }


def percentile(sorted_values: list[int], q: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = q * (len(sorted_values) - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return float(sorted_values[lo])
    return sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo)


def variant_key(variant: Variant) -> str:
    return f"{variant.env_id}::{variant.variant_id}"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_bench_toml(path: Path, requests_path: Path, target_size: int, max_tokens: int) -> None:
    relative_requests = requests_path.name if requests_path.parent == path.parent else str(requests_path)
    content = f"""# MedMarks vLLM request benchmark settings.
# Consume this with benchmarks/medmarks_vllm/bench_client/run_requests.py.

dataset_path = {json.dumps(relative_requests)}
num_prompts = {target_size}
max_tokens = {max_tokens}
stream = true
request_rate = 0
max_concurrency = 16
warmup_requests = 1
timeout_seconds = 120
retries = 2
"""
    path.write_text(content, encoding="utf-8")


def json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "model_dump"):
            return json_safe(value.model_dump())
        if isinstance(value, Mapping):
            return {str(key): json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        return str(value)


if __name__ == "__main__":
    main()
