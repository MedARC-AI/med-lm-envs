"""Bench sidecar planning and validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verifiers.utils.save_utils import make_serializable

from medarc_verifiers.cli.eval_identity import EvalPathPlan

BENCH_INDEX_FILENAME = "bench_index.json"
BENCH_INDEX_VERSION = 1


class BenchIndexError(ValueError):
    """Raised when a bench sidecar is missing, stale, or internally inconsistent."""


def build_bench_index(
    *,
    output_root: Path,
    source_config: Path,
    eval_configs: Sequence[Any],
    path_plans: Sequence[EvalPathPlan],
    plan_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    entries = [
        build_bench_index_entry(
            index=index,
            output_root=output_root,
            config=config,
            path_plan=path_plan,
            plan_payload=plan_payload,
        )
        for index, (config, path_plan, plan_payload) in enumerate(
            zip(eval_configs, path_plans, plan_payloads), start=1
        )
    ]
    payload = {
        "version": BENCH_INDEX_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "source_config": str(source_config),
        "evals": entries,
    }
    validate_bench_index(payload, output_root=output_root, require_artifacts=False)
    return payload


def build_bench_index_entry(
    *,
    index: int,
    output_root: Path,
    config: Any,
    path_plan: EvalPathPlan,
    plan_payload: Mapping[str, Any],
) -> dict[str, Any]:
    identity = path_plan.identity
    entry = {
        "index": index,
        "results_path": str(path_plan.results_path),
        "env_id": identity.env_id,
        "model": identity.model_id,
        "variant_id": identity.variant_id,
        "variant_payload": identity.variant_payload,
        "env_args": dict(config.env_args or {}),
        "sampling_args": dict(config.sampling_args or {}),
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
    }
    digest_payload = {key: value for key, value in entry.items() if key != "index"}
    entry["plan_digest"] = plan_digest({**digest_payload, "output_root": str(output_root), "plan": dict(plan_payload)})
    return entry


def read_bench_index(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME} at {path}: expected JSON object.") from exc
    if not isinstance(payload, dict):
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME} at {path}: expected JSON object.")
    return payload


def write_bench_index(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=make_serializable, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_bench_index(
    payload: Mapping[str, Any],
    *,
    output_root: Path,
    require_artifacts: bool,
) -> None:
    if payload.get("version") != BENCH_INDEX_VERSION:
        raise BenchIndexError(f"Unsupported {BENCH_INDEX_FILENAME} version: {payload.get('version')!r}.")
    entries = payload.get("evals")
    if not isinstance(entries, list):
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: 'evals' must be a list.")

    normalized_root = output_root.resolve()
    paths: list[Path] = []
    identities: list[tuple[str, str, str | None]] = []
    model_env_counts: Counter[tuple[str, str]] = Counter()
    for offset, raw_entry in enumerate(entries, start=1):
        if not isinstance(raw_entry, Mapping):
            raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: eval entry {offset} must be an object.")
        results_path = _entry_results_path(raw_entry)
        _require_under_root(results_path, normalized_root)
        paths.append(results_path.resolve())

        model = _required_string(raw_entry, "model", offset)
        env_id = _required_string(raw_entry, "env_id", offset)
        variant_id = raw_entry.get("variant_id")
        if variant_id is not None and not isinstance(variant_id, str):
            raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: eval entry {offset} variant_id must be a string.")
        identities.append((model, env_id, variant_id))
        model_env_counts[(model, env_id)] += 1

        if require_artifacts:
            _require_artifact(results_path / "metadata.json")
            _require_artifact(results_path / "results.jsonl")
            _validate_metadata_identity(results_path / "metadata.json", model=model, env_id=env_id)

    _raise_duplicates(paths, label="results_path")
    for (model, env_id), count in model_env_counts.items():
        if count > 1:
            missing_variant = [identity for identity in identities if identity[:2] == (model, env_id) and not identity[2]]
            if missing_variant:
                raise BenchIndexError(
                    f"Duplicate bench entries for model={model!r}, env_id={env_id!r} require explicit variant_id."
                )
    _raise_duplicates(identities, label="(model, env_id, variant_id)")


def find_entry_for_results_path(payload: Mapping[str, Any], results_path: Path) -> Mapping[str, Any] | None:
    target = results_path.resolve()
    entries = payload.get("evals")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, Mapping) and _entry_results_path(entry).resolve() == target:
            return entry
    return None


def plan_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_canonicalize(payload), sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _entry_results_path(entry: Mapping[str, Any]) -> Path:
    raw_path = entry.get("results_path")
    if not isinstance(raw_path, str) or not raw_path:
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: each eval entry needs a non-empty results_path.")
    return Path(raw_path)


def _required_string(entry: Mapping[str, Any], key: str, offset: int) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: eval entry {offset} needs non-empty {key}.")
    return value


def _require_under_root(path: Path, normalized_root: Path) -> None:
    try:
        path.resolve().relative_to(normalized_root)
    except ValueError as exc:
        raise BenchIndexError(
            f"Invalid {BENCH_INDEX_FILENAME}: results_path {path} is outside output root {normalized_root}."
        ) from exc


def _require_artifact(path: Path) -> None:
    if not path.is_file():
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: required artifact is missing: {path}.")


def _validate_metadata_identity(metadata_path: Path, *, model: str, env_id: str) -> None:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchIndexError(f"Invalid metadata.json at {metadata_path}: expected JSON object.") from exc
    if not isinstance(metadata, Mapping):
        raise BenchIndexError(f"Invalid metadata.json at {metadata_path}: expected JSON object.")
    for key, expected in (("model", model), ("env_id", env_id)):
        current = metadata.get(key)
        if current is not None and current != expected:
            raise BenchIndexError(
                f"{BENCH_INDEX_FILENAME} identity mismatch for {metadata_path.parent}: "
                f"{key} sidecar={expected!r} metadata={current!r}."
            )


def _raise_duplicates(values: Sequence[Any], *, label: str) -> None:
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        rendered = ", ".join(str(value) for value in duplicates)
        raise BenchIndexError(f"Invalid {BENCH_INDEX_FILENAME}: duplicate {label}: {rendered}.")


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return [_canonicalize(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


__all__ = [
    "BENCH_INDEX_FILENAME",
    "BENCH_INDEX_VERSION",
    "BenchIndexError",
    "build_bench_index",
    "build_bench_index_entry",
    "find_entry_for_results_path",
    "plan_digest",
    "read_bench_index",
    "validate_bench_index",
    "write_bench_index",
]
