"""Metadata normalization utilities for the JSONL → Parquet export pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from medarc_verifiers.export.parquet.discovery import RunRecord

logger = logging.getLogger(__name__)

VARIANT_HASH_PREFIX_LENGTH = 16


@dataclass(frozen=True, slots=True)
class NormalizedMetadata:
    """Normalized view of metadata.json contents combined with manifest details."""

    record: RunRecord
    metadata_path: Path
    raw_metadata: Mapping[str, Any]
    base_env_id: str
    model: str | None
    env_args: Mapping[str, Any]
    env_columns: Mapping[str, Any]
    variant_id: str
    num_examples: int | None
    rollouts_per_example: int | None
    sampling_args: Mapping[str, Any]


def load_normalized_metadata(record: RunRecord) -> NormalizedMetadata | None:
    """Load and normalize metadata.json for a given run record."""
    if not record.has_metadata:
        logger.debug("Run %s missing metadata.json; skipping.", record.job_id)
        return None

    try:
        payload = json.loads(record.metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse metadata for %s: %s", record.metadata_path, exc)
        return None
    if not isinstance(payload, Mapping):
        logger.debug("metadata.json for %s is not a mapping; skipping.", record.job_id)
        return None

    base_env_id = _coerce_optional_str(payload.get("env_id"))
    env_args = _ensure_mapping(payload.get("env_args"), "metadata.env_args")
    env_columns = _build_env_columns(env_args)
    variant_id = _compute_variant_id(record, env_args, base_env_id)

    model = _coerce_optional_str(payload.get("model"))
    num_examples = _coerce_optional_int(payload.get("num_examples"))
    rollouts = _coerce_optional_int(payload.get("rollouts_per_example"))
    sampling_args = _ensure_mapping(payload.get("sampling_args"), "metadata.sampling_args")

    return NormalizedMetadata(
        record=record,
        metadata_path=record.metadata_path,
        raw_metadata=payload,
        base_env_id=base_env_id or record.manifest_env_id or "",
        model=model,
        env_args=env_args,
        env_columns=env_columns,
        variant_id=variant_id,
        num_examples=num_examples,
        rollouts_per_example=rollouts,
        sampling_args=sampling_args,
    )


def _compute_variant_id(
    record: RunRecord,
    env_args: Mapping[str, Any],
    base_env_id: str | None,
) -> str:
    """Derive a stable variant identifier from env args and manifest context."""
    normalized_payload = {
        "manifest_env_id": record.manifest_env_id,
        "base_env_id": base_env_id,
        "env_args": _normalize_for_hash(env_args),
    }
    payload_json = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return digest[:VARIANT_HASH_PREFIX_LENGTH]


def _normalize_for_hash(env_args: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized: dict[str, Any] = {}
    for key in sorted(env_args):
        value = env_args[key]
        normalized[key] = value
    return normalized


def _build_env_columns(env_args: Mapping[str, Any]) -> Mapping[str, Any]:
    columns: dict[str, Any] = {}
    for key, value in env_args.items():
        if not isinstance(key, str) or not key:
            continue
        columns[f"env_{key}"] = _coerce_scalar(value)
    return columns


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    logger.debug("Expected mapping for %s but received %r; defaulting to empty mapping.", context, value)
    return {}


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) or value is None:
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return ""
        int_candidate = _try_parse_int(stripped)
        if int_candidate is not None:
            return int_candidate
        float_candidate = _try_parse_float(stripped)
        if float_candidate is not None:
            return float_candidate
        if stripped.lower() in {"true", "false"}:
            return stripped.lower() == "true"
        return stripped
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return _try_parse_int(value.strip())
    return None


def _try_parse_int(value: str) -> int | None:
    try:
        return int(value, 10)
    except (TypeError, ValueError):
        return None


def _try_parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["NormalizedMetadata", "load_normalized_metadata"]
