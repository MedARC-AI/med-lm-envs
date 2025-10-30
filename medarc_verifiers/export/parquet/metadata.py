"""Metadata normalization utilities for the JSONL → Parquet export pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from medarc_verifiers.export.parquet.discovery import RunRecord
from medarc_verifiers.export.parquet.schemas import MetadataModel

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
    try:
        metadata_model = MetadataModel.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Metadata schema validation failed for %s: %s", record.metadata_path, exc)
        return None

    env_args = metadata_model.env_args
    env_columns = _build_env_columns(env_args)
    variant_id = _compute_variant_id(record, env_args, metadata_model.env_id)

    raw_metadata = metadata_model.model_dump(mode="python")

    return NormalizedMetadata(
        record=record,
        metadata_path=record.metadata_path,
        raw_metadata=raw_metadata,
        base_env_id=(metadata_model.env_id or record.manifest_env_id or ""),
        model=metadata_model.model,
        env_args=env_args,
        env_columns=env_columns,
        variant_id=variant_id,
        num_examples=metadata_model.num_examples,
        rollouts_per_example=metadata_model.rollouts_per_example,
        sampling_args=metadata_model.sampling_args,
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
        "env_args": dict(sorted(env_args.items())),
    }
    payload_json = json.dumps(normalized_payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
    return digest[:VARIANT_HASH_PREFIX_LENGTH]


def _build_env_columns(env_args: Mapping[str, Any]) -> Mapping[str, Any]:
    columns: dict[str, Any] = {}
    for key, value in env_args.items():
        if not isinstance(key, str) or not key:
            continue
        columns[f"env_{key}"] = _coerce_scalar(value)
    return columns


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
