"""Metadata normalization utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from pydantic import BaseModel, Field, ValidationError

from medarc_verifiers.cli.eval_identity import (
    MEDARC_CONFIG_FINGERPRINT_KEY,
    MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY,
    MEDARC_VARIANT_ID_KEY,
    MEDARC_VARIANT_PAYLOAD_KEY,
)
from medarc_verifiers.cli.process.discovery import RunRecord
from medarc_verifiers.cli.process.rollout import derive_base_env_id, extract_rollout_index

logger = logging.getLogger(__name__)


class _MetadataPayload(BaseModel):
    """Lightweight schema for metadata.json rows."""

    env_id: str | None = None
    model: str | None = None
    avg_reward: float | None = None
    version_info: dict[str, str | None] | None = None
    env_args: dict[str, Any] = Field(default_factory=dict)
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    sampling_args: dict[str, Any] = Field(default_factory=dict)
    medarc_config_fingerprint: str | None = None
    medarc_config_fingerprint_payload: dict[str, Any] | None = None
    variant_id: str | None = None
    variant_payload: dict[str, Any] | None = None


@dataclass(slots=True)
class NormalizedMetadata:
    """Normalized view of metadata.json merged with manifest discovery data."""

    identity: "RunIdentity"
    record: RunRecord
    metadata_path: Path | None
    raw_metadata: Mapping[str, Any]
    manifest_env_id: str
    metadata_env_id: str | None
    base_env_id: str
    rollout_index: int
    model_id: str | None
    metadata_model: str | None
    env_args: Mapping[str, Any]
    sampling_args: Mapping[str, Any]
    num_examples: int | None
    rollouts_per_example: int | None
    variant_id: str | None
    variant_payload: Mapping[str, Any] | None
    medarc_config_fingerprint: str | None
    medarc_config_fingerprint_payload: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """Canonical identity for selecting and exporting a discovered run record."""

    model_id: str
    manifest_env_id: str
    base_env_id: str
    rollout_index: int | None
    job_run_id: str
    output_env_id: str


@dataclass(frozen=True, slots=True)
class ResolvedRunIdentity:
    """Selection-time identity that tolerates missing model ids."""

    model_id: str | None
    manifest_env_id: str
    base_env_id: str
    rollout_index: int | None
    job_run_id: str
    output_env_id: str
    variant_id: str | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedMetadataContext:
    raw_metadata: Mapping[str, Any]
    manifest_env_id: str
    metadata_env_id: str | None
    base_env_id: str
    rollout_index: int
    model_id: str | None
    metadata_model: str | None
    env_args: Mapping[str, Any]
    sampling_args: Mapping[str, Any]
    num_examples: int | None
    rollouts_per_example: int | None
    variant_id: str | None
    variant_payload: Mapping[str, Any] | None
    medarc_config_fingerprint: str | None
    medarc_config_fingerprint_payload: Mapping[str, Any] | None


def resolve_run_identity(
    record: RunRecord,
    *,
    combine_rollouts: bool = True,
) -> ResolvedRunIdentity:
    """Resolve a run identity for selection without requiring model_id."""
    context = _resolve_metadata_context(record, combine_rollouts=combine_rollouts)
    resolved_rollout_index = (
        context.rollout_index if context.rollout_index != 0 or context.manifest_env_id != context.base_env_id else None
    )
    return ResolvedRunIdentity(
        model_id=context.model_id,
        manifest_env_id=context.manifest_env_id,
        base_env_id=context.base_env_id,
        rollout_index=resolved_rollout_index,
        job_run_id=record.manifest.job_run_id,
        output_env_id=context.base_env_id or context.manifest_env_id or record.job_id,
        variant_id=context.variant_id,
    )


def load_normalized_metadata(
    record: RunRecord,
    *,
    combine_rollouts: bool = True,
) -> NormalizedMetadata:
    """Merge manifest fields with metadata.json (when present)."""
    context = _resolve_metadata_context(record, combine_rollouts=combine_rollouts)
    if not context.model_id:
        raise RuntimeError(format_missing_model_id_error(record))
    resolved_rollout_index = (
        context.rollout_index if context.rollout_index != 0 or context.manifest_env_id != context.base_env_id else None
    )
    identity = RunIdentity(
        model_id=context.model_id,
        manifest_env_id=context.manifest_env_id,
        base_env_id=context.base_env_id,
        rollout_index=resolved_rollout_index,
        job_run_id=record.manifest.job_run_id,
        output_env_id=context.base_env_id or context.manifest_env_id or record.job_id,
    )

    return NormalizedMetadata(
        identity=identity,
        record=record,
        metadata_path=record.metadata_path if record.has_metadata else None,
        raw_metadata=context.raw_metadata,
        manifest_env_id=context.manifest_env_id,
        metadata_env_id=context.metadata_env_id,
        base_env_id=context.base_env_id,
        rollout_index=identity.rollout_index or 0,
        model_id=identity.model_id,
        metadata_model=context.metadata_model,
        env_args=context.env_args,
        sampling_args=context.sampling_args,
        num_examples=context.num_examples,
        rollouts_per_example=context.rollouts_per_example,
        variant_id=context.variant_id,
        variant_payload=context.variant_payload,
        medarc_config_fingerprint=context.medarc_config_fingerprint,
        medarc_config_fingerprint_payload=context.medarc_config_fingerprint_payload,
    )


def _resolve_metadata_context(
    record: RunRecord,
    *,
    combine_rollouts: bool,
) -> _ResolvedMetadataContext:
    metadata_payload, raw_metadata = _load_metadata(record)
    _warn_manifest_metadata_result_mismatch(record, metadata_payload)
    metadata_env_id = metadata_payload.env_id if metadata_payload else None
    metadata_model = metadata_payload.model if metadata_payload else None
    env_args = _merge_mappings(
        primary=record.env_args,
        fallback=metadata_payload.env_args if metadata_payload else None,
    )
    sampling_args = _merge_mappings(
        primary=record.sampling_args,
        fallback=metadata_payload.sampling_args if metadata_payload else None,
    )
    manifest_env_id = (
        _extract_env_config_id(record.env_config) or record.manifest_env_id or metadata_env_id or record.job_id
    )
    base_env_id, rollout_index = derive_base_env_id(
        manifest_env_id,
        combine_rollouts=combine_rollouts,
    )
    if rollout_index == 0 and record.results_dir_name:
        alt_index = extract_rollout_index(record.results_dir_name)
        if alt_index:
            rollout_index = alt_index
    return _ResolvedMetadataContext(
        raw_metadata=raw_metadata,
        manifest_env_id=manifest_env_id,
        metadata_env_id=metadata_env_id,
        base_env_id=base_env_id,
        rollout_index=rollout_index,
        model_id=record.model_id or metadata_model,
        metadata_model=metadata_model,
        env_args=env_args,
        sampling_args=sampling_args,
        num_examples=_prefer_manifest_value(
            record.num_examples,
            metadata_payload.num_examples if metadata_payload else None,
        ),
        rollouts_per_example=_prefer_manifest_value(
            record.rollouts_per_example,
            metadata_payload.rollouts_per_example if metadata_payload else None,
        ),
        variant_id=_string_or_none(
            _raw_metadata_value(raw_metadata, MEDARC_VARIANT_ID_KEY, metadata_payload.variant_id if metadata_payload else None)
        ),
        variant_payload=_mapping_or_none(
            _raw_metadata_value(
                raw_metadata,
                MEDARC_VARIANT_PAYLOAD_KEY,
                metadata_payload.variant_payload if metadata_payload else None,
            )
        ),
        medarc_config_fingerprint=_string_or_none(
            _raw_metadata_value(
                raw_metadata,
                MEDARC_CONFIG_FINGERPRINT_KEY,
                metadata_payload.medarc_config_fingerprint if metadata_payload else None,
            )
        ),
        medarc_config_fingerprint_payload=_mapping_or_none(
            _raw_metadata_value(
                raw_metadata,
                MEDARC_CONFIG_FINGERPRINT_PAYLOAD_KEY,
                metadata_payload.medarc_config_fingerprint_payload if metadata_payload else None,
            )
        ),
    )


def format_missing_model_id_error(record: RunRecord) -> str:
    return (
        "Missing model_id for run "
        f"(job_run_id={record.manifest.job_run_id}, job_id={record.job_id}, "
        f"results_dir={record.results_dir}, manifest={record.manifest.manifest_path})"
    )


def _load_metadata(record: RunRecord) -> tuple[_MetadataPayload | None, Mapping[str, Any]]:
    if not record.has_metadata:
        return None, {}
    path = record.metadata_path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to read metadata for %s: %s", path, exc)
        return None, {}
    if not isinstance(payload, Mapping):
        logger.warning(
            "Invalid metadata payload type for %s: expected JSON object, got %s", path, type(payload).__name__
        )
        return None, {}
    raw_payload = dict(payload)
    try:
        model = _MetadataPayload.model_validate(raw_payload)
    except ValidationError as exc:
        logger.warning("Invalid metadata schema for %s: %s", path, exc)
        return None, _sanitize_invalid_raw_metadata(raw_payload)
    return model, raw_payload


def _sanitize_invalid_raw_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    sanitized: dict[str, Any] = {}
    version_info = payload.get("version_info")
    if isinstance(version_info, Mapping):
        sanitized["version_info"] = dict(version_info)
    endpoint_id = payload.get("endpoint_id")
    if isinstance(endpoint_id, str):
        sanitized["endpoint_id"] = endpoint_id
    base_url = payload.get("base_url")
    if isinstance(base_url, str):
        sanitized["base_url"] = base_url
    return sanitized


def _merge_mappings(
    primary: Mapping[str, Any] | None,
    *,
    fallback: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    result: MutableMapping[str, Any] = {}
    if fallback:
        result.update(fallback)
    if primary:
        result.update(primary)
    return result


def _prefer_manifest_value(primary: int | None, fallback: int | None) -> int | None:
    if primary is not None:
        return primary
    return fallback


def _raw_metadata_value(raw_metadata: Mapping[str, Any], key: str, fallback: Any) -> Any:
    if key in raw_metadata:
        return raw_metadata.get(key)
    return fallback


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _warn_manifest_metadata_result_mismatch(record: RunRecord, metadata_payload: _MetadataPayload | None) -> None:
    if metadata_payload is None:
        return

    mismatches: list[str] = []
    if _has_float_mismatch(record.avg_reward, metadata_payload.avg_reward):
        mismatches.append(f"avg_reward manifest={record.avg_reward!r} metadata={metadata_payload.avg_reward!r}")
    if _has_int_mismatch(record.num_examples, metadata_payload.num_examples):
        mismatches.append(f"num_examples manifest={record.num_examples!r} metadata={metadata_payload.num_examples!r}")
    if not mismatches:
        return

    logger.warning(
        "Manifest/metadata result mismatch for process input (job_run_id=%s, job_id=%s, metadata=%s): %s",
        record.manifest.job_run_id,
        record.job_id,
        record.metadata_path,
        "; ".join(mismatches),
    )


def _has_float_mismatch(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _has_int_mismatch(left: int | None, right: int | None) -> bool:
    if left is None or right is None:
        return False
    return left != right


def _extract_env_config_id(env_config: Mapping[str, Any] | None) -> str | None:
    if not env_config:
        return None
    value = env_config.get("id")
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


__all__ = [
    "NormalizedMetadata",
    "ResolvedRunIdentity",
    "RunIdentity",
    "format_missing_model_id_error",
    "load_normalized_metadata",
    "resolve_run_identity",
]
