"""Metadata normalization utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from pydantic import BaseModel, Field, ValidationError

from medarc_verifiers.cli.process.discovery import RunRecord
from medarc_verifiers.cli.process.rollout import derive_base_env_id, extract_rollout_index

logger = logging.getLogger(__name__)


class _MetadataPayload(BaseModel):
    """Lightweight schema for metadata.json rows."""

    env_id: str | None = None
    model: str | None = None
    env_args: dict[str, Any] = Field(default_factory=dict)
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    sampling_args: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class NormalizedMetadata:
    """Normalized view of metadata.json merged with manifest discovery data."""

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


def load_normalized_metadata(
    record: RunRecord,
    *,
    combine_rollouts: bool = True,
) -> NormalizedMetadata:
    """Merge manifest fields with metadata.json (when present)."""
    metadata_payload, raw_metadata = _load_metadata(record)
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
    # If we didn't capture a rollout index from the manifest env id,
    # try to derive it from the results directory name (common when
    # manifests keep base env id, but the on-disk folder encodes the rollout).
    if rollout_index == 0 and record.results_dir_name:
        alt_index = extract_rollout_index(record.results_dir_name)
        if alt_index:
            rollout_index = alt_index

    model_id = record.model_id or metadata_model
    num_examples = record.num_examples or (metadata_payload.num_examples if metadata_payload else None)
    rollouts_per_example = record.rollouts_per_example or (
        metadata_payload.rollouts_per_example if metadata_payload else None
    )

    return NormalizedMetadata(
        record=record,
        metadata_path=record.metadata_path if record.has_metadata else None,
        raw_metadata=raw_metadata,
        manifest_env_id=manifest_env_id,
        metadata_env_id=metadata_env_id,
        base_env_id=base_env_id,
        rollout_index=rollout_index,
        model_id=model_id,
        metadata_model=metadata_model,
        env_args=env_args,
        sampling_args=sampling_args,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
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
    try:
        model = _MetadataPayload.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Invalid metadata schema for %s: %s", path, exc)
        return None, {}
    return model, model.model_dump(mode="python")


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


def _extract_env_config_id(env_config: Mapping[str, Any] | None) -> str | None:
    if not env_config:
        return None
    value = env_config.get("id")
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed:
            return trimmed
    return None


__all__ = ["NormalizedMetadata", "load_normalized_metadata"]
