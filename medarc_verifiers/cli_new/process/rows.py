"""Row loading and enrichment utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from medarc_verifiers.cli_new.process.metadata import NormalizedMetadata

logger = logging.getLogger(__name__)

DEFAULT_DROP_COLUMNS = {"info", "sampling_args"}
PROMPT_COMPLETION_COLUMNS = {"prompt", "completion"}


def load_rows(
    metadata: NormalizedMetadata,
    *,
    include_prompt_completion: bool = False,
    keep_columns: Sequence[str] | None = None,
    drop_columns: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Load results.jsonl rows and attach manifest metadata."""
    record = metadata.record
    if not record.has_results:
        logger.debug("Run %s missing results.jsonl; skipping.", record.job_id)
        return []

    results_path = record.results_path
    keep = {column for column in keep_columns or () if column}
    drop = {column for column in drop_columns or () if column}
    drop.update(DEFAULT_DROP_COLUMNS)
    if include_prompt_completion:
        drop.difference_update(PROMPT_COMPLETION_COLUMNS)
    else:
        drop.update(PROMPT_COMPLETION_COLUMNS)

    # First pass: decode and clean rows, and count example_id occurrences to
    # detect multiple rollouts within a single JSONL (example_id repetition).
    decoded_rows: list[tuple[int, Mapping[str, Any]]] = []
    example_counts: dict[Any, int] = {}
    try:
        with results_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = _decode_line(line, results_path, line_number)
                decoded_rows.append((line_number, payload))
                ex_id = payload.get("example_id")
                # Count occurrences to infer intra-file rollout structure.
                try:
                    example_counts[ex_id] = example_counts.get(ex_id, 0) + 1
                except TypeError:
                    # Non-hashable example_id shouldn't happen (schema requires
                    # primitive), but guard just in case.
                    pass
    except ValueError:
        raise
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to read %s: %s", results_path, exc)
        return []

    multi_rollout = any(count > 1 for count in example_counts.values())

    # Second pass: enrich rows. If the file contains multiple rollouts, compute
    # a data-driven rollout_index by counting seen occurrences per example_id.
    # Otherwise, retain the suffix/dir-derived rollout_index from metadata.
    rows: list[dict[str, Any]] = []
    seen_per_example: dict[Any, int] = {}
    for line_number, payload in decoded_rows:
        cleaned = _clean_row(payload, drop=drop, keep=keep)
        if multi_rollout:
            ex_id = payload.get("example_id")
            try:
                seen = seen_per_example.get(ex_id, 0)
                rollout_index = seen  # 0-based occurrence index
                seen_per_example[ex_id] = seen + 1
            except TypeError:
                # Fallback to metadata rollout_index if example_id is unusable as key
                rollout_index = metadata.rollout_index
        else:
            rollout_index = metadata.rollout_index
        enriched = _attach_metadata(cleaned, metadata, line_number=line_number, rollout_index=rollout_index)
        rows.append(enriched)

    return rows


def _decode_line(line: str, path: Path, line_number: int) -> Mapping[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:  # pragma: no cover - explicit error path
        message = f"Failed to parse JSONL line {line_number} in {path}: {exc.msg}"
        raise ValueError(message) from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected JSON object at {path}:{line_number}")
    if "example_id" not in payload:
        env_id = payload.get("env_id")
        raise ValueError(f"Missing example_id in {path}:{line_number} (env_id={env_id!r})")
    return payload


def _clean_row(
    row: Mapping[str, Any],
    *,
    drop: set[str],
    keep: set[str],
) -> MutableMapping[str, Any]:
    cleaned: MutableMapping[str, Any] = {}
    for key, value in row.items():
        if key in drop and key not in keep:
            continue
        if key not in keep and not _is_primitive(value):
            continue
        cleaned[key] = value
    return cleaned


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _attach_metadata(
    row: MutableMapping[str, Any],
    metadata: NormalizedMetadata,
    *,
    line_number: int,
    rollout_index: int,
) -> MutableMapping[str, Any]:
    record = metadata.record

    error_value = record.reason if record.status == "failed" else None

    env_identifier = metadata.base_env_id or metadata.manifest_env_id

    row.update(
        {
            "env_id": env_identifier,
            "manifest_env_id": metadata.manifest_env_id,
            "base_env_id": metadata.base_env_id,
            "job_run_id": record.manifest.job_run_id,
            "run_id": record.job_id,
            "model_id": metadata.model_id,
            "rollout_index": rollout_index,
            "status": record.status,
            "error": error_value,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
        }
    )
    return row


__all__ = ["load_rows"]
