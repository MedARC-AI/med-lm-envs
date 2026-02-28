"""Row loading and enrichment utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from medarc_verifiers.cli.process.metadata import NormalizedMetadata

logger = logging.getLogger(__name__)

DEFAULT_DROP_COLUMNS = {"info", "sampling_args", "extras"}
PROMPT_COMPLETION_COLUMNS = {"prompt", "completion"}
# Top-level JSON fields we explicitly allow through even though they are not primitives.
# These may be absent in older results and will appear as nulls in existing parquet files.
ALLOWED_JSON_COLUMNS = {"token_usage"}


def load_rows(
    metadata: NormalizedMetadata,
    *,
    extra_columns: Sequence[str] | None = None,
    drop_columns: Sequence[str] | None = None,
    answer_column: str | None = None,
) -> list[dict[str, Any]]:
    """Load results.jsonl rows and attach manifest metadata."""
    record = metadata.record
    if not record.has_results:
        raise FileNotFoundError(
            "Missing results.jsonl for selected run "
            f"(job_run_id={record.manifest.job_run_id}, job_id={record.job_id}, path={record.results_path})"
        )

    results_path = record.results_path
    extras_keys = {column for column in extra_columns or () if column}
    drop = {column for column in drop_columns or () if column}
    drop.update(DEFAULT_DROP_COLUMNS)
    drop.update(PROMPT_COMPLETION_COLUMNS)
    decoded_rows, example_counts = _decode_results_jsonl(results_path)
    multi_rollout = _detect_multi_rollout_shape(example_counts)
    version_info_json = _encode_metadata_json_column(metadata.raw_metadata.get("version_info"))

    rows: list[dict[str, Any]] = []
    seen_per_example: dict[Any, int] = {}
    for line_number, payload in decoded_rows:
        cleaned, extras = _clean_payload_row(
            payload,
            extras_keys=extras_keys,
            drop=drop,
            answer_column=answer_column,
        )
        rollout_index = _resolve_rollout_index(
            payload,
            metadata,
            multi_rollout=multi_rollout,
            seen_per_example=seen_per_example,
        )
        if extras_keys and extras:
            cleaned["extras"] = json.dumps(extras, sort_keys=True)
        else:
            cleaned["extras"] = None
        enriched = _attach_row_metadata(
            cleaned,
            metadata,
            line_number=line_number,
            rollout_index=rollout_index,
            version_info_json=version_info_json,
        )
        rows.append(enriched)

    return rows


def _decode_results_jsonl(path: Path) -> tuple[list[tuple[int, Mapping[str, Any]]], dict[Any, int]]:
    """Decode results.jsonl and count example_id occurrences for rollout detection."""
    decoded_rows: list[tuple[int, Mapping[str, Any]]] = []
    example_counts: dict[Any, int] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                payload = _decode_line(line, path, line_number)
                decoded_rows.append((line_number, payload))
                ex_id = payload.get("example_id")
                try:
                    example_counts[ex_id] = example_counts.get(ex_id, 0) + 1
                except TypeError:
                    pass
    except ValueError:
        raise
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to read %s: %s", path, exc)
        return [], {}
    return decoded_rows, example_counts


def _detect_multi_rollout_shape(example_counts: Mapping[Any, int]) -> bool:
    return any(count > 1 for count in example_counts.values())


def _clean_payload_row(
    payload: Mapping[str, Any],
    *,
    extras_keys: set[str],
    drop: set[str],
    answer_column: str | None,
) -> tuple[MutableMapping[str, Any], Mapping[str, Any]]:
    extras = _extract_extras(payload, extras_keys=extras_keys)
    cleaned = _clean_row(payload, drop=drop, extras_keys=extras_keys)
    cleaned.pop("rollout_index", None)
    _map_answer_column(cleaned, payload, answer_column=answer_column)
    _normalize_token_usage(cleaned)
    payload_rollout_index = _coerce_rollout_index(payload.get("rollout_index"))
    if payload_rollout_index is not None:
        cleaned["rollout_index"] = payload_rollout_index
    return cleaned, extras


def _resolve_rollout_index(
    payload: Mapping[str, Any],
    metadata: NormalizedMetadata,
    *,
    multi_rollout: bool,
    seen_per_example: MutableMapping[Any, int],
) -> int:
    payload_rollout_index = _coerce_rollout_index(payload.get("rollout_index"))
    if payload_rollout_index is not None:
        return payload_rollout_index
    if not multi_rollout:
        return metadata.rollout_index

    ex_id = payload.get("example_id")
    try:
        seen = seen_per_example.get(ex_id, 0)
        seen_per_example[ex_id] = seen + 1
        return seen
    except TypeError:
        return metadata.rollout_index


def _map_answer_column(
    cleaned: MutableMapping[str, Any],
    payload: Mapping[str, Any],
    *,
    answer_column: str | None,
) -> None:
    if not answer_column or answer_column == "answer":
        return
    if "answer" in cleaned:
        return
    if answer_column not in payload:
        return
    value = payload.get(answer_column)
    if not _is_primitive(value):
        return
    cleaned["answer"] = value


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
    extras_keys: set[str],
) -> MutableMapping[str, Any]:
    cleaned: MutableMapping[str, Any] = {}

    # First pass: process top-level keys
    for key, value in row.items():
        if key in extras_keys:
            continue
        if key in drop:
            continue
        is_allowed_json = key in ALLOWED_JSON_COLUMNS and isinstance(value, Mapping)
        if not _is_primitive(value) and not is_allowed_json:
            continue
        cleaned[key] = value

    return cleaned


def _extract_extras(row: Mapping[str, Any], *, extras_keys: set[str]) -> Mapping[str, Any]:
    """Extract env-specific keys into an extras mapping (excluded from top-level columns)."""
    if not extras_keys:
        return {}
    extras: dict[str, Any] = {}
    info = row.get("info") if isinstance(row.get("info"), Mapping) else {}

    for key in sorted(extras_keys):
        if key in row:
            extras[key] = row.get(key)
            continue
        if info and key in info:
            extras[key] = info.get(key)
    # Drop null-only payloads to keep extras=None for rows without values.
    if all(value is None for value in extras.values()):
        return {}
    return extras


def _is_primitive(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _coerce_rollout_index(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _attach_row_metadata(
    row: MutableMapping[str, Any],
    metadata: NormalizedMetadata,
    *,
    line_number: int,
    rollout_index: int,
    version_info_json: str | None,
) -> MutableMapping[str, Any]:
    record = metadata.record
    identity = metadata.identity

    error_value = record.reason if record.status == "failed" else None

    row.update(
        {
            "env_id": identity.output_env_id,
            "manifest_env_id": identity.manifest_env_id,
            "base_env_id": identity.base_env_id,
            "job_run_id": record.manifest.job_run_id,
            "run_id": record.job_id,
            "model_id": identity.model_id,
            "version_info": version_info_json,
            "status": record.status,
            "error": error_value,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
        }
    )
    if "rollout_index" not in row:
        row["rollout_index"] = rollout_index
    return row


def _normalize_token_usage(row: MutableMapping[str, Any]) -> None:
    """Flatten token_usage dict into explicit columns and drop the original field."""
    if "token_usage" not in row:
        return
    usage = row.pop("token_usage", None)

    def _coerce_number(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except Exception:
            return None

    def _extract_nested(role: str, key: str) -> float | None:
        if not isinstance(usage, Mapping):
            return None
        block = usage.get(role)
        if not isinstance(block, Mapping):
            return None
        return _coerce_number(block.get(key))

    for role in ("judge", "model"):
        row[f"{role}_cost"] = None
        row[f"{role}_token_completion"] = None
        row[f"{role}_token_prompt"] = None
        row[f"{role}_token_total"] = None

    if not isinstance(usage, Mapping):
        return

    # verifiers 0.1.10+ shape:
    # token_usage = {"input_tokens": ..., "output_tokens": ...}
    if "input_tokens" in usage or "output_tokens" in usage:
        prompt_tokens = _coerce_number(usage.get("input_tokens"))
        completion_tokens = _coerce_number(usage.get("output_tokens"))
        row["model_token_prompt"] = prompt_tokens
        row["model_token_completion"] = completion_tokens
        if prompt_tokens is not None or completion_tokens is not None:
            row["model_token_total"] = float((prompt_tokens or 0.0) + (completion_tokens or 0.0))
        return

    for role in ("judge", "model"):
        row[f"{role}_cost"] = _extract_nested(role, "cost")
        row[f"{role}_token_completion"] = _extract_nested(role, "completion")
        row[f"{role}_token_prompt"] = _extract_nested(role, "prompt")
        row[f"{role}_token_total"] = _extract_nested(role, "total")


def _encode_metadata_json_column(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    try:
        return json.dumps(value, sort_keys=True)
    except (TypeError, ValueError):
        return None


__all__ = ["load_rows"]
