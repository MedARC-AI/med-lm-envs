"""Aggregation helpers for exporter process pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AggregatedEnvRows:
    """Container describing all rows for a single environment."""

    env_id: str
    base_env_id: str
    model_id: str | None
    rows: list[Mapping[str, Any]]
    column_names: tuple[str, ...]
    job_run_ids: tuple[str, ...]


def aggregate_rows_by_env(
    rows: Iterable[Mapping[str, Any]],
) -> list[AggregatedEnvRows]:
    """Group enriched rows by (model_id, base_env_id), capturing unioned schemas."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}

    for row in rows:
        base_env_id = str(row.get("base_env_id") or row.get("env_id") or "")
        env_id = str(row.get("env_id") or base_env_id)
        model_id = str(row.get("model_id") or "unknown")
        group_key = (model_id, base_env_id or env_id)
        if not group_key[1]:  # no env identifier
            logger.debug("Skipping row without env identifiers.")
            continue

        if group_key not in groups:
            groups[group_key] = {
                "env_id": env_id if env_id else base_env_id,
                "base_env_id": base_env_id,
                "model_id": model_id,
                "rows": [],
                "column_names": set(),
                "job_run_ids": set(),
            }

        group = groups[group_key]
        if not group["env_id"] and env_id:
            group["env_id"] = env_id
        if not group["base_env_id"] and base_env_id:
            group["base_env_id"] = base_env_id
        if not group["model_id"] and model_id:
            group["model_id"] = model_id
        group["rows"].append(row)
        group["column_names"].update(row.keys())
        job_run_id = row.get("job_run_id")
        if job_run_id:
            group["job_run_ids"].add(str(job_run_id))

    aggregated: list[AggregatedEnvRows] = []
    for key in sorted(groups):
        group = groups[key]
        # Preserve rollout_index as assigned during row loading; aggregation just passes rows through.
        normalized_rows: list[Mapping[str, Any]] = list(group["rows"])  # shallow copy
        _normalize_rollout_indices(normalized_rows)
        candidate_env_id = group["env_id"] or group["base_env_id"] or ""
        aggregated.append(
            AggregatedEnvRows(
                env_id=candidate_env_id,
                base_env_id=group["base_env_id"] or key[1],
                model_id=group["model_id"],
                rows=normalized_rows,
                column_names=tuple(sorted(group["column_names"])),
                job_run_ids=tuple(sorted(group["job_run_ids"])),
            )
        )
    return aggregated


def _normalize_rollout_indices(rows: list[Mapping[str, Any]]) -> None:
    values: list[int] = []
    for row in rows:
        value = row.get("rollout_index")
        if value is None:
            continue
        try:
            values.append(int(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return
    mapping = {val: idx for idx, val in enumerate(sorted(set(values)))}
    for row in rows:
        value = row.get("rollout_index")
        if value is None:
            continue
        try:
            normalized = mapping[int(value)]
        except (TypeError, ValueError, KeyError):
            continue
        try:
            row["rollout_index"] = normalized
        except TypeError:
            # Ignore non-mutable mappings.
            continue


__all__ = ["AggregatedEnvRows", "aggregate_rows_by_env"]
