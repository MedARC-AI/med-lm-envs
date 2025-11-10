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
    rows: list[Mapping[str, Any]]
    column_names: tuple[str, ...]
    job_run_ids: tuple[str, ...]


def aggregate_rows_by_env(rows: Iterable[Mapping[str, Any]]) -> list[AggregatedEnvRows]:
    """Group enriched rows by base_env_id, capturing unioned schemas."""
    groups: dict[str, dict[str, Any]] = {}

    for row in rows:
        base_env_id = str(row.get("base_env_id") or row.get("env_id") or "")
        env_id = str(row.get("env_id") or base_env_id)
        group_key = base_env_id or env_id
        if not group_key:
            logger.debug("Skipping row without env identifiers.")
            continue

        if group_key not in groups:
            groups[group_key] = {
                "env_id": env_id if env_id else base_env_id,
                "base_env_id": base_env_id,
                "rows": [],
                "column_names": set(),
                "job_run_ids": set(),
            }

        group = groups[group_key]
        if not group["env_id"] and env_id:
            group["env_id"] = env_id
        if not group["base_env_id"] and base_env_id:
            group["base_env_id"] = base_env_id
        group["rows"].append(row)
        group["column_names"].update(row.keys())
        job_run_id = row.get("job_run_id")
        if job_run_id:
            group["job_run_ids"].add(str(job_run_id))

    aggregated: list[AggregatedEnvRows] = []
    for key in sorted(groups):
        group = groups[key]
        # Normalize rollout_index across runs within the same base_env group so
        # that appended rollouts map to contiguous indices starting from 0.
        # We treat each distinct (job_run_id, original_rollout_index) as a
        # rollout cohort and remap in first-seen order.
        cohort_map: dict[tuple[str | None, int], int] = {}
        next_index = 0
        normalized_rows: list[Mapping[str, Any]] = []
        for row in group["rows"]:
            job_run_id = row.get("job_run_id")
            original = row.get("rollout_index")
            try:
                original_int = int(original) if original is not None else 0
            except Exception:  # noqa: BLE001
                original_int = 0
            cohort = (str(job_run_id) if job_run_id is not None else None, original_int)
            if cohort not in cohort_map:
                cohort_map[cohort] = next_index
                next_index += 1
            remapped = dict(row)
            remapped["rollout_index"] = cohort_map[cohort]
            normalized_rows.append(remapped)

        aggregated.append(
            AggregatedEnvRows(
                env_id=group["env_id"] or key,
                base_env_id=group["base_env_id"] or key,
                rows=normalized_rows,
                column_names=tuple(sorted(group["column_names"])),
                job_run_ids=tuple(sorted(group["job_run_ids"])),
            )
        )
    return aggregated


__all__ = ["AggregatedEnvRows", "aggregate_rows_by_env"]
