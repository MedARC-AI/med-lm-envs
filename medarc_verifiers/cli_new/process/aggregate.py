"""Aggregation helpers for exporter process pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import re

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
    rollout_re = re.compile(r"^(.*?)-rollout\d+$")
    for key in sorted(groups):
        group = groups[key]
        # Preserve rollout_index as assigned during row loading; aggregation just passes rows through.
        normalized_rows: list[Mapping[str, Any]] = list(group["rows"])  # shallow copy
        # Canonicalize env_id by stripping trailing rollout suffix if present (keep other variant parts like task)
        candidate_env_id = group["env_id"] or group["base_env_id"] or key
        m = rollout_re.match(candidate_env_id)
        out_env_id = m.group(1) if m else candidate_env_id
        aggregated.append(
            AggregatedEnvRows(
                env_id=out_env_id,
                base_env_id=group["base_env_id"] or key,
                rows=normalized_rows,
                column_names=tuple(sorted(group["column_names"])),
                job_run_ids=tuple(sorted(group["job_run_ids"])),
            )
        )
    return aggregated


__all__ = ["AggregatedEnvRows", "aggregate_rows_by_env"]
