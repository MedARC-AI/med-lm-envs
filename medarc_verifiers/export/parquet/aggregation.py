"""Aggregation helpers for grouping enriched rows by environment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


@dataclass
class AggregatedEnvRows:
    """Container for rows grouped by environment."""

    env_id: str
    rows: list[Mapping[str, Any]]
    column_names: tuple[str, ...]
    partition_columns: tuple[str, ...] = field(default_factory=tuple)
    partitions: Dict[tuple[Any, ...], list[Mapping[str, Any]]] = field(default_factory=dict)


def aggregate_rows_by_env(
    rows: Iterable[Mapping[str, Any]],
    *,
    partition_by: Sequence[str] | None = None,
) -> list[AggregatedEnvRows]:
    """Group enriched rows by env_id, unioning column names and optional partitions."""
    partition_columns = tuple(partition_by or ())
    env_groups: Dict[str, AggregatedEnvRows] = {}

    for row in rows:
        env_id = str(row.get("env_id") or "")
        if not env_id:
            logger.debug("Encountered row without env_id; skipping.")
            continue

        group = env_groups.get(env_id)
        if group is None:
            group = AggregatedEnvRows(
                env_id=env_id,
                rows=[],
                column_names=(),
                partition_columns=partition_columns,
                partitions={},
            )
            env_groups[env_id] = group

        group.rows.append(row)
        # Update column names
        group.column_names = tuple(sorted(set(group.column_names) | set(row.keys())))

        if partition_columns:
            key = tuple(row.get(column) for column in partition_columns)
            group.partitions.setdefault(key, []).append(row)

    return [env_groups[env_id] for env_id in sorted(env_groups)]


__all__ = ["AggregatedEnvRows", "aggregate_rows_by_env"]
