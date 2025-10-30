"""Parquet dataset writing helpers for MedARC exports."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datasets import Dataset

from medarc_verifiers.export.parquet.aggregation import AggregatedEnvRows

logger = logging.getLogger(__name__)

PARTITION_FILENAME_SEPARATOR = "__"
DEFAULT_DATASET_FILENAME = "data.parquet"


@dataclass(frozen=True)
class WriteSummary:
    """Summary of a dataset write action."""

    env_id: str
    row_count: int
    dataset_paths: tuple[str, ...]
    partition_columns: tuple[str, ...]
    partitions: tuple[dict[str, Any], ...]


def write_parquet_datasets(
    groups: Sequence[AggregatedEnvRows],
    output_dir: Path | str,
    *,
    dry_run: bool,
    overwrite: bool,
) -> tuple[list[WriteSummary], list[str]]:
    """Write aggregated environment rows to Parquet datasets."""
    target_dir = Path(output_dir)
    summaries: list[WriteSummary] = []
    errors: list[str] = []

    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        try:
            summary = _write_env_group(
                group,
                target_dir,
                dry_run=dry_run,
                overwrite=overwrite,
            )
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            message = f"{group.env_id}: {exc}"
            logger.warning("Failed to write dataset for %s: %s", group.env_id, exc)
            errors.append(message)

    if dry_run or not summaries:
        return summaries, errors

    env_index = {
        "environments": [
            {
                "env_id": summary.env_id,
                "row_count": summary.row_count,
                "dataset_paths": list(summary.dataset_paths),
                "partition_columns": list(summary.partition_columns),
                "partitions": list(summary.partitions),
            }
            for summary in summaries
        ],
    }
    index_path = target_dir / "env_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(env_index, handle, indent=2, sort_keys=True)

    return summaries, errors


def _write_env_group(
    group: AggregatedEnvRows,
    base_dir: Path,
    *,
    dry_run: bool,
    overwrite: bool,
) -> WriteSummary:
    env_slug = _slugify(group.env_id)
    env_dir = base_dir / env_slug
    dataset_paths: list[str] = []
    partitions_meta: list[dict[str, Any]] = []

    if dry_run:
        planned_paths = _planned_paths(group, env_dir)
        dataset_paths.extend(str(path) for path in planned_paths)
        partitions_meta = _planned_partitions_payload(group, env_dir)
        return WriteSummary(
            env_id=group.env_id,
            row_count=len(group.rows),
            dataset_paths=tuple(dataset_paths),
            partition_columns=group.partition_columns,
            partitions=tuple(partitions_meta),
        )

    if env_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory {env_dir} already exists. Use --overwrite to replace.")
        shutil.rmtree(env_dir)
    env_dir.mkdir(parents=True, exist_ok=True)

    if group.partition_columns and group.partitions:
        for key, rows in group.partitions.items():
            file_path = env_dir / _partition_filename(group.partition_columns, key)
            _write_rows_to_parquet(rows, file_path)
            dataset_paths.append(str(file_path))
            partitions_meta.append(
                {
                    "partition_values": dict(zip(group.partition_columns, key, strict=False)),
                    "path": str(file_path),
                    "row_count": len(rows),
                }
            )
    else:
        file_path = env_dir / DEFAULT_DATASET_FILENAME
        rows = group.rows if group.rows else []
        _write_rows_to_parquet(rows, file_path)
        dataset_paths.append(str(file_path))

    return WriteSummary(
        env_id=group.env_id,
        row_count=len(group.rows),
        dataset_paths=tuple(dataset_paths),
        partition_columns=group.partition_columns,
        partitions=tuple(partitions_meta),
    )


def _write_rows_to_parquet(rows: Iterable[Mapping[str, Any]], file_path: Path) -> None:
    sanitized_rows = [_sanitize_row(row) for row in rows]
    dataset = Dataset.from_list(sanitized_rows)
    dataset.to_parquet(str(file_path))


def _planned_paths(group: AggregatedEnvRows, env_dir: Path) -> list[Path]:
    if group.partition_columns and group.partitions:
        return [env_dir / _partition_filename(group.partition_columns, key) for key in group.partitions]
    return [env_dir / DEFAULT_DATASET_FILENAME]


def _planned_partitions_payload(group: AggregatedEnvRows, env_dir: Path) -> list[dict[str, Any]]:
    if not (group.partition_columns and group.partitions):
        return []
    payload: list[dict[str, Any]] = []
    for key, rows in group.partitions.items():
        file_path = env_dir / _partition_filename(group.partition_columns, key)
        payload.append(
            {
                "partition_values": dict(zip(group.partition_columns, key, strict=False)),
                "path": str(file_path),
                "row_count": len(rows),
            }
        )
    return payload


def _partition_filename(columns: Sequence[str], key: Sequence[Any]) -> str:
    parts = []
    for column, value in zip(columns, key, strict=False):
        if value is None:
            sanitized_value = "null"
        else:
            sanitized_value = _slugify(str(value))
        parts.append(f"{column}-{sanitized_value}")
    if not parts:
        return DEFAULT_DATASET_FILENAME
    return f"{PARTITION_FILENAME_SEPARATOR.join(parts)}.parquet"


_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value.strip())
    return slug or "env"


def _sanitize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in row.items():
        sanitized[key] = _sanitize_value(value)
    return sanitized


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized_mapping = {str(k): _sanitize_value(v) for k, v in value.items()}
        if not sanitized_mapping:
            return None
        return sanitized_mapping
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


__all__ = ["WriteSummary", "write_parquet_datasets"]
