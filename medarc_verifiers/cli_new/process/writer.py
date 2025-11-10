"""Parquet writer utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from medarc_verifiers.cli_new.process.aggregate import AggregatedEnvRows

logger = logging.getLogger(__name__)

EXPORTER_METADATA_KEY = b"medarc_exporter"
DEFAULT_SCHEMA_VERSION = 1


@dataclass(slots=True)
class WriterConfig:
    """Settings controlling parquet output behavior."""

    output_dir: Path
    exporter_version: str
    processed_at: str
    processed_with_args: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = DEFAULT_SCHEMA_VERSION
    dry_run: bool = False
    overwrite: bool = False


@dataclass(slots=True)
class EnvWriteSummary:
    """Summary of a single environment write."""

    env_id: str
    base_env_id: str
    output_path: Path
    row_count: int
    job_run_ids: tuple[str, ...]
    exporter_metadata: Mapping[str, Any]
    dry_run: bool


def write_env_groups(
    groups: Sequence[AggregatedEnvRows],
    config: WriterConfig,
) -> list[EnvWriteSummary]:
    """Write each aggregated environment to `<env_id>.parquet`."""
    output_dir = config.output_dir
    summaries: list[EnvWriteSummary] = []
    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for group in groups:
        summary = _write_group(group, config)
        summaries.append(summary)

    if config.dry_run or not summaries:
        return summaries

    _write_env_index(output_dir, summaries, config)
    return summaries


def _write_group(group: AggregatedEnvRows, config: WriterConfig) -> EnvWriteSummary:
    env_id = group.env_id or group.base_env_id
    base_env_id = group.base_env_id
    output_path = config.output_dir / f"{_slugify(env_id)}.parquet"
    exporter_metadata = {
        "exporter_version": config.exporter_version,
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "source_runs": list(group.job_run_ids),
        "processed_with_args": dict(config.processed_with_args),
        "env_id": env_id,
        "base_env_id": base_env_id,
    }
    row_count = len(group.rows)

    if config.dry_run:
        return EnvWriteSummary(
            env_id=env_id,
            base_env_id=base_env_id,
            output_path=output_path,
            row_count=row_count,
            job_run_ids=group.job_run_ids,
            exporter_metadata=exporter_metadata,
            dry_run=True,
        )

    if output_path.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"Output file {output_path} already exists. Use --overwrite to replace."
            )
        output_path.unlink()

    table = _build_arrow_table(group)
    metadata = table.schema.metadata or {}
    metadata = {**metadata, EXPORTER_METADATA_KEY: json.dumps(exporter_metadata, sort_keys=True).encode("utf-8")}
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, output_path)

    return EnvWriteSummary(
        env_id=env_id,
        base_env_id=base_env_id,
        output_path=output_path,
        row_count=row_count,
        job_run_ids=group.job_run_ids,
        exporter_metadata=exporter_metadata,
        dry_run=False,
    )


def _build_arrow_table(group: AggregatedEnvRows) -> pa.Table:
    if not group.rows:
        logger.debug("Group %s has no rows; writing empty table.", group.base_env_id)
        columns = list(group.column_names) if group.column_names else []
        arrays = [pa.array([], type=pa.null()) for _ in columns]
        return pa.Table.from_arrays(arrays, names=columns)

    df = pl.DataFrame(group.rows)
    for column in group.column_names:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(column))
    ordered_columns = list(group.column_names) if group.column_names else df.columns
    df = df.select(ordered_columns)
    return df.to_arrow()


def _write_env_index(
    output_dir: Path,
    summaries: Sequence[EnvWriteSummary],
    config: WriterConfig,
) -> None:
    payload = {
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "exporter_version": config.exporter_version,
        "environments": [
            {
                "env_id": summary.env_id,
                "base_env_id": summary.base_env_id,
                "path": summary.output_path.as_posix(),
                "row_count": summary.row_count,
                "job_run_ids": list(summary.job_run_ids),
                "exporter_metadata": summary.exporter_metadata,
            }
            for summary in summaries
        ],
    }
    index_path = output_dir / "env_index.json"
    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value.strip())
    return slug or "env"


__all__ = ["EnvWriteSummary", "WriterConfig", "write_env_groups"]
