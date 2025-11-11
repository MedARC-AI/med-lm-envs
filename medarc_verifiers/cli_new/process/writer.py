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

    if output_path.exists() and not config.overwrite:
        # Append-merge semantics: read existing file, drop duplicate job_run_ids,
        # union schemas, concatenate, and rewrite file.
        try:
            existing_table = pq.read_table(output_path)
        except Exception as exc:  # noqa: BLE001
            raise FileExistsError(
                f"Failed to read existing output file {output_path}: {exc}. To force regeneration, use --overwrite."
            ) from exc

        # Convert both existing and new data to Polars for easy schema union and filtering
        existing_df = pl.from_arrow(existing_table)
        new_df = pl.from_arrow(_build_arrow_table(group))

        # Deduplicate by job_run_id when possible
        existing_job_ids: set[str] = set()
        if "job_run_id" in existing_df.columns:
            try:
                existing_job_ids = set(str(x) for x in existing_df["job_run_id"].drop_nulls().unique().to_list())
            except Exception:  # pragma: no cover - defensive
                existing_job_ids = set()

        if "job_run_id" in new_df.columns and existing_job_ids:
            new_df = new_df.filter(~pl.col("job_run_id").is_in(list(existing_job_ids)))

        # If nothing new to add, reuse existing file/stats and metadata
        if new_df.height == 0:
            # Attempt to recover existing exporter metadata from file
            existing_meta_raw = existing_table.schema.metadata or {}
            existing_export_meta: dict[str, Any] | None = None
            if EXPORTER_METADATA_KEY in existing_meta_raw:
                try:
                    existing_export_meta = json.loads(existing_meta_raw[EXPORTER_METADATA_KEY].decode("utf-8"))
                except Exception:  # pragma: no cover - malformed metadata
                    existing_export_meta = None
            # Fall back to computing job_run_ids from data if needed
            if existing_export_meta is None:
                combined_job_ids = sorted(
                    str(x)
                    for x in (
                        existing_df["job_run_id"].drop_nulls().unique().to_list()
                        if "job_run_id" in existing_df.columns
                        else []
                    )
                )
                existing_export_meta = {
                    "exporter_version": config.exporter_version,
                    "processed_at": config.processed_at,
                    "schema_version": config.schema_version,
                    "source_runs": combined_job_ids,
                    "processed_with_args": dict(config.processed_with_args),
                    "env_id": env_id,
                    "base_env_id": base_env_id,
                }

            return EnvWriteSummary(
                env_id=env_id,
                base_env_id=base_env_id,
                output_path=output_path,
                row_count=existing_df.height,
                job_run_ids=tuple(existing_export_meta.get("source_runs", [])),
                exporter_metadata=existing_export_meta,
                dry_run=False,
            )

        # Union schemas: ensure both frames share the same columns
        union_cols = list(sorted(set(existing_df.columns) | set(new_df.columns)))
        for col in union_cols:
            if col not in existing_df.columns:
                existing_df = existing_df.with_columns(pl.lit(None).alias(col))
            if col not in new_df.columns:
                new_df = new_df.with_columns(pl.lit(None).alias(col))
        existing_df = existing_df.select(union_cols)
        new_df = new_df.select(union_cols)

        combined_df = pl.concat([existing_df, new_df], how="vertical_relaxed")

        # Compute combined job_run_ids from data (authoritative)
        if "job_run_id" in combined_df.columns:
            combined_job_ids = sorted(str(x) for x in combined_df["job_run_id"].drop_nulls().unique().to_list())
        else:
            combined_job_ids = sorted(set(existing_job_ids) | set(group.job_run_ids))

        # Build updated metadata and write back
        combined_table = combined_df.to_arrow()
        meta_raw = combined_table.schema.metadata or {}
        merged_exporter_meta = {
            "exporter_version": config.exporter_version,
            "processed_at": config.processed_at,
            "schema_version": config.schema_version,
            "source_runs": combined_job_ids,
            "processed_with_args": dict(config.processed_with_args),
            "env_id": env_id,
            "base_env_id": base_env_id,
        }
        meta_raw = {**meta_raw, EXPORTER_METADATA_KEY: json.dumps(merged_exporter_meta, sort_keys=True).encode("utf-8")}
        combined_table = combined_table.replace_schema_metadata(meta_raw)
        pq.write_table(combined_table, output_path)

        return EnvWriteSummary(
            env_id=env_id,
            base_env_id=base_env_id,
            output_path=output_path,
            row_count=combined_df.height,
            job_run_ids=tuple(combined_job_ids),
            exporter_metadata=merged_exporter_meta,
            dry_run=False,
        )

    # Fresh write (no existing file or explicit overwrite)
    if output_path.exists():
        # Explicit overwrite path: remove and write anew
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
    # Load existing index if present to preserve other environments
    index_path = output_dir / "env_index.json"
    existing: dict[str, Any] = {}
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle) or {}
        except Exception:  # pragma: no cover - tolerate bad index
            existing = {}

    env_map: dict[str, Any] = {}
    # Seed from existing
    for item in existing.get("environments", []) or []:
        env_id = str(item.get("env_id") or item.get("base_env_id") or "")
        if env_id:
            env_map[env_id] = item

    # Apply/replace with new summaries
    for summary in summaries:
        env_map[summary.env_id] = {
            "env_id": summary.env_id,
            "base_env_id": summary.base_env_id,
            "path": summary.output_path.as_posix(),
            "row_count": summary.row_count,
            "job_run_ids": list(summary.job_run_ids),
            "exporter_metadata": summary.exporter_metadata,
        }

    payload = {
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "exporter_version": config.exporter_version,
        "environments": [env_map[k] for k in sorted(env_map.keys())],
    }

    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value.strip())
    return slug or "env"


__all__ = ["EnvWriteSummary", "WriterConfig", "write_env_groups"]
