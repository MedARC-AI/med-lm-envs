"""Parquet writer utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from medarc_verifiers.cli_new.process.aggregate import AggregatedEnvRows

logger = logging.getLogger(__name__)

EXPORTER_METADATA_KEY = b"medarc_exporter"
DEFAULT_SCHEMA_VERSION = 1
ALLOWED_COLUMNS: tuple[str, ...] = (
    "env_id",
    "error",
    "example_id",
    "answer",
    "generation_ms",
    "job_run_id",
    "judge_cost",
    "judge_token_completion",
    "judge_token_prompt",
    "judge_token_total",
    "model_cost",
    "model_id",
    "model_token_completion",
    "model_token_prompt",
    "model_token_total",
    "reward",
    "rollout_index",
    "run_id",
    "scoring_ms",
    "status",
    "task",
    "total_ms",
)


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
    append: bool = True


@dataclass(slots=True)
class EnvWriteSummary:
    """Summary of a single environment write."""

    env_id: str
    base_env_id: str
    model_id: str
    output_path: Path
    row_count: int
    job_run_ids: tuple[str, ...]
    exporter_metadata: Mapping[str, Any]
    dry_run: bool


def write_env_groups(
    groups: Sequence[AggregatedEnvRows] | Iterable[AggregatedEnvRows],
    config: WriterConfig,
    *,
    write_index: bool = True,
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

    if write_index:
        _write_env_index(output_dir, summaries, config)
        write_hf_dataset_config(summaries, config)
    return summaries


def write_env_index(
    summaries: Sequence[EnvWriteSummary],
    config: WriterConfig,
) -> None:
    """Write env_index.json from collected summaries."""
    if config.dry_run or not summaries:
        return
    _write_env_index(config.output_dir, summaries, config)


def write_hf_dataset_config(
    summaries: Sequence[EnvWriteSummary],
    config: WriterConfig,
) -> None:
    """Emit Hugging Face datasets metadata (dataset_infos.json) for the Parquet files."""
    if config.dry_run or not summaries:
        return

    data_files: dict[str, list[str]] = {"train": []}
    split_name_map: dict[str, str] = {}
    env_row_counts: dict[str, int] = {}
    for summary in summaries:
        rel_path = summary.output_path.relative_to(config.output_dir).as_posix()
        data_files["train"].append(rel_path)
        env_row_counts[summary.base_env_id] = env_row_counts.get(summary.base_env_id, 0) + int(summary.row_count)
        split_name = _sanitize_split_name(summary.base_env_id)
        split_name_map[split_name] = summary.base_env_id
        data_files.setdefault(split_name, []).append(rel_path)

    # Build minimal split info
    splits: dict[str, dict[str, Any]] = {
        "train": {
            "name": "train",
            "num_bytes": None,
            "num_examples": sum(int(s.row_count) for s in summaries),
            "dataset_name": "default",
        }
    }
    for env_id, count in sorted(env_row_counts.items()):
        splits[env_id] = {
            "name": env_id,
            "num_bytes": None,
            "num_examples": count,
            "dataset_name": "default",
        }

    dataset_info = {
        "builder_name": "parquet",
        "config_name": "default",
        "config_description": "MedARC processed outputs grouped by model and environment.",
        "dataset_size": None,
        "download_checksums": None,
        "download_size": None,
        "features": None,
        "homepage": None,
        "license": None,
        "splits": splits,
        "data_files": data_files,
        "version": "0.0.0",
        "extras": {
            "processed_at": config.processed_at,
            "processed_with_args": dict(config.processed_with_args),
            "env_id_map": split_name_map,
        },
    }

    payload = {"default": dataset_info}

    config_path = config.output_dir / "dataset_infos.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

def _write_group(group: AggregatedEnvRows, config: WriterConfig) -> EnvWriteSummary:
    env_id = group.env_id or group.base_env_id
    base_env_id = group.base_env_id
    model_id = group.model_id or "unknown"
    model_dir = config.output_dir / _slugify(model_id)
    if not config.dry_run:
        model_dir.mkdir(parents=True, exist_ok=True)
    output_path = model_dir / f"{_slugify(env_id)}.parquet"
    exporter_metadata = {
        "exporter_version": config.exporter_version,
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "source_runs": list(group.job_run_ids),
        "processed_with_args": dict(config.processed_with_args),
        "env_id": env_id,
        "base_env_id": base_env_id,
        "model_id": model_id,
        "append": False,
    }
    row_count = len(group.rows)

    if config.dry_run:
        return EnvWriteSummary(
            env_id=env_id,
            base_env_id=base_env_id,
            model_id=model_id,
            output_path=output_path,
            row_count=row_count,
            job_run_ids=group.job_run_ids,
            exporter_metadata=exporter_metadata,
            dry_run=True,
        )

    if output_path.exists() and not config.overwrite:
        if not config.append:
            raise FileExistsError(
                f"Output file {output_path} exists and append is disabled. Use --overwrite to replace or enable append to merge."
            )
        # Append-merge semantics: read existing file, drop duplicate job_run_ids,
        # union schemas, concatenate, and rewrite file.
        try:
            existing_table = pq.read_table(output_path)
        except Exception as exc:  # noqa: BLE001
            raise FileExistsError(
                f"Failed to read existing output file {output_path}: {exc}. To force regeneration, use --overwrite."
            ) from exc

        # Convert both existing and new data to Polars and normalize schema to a fixed column set
        existing_df = _normalize_columns(pl.from_arrow(existing_table))
        new_df = _normalize_columns(pl.from_arrow(_build_arrow_table(group)))

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
                    "model_id": model_id,
                }

            return EnvWriteSummary(
                env_id=env_id,
                base_env_id=base_env_id,
                model_id=model_id,
                output_path=output_path,
                row_count=existing_df.height,
                job_run_ids=tuple(existing_export_meta.get("source_runs", [])),
                exporter_metadata=existing_export_meta,
                dry_run=False,
            )

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
            "model_id": model_id,
            "append": True,
        }
        meta_raw = {**meta_raw, EXPORTER_METADATA_KEY: json.dumps(merged_exporter_meta, sort_keys=True).encode("utf-8")}
        combined_table = combined_table.replace_schema_metadata(meta_raw)
        pq.write_table(combined_table, output_path)

        return EnvWriteSummary(
            env_id=env_id,
            base_env_id=base_env_id,
            model_id=model_id,
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
        model_id=model_id,
        output_path=output_path,
        row_count=row_count,
        job_run_ids=group.job_run_ids,
        exporter_metadata=exporter_metadata,
        dry_run=False,
    )


def _build_arrow_table(group: AggregatedEnvRows) -> pa.Table:
    if not group.rows:
        logger.debug("Group %s has no rows; writing empty table.", group.base_env_id)
        arrays = [pa.array([], type=pa.null()) for _ in ALLOWED_COLUMNS]
        return pa.Table.from_arrays(arrays, names=list(ALLOWED_COLUMNS))

    # Use full-length schema inference so late non-null values don't clash with
    # early all-null samples (default inference length is limited).
    df = pl.DataFrame(group.rows, infer_schema_length=None)
    df = _normalize_columns(df)
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

    # Flattened list of environments (now per model) for backward compatibility.
    existing_envs: dict[tuple[str, str], dict[str, Any]] = {}
    for item in existing.get("environments", []) or []:
        env_id = str(item.get("env_id") or item.get("base_env_id") or "")
        model_id = str(item.get("model_id") or "")
        if env_id and model_id:
            existing_envs[(env_id, model_id)] = dict(item)

    env_entries: dict[tuple[str, str], dict[str, Any]] = dict(existing_envs)
    for summary in summaries:
        key = (summary.env_id, summary.model_id)
        env_entries[key] = {
            "env_id": summary.env_id,
            "base_env_id": summary.base_env_id,
            "model_id": summary.model_id,
            "path": summary.output_path.as_posix(),
            "row_count": summary.row_count,
            "job_run_ids": list(summary.job_run_ids),
            "exporter_metadata": summary.exporter_metadata,
        }
    env_entries_list = [env_entries[k] for k in sorted(env_entries.keys())]

    # Grouped view by environment to ease reconstruction for winrates / HF metadata
    env_groups: dict[str, dict[str, Any]] = {}
    for entry in env_entries_list:
        env_id = entry["env_id"]
        group = env_groups.setdefault(
            env_id,
            {
                "env_id": env_id,
                "base_env_id": entry["base_env_id"],
                "paths": [],
                "row_count": 0,
            },
        )
        group["paths"].append(
            {
                "model_id": entry["model_id"],
                "path": entry["path"],
                "row_count": entry["row_count"],
                "job_run_ids": entry["job_run_ids"],
            }
        )
        try:
            group["row_count"] += int(entry["row_count"])
        except Exception:
            pass

    # Group by model for convenience (e.g., browsing output folders)
    model_groups: dict[str, dict[str, Any]] = {}
    for entry in env_entries_list:
        model_id = entry["model_id"]
        group = model_groups.setdefault(
            model_id,
            {
                "model_id": model_id,
                "path": _slugify(model_id),
                "environments": [],
            },
        )
        group["environments"].append(
            {
                "env_id": entry["env_id"],
                "base_env_id": entry["base_env_id"],
                "path": entry["path"],
                "row_count": entry["row_count"],
                "job_run_ids": entry["job_run_ids"],
            }
        )

    payload = {
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "exporter_version": config.exporter_version,
        "environments": env_entries_list,
        "env_groups": [env_groups[k] for k in sorted(env_groups.keys())],
        "models": [model_groups[k] for k in sorted(model_groups.keys())],
    }

    with index_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


_SLUG_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def _slugify(value: str) -> str:
    slug = _SLUG_PATTERN.sub("_", value.strip())
    return slug or "env"


def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict output schema to a fixed set of columns for cross-env compatibility."""
    out = df.clone()
    for col in ALLOWED_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).alias(col))
    return out.select(list(ALLOWED_COLUMNS))


__all__ = ["EnvWriteSummary", "WriterConfig", "write_env_groups", "write_env_index"]


def _sanitize_split_name(name: str) -> str:
    """Datasets split names must match ^\\w+(\\.\\w+)*$; replace disallowed chars."""
    sanitized = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            sanitized.append(ch)
        elif ch == ".":
            sanitized.append(".")
        else:
            sanitized.append("_")
    out = "".join(sanitized).strip("_")
    return out or "env"
