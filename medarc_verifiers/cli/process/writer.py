"""Parquet writer utilities for exporter process pipeline."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from medarc_verifiers.cli.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli.utils.json_io import write_json
from medarc_verifiers.cli.utils.shared import slugify_filename_component

logger = logging.getLogger(__name__)

EXPORTER_METADATA_KEY = b"medarc_exporter"
DEFAULT_SCHEMA_VERSION = 1
ALLOWED_COLUMNS: tuple[str, ...] = (
    "env_id",
    "error",
    "example_id",
    "answer",
    "extras",
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

EXPECTED_POLARS_DTYPES: dict[str, pl.DataType] = {
    "env_id": pl.String,
    "error": pl.String,
    "example_id": pl.Int64,
    "answer": pl.String,
    "extras": pl.String,
    "generation_ms": pl.Float64,
    "job_run_id": pl.String,
    "judge_cost": pl.Float64,
    "judge_token_completion": pl.Float64,
    "judge_token_prompt": pl.Float64,
    "judge_token_total": pl.Float64,
    "model_cost": pl.Float64,
    "model_id": pl.String,
    "model_token_completion": pl.Float64,
    "model_token_prompt": pl.Float64,
    "model_token_total": pl.Float64,
    "reward": pl.Float64,
    "rollout_index": pl.Int64,
    "run_id": pl.String,
    "scoring_ms": pl.Float64,
    "status": pl.String,
    "task": pl.String,
    "total_ms": pl.Float64,
}

EXPECTED_ARROW_SCHEMA = pa.schema(
    [
        pa.field("env_id", pa.large_string()),
        pa.field("error", pa.large_string()),
        pa.field("example_id", pa.int64()),
        pa.field("answer", pa.large_string()),
        pa.field("extras", pa.large_string()),
        pa.field("generation_ms", pa.float64()),
        pa.field("job_run_id", pa.large_string()),
        pa.field("judge_cost", pa.float64()),
        pa.field("judge_token_completion", pa.float64()),
        pa.field("judge_token_prompt", pa.float64()),
        pa.field("judge_token_total", pa.float64()),
        pa.field("model_cost", pa.float64()),
        pa.field("model_id", pa.large_string()),
        pa.field("model_token_completion", pa.float64()),
        pa.field("model_token_prompt", pa.float64()),
        pa.field("model_token_total", pa.float64()),
        pa.field("reward", pa.float64()),
        pa.field("rollout_index", pa.int64()),
        pa.field("run_id", pa.large_string()),
        pa.field("scoring_ms", pa.float64()),
        pa.field("status", pa.large_string()),
        pa.field("task", pa.large_string()),
        pa.field("total_ms", pa.float64()),
    ]
)


def _validate_allowed_columns() -> None:
    allowed = set(ALLOWED_COLUMNS)
    expected = set(EXPECTED_POLARS_DTYPES)
    if allowed != expected:
        msg = (
            "ALLOWED_COLUMNS and EXPECTED_POLARS_DTYPES keys are out of sync: "
            f"allowed_only={sorted(allowed - expected)}, "
            f"expected_only={sorted(expected - allowed)}"
        )
        raise ValueError(msg)

    schema_fields = {field.name for field in EXPECTED_ARROW_SCHEMA}
    if allowed != schema_fields:
        msg = (
            "ALLOWED_COLUMNS and EXPECTED_ARROW_SCHEMA fields are out of sync: "
            f"allowed_only={sorted(allowed - schema_fields)}, "
            f"schema_only={sorted(schema_fields - allowed)}"
        )
        raise ValueError(msg)


_validate_allowed_columns()


@dataclass(slots=True)
class WriterConfig:
    """Settings controlling parquet output behavior."""

    output_dir: Path
    processed_at: str
    processed_with_args: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = DEFAULT_SCHEMA_VERSION
    dry_run: bool = False


@dataclass(slots=True)
class EnvWriteSummary:
    """Summary of a single environment write."""

    env_id: str
    base_env_id: str
    model_id: str
    output_path: Path
    row_count: int
    job_run_ids: tuple[str, ...]
    job_run_ids_added: tuple[str, ...]
    job_run_ids_replaced: tuple[str, ...]
    job_run_ids_unchanged: tuple[str, ...]
    exporter_metadata: Mapping[str, Any]
    dry_run: bool
    action: str
    changed: bool


def write_env_groups(
    groups: Sequence[AggregatedEnvRows] | Iterable[AggregatedEnvRows],
    config: WriterConfig,
    *,
    write_index: bool = True,
    run_metadata: Mapping[str, Mapping[str, Any]] | None = None,
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
        write_hf_dataset_config(summaries, config)
        _write_env_index(output_dir, summaries, config, run_metadata=run_metadata)
    return summaries


def write_env_index(
    summaries: Sequence[EnvWriteSummary],
    config: WriterConfig,
    *,
    run_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    """Write env_index.json from collected summaries."""
    if config.dry_run or not summaries:
        return False
    return _write_env_index(config.output_dir, summaries, config, run_metadata=run_metadata)


def write_hf_dataset_config(
    summaries: Sequence[EnvWriteSummary],
    config: WriterConfig,
) -> bool:
    """Emit Hugging Face datasets metadata (dataset_infos.json) for the Parquet files."""
    if config.dry_run or not summaries:
        return False

    parquet_paths = _collect_parquet_paths(config.output_dir)
    data_files: dict[str, list[str]] = {"train": parquet_paths}

    # Build minimal split info
    splits: dict[str, dict[str, Any]] = {
        "train": {
            "name": "train",
            "num_bytes": None,
            "num_examples": None,
            "dataset_name": "default",
        }
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
        },
    }

    payload = {"default": dataset_info}

    config_path = config.output_dir / "dataset_infos.json"
    return write_json(config_path, payload)


def _write_group(group: AggregatedEnvRows, config: WriterConfig) -> EnvWriteSummary:
    env_id = group.env_id or group.base_env_id
    if not env_id:
        raise ValueError("env_id is required for parquet output.")
    model_id = group.model_id
    if not model_id:
        raise ValueError("model_id is required for parquet output.")
    output_path = build_output_path(config.output_dir, model_id=model_id, env_id=env_id)
    if not config.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    exporter_metadata = {
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "source_runs": list(group.job_run_ids),
        "processed_with_args": dict(config.processed_with_args),
        "env_id": env_id,
        "model_id": model_id,
    }
    row_count = len(group.rows)
    job_run_ids_set = set(group.job_run_ids)

    if config.dry_run:
        action = "updated" if file_exists else "created"
        return EnvWriteSummary(
            env_id=env_id,
            base_env_id=group.base_env_id,
            model_id=model_id,
            output_path=output_path,
            row_count=row_count,
            job_run_ids=group.job_run_ids,
            job_run_ids_added=tuple(sorted(job_run_ids_set)),
            job_run_ids_replaced=(),
            job_run_ids_unchanged=(),
            exporter_metadata=exporter_metadata,
            dry_run=True,
            action=action,
            changed=row_count > 0,
        )

    table = _build_arrow_table(group)
    metadata = table.schema.metadata or {}
    metadata = {**metadata, EXPORTER_METADATA_KEY: json.dumps(exporter_metadata, sort_keys=True).encode("utf-8")}
    table = table.replace_schema_metadata(metadata)
    _write_parquet_atomic(table, output_path)

    return EnvWriteSummary(
        env_id=env_id,
        base_env_id=group.base_env_id,
        model_id=model_id,
        output_path=output_path,
        row_count=row_count,
        job_run_ids=group.job_run_ids,
        job_run_ids_added=tuple(sorted(job_run_ids_set)),
        job_run_ids_replaced=(),
        job_run_ids_unchanged=(),
        exporter_metadata=exporter_metadata,
        dry_run=False,
        action="updated" if file_exists else "created",
        changed=True,
    )


def _build_arrow_table(group: AggregatedEnvRows) -> pa.Table:
    if not group.rows:
        logger.debug("Group %s has no rows; writing empty table.", group.base_env_id)
        arrays = [pa.array([], type=EXPECTED_ARROW_SCHEMA.field(col).type) for col in ALLOWED_COLUMNS]
        return pa.Table.from_arrays(arrays, schema=EXPECTED_ARROW_SCHEMA)

    # Use full-length schema inference so late non-null values don't clash with
    # early all-null samples (default inference length is limited).
    df = pl.DataFrame(group.rows, infer_schema_length=None)
    df = _normalize_columns(df)
    table = df.to_arrow()
    # Ensure shard schemas are consistent across environments even when a column
    # is entirely null within a single shard (Arrow would otherwise use `null`).
    return table.cast(EXPECTED_ARROW_SCHEMA)


def _write_env_index(
    output_dir: Path,
    summaries: Sequence[EnvWriteSummary],
    config: WriterConfig,
    *,
    run_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> bool:
    # Load existing index if present to preserve other environments
    index_path = output_dir / "env_index.json"
    existing: dict[str, Any] = {}
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle) or {}
        except Exception:  # pragma: no cover - tolerate bad index
            existing = {}

    existing_files: dict[str, dict[str, Any]] = {}
    existing_runs: dict[str, dict[str, Any]] = {}
    if existing.get("version") == 2:
        raw_files = existing.get("files") or {}
        if isinstance(raw_files, Mapping):
            existing_files = {
                _normalize_index_file_key(output_dir, str(path)): dict(entry)
                for path, entry in raw_files.items()
                if isinstance(entry, Mapping)
            }
        raw_runs = existing.get("runs") or {}
        if isinstance(raw_runs, Mapping):
            existing_runs = {str(k): dict(v) for k, v in raw_runs.items() if isinstance(v, Mapping)}
    else:
        existing_files = {}
        existing_runs = {}

    files: dict[str, dict[str, Any]] = dict(existing_files)
    runs: dict[str, dict[str, Any]] = {run_id: _filter_run_entry(payload) for run_id, payload in existing_runs.items()}
    run_metadata = run_metadata or {}
    for summary in summaries:
        path_str = _relative_output_path(output_dir, summary.output_path)
        timestamps: list[str] = []
        files[path_str] = {
            "env_id": summary.env_id,
            "model_id": summary.model_id,
            "row_count": summary.row_count,
        }
        for job_run_id in summary.job_run_ids:
            prior = runs.get(job_run_id, {})
            meta = run_metadata.get(job_run_id, {})
            created_at = meta.get("created_at") or prior.get("created_at")
            updated_at = meta.get("updated_at") or prior.get("updated_at") or created_at
            runs[job_run_id] = {
                "created_at": created_at,
                "updated_at": updated_at,
                "config_checksum": meta.get("config_checksum") or prior.get("config_checksum"),
            }
            if updated_at:
                timestamps.append(str(updated_at))
        files[path_str]["updated_at"] = _max_timestamp(timestamps)

    payload = {
        "version": 2,
        "processed_at": config.processed_at,
        "schema_version": config.schema_version,
        "processed_with_args": dict(config.processed_with_args),
        "runs": runs,
        "files": files,
    }

    return write_json(index_path, payload)


def _relative_output_path(output_dir: Path, path: Path) -> str:
    try:
        rel_path = path.resolve().relative_to(output_dir.resolve())
    except ValueError:
        rel_path = path
    return rel_path.as_posix()


def _normalize_index_file_key(output_dir: Path, raw_path: str) -> str:
    raw_path = raw_path.strip()
    if not raw_path:
        return raw_path
    path = Path(raw_path)
    if path.is_absolute():
        try:
            output_root = output_dir.resolve()
            path = path.resolve().relative_to(output_root)
        except Exception:
            return path.as_posix()
    return path.as_posix()


def _collect_parquet_paths(output_dir: Path) -> list[str]:
    paths = []
    for path in output_dir.rglob("*.parquet"):
        if not path.is_file():
            continue
        paths.append(_relative_output_path(output_dir, path))
    return sorted(paths)


def _max_timestamp(values: Iterable[str]) -> str | None:
    best_dt: datetime | None = None
    best_raw: str | None = None
    for raw in values:
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except Exception:
            continue
        if best_dt is None or dt > best_dt:
            best_dt = dt
            best_raw = str(raw)
    return best_raw


def _filter_run_entry(payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = ("created_at", "updated_at", "config_checksum")
    return {key: payload.get(key) for key in allowed if key in payload}


def _write_parquet_atomic(table: pa.Table, path: Path) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        pq.write_table(table, tmp_path, use_content_defined_chunking=True, write_page_index=True)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _normalize_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Restrict output schema to a fixed set of columns for cross-env compatibility."""
    out = df.clone()
    for col in ALLOWED_COLUMNS:
        if col not in out.columns:
            out = out.with_columns(pl.lit(None).alias(col))
    out = out.select(list(ALLOWED_COLUMNS))
    out = out.with_columns(
        [pl.col(col).cast(EXPECTED_POLARS_DTYPES[col], strict=False).alias(col) for col in ALLOWED_COLUMNS]
    )
    return out


def build_output_path(output_dir: Path, *, model_id: str, env_id: str) -> Path:
    """Return the canonical parquet output path for a (model_id, env_id) dataset."""
    if not model_id:
        raise ValueError("model_id is required for output path.")
    if not env_id:
        raise ValueError("env_id is required for output path.")
    model_dir = output_dir / slugify_filename_component(model_id)
    return model_dir / f"{slugify_filename_component(env_id)}.parquet"


__all__ = ["EnvWriteSummary", "WriterConfig", "build_output_path", "write_env_groups", "write_env_index"]
