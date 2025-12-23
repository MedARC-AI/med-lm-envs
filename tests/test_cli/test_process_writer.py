from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq
import pytest

from medarc_verifiers.cli.process import writer
from medarc_verifiers.cli.process.aggregate import aggregate_rows_by_env
from medarc_verifiers.cli.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli.process.writer import EXPORTER_METADATA_KEY, WriterConfig, write_env_groups


def _group_for_env() -> AggregatedEnvRows:
    rows = [
        {
            "env_id": "demo-env-variant",
            "base_env_id": "demo-env",
            "example_id": "ex-1",
            "job_run_id": "run-1",
            "model_id": "demo-model",
            "score": 1.0,
        },
        {
            "env_id": "demo-env-variant",
            "base_env_id": "demo-env",
            "example_id": "ex-2",
            "job_run_id": "run-2",
            "model_id": "demo-model",
            "score": 0.5,
        },
    ]
    return aggregate_rows_by_env(rows)[0]


def test_write_env_groups_creates_parquet_and_index(tmp_path: Path) -> None:
    group = _group_for_env()
    config = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-01T00:00:00Z",
        processed_with_args={"include_prompt_completion": False},
    )

    summaries = write_env_groups([group], config)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.output_path.exists()
    assert summary.env_id == "demo-env-variant"
    assert summary.model_id == "demo-model"
    assert summary.output_path.parent.name == "demo-model"

    table = pq.read_table(summary.output_path)
    metadata = table.schema.metadata or {}
    assert EXPORTER_METADATA_KEY in metadata
    embedded = json.loads(metadata[EXPORTER_METADATA_KEY])
    assert embedded["source_runs"] == list(group.job_run_ids)
    assert embedded["processed_with_args"] == {"include_prompt_completion": False}

    index_path = tmp_path / "env_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    env_entry = payload["environments"][0]
    assert env_entry["row_count"] == len(group.rows)
    assert env_entry["env_id"] == "demo-env-variant"
    assert env_entry["base_env_id"] == "demo-env"
    assert env_entry["model_id"] == "demo-model"
    assert payload["env_groups"][0]["paths"][0]["path"].endswith("demo-model/demo-env-variant.parquet")
    ds_infos = json.loads((tmp_path / "dataset_infos.json").read_text(encoding="utf-8"))
    assert "default" in ds_infos
    assert "train" in ds_infos["default"]["data_files"]


def test_write_env_groups_dry_run(tmp_path: Path) -> None:
    group = _group_for_env()
    config = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-01T00:00:00Z",
        dry_run=True,
    )
    summaries = write_env_groups([group], config)
    assert summaries[0].dry_run is True
    assert not summaries[0].output_path.exists()
    assert not (tmp_path / "env_index.json").exists()


def test_write_env_groups_respects_overwrite_flag(tmp_path: Path) -> None:
    group = _group_for_env()
    config = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-01T00:00:00Z",
    )
    write_env_groups([group], config)

    # Default behavior is append=True; set append=False to require explicit overwrite
    config.append = False
    with pytest.raises(FileExistsError):
        write_env_groups([group], config)

    config.overwrite = True
    write_env_groups([group], config)


def test_write_env_groups_appends_and_deduplicates(tmp_path: Path) -> None:
    # First write with two rows (run-1, run-2)
    group1 = _group_for_env()
    cfg = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-01T00:00:00Z",
        processed_with_args={"include_prompt_completion": False},
        append=True,
    )
    summaries1 = write_env_groups([group1], cfg)
    out_path = summaries1[0].output_path
    assert out_path.exists()

    # Second write: one overlapping run (run-2) and one new (run-3)
    rows2 = [
        {
            "env_id": "demo-env-variant",
            "base_env_id": "demo-env",
            "example_id": "ex-3",
            "job_run_id": "run-2",
            "model_id": "demo-model",
            "score": 0.7,
        },
        {
            "env_id": "demo-env-variant",
            "base_env_id": "demo-env",
            "example_id": "ex-4",
            "job_run_id": "run-3",
            "model_id": "demo-model",
            "score": 0.9,
        },
    ]
    group2 = aggregate_rows_by_env(rows2)[0]
    write_env_groups([group2], cfg)

    # Read back and check rows and metadata
    table = pq.read_table(out_path)
    df = table.to_pandas()
    # Expect three unique job_run_ids: run-1, run-2, run-3
    assert set(df["job_run_id"]) == {"run-1", "run-2", "run-3"}
    assert len(df) == 3

    meta = table.schema.metadata or {}
    assert EXPORTER_METADATA_KEY in meta
    embedded = json.loads(meta[EXPORTER_METADATA_KEY])
    assert embedded["append"] is True
    assert set(embedded["source_runs"]) == {"run-1", "run-2", "run-3"}


def test_write_env_groups_overwrite_rebuilds_fresh(tmp_path: Path) -> None:
    # Initial write with two rows
    group1 = _group_for_env()
    cfg1 = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-01T00:00:00Z",
        append=True,
    )
    summaries1 = write_env_groups([group1], cfg1)
    out_path = summaries1[0].output_path
    assert out_path.exists()

    # Second write with overwrite=True and append=False, using a different single-row group
    rows2 = [
        {
            "env_id": "demo-env-variant",
            "base_env_id": "demo-env",
            "example_id": "ex-5",
            "job_run_id": "run-99",
            "model_id": "demo-model",
            "score": 0.2,
        }
    ]
    group2 = aggregate_rows_by_env(rows2)[0]
    cfg2 = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-02T00:00:00Z",
        append=False,
        overwrite=True,
    )
    write_env_groups([group2], cfg2)

    # Expect file to be rebuilt with only the new row
    table = pq.read_table(out_path)
    df = table.to_pandas()
    assert list(df["job_run_id"]) == ["run-99"]
    assert len(df) == 1

    meta = table.schema.metadata or {}
    assert EXPORTER_METADATA_KEY in meta
    embedded = json.loads(meta[EXPORTER_METADATA_KEY])
    # Fresh write path should mark append False and reflect only the new source run
    assert embedded.get("append") is False
    assert embedded["source_runs"] == ["run-99"]


def test_build_arrow_table_infers_schema_beyond_default_window() -> None:
    # Schema inference should consider all rows so late non-null values don't error.
    rows = []
    for idx in range(120):
        rows.append(
            {
                "env_id": "demo-env",
                "base_env_id": "demo-env",
                "example_id": f"ex-{idx}",
                "job_run_id": f"run-{idx}",
                "judge_cost": None,
            }
        )
    rows.append(
        {
            "env_id": "demo-env",
            "base_env_id": "demo-env",
            "example_id": "ex-final",
            "job_run_id": "run-final",
            "judge_cost": 0.0,
        }
    )
    group = aggregate_rows_by_env(rows)[0]

    table = writer._build_arrow_table(group)
    df = pl.from_arrow(table)

    assert df.schema["judge_cost"] == pl.Float64
    assert df["judge_cost"].drop_nulls().to_list() == [0.0]
