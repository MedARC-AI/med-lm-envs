from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from medarc_verifiers.cli.process import writer
from medarc_verifiers.cli.process.aggregate import aggregate_rows_by_env
from medarc_verifiers.cli.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli.process.writer import EXPORTER_METADATA_KEY, WriterConfig, write_env_groups


def _group_for_env() -> AggregatedEnvRows:
    rows = [
        {
            "env_id": "demo-env",
            "base_env_id": "demo-env",
            "example_id": "ex-1",
            "job_run_id": "run-1",
            "model_id": "demo-model",
            "score": 1.0,
        },
        {
            "env_id": "demo-env",
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
        processed_at="2024-01-01T00:00:00Z",
        processed_with_args={},
    )

    summaries = write_env_groups([group], config)
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary.output_path.exists()
    assert summary.env_id == "demo-env"
    assert summary.model_id == "demo-model"
    assert summary.output_path.parent.name == "demo-model"

    table = pq.read_table(summary.output_path)
    metadata = table.schema.metadata or {}
    assert EXPORTER_METADATA_KEY in metadata
    embedded = json.loads(metadata[EXPORTER_METADATA_KEY])
    assert embedded["source_runs"] == list(group.job_run_ids)
    assert embedded["processed_with_args"] == {}

    index_path = tmp_path / "env_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    rel_path = summary.output_path.relative_to(tmp_path).as_posix()
    file_entry = payload["files"][rel_path]
    assert file_entry["row_count"] == len(group.rows)
    assert file_entry["env_id"] == "demo-env"
    assert file_entry["model_id"] == "demo-model"
    assert file_entry["updated_at"] is None
    assert "output_path" not in payload["runs"]["run-1"]
    ds_infos = json.loads((tmp_path / "dataset_infos.json").read_text(encoding="utf-8"))
    assert "default" in ds_infos
    assert "train" in ds_infos["default"]["data_files"]
    assert set(ds_infos["default"]["data_files"].keys()) == {"train"}
    assert rel_path in ds_infos["default"]["data_files"]["train"]


def test_write_env_groups_writes_variant_path_and_metadata(tmp_path: Path) -> None:
    rows = [
        {
            "env_id": "demo-env",
            "base_env_id": "demo-env",
            "example_id": "ex-1",
            "job_run_id": "run-1",
            "model_id": "demo-model",
            "variant_id": "env_args.shuffle_seed-1618",
            "variant_payload": json.dumps({"env_args": {"shuffle_seed": 1618}}),
            "reward": 1.0,
        }
    ]
    group = aggregate_rows_by_env(rows)[0]
    config = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
        processed_with_args={},
    )

    summaries = write_env_groups([group], config)

    summary = summaries[0]
    rel_path = summary.output_path.relative_to(tmp_path).as_posix()
    assert rel_path == "demo-model/demo-env__variants/env_args.shuffle_seed-1618.parquet"
    assert summary.variant_id == "env_args.shuffle_seed-1618"
    assert summary.variant_payload == {"env_args": {"shuffle_seed": 1618}}

    table = pq.read_table(summary.output_path)
    assert table.column("variant_id").to_pylist() == ["env_args.shuffle_seed-1618"]
    assert [json.loads(value) for value in table.column("variant_payload").to_pylist()] == [
        {"env_args": {"shuffle_seed": 1618}}
    ]
    embedded = json.loads((table.schema.metadata or {})[EXPORTER_METADATA_KEY])
    assert embedded["variant_id"] == "env_args.shuffle_seed-1618"
    assert embedded["variant_payload"] == {"env_args": {"shuffle_seed": 1618}}

    payload = json.loads((tmp_path / "env_index.json").read_text(encoding="utf-8"))
    file_entry = payload["files"][rel_path]
    assert file_entry["env_id"] == "demo-env"
    assert file_entry["base_env_id"] == "demo-env"
    assert file_entry["model_id"] == "demo-model"
    assert file_entry["variant_id"] == "env_args.shuffle_seed-1618"
    assert file_entry["variant_payload"] == {"env_args": {"shuffle_seed": 1618}}


def test_write_env_groups_dry_run(tmp_path: Path) -> None:
    group = _group_for_env()
    config = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
        dry_run=True,
    )
    summaries = write_env_groups([group], config)
    assert summaries[0].dry_run is True
    assert not summaries[0].output_path.exists()
    assert not (tmp_path / "env_index.json").exists()


def test_write_env_groups_dataset_infos_includes_existing_parquets(tmp_path: Path) -> None:
    rows1 = [
        {
            "env_id": "env-a",
            "base_env_id": "env-a",
            "example_id": "ex-1",
            "job_run_id": "run-1",
            "model_id": "model-a",
        }
    ]
    rows2 = [
        {
            "env_id": "env-b",
            "base_env_id": "env-b",
            "example_id": "ex-2",
            "job_run_id": "run-2",
            "model_id": "model-b",
        }
    ]
    group1 = aggregate_rows_by_env(rows1)[0]
    group2 = aggregate_rows_by_env(rows2)[0]
    cfg = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
    )
    write_env_groups([group1], cfg)
    write_env_groups([group2], cfg)

    ds_infos = json.loads((tmp_path / "dataset_infos.json").read_text(encoding="utf-8"))
    data_files = set(ds_infos["default"]["data_files"]["train"])
    assert data_files == {"model-a/env-a.parquet", "model-b/env-b.parquet"}


def test_write_env_groups_overwrite_rebuilds_fresh(tmp_path: Path) -> None:
    # Initial write with two rows
    group1 = _group_for_env()
    cfg1 = WriterConfig(output_dir=tmp_path, processed_at="2024-01-01T00:00:00Z")
    summaries1 = write_env_groups([group1], cfg1)
    out_path = summaries1[0].output_path
    assert out_path.exists()

    # Second write uses a different single-row group and overwrites the file
    rows2 = [
        {
            "env_id": "demo-env",
            "base_env_id": "demo-env",
            "example_id": "ex-5",
            "job_run_id": "run-99",
            "model_id": "demo-model",
            "score": 0.2,
        }
    ]
    group2 = aggregate_rows_by_env(rows2)[0]
    cfg2 = WriterConfig(output_dir=tmp_path, processed_at="2024-01-02T00:00:00Z")
    write_env_groups([group2], cfg2)

    # Expect file to be rebuilt with only the new row
    table = pq.read_table(out_path)
    df = table.to_pandas()
    assert list(df["job_run_id"]) == ["run-99"]
    assert len(df) == 1

    meta = table.schema.metadata or {}
    assert EXPORTER_METADATA_KEY in meta
    embedded = json.loads(meta[EXPORTER_METADATA_KEY])
    assert embedded["source_runs"] == ["run-99"]


def test_write_env_groups_normalizes_absolute_index_keys(tmp_path: Path) -> None:
    index_path = tmp_path / "env_index.json"
    abs_path = (tmp_path / "demo-model" / "demo-env.parquet").resolve()
    legacy_payload = {
        "version": 2,
        "processed_at": "2024-01-01T00:00:00Z",
        "schema_version": 1,
        "processed_with_args": {},
        "runs": {},
        "files": {
            abs_path.as_posix(): {
                "env_id": "demo-env",
                "model_id": "demo-model",
                "row_count": 0,
            }
        },
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(legacy_payload), encoding="utf-8")

    group = _group_for_env()
    config = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
        processed_with_args={},
    )
    summaries = write_env_groups([group], config)
    rel_path = summaries[0].output_path.relative_to(tmp_path).as_posix()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert rel_path in payload["files"]
    assert all(not Path(path).is_absolute() for path in payload["files"].keys())


def test_write_env_groups_does_not_leave_temp_files(tmp_path: Path) -> None:
    group = _group_for_env()
    config = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
    )
    write_env_groups([group], config)
    assert not list(tmp_path.rglob("*.tmp"))


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


def test_build_arrow_table_drops_unexpected_flat_token_columns() -> None:
    rows = [
        {
            "env_id": "demo-env",
            "base_env_id": "demo-env",
            "example_id": "ex-1",
            "job_run_id": "run-1",
            "model_id": "demo-model",
            "input_tokens": 12,  # Should be dropped by fixed schema selection.
            "output_tokens": 8,  # Should be dropped by fixed schema selection.
            "model_token_prompt": 12.0,
            "model_token_completion": 8.0,
            "model_token_total": 20.0,
        }
    ]
    group = aggregate_rows_by_env(rows)[0]

    table = writer._build_arrow_table(group)
    assert table.schema.names == list(writer.ALLOWED_COLUMNS)

    df = pl.from_arrow(table)
    assert "input_tokens" not in df.columns
    assert "output_tokens" not in df.columns
    assert df["model_token_prompt"].drop_nulls().to_list() == [12.0]
    assert df["model_token_completion"].drop_nulls().to_list() == [8.0]
    assert df["model_token_total"].drop_nulls().to_list() == [20.0]
