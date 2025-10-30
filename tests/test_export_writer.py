from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.export.parquet.aggregation import AggregatedEnvRows
from medarc_verifiers.export.parquet.writer import write_parquet_datasets


def _build_group(
    env_id: str,
    rows: list[dict],
    *,
    partition_columns: tuple[str, ...] = (),
    partitions: dict[tuple, list[dict]] | None = None,
) -> AggregatedEnvRows:
    return AggregatedEnvRows(
        env_id=env_id,
        rows=rows,
        column_names=tuple(sorted({key for row in rows for key in row})),
        partition_columns=partition_columns,
        partitions=partitions or {},
    )


def test_write_parquet_datasets_creates_files(tmp_path: Path):
    rows = [
        {"env_id": "env-a", "value": 1},
        {"env_id": "env-a", "value": 2},
    ]
    group = _build_group("env-a", rows)

    output_dir = tmp_path / "exports"
    summaries, errors = write_parquet_datasets([group], output_dir, dry_run=False, overwrite=False)

    assert errors == []
    assert len(summaries) == 1
    dataset_path = Path(summaries[0].dataset_paths[0])
    assert dataset_path.exists()

    index_path = output_dir / "env_index.json"
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["environments"][0]["env_id"] == "env-a"


def test_write_parquet_datasets_respects_partitions(tmp_path: Path):
    rows = [
        {"env_id": "env-a", "model": "m1", "value": 1},
        {"env_id": "env-a", "model": "m2", "value": 2},
        {"env_id": "env-a", "model": "m1", "value": 3},
    ]
    partitions = {
        ("m1",): [rows[0], rows[2]],
        ("m2",): [rows[1]],
    }
    group = _build_group("env-a", rows, partition_columns=("model",), partitions=partitions)

    output_dir = tmp_path / "exports"
    summaries, errors = write_parquet_datasets([group], output_dir, dry_run=False, overwrite=False)

    assert errors == []
    dataset_paths = {Path(path).name for path in summaries[0].dataset_paths}
    assert dataset_paths == {"model-m1.parquet", "model-m2.parquet"}


def test_write_parquet_datasets_supports_dry_run(tmp_path: Path):
    rows = [{"env_id": "env-a", "value": 1}]
    group = _build_group("env-a", rows)

    output_dir = tmp_path / "exports"
    summaries, errors = write_parquet_datasets([group], output_dir, dry_run=True, overwrite=False)

    assert errors == []
    assert (output_dir / "env_index.json").exists() is False
    assert summaries[0].dataset_paths[0].endswith("data.parquet")
    assert not Path(summaries[0].dataset_paths[0]).exists()


def test_write_parquet_datasets_errors_without_overwrite(tmp_path: Path):
    rows = [{"env_id": "env-a", "value": 1}]
    group = _build_group("env-a", rows)

    output_dir = tmp_path / "exports"
    write_parquet_datasets([group], output_dir, dry_run=False, overwrite=False)
    summaries, errors = write_parquet_datasets([group], output_dir, dry_run=False, overwrite=False)

    assert summaries == []
    assert len(errors) == 1
    assert "already exists" in errors[0]


def test_write_parquet_datasets_sanitizes_empty_dicts(tmp_path: Path):
    rows = [
        {"env_id": "env-a", "sampling_args": {"extra_body": {}}},
    ]
    group = _build_group("env-a", rows)

    output_dir = tmp_path / "exports"
    summaries, errors = write_parquet_datasets([group], output_dir, dry_run=False, overwrite=False)

    assert errors == []
    dataset_path = Path(summaries[0].dataset_paths[0])
    assert dataset_path.exists()
