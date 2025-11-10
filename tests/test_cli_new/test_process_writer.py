from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from medarc_verifiers.cli_new.process.aggregate import aggregate_rows_by_env
from medarc_verifiers.cli_new.process.aggregate import AggregatedEnvRows
from medarc_verifiers.cli_new.process.writer import (
    EXPORTER_METADATA_KEY,
    WriterConfig,
    write_env_groups,
)


def _group_for_env() -> AggregatedEnvRows:
    rows = [
        {"env_id": "demo-env-variant", "base_env_id": "demo-env", "example_id": "ex-1", "job_run_id": "run-1", "score": 1.0},
        {"env_id": "demo-env-variant", "base_env_id": "demo-env", "example_id": "ex-2", "job_run_id": "run-2", "score": 0.5},
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

    with pytest.raises(FileExistsError):
        write_env_groups([group], config)

    config.overwrite = True
    write_env_groups([group], config)
