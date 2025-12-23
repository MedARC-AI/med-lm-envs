from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from medarc_verifiers.cli.process import hf_sync
from medarc_verifiers.cli.process.aggregate import aggregate_rows_by_env
from medarc_verifiers.cli.process.writer import WriterConfig, write_env_groups


def _build_dataset(job_run_id: str, reward: float, started_at: str) -> Dataset:
    return Dataset.from_dict(
        {
            "job_run_id": [job_run_id],
            "example_id": ["ex-1"],
            "rollout_index": [0],
            "base_env_id": ["env-a"],
            "env_id": ["env-a"],
            "reward": [reward],
            "started_at": [started_at],
        }
    )


def test_merge_split_append_deduplicates_row_key() -> None:
    existing = _build_dataset("run-1", 0.5, "2024-01-01T00:00:00Z")
    new = _build_dataset("run-1", 0.7, "2024-01-02T00:00:00Z")

    merged = hf_sync._merge_split(new, existing, strategy="append")
    assert len(merged) == 1
    row = merged[0]
    assert row["reward"] == 0.5  # existing preserved


def test_merge_split_update_prefers_latest_started_at() -> None:
    existing = _build_dataset("run-1", 0.5, "2024-01-01T00:00:00Z")
    new = _build_dataset("run-2", 0.7, "2024-01-02T00:00:00Z")

    merged = hf_sync._merge_split(new, existing, strategy="update")
    assert len(merged) == 1
    row = merged[0]
    assert row["reward"] == 0.7
    assert row["job_run_id"] == "run-2"


def test_sync_to_hub_dry_run_builds_summary(tmp_path: Path) -> None:
    rows = [
        {"base_env_id": "env-a", "env_id": "env-a", "job_run_id": "run-1", "example_id": "ex-1", "rollout_index": 0}
    ]
    group = aggregate_rows_by_env(rows)[0]
    config = WriterConfig(
        output_dir=tmp_path,
        exporter_version="0.1.0",
        processed_at="2024-01-01T00:00:00Z",
    )
    summaries = write_env_groups([group], config)

    hf_config = hf_sync.HFSyncConfig(
        repo_id="local/test",
        merge_strategy="append",
        dry_run=True,
    )
    summary = hf_sync.sync_to_hub(summaries, hf_config)
    assert summary is not None
    assert summary.total_rows == len(rows)
    assert summary.splits[0].env_id == "env-a"
    assert summary.splits[0].base_env_id == "env-a"
