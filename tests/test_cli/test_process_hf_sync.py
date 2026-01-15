from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli import hf as hf_sync
from medarc_verifiers.cli.process.aggregate import aggregate_rows_by_env
from medarc_verifiers.cli.process.writer import WriterConfig, write_env_groups


def test_sync_to_hub_dry_run_builds_summary(tmp_path: Path) -> None:
    rows = [
        {"base_env_id": "env-a", "env_id": "env-a", "job_run_id": "run-1", "example_id": "ex-1", "rollout_index": 0}
    ]
    group = aggregate_rows_by_env(rows)[0]
    config = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
    )
    summaries = write_env_groups([group], config)

    hf_config = hf_sync.HFSyncConfig(
        repo_id="local/test",
        dry_run=True,
    )
    summary = hf_sync.sync_to_hub(
        summaries,
        hf_config,
        output_dir=tmp_path,
        metadata_paths=[tmp_path / "env_index.json", tmp_path / "dataset_infos.json"],
    )
    assert summary is not None
    assert summary.total_rows == len(rows)
    assert summary.total_files == 3
    assert "env_index.json" in summary.files
    assert "dataset_infos.json" in summary.files


def test_sync_to_hub_uses_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    rows = [
        {"base_env_id": "env-a", "env_id": "env-a", "job_run_id": "run-1", "example_id": "ex-1", "rollout_index": 0}
    ]
    group = aggregate_rows_by_env(rows)[0]
    config = WriterConfig(
        output_dir=tmp_path,
        processed_at="2024-01-01T00:00:00Z",
    )
    summaries = write_env_groups([group], config)

    captured: dict[str, object] = {}

    class FakeOp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["op"] = (args, kwargs)

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            captured["token"] = token

        def create_repo(self, **_kwargs: object) -> None:
            captured["create_repo"] = True

        def create_commit(self, **_kwargs: object) -> None:
            captured["create_commit"] = True

    import types
    import sys

    fake_module = types.SimpleNamespace(CommitOperationAdd=FakeOp, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    hf_config = hf_sync.HFSyncConfig(
        repo_id="local/test",
        dry_run=False,
        token="secret-token",
        private=True,
    )
    summary = hf_sync.sync_to_hub(
        summaries,
        hf_config,
        output_dir=tmp_path,
        metadata_paths=[tmp_path / "env_index.json"],
    )
    assert summary is not None
    assert captured["token"] == "secret-token"
    assert captured.get("create_commit") is True


def test_sync_to_hub_does_not_double_prefix_metadata_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    output_dir = Path("runs") / "processed"
    rows = [
        {"base_env_id": "env-a", "env_id": "env-a", "job_run_id": "run-1", "example_id": "ex-1", "rollout_index": 0}
    ]
    group = aggregate_rows_by_env(rows)[0]
    config = WriterConfig(
        output_dir=output_dir,
        processed_at="2024-01-01T00:00:00Z",
    )
    summaries = write_env_groups([group], config)

    hf_config = hf_sync.HFSyncConfig(
        repo_id="local/test",
        dry_run=True,
    )
    summary = hf_sync.sync_to_hub(
        summaries,
        hf_config,
        output_dir=output_dir,
        metadata_paths=[output_dir / "env_index.json", output_dir / "dataset_infos.json"],
    )
    assert summary is not None
    assert "env_index.json" in summary.files
    assert "dataset_infos.json" in summary.files
    assert "runs/processed/env_index.json" not in summary.files
