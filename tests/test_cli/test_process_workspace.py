from __future__ import annotations

import json
from pathlib import Path

import pytest

from medarc_verifiers.cli.hf import HFSyncConfig
from medarc_verifiers.cli.process import workspace


def _write_snapshot(snapshot_dir: Path, *, content: str = "remote") -> Path:
    parquet_path = snapshot_dir / "model-a" / "env-a.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_text(content, encoding="utf-8")
    env_index = {
        "version": 2,
        "processed_at": "2024-01-01T00:00:00Z",
        "schema_version": 1,
        "processed_with_args": {},
        "runs": {},
        "files": {"model-a/env-a.parquet": {"env_id": "env-a", "base_env_id": "env-a", "model_id": "model-a"}},
    }
    (snapshot_dir / "env_index.json").write_text(json.dumps(env_index), encoding="utf-8")
    return parquet_path


def test_prepare_hf_baseline_pull_copies_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    parquet_path = _write_snapshot(snapshot_dir)

    def _fake_download_hf_repo(**_kwargs) -> Path:
        return snapshot_dir

    monkeypatch.setattr(workspace, "download_hf_repo", _fake_download_hf_repo)
    hf_config = HFSyncConfig(repo_id="demo/repo")
    output_dir = tmp_path / "output"

    result = workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=hf_config,
        pull_policy="pull",
        is_tty=False,
        prompt_func=None,
    )

    copied = output_dir / parquet_path.relative_to(snapshot_dir)
    assert copied.exists()
    assert copied in result.files_copied


def test_prepare_hf_baseline_pull_keeps_unrelated_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir)

    def _fake_download_hf_repo(**_kwargs) -> Path:
        return snapshot_dir

    monkeypatch.setattr(workspace, "download_hf_repo", _fake_download_hf_repo)
    hf_config = HFSyncConfig(repo_id="demo/repo")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    local_path = output_dir / "local.txt"
    local_path.write_text("local", encoding="utf-8")

    workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=hf_config,
        pull_policy="pull",
        is_tty=False,
        prompt_func=None,
    )

    assert local_path.exists()
    assert (output_dir / "model-a" / "env-a.parquet").exists()


def test_prepare_hf_baseline_clean_replaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir, content="remote")

    def _fake_download_hf_repo(**_kwargs) -> Path:
        return snapshot_dir

    monkeypatch.setattr(workspace, "download_hf_repo", _fake_download_hf_repo)
    hf_config = HFSyncConfig(repo_id="demo/repo")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    local_path = output_dir / "model-a" / "env-a.parquet"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text("local", encoding="utf-8")

    workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=hf_config,
        pull_policy="clean",
        is_tty=False,
        prompt_func=None,
    )

    assert local_path.read_text(encoding="utf-8") == "remote"


def test_prepare_hf_baseline_prompt_conflict_overwrite(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir, content="remote")

    def _fake_download_hf_repo(**_kwargs) -> Path:
        return snapshot_dir

    monkeypatch.setattr(workspace, "download_hf_repo", _fake_download_hf_repo)
    hf_config = HFSyncConfig(repo_id="demo/repo")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    local_path = output_dir / "model-a" / "env-a.parquet"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text("local", encoding="utf-8")

    responses = iter(["pull", "y"])

    def _prompt(_message: str) -> str:
        return next(responses)

    workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=hf_config,
        pull_policy="prompt",
        is_tty=True,
        prompt_func=_prompt,
    )

    assert local_path.read_text(encoding="utf-8") == "remote"


def test_prepare_hf_baseline_pull_skips_when_local_baseline_present(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_snapshot(output_dir)

    def _fail_download(**_kwargs) -> Path:
        raise AssertionError("download_hf_repo should not be called when local baseline exists")

    monkeypatch.setattr(workspace, "download_hf_repo", _fail_download)
    hf_config = HFSyncConfig(repo_id="demo/repo")

    result = workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=hf_config,
        pull_policy="pull",
        is_tty=False,
        prompt_func=None,
    )

    assert result.policy == "pull"


def test_prepare_hf_baseline_pull_downloads_when_file_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    parquet_path = _write_snapshot(snapshot_dir, content="remote")
    _write_snapshot(output_dir, content="local")
    (output_dir / parquet_path.relative_to(snapshot_dir)).unlink()

    def _fake_download_hf_repo(**_kwargs) -> Path:
        return snapshot_dir

    monkeypatch.setattr(workspace, "download_hf_repo", _fake_download_hf_repo)
    hf_config = HFSyncConfig(repo_id="demo/repo")

    result = workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=hf_config,
        pull_policy="pull",
        is_tty=False,
        prompt_func=None,
    )

    restored = output_dir / parquet_path.relative_to(snapshot_dir)
    assert restored.exists()
    assert restored in result.files_copied
