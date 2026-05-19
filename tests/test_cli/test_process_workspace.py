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


def test_prepare_output_workspace_clean_skips_hf_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "stale.txt"
    sentinel.write_text("stale", encoding="utf-8")

    def _fail_prepare_hf_baseline(**_kwargs) -> workspace.BaselineResult:
        raise AssertionError("prepare_hf_baseline should not run when clean=True")

    monkeypatch.setattr(workspace, "prepare_hf_baseline", _fail_prepare_hf_baseline)

    result = workspace.prepare_output_workspace(
        output_dir=output_dir,
        hf_config=HFSyncConfig(repo_id="demo/repo"),
        pull_policy="pull",
        clean=True,
        assume_yes=True,
        is_tty=False,
        prompt_func=None,
    )

    assert result.cleaned is True
    assert result.baseline_result is None
    assert not sentinel.exists()


def test_prepare_output_workspace_runs_hf_baseline_before_local_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    parquet_path = _write_snapshot(snapshot_dir)

    def _fake_prepare_hf_baseline(**_kwargs) -> workspace.BaselineResult:
        copied = output_dir / parquet_path.relative_to(snapshot_dir)
        copied.parent.mkdir(parents=True, exist_ok=True)
        copied.write_text(parquet_path.read_text(encoding="utf-8"), encoding="utf-8")
        (output_dir / "env_index.json").write_text((snapshot_dir / "env_index.json").read_text(encoding="utf-8"))
        return workspace.BaselineResult(policy="pull", files_copied=[copied], snapshot_dir=snapshot_dir)

    monkeypatch.setattr(workspace, "prepare_hf_baseline", _fake_prepare_hf_baseline)

    result = workspace.prepare_output_workspace(
        output_dir=output_dir,
        hf_config=HFSyncConfig(repo_id="demo/repo"),
        pull_policy="pull",
        clean=False,
        assume_yes=False,
        is_tty=False,
        prompt_func=None,
    )

    assert result.cleaned is False
    assert result.baseline_result is not None
    assert (output_dir / "env_index.json").exists()
    assert (output_dir / "model-a" / "env-a.parquet").exists()


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
    monkeypatch.setattr(workspace, "compute_pending_parquet_uploads", lambda **_kwargs: set())
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


def test_prepare_hf_baseline_prompt_offers_upload_when_pending_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_snapshot(output_dir, content="local")

    prompts: list[str] = []

    def _prompt(message: str) -> str:
        prompts.append(message)
        return "upload"

    def _fail_download(**_kwargs) -> Path:
        raise AssertionError("download_hf_repo should not be called for upload recovery")

    monkeypatch.setattr(workspace, "download_hf_repo", _fail_download)
    monkeypatch.setattr(
        workspace,
        "compute_pending_parquet_uploads",
        lambda **_kwargs: {"model-a/env-a.parquet"},
    )

    result = workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=HFSyncConfig(repo_id="demo/repo"),
        pull_policy="prompt",
        is_tty=True,
        prompt_func=_prompt,
    )

    assert result.policy == "continue-upload"
    assert result.pending_parquet_uploads == {"model-a/env-a.parquet"}
    assert prompts and "upload" in prompts[0]


def test_prepare_hf_baseline_continue_upload_skips_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_snapshot(output_dir, content="local")

    def _fail_download(**_kwargs) -> Path:
        raise AssertionError("download_hf_repo should not be called for continue-upload")

    monkeypatch.setattr(workspace, "download_hf_repo", _fail_download)
    monkeypatch.setattr(
        workspace,
        "compute_pending_parquet_uploads",
        lambda **_kwargs: {"model-a/env-a.parquet"},
    )

    result = workspace.prepare_hf_baseline(
        output_dir=output_dir,
        hf_config=HFSyncConfig(repo_id="demo/repo"),
        pull_policy="continue-upload",
        is_tty=False,
        prompt_func=None,
    )

    assert result.policy == "continue-upload"
    assert result.pending_parquet_uploads == {"model-a/env-a.parquet"}


def test_prepare_hf_baseline_prompt_hides_upload_when_recovery_check_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_snapshot(output_dir, content="local")

    prompts: list[str] = []

    def _prompt(message: str) -> str:
        prompts.append(message)
        return "pull"

    monkeypatch.setattr(
        workspace,
        "compute_pending_parquet_uploads",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("hf down")),
    )
    monkeypatch.setattr(workspace, "download_hf_repo", lambda **_kwargs: tmp_path / "snapshot")

    with caplog.at_level("WARNING"):
        result = workspace.prepare_hf_baseline(
            output_dir=output_dir,
            hf_config=HFSyncConfig(repo_id="demo/repo"),
            pull_policy="prompt",
            is_tty=True,
            prompt_func=_prompt,
        )

    assert result.policy == "pull"
    assert result.pending_parquet_uploads == set()
    assert prompts and "upload" not in prompts[0]
    assert "HF upload recovery check failed before prompt" in caplog.text


def test_prepare_hf_baseline_continue_upload_empty_dir_warns_and_pulls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _write_snapshot(snapshot_dir, content="remote")
    monkeypatch.setattr(workspace, "download_hf_repo", lambda **_kwargs: snapshot_dir)

    with caplog.at_level("WARNING"):
        result = workspace.prepare_hf_baseline(
            output_dir=tmp_path / "output",
            hf_config=HFSyncConfig(repo_id="demo/repo"),
            pull_policy="continue-upload",
            is_tty=False,
            prompt_func=None,
        )

    assert result.policy == "pull"
    assert "falling back to pull" in caplog.text


def test_prepare_hf_baseline_continue_upload_degrades_when_recovery_check_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_snapshot(output_dir, content="local")

    monkeypatch.setattr(
        workspace,
        "compute_pending_parquet_uploads",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("hf down")),
    )

    with caplog.at_level("WARNING"):
        result = workspace.prepare_hf_baseline(
            output_dir=output_dir,
            hf_config=HFSyncConfig(repo_id="demo/repo"),
            pull_policy="continue-upload",
            is_tty=False,
            prompt_func=None,
        )

    assert result.policy == "continue-upload"
    assert result.pending_parquet_uploads == set()
    assert "uploading only current touched files" in caplog.text


@pytest.mark.parametrize("exc_type", [EOFError, KeyboardInterrupt])
def test_prepare_hf_baseline_prompt_aborts_cleanly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exc_type: type[BaseException],
) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    _write_snapshot(output_dir, content="local")
    monkeypatch.setattr(
        workspace,
        "compute_pending_parquet_uploads",
        lambda **_kwargs: {"model-a/env-a.parquet"},
    )

    def _prompt(_message: str) -> str:
        raise exc_type

    with pytest.raises(RuntimeError, match="Aborted HF baseline selection."):
        workspace.prepare_hf_baseline(
            output_dir=output_dir,
            hf_config=HFSyncConfig(repo_id="demo/repo"),
            pull_policy="prompt",
            is_tty=True,
            prompt_func=_prompt,
        )


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


def test_hf_baseline_rejects_absolute_env_index_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "env_index.json").write_text(
        json.dumps(
            {
                "version": 2,
                "files": {"/abs/path.parquet": {"env_id": "env-a", "model_id": "model-a"}},
            }
        ),
        encoding="utf-8",
    )
    assert workspace._has_complete_hf_baseline(output_dir) is False


def test_hf_baseline_rejects_traversal_env_index_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "env_index.json").write_text(
        json.dumps(
            {
                "version": 2,
                "files": {"../escape.parquet": {"env_id": "env-a", "model_id": "model-a"}},
            }
        ),
        encoding="utf-8",
    )
    assert workspace._has_complete_hf_baseline(output_dir) is False
