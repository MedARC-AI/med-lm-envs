from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import hf as hf_sync
from medarc_verifiers.cli.hf import sync as hf_sync_impl
from medarc_verifiers.cli.process.aggregate import aggregate_rows_by_env
from medarc_verifiers.cli.process.writer import WriterConfig, write_env_groups


def test_sync_to_hub_dry_run_returns_none(tmp_path: Path) -> None:
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
    assert summary is None


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


def test_sync_to_hub_dry_run_with_relative_output_paths_returns_none(
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
    assert summary is None


@pytest.mark.parametrize(
    ("remote_case", "expected_pending"),
    [
        ("missing", {"model-a/env-a.parquet"}),
        ("match", set()),
        ("mismatch", {"model-a/env-a.parquet"}),
        ("no-lfs", {"model-a/env-a.parquet"}),
    ],
)
def test_compute_pending_parquet_uploads_detects_remote_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    remote_case: str,
    expected_pending: set[str],
) -> None:
    parquet_path = tmp_path / "model-a" / "env-a.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_text("local-data", encoding="utf-8")
    local_sha = hashlib.sha256(parquet_path.read_bytes()).hexdigest()

    class FakeLFS:
        def __init__(self, sha256: str | None) -> None:
            self.sha256 = sha256

    class FakeTreeEntry:
        def __init__(self, path: str, lfs: object | None) -> None:
            self.path = path
            self.lfs = lfs

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def list_repo_tree(self, **_kwargs: object) -> list[FakeTreeEntry]:
            if remote_case == "missing":
                return []
            if remote_case == "no-lfs":
                return [FakeTreeEntry("model-a/env-a.parquet", None)]
            sha256 = local_sha if remote_case == "match" else "0" * 64
            return [FakeTreeEntry("model-a/env-a.parquet", FakeLFS(sha256))]

    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))

    pending = hf_sync.compute_pending_parquet_uploads(
        output_dir=tmp_path,
        repo_id="demo/repo",
        branch="main",
        token="secret-token",
    )

    assert pending == expected_pending


def test_sync_to_hub_explicit_files_uploads_exact_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "keep.parquet").write_text("1", encoding="utf-8")
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {}

    class FakeOp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured.setdefault("ops", []).append((args, kwargs))

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            captured["token"] = token

        def create_repo(self, **_kwargs: object) -> None:
            captured["create_repo"] = True

        def create_commit(self, **kwargs: object) -> None:
            captured["create_commit"] = kwargs

    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(CommitOperationAdd=FakeOp, HfApi=FakeApi))

    summary = hf_sync.sync_to_hub(
        [],
        hf_sync.HFSyncConfig(repo_id="local/test", token="secret-token"),
        output_dir=tmp_path,
        files=[tmp_path / "keep.parquet", "meta.json"],
    )

    assert summary is not None
    assert summary.files == ["keep.parquet", "meta.json"]
    assert summary.total_files == 2
    assert summary.total_rows == 0
    assert captured["token"] == "secret-token"
    assert captured.get("create_commit") is not None


def test_sync_to_hub_explicit_files_respects_dry_run(tmp_path: Path) -> None:
    (tmp_path / "keep.parquet").write_text("1", encoding="utf-8")

    summary = hf_sync.sync_to_hub(
        [],
        hf_sync.HFSyncConfig(repo_id="local/test", dry_run=True),
        output_dir=tmp_path,
        files=["keep.parquet"],
    )

    assert summary is None


@pytest.mark.parametrize("bad_path", ["/tmp/escape.txt", "../escape.txt"])
def test_sync_files_to_hub_rejects_unsafe_paths(tmp_path: Path, bad_path: str) -> None:
    with pytest.raises(ValueError, match="output_dir|traversal"):
        hf_sync.sync_files_to_hub(
            repo_id="local/test",
            output_dir=tmp_path,
            files=[bad_path],
            token=None,
            private=False,
            message="msg",
            dry_run=True,
        )


def test_transient_hf_errors_include_statuses_timeouts_and_transport() -> None:
    import httpx

    class StatusError(Exception):
        def __init__(self, status_code: int) -> None:
            super().__init__(f"status={status_code}")
            self.response = SimpleNamespace(status_code=status_code)

    assert hf_sync_impl._is_transient_hf_error(StatusError(429)) is True
    assert hf_sync_impl._is_transient_hf_error(StatusError(503)) is True
    assert hf_sync_impl._is_transient_hf_error(httpx.TimeoutException("timeout")) is True
    assert hf_sync_impl._is_transient_hf_error(httpx.TransportError("transport")) is True


def test_compute_pending_parquet_uploads_retries_without_expand_on_compat_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "model-a" / "env-a.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path.write_text("local-data", encoding="utf-8")

    calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            self.token = token

        def list_repo_tree(self, **kwargs: object):
            calls.append(kwargs)
            if "expand" in kwargs:
                raise TypeError("unexpected keyword argument 'expand'")
            return [SimpleNamespace(path="model-a/env-a.parquet", lfs=None)]

    import sys

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))

    pending = hf_sync.compute_pending_parquet_uploads(
        output_dir=tmp_path,
        repo_id="demo/repo",
        branch="main",
        token="secret-token",
    )

    assert pending == {"model-a/env-a.parquet"}
    assert len(calls) == 2
    assert "expand" in calls[0]
    assert "expand" not in calls[1]


def test_sync_files_to_hub_creates_repo_with_confirmation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Create a dummy file to upload
    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {"create_commit_calls": 0}

    class FakeResponse:
        status_code = 404

    class FakeRepoNotFound(Exception):
        def __init__(self) -> None:
            super().__init__("Repository Not Found")
            self.response = FakeResponse()

    class FakeOp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["op"] = (args, kwargs)

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            captured["token"] = token
            self._created = False

        def create_repo(self, **kwargs: object) -> None:
            captured["create_repo"] = kwargs
            self._created = True

        def create_commit(self, **_kwargs: object) -> None:
            captured["create_commit_calls"] = int(captured["create_commit_calls"]) + 1
            if not self._created:
                raise FakeRepoNotFound()
            captured["create_commit"] = True

    import sys
    import types

    fake_module = types.SimpleNamespace(CommitOperationAdd=FakeOp, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    hf_sync.sync_files_to_hub(
        repo_id="local/missing",
        output_dir=tmp_path,
        files=["artifact.json"],
        token="secret-token",
        private=True,
        message="msg",
        dry_run=False,
        is_tty=True,
        assume_yes=False,
        prompt_func=lambda _prompt: "y",
    )

    assert captured.get("create_repo") is not None
    assert captured.get("create_commit") is True
    assert captured["create_commit_calls"] == 2


def test_sync_files_to_hub_skips_when_repo_creation_declined(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "artifact.json").write_text("{}", encoding="utf-8")

    captured: dict[str, object] = {"create_commit_calls": 0}

    class FakeResponse:
        status_code = 404

    class FakeRepoNotFound(Exception):
        def __init__(self) -> None:
            super().__init__("Repository Not Found")
            self.response = FakeResponse()

    class FakeOp:
        def __init__(self, *args: object, **kwargs: object) -> None:
            captured["op"] = (args, kwargs)

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            captured["token"] = token

        def create_repo(self, **kwargs: object) -> None:
            captured["create_repo"] = kwargs

        def create_commit(self, **_kwargs: object) -> None:
            captured["create_commit_calls"] = int(captured["create_commit_calls"]) + 1
            raise FakeRepoNotFound()

    import sys
    import types

    fake_module = types.SimpleNamespace(CommitOperationAdd=FakeOp, HfApi=FakeApi)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with caplog.at_level("WARNING"):
        uploaded = hf_sync.sync_files_to_hub(
            repo_id="local/missing",
            output_dir=tmp_path,
            files=["artifact.json"],
            token="secret-token",
            private=True,
            message="msg",
            dry_run=False,
            is_tty=True,
            assume_yes=False,
            prompt_func=lambda _prompt: "n",
        )

    assert uploaded is False
    assert captured["create_commit_calls"] == 1
    assert captured.get("create_repo") is None
    assert "skipping upload because repo creation was declined" in caplog.text
