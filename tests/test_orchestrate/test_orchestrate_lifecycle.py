import json
import os
import sys
import types
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate import lifecycle as lifecycle_module
from medarc_verifiers.orchestrate.lifecycle import (
    materialize_image,
    materialized_image_path,
    resolve_image_digest,
    run_prepare,
    run_teardown,
)


def test_materialize_image_skips_existing_final_image(tmp_path: Path, monkeypatch) -> None:
    final = tmp_path / "image.sqsh"
    final.write_text("already materialized", encoding="utf-8")
    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.shutil.which", lambda name: None)

    result = materialize_image(source="vllm/vllm-openai:v0.12.0", final_path=final)

    assert result["skipped"] is True
    assert result["image_path"] == str(final)


def test_materialize_image_rejects_missing_absolute_sqsh(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqsh"

    with pytest.raises(RuntimeError, match="does not exist"):
        materialize_image(source=str(missing), final_path=missing)


def test_resolve_image_digest_uses_registry_bearer_challenge(monkeypatch) -> None:
    calls: list[tuple[str, dict[str, str] | None, dict[str, str] | None]] = []

    class Response:
        def __init__(self, status_code: int, *, headers=None, payload=None) -> None:
            self.status_code = status_code
            self.headers = headers or {}
            self._payload = payload or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._payload

    class Client:
        def __init__(self, timeout) -> None:
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def get(self, url, *, headers=None, params=None):
            calls.append((url, headers, params))
            if url.endswith("/v2/vllm/vllm-openai/manifests/latest") and headers.get("Authorization") is None:
                return Response(
                    401,
                    headers={
                        "www-authenticate": (
                            'Bearer realm="https://auth.docker.io/token",'
                            'service="registry.docker.io",scope="repository:vllm/vllm-openai:pull"'
                        )
                    },
                )
            if url == "https://auth.docker.io/token":
                return Response(200, payload={"token": "registry-token"})
            return Response(200, headers={"docker-content-digest": "sha256:abc123"})

    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.httpx.Client", Client)

    resolved = resolve_image_digest("vllm/vllm-openai:latest")

    assert resolved == "docker.io/vllm/vllm-openai@sha256:abc123"
    assert calls[1][2] == {
        "service": "registry.docker.io",
        "scope": "repository:vllm/vllm-openai:pull",
    }
    assert calls[2][1]["Authorization"] == "Bearer registry-token"


def test_materialize_latest_resolves_digest_imports_digest_path_and_updates_symlink(tmp_path: Path, monkeypatch) -> None:
    final = tmp_path / "latest.sqsh"
    imported_paths: list[Path] = []

    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.shutil.which", lambda name: "/usr/bin/enroot")
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.lifecycle.resolve_image_digest",
        lambda source: "docker.io/vllm/vllm-openai@sha256:abc123",
    )

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, check, capture_output, text):
        del check, capture_output, text
        tmp_prefix = Path(command[command.index("--output") + 1])
        produced = tmp_prefix
        produced.write_text("sqsh", encoding="utf-8")
        imported_paths.append(produced)
        assert command[-1] == "docker://vllm/vllm-openai:latest"
        return Completed()

    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.subprocess.run", fake_run)

    result = materialize_image(source="vllm/vllm-openai:latest", final_path=final)

    assert result["resolved_source"] == "docker.io/vllm/vllm-openai@sha256:abc123"
    assert result["image_path"] == str(final)
    assert result["resolved_image_path"].endswith(".sqsh")
    assert Path(result["resolved_image_path"]).is_file()
    assert final.is_symlink()
    assert final.resolve() == Path(result["resolved_image_path"])
    assert (tmp_path / "latest").is_symlink()
    assert imported_paths


def test_materialize_latest_does_not_self_link_digest_path(tmp_path: Path, monkeypatch) -> None:
    resolved_source = "docker.io/vllm/vllm-openai@sha256:abc123"
    final = materialized_image_path(resolved_source, tmp_path)

    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.shutil.which", lambda name: "/usr/bin/enroot")
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.lifecycle.resolve_image_digest",
        lambda source: resolved_source,
    )

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(command, check, capture_output, text):
        del check, capture_output, text
        tmp_prefix = Path(command[command.index("--output") + 1])
        tmp_prefix.write_text("sqsh", encoding="utf-8")
        return Completed()

    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.subprocess.run", fake_run)

    result = materialize_image(source="vllm/vllm-openai:latest", final_path=final)

    assert result["resolved_image_path"] == str(final)
    assert final.is_file()
    assert not final.is_symlink()
    assert (tmp_path / "latest").is_symlink()
    assert (tmp_path / "latest").resolve() == final


def test_lock_dir_recovers_stale_slurm_owner(tmp_path: Path, monkeypatch) -> None:
    lock_dir = tmp_path / ".locks" / "latest.lock"
    lock_dir.mkdir(parents=True)
    (lock_dir / "owner.json").write_text(json.dumps({"slurm_job_id": "123", "host": "other"}), encoding="utf-8")
    called = False

    class Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.shutil.which", lambda name: "/usr/bin/squeue")
    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.subprocess.run", lambda *args, **kwargs: Completed())

    def work() -> None:
        nonlocal called
        called = True
        assert (lock_dir / "owner.json").is_file()

    lifecycle_module._with_lock_dir(lock_dir, work)

    assert called is True
    assert not lock_dir.exists()


def test_prepare_prefetch_loads_env_and_writes_result(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    result_path = tmp_path / "runtime" / "prepare_result.json"
    hf_home = tmp_path / "outputs" / "orchestrate" / "run" / "hf"
    hub_cache = hf_home / "hub"
    env_file.write_text("HF_TOKEN=from-env-file\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id, cache_dir, local_files_only, resume_download):
        captured.update(
            {
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "local_files_only": local_files_only,
                "resume_download": resume_download,
                "hf_token": os.environ.get("HF_TOKEN"),
                "hf_home": os.environ.get("HF_HOME"),
            }
        )
        snapshot = Path(cache_dir) / "models--Foo--Bar" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        return str(snapshot)

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download, get_token=lambda: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    rc = run_prepare(
        model="Foo/Bar",
        result_path=result_path,
        env_file=env_file,
        hf_home=hf_home,
        hub_cache=hub_cache,
        prefetch_model_flag=True,
        materialize_image_flag=False,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert captured["repo_id"] == "Foo/Bar"
    assert captured["cache_dir"] == str(hub_cache)
    assert captured["hf_token"] == "from-env-file"
    assert captured["hf_home"] == str(hf_home)
    assert result["model"]["snapshot_path"].endswith("/models--Foo--Bar/snapshots/abc123")
    assert result["model"]["commit_hash"] == "abc123"


def test_prepare_model_implies_prefetch_and_result_is_optional(tmp_path: Path, monkeypatch, capsys) -> None:
    hub_cache = tmp_path / "outputs" / "orchestrate" / "run" / "hf" / "hub"
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id, cache_dir, local_files_only, resume_download):
        captured.update({"repo_id": repo_id, "cache_dir": cache_dir})
        snapshot = Path(cache_dir) / "models--Foo--Bar" / "snapshots" / "abc123"
        snapshot.mkdir(parents=True)
        return str(snapshot)

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download, get_token=lambda: None)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    rc = run_prepare(
        model="Foo/Bar",
        result_path=None,
        env_file=None,
        hub_cache=hub_cache,
    )

    assert rc == 0
    assert captured["repo_id"] == "Foo/Bar"
    assert "prepare completed model=Foo/Bar" in capsys.readouterr().out
    assert not (tmp_path / "runtime" / "prepare_result.json").exists()


def test_prepare_requires_operation_and_model_cache(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_prepare(model=None, result_path=None, env_file=None)

    with pytest.raises(ValueError, match="--hub-cache or --hf-home"):
        run_prepare(model="Foo/Bar", result_path=None, env_file=None)


def test_prepare_image_requires_output_for_non_absolute_image(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--image-dir or --image-output"):
        run_prepare(model=None, image="vllm/vllm-openai:latest", result_path=None, env_file=None)


def test_prepare_materialize_configures_enroot_paths_under_image_cache(tmp_path: Path, monkeypatch) -> None:
    image_dir = tmp_path / "images"
    result_path = tmp_path / "runtime" / "prepare_result.json"
    captured: dict[str, object] = {}

    def fake_materialize_image(*, source, final_path, latest_link):
        captured.update(
            {
                "source": source,
                "final_path": final_path,
                "latest_link": latest_link,
                "enroot_cache": os.environ.get("ENROOT_CACHE_PATH"),
                "enroot_data": os.environ.get("ENROOT_DATA_PATH"),
                "enroot_runtime": os.environ.get("ENROOT_RUNTIME_PATH"),
            }
        )
        return {"source": source, "image_path": str(final_path), "skipped": True}

    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr("medarc_verifiers.orchestrate.lifecycle.materialize_image", fake_materialize_image)

    rc = run_prepare(
        model="Foo/Bar",
        result_path=result_path,
        env_file=None,
        image="vllm/vllm-openai:v0.12.0",
        image_dir=image_dir,
        prefetch_model_flag=False,
        materialize_image_flag=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert captured["source"] == "vllm/vllm-openai:v0.12.0"
    assert str(captured["enroot_cache"]).startswith(str(image_dir / ".enroot" / "cache-"))
    assert str(captured["enroot_data"]).startswith(str(image_dir / ".enroot" / "data-"))
    assert str(captured["enroot_runtime"]).endswith("-123")
    assert Path(str(captured["enroot_cache"])).is_dir()
    assert result["enroot"]["ENROOT_CACHE_PATH"] == captured["enroot_cache"]


def test_teardown_deletes_only_isolated_repo_cache(tmp_path: Path) -> None:
    hub_cache = tmp_path / "outputs" / "orchestrate" / "run" / "hf" / "hub"
    repo_dir = hub_cache / "models--Foo--Bar"
    repo_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    result_path = tmp_path / "runtime" / "teardown_result.json"

    rc = run_teardown(
        result_path=result_path,
        model="Foo/Bar",
        env_file=None,
        hub_cache=hub_cache,
        remove_model_weights=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert not repo_dir.exists()
    assert str(repo_dir) in result["removed"]


def test_teardown_rejects_shared_cache_model_deletion(tmp_path: Path) -> None:
    result_path = tmp_path / "runtime" / "teardown_result.json"

    rc = run_teardown(
        result_path=result_path,
        model="Foo/Bar",
        env_file=None,
        hub_cache=tmp_path / "shared" / "hf" / "hub",
        remove_model_weights=True,
    )

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert result["state"] == "failed"
    assert "isolated" in result["error"]
