import json
import os
import sys
import types
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import ExecutionAllocation, ensure_run_bundle, load_task_spec, write_execution_allocation
from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate import lifecycle as lifecycle_module
from medarc_verifiers.orchestrate.lifecycle import materialize_image, resolve_image_digest, run_construct, run_teardown


def _task(tmp_path: Path) -> TaskSpec:
    suite = tmp_path / "suite.toml"
    suite.write_text(
        """
[[eval]]
env_id = "medqa"
num_examples = 1
rollouts_per_example = 1
""".lstrip(),
        encoding="utf-8",
    )
    return TaskSpec(
        task_id="foo:suite",
        model_key="foo",
        model_id="Foo/Bar",
        orchestrate={
            "vllm": {"gpus": 1, "tensor_parallel_size": 1, "serve": {}},
            "container": {"image": "vllm/vllm-openai:v0.12.0", "container_port": 8000},
            "pyxis": {},
        },
        suite_path=suite,
        target_endpoint_id="foo",
        generated_eval_config={
            "endpoint_id": "foo",
            "endpoints_path": str(tmp_path / "endpoints.toml"),
            "output_dir": ".",
            "eval": [{"env_id": "medqa", "num_examples": 1, "rollouts_per_example": 1}],
        },
    )


def _bundle_with_lifecycle(tmp_path: Path, *, isolated: bool = True):
    hf_home = tmp_path / ("outputs/orchestrate/run/hf" if isolated else "shared/hf")
    hub_cache = hf_home / "hub"
    image_dir = tmp_path / "images"
    bundle = ensure_run_bundle(
        tasks=[_task(tmp_path)],
        output_root=tmp_path / "outputs" / "orchestrate" / "run",
        mode="slurm",
        runtime="pyxis",
        construct_cache_by_task={
            "foo:suite": {
                "hf_home": str(hf_home),
                "hub_cache": str(hub_cache),
                "image_dir": str(image_dir),
                "isolated": isolated,
            }
        },
        teardown_by_task={"foo:suite": {"remove_model_weights": True, "remove_images": False}},
        container_image_by_task={"foo:suite": str(image_dir / "vllm.sqsh")},
    )
    task_bundle = bundle.tasks["foo:suite"]
    write_execution_allocation(
        task_bundle.paths.construct_allocation_path,
        ExecutionAllocation(task_id="foo:suite"),
    )
    write_execution_allocation(
        task_bundle.paths.teardown_allocation_path,
        ExecutionAllocation(task_id="foo:suite"),
    )
    return task_bundle


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


def test_construct_prefetch_loads_env_and_writes_snapshot_result(tmp_path: Path, monkeypatch) -> None:
    task_bundle = _bundle_with_lifecycle(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from-env-file\n", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_snapshot_download(repo_id, cache_dir, local_files_only, resume_download):
        import os

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

    rc = run_construct(
        task_path=task_bundle.paths.task_spec_path,
        allocation_path=task_bundle.paths.construct_allocation_path,
        env_file=env_file,
        prefetch_model_flag=True,
        materialize_image_flag=False,
    )

    result = json.loads(task_bundle.paths.construct_result_path.read_text(encoding="utf-8"))
    spec = load_task_spec(task_bundle.paths.task_spec_path)
    assert rc == 0
    assert captured["repo_id"] == "Foo/Bar"
    assert captured["cache_dir"] == spec.construct_cache["hub_cache"]
    assert captured["hf_token"] == "from-env-file"
    assert captured["hf_home"] == spec.construct_cache["hf_home"]
    assert result["model"]["snapshot_path"].endswith("/models--Foo--Bar/snapshots/abc123")
    assert result["model"]["commit_hash"] == "abc123"


def test_construct_materialize_configures_enroot_paths_under_image_cache(tmp_path: Path, monkeypatch) -> None:
    task_bundle = _bundle_with_lifecycle(tmp_path)
    spec = load_task_spec(task_bundle.paths.task_spec_path)
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

    rc = run_construct(
        task_path=task_bundle.paths.task_spec_path,
        allocation_path=task_bundle.paths.construct_allocation_path,
        env_file=None,
        prefetch_model_flag=False,
        materialize_image_flag=True,
    )

    result = json.loads(task_bundle.paths.construct_result_path.read_text(encoding="utf-8"))
    image_dir = Path(str(spec.construct_cache["image_dir"]))
    assert rc == 0
    assert captured["source"] == "vllm/vllm-openai:v0.12.0"
    assert str(captured["enroot_cache"]).startswith(str(image_dir / ".enroot" / "cache-"))
    assert str(captured["enroot_data"]).startswith(str(image_dir / ".enroot" / "data-"))
    assert str(captured["enroot_runtime"]).endswith("-123")
    assert Path(str(captured["enroot_cache"])).is_dir()
    assert result["enroot"]["ENROOT_CACHE_PATH"] == captured["enroot_cache"]


def test_teardown_deletes_only_isolated_repo_cache(tmp_path: Path) -> None:
    task_bundle = _bundle_with_lifecycle(tmp_path, isolated=True)
    spec = load_task_spec(task_bundle.paths.task_spec_path)
    repo_dir = Path(spec.construct_cache["hub_cache"]) / "models--Foo--Bar"
    repo_dir.mkdir(parents=True)
    (repo_dir / "refs").mkdir()
    task_bundle.paths.construct_result_path.write_text(
        json.dumps({"model": {"repo_id": "Foo/Bar", "hub_cache": spec.construct_cache["hub_cache"]}}),
        encoding="utf-8",
    )

    rc = run_teardown(
        task_path=task_bundle.paths.task_spec_path,
        allocation_path=task_bundle.paths.teardown_allocation_path,
        env_file=None,
    )

    result = json.loads(task_bundle.paths.teardown_result_path.read_text(encoding="utf-8"))
    assert rc == 0
    assert not repo_dir.exists()
    assert str(repo_dir) in result["removed"]


def test_teardown_rejects_shared_cache_model_deletion(tmp_path: Path) -> None:
    task_bundle = _bundle_with_lifecycle(tmp_path, isolated=False)

    rc = run_teardown(
        task_path=task_bundle.paths.task_spec_path,
        allocation_path=task_bundle.paths.teardown_allocation_path,
        env_file=None,
    )

    result = json.loads(task_bundle.paths.teardown_result_path.read_text(encoding="utf-8"))
    assert rc == 1
    assert result["state"] == "failed"
    assert "isolated" in result["error"]
