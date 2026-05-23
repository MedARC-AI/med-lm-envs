import json
import sys
import types
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import ExecutionAllocation, ensure_run_bundle, load_task_spec, write_execution_allocation
from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.lifecycle import materialize_image, run_construct, run_teardown


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
