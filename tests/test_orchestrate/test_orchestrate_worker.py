import sys
import types
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import ExecutionAllocation, ensure_run_bundle, load_task_spec
from medarc_verifiers.orchestrate.config import expand_tasks, load_plan
from medarc_verifiers.orchestrate.state import TaskManifest
from medarc_verifiers.orchestrate.runtime import RuntimeLaunchError
from medarc_verifiers.orchestrate.podman_vllm import PodmanRuntimeAdapter
from medarc_verifiers.orchestrate.worker import _load_runtime_env, main as worker_main


def _write_job_config(path: Path) -> None:
    path.write_text(
        """
model = "Foo/Bar"

[[eval]]
env_id = "medqa"
num_examples = 1
rollouts_per_example = 1
""".lstrip(),
        encoding="utf-8",
    )


def _write_orchestrate_config(
    path: Path,
    *,
    gpus: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int | None = 1,
) -> None:
    dp_line = f"data_parallel_size = {data_parallel_size}\n" if data_parallel_size is not None else ""
    path.write_text(
        f"""
schema_version = 1

[[model]]
id = "Foo/Bar"

[model.vllm]
gpus = {gpus}
tensor_parallel_size = {tensor_parallel_size}
{dp_line}
[model.vllm.serve]

[model.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )


def _bundle(
    tmp_path: Path,
    *,
    gpus: int = 1,
    tensor_parallel_size: int = 1,
    data_parallel_size: int | None = 1,
    allocated_gpus: int = 1,
):
    job_cfg = tmp_path / "job.toml"
    orchestrate_cfg = tmp_path / "orchestrate.toml"
    plan_path = tmp_path / "plan.yaml"
    _write_job_config(job_cfg)
    _write_orchestrate_config(
        orchestrate_cfg,
        gpus=gpus,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\n",
        encoding="utf-8",
    )
    tasks = expand_tasks(load_plan(plan_path))
    return tasks, ensure_run_bundle(
        tasks=tasks,
        run_id="bundle",
        output_root=tmp_path / "outputs",
        mode="slurm",
        runtime="pyxis",
        allocation_defaults={
            tasks[0].task_id: ExecutionAllocation(
                task_id=tasks[0].task_id,
                allocated_gpus=allocated_gpus,
                server_port=8000,
            )
        },
    )


def test_ensure_run_bundle_rejects_output_root_from_different_run_id(tmp_path: Path) -> None:
    tasks, _ = _bundle(tmp_path)

    with pytest.raises(ValueError, match="belongs to run_id=bundle, not fresh-run"):
        ensure_run_bundle(
            tasks=tasks,
            run_id="fresh-run",
            output_root=tmp_path / "outputs",
            mode="slurm",
            runtime="pyxis",
        )


def test_ensure_run_bundle_rejects_orphaned_task_bundle_artifacts(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    orchestrate_cfg = tmp_path / "orchestrate.toml"
    plan_path = tmp_path / "plan.yaml"
    _write_job_config(job_cfg)
    _write_orchestrate_config(orchestrate_cfg)
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\n", encoding="utf-8"
    )
    tasks = expand_tasks(load_plan(plan_path))
    orphan_root = tmp_path / "outputs" / "tasks" / "orphan-task"
    orphan_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="contains orchestrate task bundle artifacts without a run manifest"):
        ensure_run_bundle(
            tasks=tasks,
            run_id="bundle",
            output_root=tmp_path / "outputs",
            mode="slurm",
            runtime="pyxis",
        )


def test_load_task_spec_rejects_bundled_eval_config_checksum_mismatch(tmp_path: Path) -> None:
    tasks, bundle = _bundle(tmp_path)
    task_bundle = bundle.tasks[tasks[0].task_id]
    Path(task_bundle.spec.bundled_eval_config_path).write_text('model = "changed"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Bundled eval config checksum mismatch"):
        load_task_spec(Path(task_bundle.spec.output_paths.task_spec_path))


def test_worker_cli_loads_task_and_allocation(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path)
    task_bundle = bundle.tasks[tasks[0].task_id]
    captured: dict[str, object] = {}

    async def fake_run(self, *, manifest=None):
        captured["task_id"] = self._spec.task_id
        captured["server_port"] = self._allocation.server_port
        return manifest or TaskManifest(
            task_id=self._spec.task_id,
            config_path=self._spec.bundled_eval_config_path,
            model_key=self._spec.model_key,
            model_id=self._spec.model_id,
        )

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker._build_runtime_adapter", lambda runtime: object())
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.TaskWorker.run", fake_run)

    rc = worker_main(
        [
            "--task",
            task_bundle.spec.output_paths.task_spec_path,
            "--allocation",
            task_bundle.spec.output_paths.allocation_path,
            "--runtime",
            "pyxis",
            "--run-id",
            "bundle-job-foo",
            "--no-uv-run",
        ]
    )

    assert rc == 0
    assert captured == {"task_id": tasks[0].task_id, "server_port": 8000}


def test_worker_cli_persists_failed_state(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path)
    task_bundle = bundle.tasks[tasks[0].task_id]

    async def fake_run(self, *, manifest=None):
        raise RuntimeLaunchError("boom")

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker._build_runtime_adapter", lambda runtime: object())
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.TaskWorker.run", fake_run)

    rc = worker_main(
        [
            "--task",
            task_bundle.spec.output_paths.task_spec_path,
            "--allocation",
            task_bundle.spec.output_paths.allocation_path,
            "--runtime",
            "pyxis",
            "--run-id",
            "bundle-job-foo",
            "--no-uv-run",
        ]
    )

    assert rc == 1
    state_payload = Path(task_bundle.spec.output_paths.state_path).read_text(encoding="utf-8")
    result_payload = Path(task_bundle.paths.runtime_dir / "result.json").read_text(encoding="utf-8")
    assert '"state": "failed"' in state_payload
    assert '"failure_reason": "serve_launch_failed"' in result_payload


def test_worker_cli_rejects_allocation_incompatible_with_tp(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path, gpus=3, tensor_parallel_size=3, data_parallel_size=None, allocated_gpus=4)
    task_bundle = bundle.tasks[tasks[0].task_id]

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker._build_runtime_adapter", lambda runtime: object())

    rc = worker_main(
        [
            "--task",
            task_bundle.spec.output_paths.task_spec_path,
            "--allocation",
            task_bundle.spec.output_paths.allocation_path,
            "--runtime",
            "pyxis",
            "--run-id",
            "bundle-job-foo",
            "--no-uv-run",
        ]
    )

    assert rc == 1
    state_payload = Path(task_bundle.spec.output_paths.state_path).read_text(encoding="utf-8")
    result_payload = Path(task_bundle.paths.runtime_dir / "result.json").read_text(encoding="utf-8")
    assert '"state": "failed"' in state_payload
    assert '"failure_reason": "unexpected_exception"' in result_payload
    assert "allocated_gpus=4 must be divisible by tensor_parallel_size=3" in result_payload


def test_worker_cli_infers_allocated_gpus_from_visible_devices(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path, gpus=1, tensor_parallel_size=1, data_parallel_size=None, allocated_gpus=1)
    task_bundle = bundle.tasks[tasks[0].task_id]
    allocation_path = Path(task_bundle.spec.output_paths.allocation_path)
    allocation_path.write_text(
        f'{{"task_id": "{tasks[0].task_id}", "server_port": 8000, "gpu_ids": []}}',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    async def fake_run(self, *, manifest=None):
        captured["allocated_gpus"] = self._allocation.allocated_gpus
        return manifest or TaskManifest(
            task_id=self._spec.task_id,
            config_path=self._spec.bundled_eval_config_path,
            model_key=self._spec.model_key,
            model_id=self._spec.model_id,
        )

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker._build_runtime_adapter", lambda runtime: object())
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.TaskWorker.run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3,5")

    rc = worker_main(
        [
            "--task",
            task_bundle.spec.output_paths.task_spec_path,
            "--allocation",
            task_bundle.spec.output_paths.allocation_path,
            "--runtime",
            "pyxis",
            "--run-id",
            "bundle-job-foo",
            "--no-uv-run",
        ]
    )

    assert rc == 0
    assert captured["allocated_gpus"] == 4


def test_load_runtime_env_falls_back_to_huggingface_login(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path)
    task_bundle = bundle.tasks[tasks[0].task_id]
    fake_module = types.SimpleNamespace(get_token=lambda: "hf-login-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    env = _load_runtime_env(
        task_bundle.spec,
        allocation=ExecutionAllocation(task_id=tasks[0].task_id),
        options=type("Options", (), {"env_file": None})(),
    )

    assert env["HF_TOKEN"] == "hf-login-token"


def test_load_runtime_env_prefers_explicit_hf_token_over_huggingface_login(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path)
    task_bundle = bundle.tasks[tasks[0].task_id]
    fake_module = types.SimpleNamespace(get_token=lambda: "hf-login-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    env = _load_runtime_env(
        task_bundle.spec,
        allocation=ExecutionAllocation(task_id=tasks[0].task_id, runtime_env={"HF_TOKEN": "explicit-token"}),
        options=type("Options", (), {"env_file": None})(),
    )

    assert env["HF_TOKEN"] == "explicit-token"


def test_load_runtime_env_skips_huggingface_login_when_library_missing(tmp_path: Path, monkeypatch) -> None:
    tasks, bundle = _bundle(tmp_path)
    task_bundle = bundle.tasks[tasks[0].task_id]
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    original_import_module = __import__("importlib").import_module

    def fake_import_module(name: str, package=None):
        if name == "huggingface_hub":
            raise ImportError("missing")
        return original_import_module(name, package)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.importlib.import_module", fake_import_module)

    env = _load_runtime_env(
        task_bundle.spec,
        allocation=ExecutionAllocation(task_id=tasks[0].task_id),
        options=type("Options", (), {"env_file": None})(),
    )

    assert "HF_TOKEN" not in env


def test_podman_runtime_adapter_launch_builds_expected_command(monkeypatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = "container-123\n"
        stderr = ""

    def fake_run(command, check, capture_output, text):
        calls.append(command)
        return Result()

    monkeypatch.setattr("medarc_verifiers.orchestrate.podman_vllm.subprocess.run", fake_run)
    adapter = PodmanRuntimeAdapter()

    handle = adapter.launch(
        task_id="task-1",
        model_id="Foo/Bar",
        container_args=["--model", "Foo/Bar"],
        image="fake",
        container_port=8000,
        volume_mounts=["/host:/container:ro"],
        gpus_required=2,
        gpu_ids=[0, 1],
        server_port=8100,
        env={"HF_TOKEN": "secret"},
        labels={"orchestrator.task_id": "task-1"},
        name="podman-task-1",
        ipc_mode="host",
    )

    assert handle.identifier == "container-123"
    command = calls[0]
    assert command[0:4] == ["podman", "run", "--detach", "--replace"]
    assert "--publish" in command
    assert "127.0.0.1:8100:8000" in command
    assert "--volume" in command
    assert "/host:/container:ro" in command
    assert "--env" in command
    assert "HF_TOKEN=secret" in command
    assert "--label" in command
    assert "orchestrator.managed=true" in command
    assert "orchestrator.task_id=task-1" in command
    assert "--device" in command
    assert "nvidia.com/gpu=0,1" in command
    assert command[-3:] == ["fake", "--model", "Foo/Bar"]
