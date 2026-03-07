from pathlib import Path

from medarc_verifiers.orchestrate.bundle import ExecutionAllocation, ensure_run_bundle
from medarc_verifiers.orchestrate.config import expand_tasks, load_plan
from medarc_verifiers.orchestrate.state import TaskManifest
from medarc_verifiers.orchestrate.runtime import RuntimeLaunchError
from medarc_verifiers.orchestrate.podman_vllm import PodmanRuntimeAdapter
from medarc_verifiers.orchestrate.worker import main as worker_main


def _write_job_config(path: Path) -> None:
    path.write_text(
        (
            "models:\n"
            "  foo:\n"
            "    model: Foo/Bar\n"
            "orchestrate:\n"
            "  vllm-container:\n"
            "    image: fake\n"
            "  foo:\n"
            "    gpus: 1\n"
            "    tensor_parallel_size: 1\n"
            "    data_parallel_size: 1\n"
            "    serve: {}\n"
        ),
        encoding="utf-8",
    )


def _bundle(tmp_path: Path):
    job_cfg = tmp_path / "job.yaml"
    plan_path = tmp_path / "plan.yaml"
    _write_job_config(job_cfg)
    plan_path.write_text(f"job_configs:\n  - {job_cfg.name}\n", encoding="utf-8")
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
                allocated_gpus=1,
                server_port=8000,
            )
        },
    )


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
    assert "orchestrator.task_id=task-1" in command
    assert "--device" in command
    assert "nvidia.com/gpu=0,1" in command
    assert command[-3:] == ["fake", "--model", "Foo/Bar"]
