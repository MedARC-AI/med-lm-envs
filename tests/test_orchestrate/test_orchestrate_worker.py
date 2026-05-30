import asyncio
import sys
import threading
import types
from pathlib import Path

from medarc_verifiers.orchestrate.podman_vllm import PodmanRuntimeAdapter
from medarc_verifiers.orchestrate.runtime import RuntimeHandle, RuntimeLaunchError
from medarc_verifiers.orchestrate.state import TaskManifest
from medarc_verifiers.orchestrate.topology import ResolvedTopology
from medarc_verifiers.orchestrate.worker import (
    LaunchInputs,
    TaskWorker,
    WorkerOptions,
    _effective_serve_args,
    _parse_vllm_image_version,
    main as launch_main,
)


def _launch_args(tmp_path: Path) -> list[str]:
    task_root = tmp_path / "task"
    eval_config = task_root / "eval-config.toml"
    eval_config.parent.mkdir(parents=True)
    eval_config.write_text("[[eval]]\nenv_id = \"medqa\"\n", encoding="utf-8")
    return [
        "--task-id",
        "foo-medqa",
        "--model",
        "Foo/Bar",
        "--endpoint-id",
        "foo",
        "--image",
        "fake",
        "--gpus",
        "1",
        "--runtime",
        "pyxis",
        "--runtime-dir",
        str(task_root / "runtime"),
        "--serve-dir",
        str(task_root / "serve"),
        "--ready-file",
        str(task_root / "runtime" / "ready.json"),
        "--host-port",
        "18734",
        "--",
        "medarc-eval",
        "bench",
        "--config",
        str(eval_config),
        "--api-base-url",
        "http://127.0.0.1:18734/v1",
        "--provider",
        "local",
        "--output-dir",
        str(task_root / "bench"),
    ]


def test_launch_cli_uses_explicit_inputs_and_literal_bench_argv(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run(self, *, manifest=None):
        captured["launch"] = self._launch
        return manifest or TaskManifest(
            task_id=self._launch.task_id,
            config_path=str(tmp_path / "task" / "eval-config.toml"),
            model_key=self._launch.endpoint_id,
            model_id=self._launch.model_id,
        )

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.build_runtime_adapter", lambda runtime: object())
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.TaskWorker.run", fake_run)

    rc = launch_main(_launch_args(tmp_path))

    assert rc == 0
    launch = captured["launch"]
    assert isinstance(launch, LaunchInputs)
    assert launch.task_id == "foo-medqa"
    assert launch.model_id == "Foo/Bar"
    assert launch.endpoint_id == "foo"
    assert launch.host_port == 18734
    assert launch.bench_argv[:2] == ("medarc-eval", "bench")
    assert "--config" in launch.bench_argv


def test_launch_cli_persists_failed_state_without_task_or_allocation(tmp_path: Path, monkeypatch) -> None:
    async def fake_run(self, *, manifest=None):
        raise RuntimeLaunchError("boom")

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.build_runtime_adapter", lambda runtime: object())
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.TaskWorker.run", fake_run)

    rc = launch_main(_launch_args(tmp_path))

    assert rc == 1
    state_payload = (tmp_path / "task" / "runtime" / "state.json").read_text(encoding="utf-8")
    result_payload = (tmp_path / "task" / "runtime" / "result.json").read_text(encoding="utf-8")
    assert '"state": "failed"' in state_payload
    assert '"failure_reason": "serve_launch_failed"' in result_payload


def test_launch_cli_rejects_incompatible_explicit_topology(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.build_runtime_adapter", lambda runtime: object())
    args = _launch_args(tmp_path)
    args[args.index("--gpus") + 1] = "3"
    bench_separator = args.index("--")
    args[bench_separator:bench_separator] = ["--tensor-parallel-size", "3"]

    rc = launch_main(args)

    assert rc == 1
    result_payload = (tmp_path / "task" / "runtime" / "result.json").read_text(encoding="utf-8")
    assert '"failure_reason": "unexpected_exception"' in result_payload


def test_launch_cli_infers_allocated_gpus_from_visible_devices(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run(self, *, manifest=None):
        captured["allocated_gpus"] = self._launch.allocated_gpus
        return manifest or TaskManifest(
            task_id=self._launch.task_id,
            config_path=str(tmp_path / "task" / "eval-config.toml"),
            model_key=self._launch.endpoint_id,
            model_id=self._launch.model_id,
        )

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.build_runtime_adapter", lambda runtime: object())
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.TaskWorker.run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,3,5")

    rc = launch_main(_launch_args(tmp_path))

    assert rc == 0
    assert captured["allocated_gpus"] == 4


def test_task_worker_shutdown_before_launch_skips_runtime(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    class Streamer:
        def start(self) -> None:
            calls.append("stream-start")

        def stop(self, *, timeout: float = 2.0) -> None:
            calls.append("stream-stop")

        def is_alive(self) -> bool:
            return False

    class Runtime:
        def launch(self, **_kwargs):
            calls.append("launch")
            return RuntimeHandle(base_url="http://127.0.0.1:9999/v1", identifier="serve-1")

        def stream_logs(self, handle, sink):
            calls.append(f"stream:{handle.identifier}")
            return Streamer()

        def teardown(self, handle) -> None:
            calls.append(f"teardown:{handle.identifier}")

    async def fail_readiness(*_args, **_kwargs):
        raise AssertionError("readiness should not run after shutdown was requested")

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fail_readiness)
    task_root = tmp_path / "task"
    launch = LaunchInputs(
        task_id="foo-medqa",
        model_id="Foo/Bar",
        endpoint_id="foo",
        image="fake",
        gpus=1,
        runtime_dir=task_root / "runtime",
        serve_dir=task_root / "serve",
        ready_file=task_root / "runtime" / "ready.json",
        bench_argv=("medarc-eval", "bench"),
    )
    worker = TaskWorker(launch, options=WorkerOptions(runtime="pyxis", readiness_timeout_s=60), runtime_adapter=Runtime())

    async def run_worker() -> None:
        await worker.request_shutdown()
        await worker.run()

    asyncio.run(run_worker())

    assert calls == []
    assert '"state": "cancelled"' in (task_root / "runtime" / "state.json").read_text(encoding="utf-8")
    assert '"failure_reason": "shutdown_requested"' in (task_root / "runtime" / "result.json").read_text(
        encoding="utf-8"
    )


def test_task_worker_shutdown_during_runtime_launch_records_cancelled(tmp_path: Path, monkeypatch) -> None:
    launch_started = threading.Event()
    release_launch = threading.Event()

    class Runtime:
        def launch(self, **_kwargs):
            launch_started.set()
            release_launch.wait(timeout=1)
            return RuntimeHandle(base_url="http://127.0.0.1:9999/v1", identifier="serve-1")

        def stream_logs(self, handle, sink):
            raise AssertionError("logs should not start when launch shutdown grace expires")

        def teardown(self, handle) -> None:
            raise AssertionError("no handle is available before launch returns")

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.LAUNCH_SHUTDOWN_GRACE_S", 0.01)
    task_root = tmp_path / "task"
    launch = LaunchInputs(
        task_id="foo-medqa",
        model_id="Foo/Bar",
        endpoint_id="foo",
        image="fake",
        gpus=1,
        runtime_dir=task_root / "runtime",
        serve_dir=task_root / "serve",
        ready_file=task_root / "runtime" / "ready.json",
        bench_argv=("medarc-eval", "bench"),
    )
    worker = TaskWorker(launch, options=WorkerOptions(runtime="pyxis", readiness_timeout_s=60), runtime_adapter=Runtime())

    async def run_worker() -> None:
        run_task = asyncio.create_task(worker.run())
        await asyncio.to_thread(launch_started.wait, 1)
        await worker.request_shutdown()
        await run_task

    try:
        asyncio.run(run_worker())
    finally:
        release_launch.set()

    assert '"state": "cancelled"' in (task_root / "runtime" / "state.json").read_text(encoding="utf-8")
    assert '"failure_reason": "shutdown_requested"' in (task_root / "runtime" / "result.json").read_text(
        encoding="utf-8"
    )


def test_task_worker_shutdown_during_stuck_runtime_launch_returns_before_thread_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    launch_started = threading.Event()
    release_launch = threading.Event()

    class Runtime:
        def launch(self, **_kwargs):
            launch_started.set()
            release_launch.wait()
            return RuntimeHandle(base_url="http://127.0.0.1:9999/v1", identifier="serve-1")

        def stream_logs(self, handle, sink):
            raise AssertionError("logs should not start when launch shutdown grace expires")

        def teardown(self, handle) -> None:
            pass

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.LAUNCH_SHUTDOWN_GRACE_S", 0.01)
    task_root = tmp_path / "task"
    launch = LaunchInputs(
        task_id="foo-medqa",
        model_id="Foo/Bar",
        endpoint_id="foo",
        image="fake",
        gpus=1,
        runtime_dir=task_root / "runtime",
        serve_dir=task_root / "serve",
        ready_file=task_root / "runtime" / "ready.json",
        bench_argv=("medarc-eval", "bench"),
    )
    worker = TaskWorker(launch, options=WorkerOptions(runtime="pyxis", readiness_timeout_s=60), runtime_adapter=Runtime())

    async def run_worker() -> None:
        run_task = asyncio.create_task(worker.run())
        await asyncio.to_thread(launch_started.wait, 1)
        await worker.request_shutdown()
        await asyncio.wait_for(run_task, timeout=1)

    try:
        asyncio.run(run_worker())
    finally:
        release_launch.set()

    assert '"state": "cancelled"' in (task_root / "runtime" / "state.json").read_text(encoding="utf-8")
    assert '"failure_reason": "shutdown_requested"' in (task_root / "runtime" / "result.json").read_text(
        encoding="utf-8"
    )


def test_task_worker_shutdown_during_runtime_launch_error_records_cancelled(tmp_path: Path) -> None:
    launch_started = threading.Event()
    release_launch = threading.Event()

    class Runtime:
        def launch(self, **_kwargs):
            launch_started.set()
            release_launch.wait(timeout=1)
            raise RuntimeLaunchError("allocation went away")

        def stream_logs(self, handle, sink):
            raise AssertionError("logs should not start when launch fails during shutdown")

        def teardown(self, handle) -> None:
            raise AssertionError("no handle is available when launch raises")

    task_root = tmp_path / "task"
    launch = LaunchInputs(
        task_id="foo-medqa",
        model_id="Foo/Bar",
        endpoint_id="foo",
        image="fake",
        gpus=1,
        runtime_dir=task_root / "runtime",
        serve_dir=task_root / "serve",
        ready_file=task_root / "runtime" / "ready.json",
        bench_argv=("medarc-eval", "bench"),
    )
    worker = TaskWorker(launch, options=WorkerOptions(runtime="pyxis", readiness_timeout_s=60), runtime_adapter=Runtime())

    async def run_worker() -> None:
        run_task = asyncio.create_task(worker.run())
        await asyncio.to_thread(launch_started.wait, 1)
        await worker.request_shutdown()
        release_launch.set()
        await run_task

    asyncio.run(run_worker())

    assert '"state": "cancelled"' in (task_root / "runtime" / "state.json").read_text(encoding="utf-8")
    assert '"failure_reason": "shutdown_requested"' in (task_root / "runtime" / "result.json").read_text(
        encoding="utf-8"
    )


def test_effective_serve_args_keeps_async_scheduling_for_latest_vllm_image() -> None:
    topology = ResolvedTopology(gpus=2, allocated_gpus=4, tensor_parallel_size=2, data_parallel_size=2, vllm_world_size=4)

    serve_args = _effective_serve_args(
        {"async_scheduling": True, "gpu_memory_utilization": 0.9},
        topology=topology,
        image="vllm/vllm-openai:latest",
    )

    assert serve_args["async_scheduling"] is True


def test_effective_serve_args_disables_async_scheduling_for_old_vllm_image() -> None:
    topology = ResolvedTopology(gpus=2, allocated_gpus=4, tensor_parallel_size=2, data_parallel_size=2, vllm_world_size=4)

    serve_args = _effective_serve_args(
        {"async_scheduling": True, "gpu_memory_utilization": 0.9},
        topology=topology,
        image="vllm/vllm-openai:v0.19.2",
    )

    assert serve_args["async_scheduling"] is False


def test_effective_serve_args_keeps_async_scheduling_for_vllm_0_20_and_newer() -> None:
    topology = ResolvedTopology(gpus=2, allocated_gpus=4, tensor_parallel_size=2, data_parallel_size=2, vllm_world_size=4)

    serve_args = _effective_serve_args(
        {"async_scheduling": True, "gpu_memory_utilization": 0.9},
        topology=topology,
        image="vllm/vllm-openai:v0.20.0",
    )

    assert serve_args["async_scheduling"] is True


def test_parse_vllm_image_version_ignores_unversioned_materialized_image() -> None:
    assert _parse_vllm_image_version("/data/pyxis/vllm/latest.sqsh") is None


def test_load_explicit_runtime_env_falls_back_to_huggingface_login(tmp_path: Path, monkeypatch) -> None:
    from medarc_verifiers.orchestrate.env import load_explicit_runtime_env

    fake_module = types.SimpleNamespace(get_token=lambda: "hf-login-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    env = load_explicit_runtime_env(repo_root=tmp_path)

    assert env["HF_TOKEN"] == "hf-login-token"
    assert env["SAFETENSORS_FAST_GPU"] == "1"


def test_load_explicit_runtime_env_keeps_explicit_safetensors_override(tmp_path: Path) -> None:
    from medarc_verifiers.orchestrate.env import load_explicit_runtime_env

    env = load_explicit_runtime_env(repo_root=tmp_path, env_overrides={"SAFETENSORS_FAST_GPU": "0"})

    assert env["SAFETENSORS_FAST_GPU"] == "0"


def test_load_explicit_runtime_env_prefers_explicit_hf_token_over_huggingface_login(tmp_path: Path, monkeypatch) -> None:
    from medarc_verifiers.orchestrate.env import load_explicit_runtime_env

    fake_module = types.SimpleNamespace(get_token=lambda: "hf-login-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    env = load_explicit_runtime_env(repo_root=tmp_path, env_overrides={"HF_TOKEN": "explicit-token"})

    assert env["HF_TOKEN"] == "explicit-token"


def test_load_explicit_runtime_env_skips_huggingface_login_when_library_missing(tmp_path: Path, monkeypatch) -> None:
    from medarc_verifiers.orchestrate.env import load_explicit_runtime_env

    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    original_import_module = __import__("importlib").import_module

    def fake_import_module(name: str, package=None):
        if name == "huggingface_hub":
            raise ImportError("missing")
        return original_import_module(name, package)

    monkeypatch.setattr("medarc_verifiers.orchestrate.env.importlib.import_module", fake_import_module)

    env = load_explicit_runtime_env(repo_root=tmp_path)

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
