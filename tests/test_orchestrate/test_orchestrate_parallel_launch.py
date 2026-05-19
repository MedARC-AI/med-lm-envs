import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bundle import load_runtime_state
from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceError, ResourceManager
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.runtime import RuntimeHandle
from medarc_verifiers.orchestrate.slurm.plan import slug_task_id
from medarc_verifiers.orchestrate.task_naming import task_root_for_id


class DummyResourceManager:
    def __init__(self) -> None:
        self._next_port = 8000
        self._gpu_reservations: set[int] = set()

    def reserve_gpus(
        self,
        task_id: str,
        *,
        count: int,
        min_free_gb=None,
        require_contiguous: bool = False,
    ):
        free = [idx for idx in range(4) if idx not in self._gpu_reservations]
        if len(free) < count:
            raise ResourceError("Insufficient free GPUs for reservation.")
        selection = free[:count]
        self._gpu_reservations.update(selection)
        return selection

    def reserve_port(self, task_id: str) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def release_gpus(self, indices):
        for idx in indices:
            self._gpu_reservations.discard(idx)

    def release_port(self, port: int) -> None:
        return None


class PortOnlyDummyResourceManager:
    def __init__(self) -> None:
        self._next_port = 8000

    def reserve_gpus(
        self,
        task_id: str,
        *,
        count: int,
        min_free_gb=None,
        require_contiguous: bool = False,
    ):
        return []

    def reserve_port(self, task_id: str) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def release_gpus(self, indices):
        return None

    def release_port(self, port: int) -> None:
        return None


class FakeLogStreamer:
    def start(self) -> None:
        return None

    def stop(self, *, timeout: float = 2.0) -> None:
        return None

    def is_alive(self) -> bool:
        return False


class FakeRuntimeAdapter:
    def __init__(self) -> None:
        self.launch_calls: list[dict[str, object]] = []

    def launch(self, **kwargs) -> RuntimeHandle:
        self.launch_calls.append(kwargs)
        port = int(kwargs["server_port"])
        task_id = str(kwargs["task_id"])
        return RuntimeHandle(base_url=f"http://127.0.0.1:{port}/v1", identifier=f"handle-{task_id}")

    def stream_logs(self, handle: RuntimeHandle, sink: Path) -> FakeLogStreamer:
        return FakeLogStreamer()

    def teardown(self, handle: RuntimeHandle) -> None:
        return None


def _task(
    tmp_path: Path,
    task_id: str,
    *,
    gpus: int = 2,
    tensor_parallel_size: int = 2,
    data_parallel_size: int = 1,
    async_scheduling: bool = False,
    gpu_memory_utilization: float | None = None,
) -> TaskSpec:
    job_config_path = tmp_path / f"{task_id}.toml"
    job_config_path.write_text(
        f"""
model = "Foo/{task_id}"

[[eval]]
env_id = "medqa"
num_examples = 1
rollouts_per_example = 1
""".lstrip(),
        encoding="utf-8",
    )
    serve_args: dict[str, object] = {}
    if async_scheduling:
        serve_args["async_scheduling"] = True
    if gpu_memory_utilization is not None:
        serve_args["gpu_memory_utilization"] = gpu_memory_utilization
    return TaskSpec(
        task_id=task_id,
        job_config_path=job_config_path,
        model_key=task_id.replace("/", "-"),
        model_id=f"Foo/{task_id}",
        orchestrate={
            "container": {"image": "fake"},
            "vllm": {
                "gpus": gpus,
                "tensor_parallel_size": tensor_parallel_size,
                "data_parallel_size": data_parallel_size,
                "serve": serve_args,
            },
        },
    )


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "resource_manager"),
    [("docker", DummyResourceManager()), ("podman", DummyResourceManager()), ("pyxis", PortOnlyDummyResourceManager())],
)
async def test_parallel_launch_runs_concurrently(
    tmp_path: Path,
    monkeypatch,
    runtime: str,
    resource_manager: ResourceManager,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job-1.toml", tmp_path / "job-2.toml"])
    tasks = [_task(tmp_path, "task-1"), _task(tmp_path, "task-2")]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=2,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        resource_manager,
        options=options,
        runtime=runtime,
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    first_readiness_started = asyncio.Event()
    first_readiness_done = asyncio.Event()
    readiness_overlapped = False

    async def fake_wait_for_readiness_async(*args, **kwargs):
        nonlocal readiness_overlapped
        await asyncio.sleep(0.2)
        if not first_readiness_started.is_set():
            first_readiness_started.set()
            await asyncio.sleep(0.2)
            first_readiness_done.set()
        else:
            if not first_readiness_done.is_set():
                readiness_overlapped = True
            await asyncio.sleep(0.2)

        class Result:
            ready = True
            elapsed_s = 0.2
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    assert readiness_overlapped
    assert [call["server_port"] for call in adapter.launch_calls] == [8000, 8001]


@pytest.mark.asyncio
async def test_parallel_launch_records_gpu_accounting_and_dp_args(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    tasks = [_task(tmp_path, "task-1", gpus=8, tensor_parallel_size=2, data_parallel_size=4)]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
        allocated_gpu_count=8,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        PortOnlyDummyResourceManager(),
        options=options,
        runtime="pyxis",
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)
    await runner._run_async()

    launch = adapter.launch_calls[0]
    assert launch["gpus_required"] == 8
    assert launch["container_args"] == [
        "--model",
        "Foo/task-1",
        "--tensor-parallel-size",
        "2",
        "--data-parallel-size",
        "4",
        "--gpu-memory-utilization",
        "0.87",
    ]

    manifest = json.loads((options.output_root / "tasks" / "task-1" / "runtime" / "task_manifest.json").read_text())
    assert manifest["bench_run_id"] is None
    assert manifest["allocated_gpus"] == 8
    assert manifest["gpus"] == 8
    assert manifest["tensor_parallel_size"] == 2
    assert manifest["data_parallel_size"] == 4
    assert manifest["vllm_world_size"] == 8
    assert manifest["allocated_gpu_hours"] is not None
    assert manifest["gpu_hours"] is not None


@pytest.mark.asyncio
async def test_parallel_launch_does_not_duplicate_task_slug_in_bench_run_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task = _task(tmp_path, "task-1", gpus=1, tensor_parallel_size=1, data_parallel_size=1)
    plan = PlanConfig(job_configs=[task.job_config_path])
    options = OrchestratorOptions(
        run_id="run-1-task-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        [task],
        DummyResourceManager(),
        options=options,
        runtime="docker",
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    manifest = json.loads((options.output_root / "tasks" / "task-1" / "runtime" / "task_manifest.json").read_text())
    assert manifest["bench_run_id"] is None


@pytest.mark.asyncio
async def test_parallel_launch_does_not_duplicate_hashed_slurm_task_slug_in_bench_run_id(
    tmp_path: Path,
    monkeypatch,
) -> None:
    task_id = "job-qwen-3.0_6b-thinking:qwen-3.0_6b-thinking"
    task = _task(tmp_path, task_id, gpus=1, tensor_parallel_size=1, data_parallel_size=1)
    slurm_style_run_id = f"qwen-small-test-20260315-171932-{slug_task_id(task_id)}"
    plan = PlanConfig(job_configs=[task.job_config_path])
    options = OrchestratorOptions(
        run_id=slurm_style_run_id,
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        [task],
        DummyResourceManager(),
        options=options,
        runtime="docker",
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    manifest = json.loads(
        (task_root_for_id(options.output_root, task.task_id) / "runtime" / "task_manifest.json").read_text()
    )
    assert manifest["bench_run_id"] is None


@pytest.mark.asyncio
async def test_parallel_launch_disables_async_scheduling_when_dp_is_active(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    tasks = [_task(tmp_path, "task-1", gpus=8, tensor_parallel_size=2, data_parallel_size=4, async_scheduling=True)]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
        allocated_gpu_count=8,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        PortOnlyDummyResourceManager(),
        options=options,
        runtime="pyxis",
        runtime_adapter=adapter,
        use_dashboard=False,
    )
    log_messages: list[str] = []

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    original_log = runner._dashboard.log
    runner._dashboard.log = lambda message: log_messages.append(message)
    try:
        await runner._run_async()
    finally:
        runner._dashboard.log = original_log

    launch = adapter.launch_calls[0]
    assert "--async-scheduling" not in launch["container_args"]
    assert any("async_scheduling_disabled=true" in message for message in log_messages)
    assert any("reason=data_parallel_size_gt_1" in message for message in log_messages)

    request_payload = json.loads(
        (options.output_root / "tasks" / "task-1" / "serve" / "container_create_request.json").read_text()
    )
    assert "--async-scheduling" not in request_payload["command"]


@pytest.mark.asyncio
async def test_parallel_launch_adjusts_gpu_memory_utilization_for_dp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    tasks = [
        _task(tmp_path, "task-1", gpus=8, tensor_parallel_size=2, data_parallel_size=4, gpu_memory_utilization=0.9)
    ]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
        allocated_gpu_count=8,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        PortOnlyDummyResourceManager(),
        options=options,
        runtime="pyxis",
        runtime_adapter=adapter,
        use_dashboard=False,
    )
    log_messages: list[str] = []

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    original_log = runner._dashboard.log
    runner._dashboard.log = lambda message: log_messages.append(message)
    try:
        await runner._run_async()
    finally:
        runner._dashboard.log = original_log

    launch = adapter.launch_calls[0]
    assert "--gpu-memory-utilization" in launch["container_args"]
    idx = launch["container_args"].index("--gpu-memory-utilization")
    assert launch["container_args"][idx + 1] == "0.87"
    assert any("gpu_memory_utilization_adjusted=true" in message for message in log_messages)

    request_payload = json.loads(
        (options.output_root / "tasks" / "task-1" / "serve" / "container_create_request.json").read_text()
    )
    idx = request_payload["command"].index("--gpu-memory-utilization")
    assert request_payload["command"][idx + 1] == "0.87"


@pytest.mark.asyncio
async def test_parallel_launch_defaults_gpu_memory_utilization_for_dp_when_unset(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    tasks = [_task(tmp_path, "task-1", gpus=8, tensor_parallel_size=2, data_parallel_size=4)]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
        allocated_gpu_count=8,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        PortOnlyDummyResourceManager(),
        options=options,
        runtime="pyxis",
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    launch = adapter.launch_calls[0]
    assert "--gpu-memory-utilization" in launch["container_args"]
    idx = launch["container_args"].index("--gpu-memory-utilization")
    assert launch["container_args"][idx + 1] == "0.87"


@pytest.mark.asyncio
async def test_parallel_launch_keeps_gpu_memory_utilization_unchanged_without_dp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    tasks = [
        _task(tmp_path, "task-1", gpus=1, tensor_parallel_size=1, data_parallel_size=1, gpu_memory_utilization=0.9)
    ]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        DummyResourceManager(),
        options=options,
        runtime="docker",
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    launch = adapter.launch_calls[0]
    idx = launch["container_args"].index("--gpu-memory-utilization")
    assert launch["container_args"][idx + 1] == "0.9"


@pytest.mark.asyncio
async def test_parallel_launch_keeps_async_scheduling_when_dp_is_one(
    tmp_path: Path,
    monkeypatch,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    tasks = [_task(tmp_path, "task-1", gpus=1, tensor_parallel_size=1, data_parallel_size=1, async_scheduling=True)]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        tasks,
        DummyResourceManager(),
        options=options,
        runtime="docker",
        runtime_adapter=adapter,
        use_dashboard=False,
    )

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.1
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(*args, **kwargs):
        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    launch = adapter.launch_calls[0]
    assert "--async-scheduling" in launch["container_args"]


@pytest.mark.asyncio
async def test_runner_uses_task_local_bench_output_without_restart_flags(tmp_path: Path, monkeypatch) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"])
    endpoints_path = tmp_path / "endpoints with spaces.json"
    task = replace(
        _task(tmp_path, "task-1", gpus=1, tensor_parallel_size=1, data_parallel_size=1),
        endpoints_path=endpoints_path,
    )
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs with spaces",
        readiness_timeout_s=1,
        max_parallel=1,
    )
    adapter = FakeRuntimeAdapter()
    runner = OrchestratorRunner(
        plan,
        [task],
        PortOnlyDummyResourceManager(),
        options=options,
        runtime="pyxis",
        runtime_adapter=adapter,
        use_dashboard=False,
    )
    captured: dict[str, object] = {}

    async def fake_wait_for_readiness_async(*args, **kwargs):
        class Result:
            ready = True
            elapsed_s = 0.2
            attempts = 1
            last_error = None

        return Result()

    async def fake_start_benchmark(command, **kwargs):
        captured["command"] = command

        class Proc:
            pass

        return Proc()

    async def fake_wait_benchmark(proc):
        class Result:
            exit_code = 0
            duration_s = 0.0
            terminated = False

        return Result()

    async def fake_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.worker.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    command = captured["command"]
    assert "--config" in command
    assert str(options.output_root / "tasks" / "task-1" / "eval-config.toml") in command
    assert "--output-dir" in command
    assert str(options.output_root / "tasks" / "task-1" / "bench") in command
    assert "--endpoints-path" in command
    assert str(endpoints_path) in command
    assert "--restart" not in command
    assert "--run-id" not in command
    assert "--on-complete" not in command

    bundled_payload = (options.output_root / "tasks" / "task-1" / "eval-config.toml").read_text(encoding="utf-8")
    assert "restart" not in bundled_payload
    manifest = json.loads((options.output_root / "tasks" / "task-1" / "runtime" / "task_manifest.json").read_text())
    assert manifest["bench_run_dir"] is None
    assert manifest["restart_source"] is None
    state = load_runtime_state(options.output_root / "tasks" / "task-1" / "runtime" / "state.json")
    assert state is not None
    assert state.restart_source is None
