import asyncio
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceError, ResourceManager
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.runtime import RuntimeHandle


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


def _task(tmp_path: Path, task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        job_config_path=tmp_path / f"{task_id}.yaml",
        model_key="foo",
        model_id=f"Foo/{task_id}",
        orchestrate={
            "restart": "runs/raw/old-run",
            "vllm-container": {"image": "fake"},
            "foo": {"gpus": 2, "tensor_parallel_size": 2, "serve": {}},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("runtime", "resource_manager"),
    [("docker", DummyResourceManager()), ("pyxis", PortOnlyDummyResourceManager())],
)
async def test_parallel_launch_runs_concurrently(
    tmp_path: Path,
    monkeypatch,
    runtime: str,
    resource_manager: ResourceManager,
) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job-1.yaml", tmp_path / "job-2.yaml"])
    tasks = [_task(tmp_path, "task-1"), _task(tmp_path, "task-2")]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=2,
    )
    adapter = FakeRuntimeAdapter()
    bench_commands: list[list[str]] = []
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
        bench_commands.append(list(args[0]))

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

    monkeypatch.setattr("medarc_verifiers.orchestrate.run.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run._register_signal_handlers", lambda loop, handler: None)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.asyncio.to_thread", fake_to_thread)

    await runner._run_async()

    assert readiness_overlapped
    assert [call["server_port"] for call in adapter.launch_calls] == [8000, 8001]
    assert all("--api-base-url" in command for command in bench_commands)
    assert all("--provider" in command and "local" in command for command in bench_commands)
    assert all("--on-complete" not in command for command in bench_commands)
    assert all("--restart" not in command for command in bench_commands)
