import asyncio
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceError
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner


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

    def cooldown_gpus(self, seconds: float = 5.0) -> None:
        return None


class FakeContainer:
    def __init__(self, name: str):
        self.id = f"fake-{name}"

    def logs(self, stream: bool, follow: bool):
        return iter(())

    def wait(self, timeout: float | None = None):
        return {"StatusCode": 0}

    def stop(self, timeout: float = 10):
        return None

    def remove(self, v: bool = True, force: bool = True):
        return None


@pytest.mark.asyncio
async def test_parallel_launch_runs_concurrently(tmp_path: Path, monkeypatch) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job-1.yaml", tmp_path / "job-2.yaml"])
    tasks = [
        TaskSpec(
            task_id="task-1",
            job_config_path=tmp_path / "job-1.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={
                "vllm-docker": {"image": "fake"},
                "foo": {"gpus": 2, "tensor_parallel_size": 2, "serve": {}},
            },
        ),
        TaskSpec(
            task_id="task-2",
            job_config_path=tmp_path / "job-2.yaml",
            model_key="foo",
            model_id="Foo/Baz",
            orchestrate={
                "vllm-docker": {"image": "fake"},
                "foo": {"gpus": 2, "tensor_parallel_size": 2, "serve": {}},
            },
        ),
    ]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=2,
    )
    runner = OrchestratorRunner(plan, tasks, DummyResourceManager(), options=options, use_dashboard=False)

    def fake_create_and_start_container(**kwargs):
        return FakeContainer(kwargs["name"])

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

    async def fake_to_thread(func, /, *args, **kwargs):
        await asyncio.sleep(0.2)
        return func(*args, **kwargs)

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

    monkeypatch.setattr("medarc_verifiers.orchestrate.run.create_and_start_container", fake_create_and_start_container)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.wait_for_readiness_async", fake_wait_for_readiness_async)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.start_benchmark", fake_start_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.wait_benchmark", fake_wait_benchmark)
    monkeypatch.setattr("medarc_verifiers.orchestrate.run.asyncio.to_thread", fake_to_thread)
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.docker_vllm.create_and_start_container",
        fake_create_and_start_container,
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.docker_vllm.wait_for_readiness_async",
        fake_wait_for_readiness_async,
    )

    await runner._run_async()

    assert readiness_overlapped
