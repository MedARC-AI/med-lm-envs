import asyncio
import sys
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.bench import start_benchmark, terminate_benchmark, wait_benchmark
from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.scheduler import TaskScheduler


class DummyResourceManager:
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
        return list(range(count))

    def reserve_port(self, task_id: str) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def release_gpus(self, indices):
        return None

    def release_port(self, port: int) -> None:
        return None


class ShutdownResourceManager:
    def __init__(self, shutdown_event: asyncio.Event) -> None:
        self._shutdown_event = shutdown_event
        self.released_gpus = False
        self.released_port = False
        self._next_port = 8100

    def reserve_gpus(
        self,
        task_id: str,
        *,
        count: int,
        min_free_gb=None,
        require_contiguous: bool = False,
    ):
        self._shutdown_event.set()
        return list(range(count))

    def reserve_port(self, task_id: str) -> int:
        port = self._next_port
        self._next_port += 1
        return port

    def release_gpus(self, indices):
        self.released_gpus = True

    def release_port(self, port: int) -> None:
        self.released_port = True


@pytest.mark.asyncio
async def test_scheduler_shutdown_stops_new_tasks(tmp_path: Path) -> None:
    tasks = [
        TaskSpec(
            task_id="task-1",
            job_config_path=tmp_path / "job-1.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 1}},
        ),
        TaskSpec(
            task_id="task-2",
            job_config_path=tmp_path / "job-2.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 1}},
        ),
    ]
    shutdown_event = asyncio.Event()
    scheduler = TaskScheduler(DummyResourceManager(), max_parallel=1)
    started: list[str] = []
    started_event = asyncio.Event()
    finish_event = asyncio.Event()

    async def runner(task: TaskSpec, allocation) -> None:
        started.append(task.task_id)
        if task.task_id == "task-1":
            started_event.set()
            await finish_event.wait()

    run_task = asyncio.create_task(scheduler.run(tasks, runner, shutdown_event=shutdown_event))
    await started_event.wait()
    shutdown_event.set()
    finish_event.set()
    await run_task

    assert started == ["task-1"]


@pytest.mark.asyncio
async def test_scheduler_shutdown_during_allocation(tmp_path: Path) -> None:
    tasks = [
        TaskSpec(
            task_id="task-1",
            job_config_path=tmp_path / "job-1.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 1}},
        )
    ]
    shutdown_event = asyncio.Event()
    resource_manager = ShutdownResourceManager(shutdown_event)
    scheduler = TaskScheduler(resource_manager, max_parallel=1)
    started = asyncio.Event()

    async def runner(task: TaskSpec, allocation) -> None:
        started.set()

    await asyncio.wait_for(
        asyncio.create_task(scheduler.run(tasks, runner, shutdown_event=shutdown_event)),
        timeout=1.0,
    )

    assert started.is_set() is False
    assert resource_manager.released_gpus is True
    assert resource_manager.released_port is True


@pytest.mark.asyncio
async def test_benchmark_termination_ends_process(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "import time; time.sleep(999)"]
    bench_proc = await start_benchmark(
        command,
        cwd=tmp_path,
        env=None,
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
    )
    await asyncio.sleep(0.2)
    await terminate_benchmark(bench_proc, term_timeout_s=1.0)
    result = await wait_benchmark(bench_proc)

    assert result.terminated is True
    assert result.exit_code != 0


@pytest.mark.asyncio
async def test_runner_shutdown_state_machine(tmp_path: Path) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.yaml"])
    tasks = [
        TaskSpec(
            task_id="task-1",
            job_config_path=tmp_path / "job-1.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 1}},
        )
    ]
    options = OrchestratorOptions(
        run_id="run-1",
        output_root=tmp_path / "outputs",
        readiness_timeout_s=1,
        max_parallel=1,
    )
    runner = OrchestratorRunner(plan, tasks, DummyResourceManager(), options=options, use_dashboard=False)
    loop = asyncio.get_running_loop()
    runner_task = asyncio.create_task(asyncio.sleep(999))

    runner._handle_shutdown(runner_task, loop)
    assert runner._shutdown.is_set() is True
    assert runner._shutdown_mode == "graceful"

    runner._handle_shutdown(runner_task, loop)
    await asyncio.sleep(0)
    assert runner._shutdown_mode == "force"
    assert runner_task.cancelled() is True
