import asyncio
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceError
from medarc_verifiers.orchestrate.scheduler import TaskScheduler


class DummyResourceManager:
    def reserve_gpus(
        self,
        task_id: str,
        *,
        count: int,
        min_free_gb=None,
        require_contiguous: bool = False,
    ):
        raise ResourceError("No GPUs available.")

    def reserve_port(self, task_id: str) -> int:
        raise AssertionError("reserve_port should not be called when GPUs are unavailable.")

    def release_gpus(self, indices):
        return None

    def release_port(self, port: int) -> None:
        return None


@pytest.mark.asyncio
async def test_scheduler_cancellation_cleans_waiters(tmp_path: Path) -> None:
    tasks = [
        TaskSpec(
            task_id="task-1",
            job_config_path=tmp_path / "job-1.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 1}},
        )
    ]
    scheduler = TaskScheduler(DummyResourceManager(), max_parallel=1)

    async def runner(task: TaskSpec, allocation) -> None:
        return None

    run_task = asyncio.create_task(scheduler.run(tasks, runner))
    await asyncio.sleep(0.05)
    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task
    await asyncio.sleep(0)

    pending = [
        task
        for task in asyncio.all_tasks()
        if not task.done() and task is not asyncio.current_task()
    ]
    assert pending == []
