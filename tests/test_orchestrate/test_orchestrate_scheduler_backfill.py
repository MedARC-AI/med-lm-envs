import asyncio
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceError
from medarc_verifiers.orchestrate.scheduler import TaskScheduler


class DummyResourceManager:
    def __init__(self, total_gpus: int = 4) -> None:
        self._total_gpus = total_gpus
        self._next_port = 9000
        self._gpu_reservations: set[int] = set()

    def reserve_gpus(
        self,
        task_id: str,
        *,
        count: int,
        min_free_gb=None,
        require_contiguous: bool = False,
    ):
        free = [idx for idx in range(self._total_gpus) if idx not in self._gpu_reservations]
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


@pytest.mark.asyncio
async def test_scheduler_backfill_prefers_large_then_small(tmp_path: Path) -> None:
    tasks = [
        TaskSpec(
            task_id="task-2gpu",
            job_config_path=tmp_path / "job-2gpu.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 2}},
        ),
        TaskSpec(
            task_id="task-1gpu-a",
            job_config_path=tmp_path / "job-1gpu-a.yaml",
            model_key="foo",
            model_id="Foo/Baz",
            orchestrate={"foo": {"gpus": 1}},
        ),
        TaskSpec(
            task_id="task-1gpu-b",
            job_config_path=tmp_path / "job-1gpu-b.yaml",
            model_key="foo",
            model_id="Foo/Qux",
            orchestrate={"foo": {"gpus": 1}},
        ),
    ]
    scheduler = TaskScheduler(DummyResourceManager(), max_parallel=4)
    started: list[str] = []
    start_events = {task.task_id: asyncio.Event() for task in tasks}
    finish_event = asyncio.Event()

    async def runner(task: TaskSpec, allocation) -> None:
        started.append(task.task_id)
        start_events[task.task_id].set()
        await finish_event.wait()

    run_task = asyncio.create_task(scheduler.run(tasks, runner))
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in start_events.values())),
        timeout=1.0,
    )
    finish_event.set()
    await run_task

    assert started[0] == "task-2gpu"
    assert set(started) == {"task-2gpu", "task-1gpu-a", "task-1gpu-b"}
