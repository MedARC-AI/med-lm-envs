import asyncio
import time
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceError
from medarc_verifiers.orchestrate.scheduler import TaskScheduler


class DummyResourceManager:
    def __init__(self, allow_large: asyncio.Event, total_gpus: int = 4) -> None:
        self._allow_large = allow_large
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
        if count == self._total_gpus and not self._allow_large.is_set():
            raise ResourceError("Large reservation blocked.")
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
async def test_scheduler_retries_blocked_task_on_release(tmp_path: Path) -> None:
    allow_large = asyncio.Event()
    tasks = [
        TaskSpec(
            task_id="task-4gpu",
            job_config_path=tmp_path / "job-4gpu.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            orchestrate={"foo": {"gpus": 4}},
        ),
        TaskSpec(
            task_id="task-1gpu",
            job_config_path=tmp_path / "job-1gpu.yaml",
            model_key="foo",
            model_id="Foo/Baz",
            orchestrate={"foo": {"gpus": 1}},
        ),
    ]
    scheduler = TaskScheduler(DummyResourceManager(allow_large), max_parallel=4)
    one_started = asyncio.Event()
    four_started = asyncio.Event()
    allow_release = asyncio.Event()
    release_time: float | None = None
    four_started_time: float | None = None

    async def runner(task: TaskSpec, allocation) -> None:
        nonlocal release_time, four_started_time
        if task.task_id == "task-1gpu":
            one_started.set()
            await allow_release.wait()
            allow_large.set()
            release_time = time.monotonic()
            return
        four_started_time = time.monotonic()
        four_started.set()

    run_task = asyncio.create_task(scheduler.run(tasks, runner))
    await asyncio.wait_for(one_started.wait(), timeout=1.0)
    await asyncio.sleep(0.05)
    assert four_started.is_set() is False
    allow_release.set()
    await asyncio.wait_for(four_started.wait(), timeout=1.0)
    await run_task

    assert release_time is not None
    assert four_started_time is not None
    assert (four_started_time - release_time) < 0.5
