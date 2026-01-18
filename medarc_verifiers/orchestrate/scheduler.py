"""Scheduler skeleton for orchestrator tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable

from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import ResourceManager, ResourceError


@dataclass(frozen=True)
class Allocation:
    gpu_ids: list[int]
    port: int


TaskRunner = Callable[[TaskSpec, Allocation], Awaitable[None]]


class TaskScheduler:
    """Queue-based scheduler with a concurrency cap."""

    def __init__(self, resource_manager: ResourceManager, *, max_parallel: int = 1) -> None:
        self._resource_manager = resource_manager
        self._max_parallel = max_parallel

    async def run(self, tasks: Iterable[TaskSpec], runner: TaskRunner) -> None:
        queue: asyncio.Queue[TaskSpec] = asyncio.Queue()
        for task in tasks:
            queue.put_nowait(task)
        semaphore = asyncio.Semaphore(self._max_parallel)

        async def worker() -> None:
            while True:
                task = await queue.get()
                try:
                    async with semaphore:
                        try:
                            allocation = self._allocate(task)
                        except ResourceError:
                            await asyncio.sleep(1.0)
                            queue.put_nowait(task)
                            continue
                        try:
                            try:
                                await runner(task, allocation)
                            except Exception:
                                pass
                        finally:
                            self._release(allocation)
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self._max_parallel)]
        await queue.join()
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def _allocate(self, task: TaskSpec) -> Allocation:
        gpus_required = int(task.vllm.get(task.model_key, {}).get("gpus", 1))
        min_free_gb = task.vllm.get(task.model_key, {}).get("memory_min_gb")
        gpu_ids = self._resource_manager.reserve_gpus(task.task_id, count=gpus_required, min_free_gb=min_free_gb)
        try:
            port = self._resource_manager.reserve_port(task.task_id)
        except Exception:
            self._resource_manager.release_gpus(gpu_ids)
            raise
        return Allocation(gpu_ids=gpu_ids, port=port)

    def _release(self, allocation: Allocation) -> None:
        self._resource_manager.release_port(allocation.port)
        self._resource_manager.cooldown_gpus()
        self._resource_manager.release_gpus(allocation.gpu_ids)


__all__ = ["Allocation", "TaskRunner", "TaskScheduler"]
