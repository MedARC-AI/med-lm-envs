"""Scheduler skeleton for orchestrator tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
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

    async def run(
        self,
        tasks: Iterable[TaskSpec],
        runner: TaskRunner,
        *,
        shutdown_event: asyncio.Event | None = None,
    ) -> None:
        queue: asyncio.PriorityQueue[tuple[int, float, int, TaskSpec]] = asyncio.PriorityQueue()
        sequence = 0
        retry_counts: dict[str, int] = {}
        for task in tasks:
            priority = self._task_priority(task)
            queue.put_nowait((priority, 0.0, sequence, task))
            sequence += 1
        semaphore = asyncio.Semaphore(self._max_parallel)
        active = 0
        active_cond = asyncio.Condition()

        async def worker() -> None:
            nonlocal active
            nonlocal sequence
            while True:
                if shutdown_event and shutdown_event.is_set():
                    return
                if shutdown_event:
                    get_task = asyncio.create_task(queue.get())
                    shutdown_task = asyncio.create_task(shutdown_event.wait())
                    done, pending = await asyncio.wait(
                        {get_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    for pending_task in pending:
                        pending_task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    shutdown_requested = shutdown_task in done
                    got_task = get_task in done
                    if not got_task:
                        return
                    priority, ready_at, seq, task = get_task.result()
                    if shutdown_requested:
                        queue.put_nowait((priority, ready_at, seq, task))
                        queue.task_done()
                        return
                else:
                    priority, ready_at, seq, task = await queue.get()
                try:
                    if shutdown_event and shutdown_event.is_set():
                        queue.put_nowait((priority, ready_at, seq, task))
                        return
                    now = time.monotonic()
                    if now < ready_at:
                        await asyncio.sleep(min(0.5, ready_at - now))
                        queue.put_nowait((priority, ready_at, seq, task))
                        continue
                    async with semaphore:
                        if shutdown_event and shutdown_event.is_set():
                            queue.put_nowait((priority, ready_at, seq, task))
                            return
                        try:
                            allocation = self._allocate(task)
                        except ResourceError:
                            if shutdown_event and shutdown_event.is_set():
                                queue.put_nowait((priority, ready_at, seq, task))
                                return
                            attempts = retry_counts.get(task.task_id, 0) + 1
                            retry_counts[task.task_id] = attempts
                            backoff_s = min(10.0, 0.5 * (2 ** min(attempts - 1, 4)))
                            queue.put_nowait((priority, time.monotonic() + backoff_s, sequence, task))
                            sequence += 1
                            continue
                        async with active_cond:
                            active += 1
                        try:
                            try:
                                await runner(task, allocation)
                            except Exception:
                                pass
                        finally:
                            self._release(allocation)
                            async with active_cond:
                                active -= 1
                                active_cond.notify_all()
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(self._max_parallel)]
        if shutdown_event:
            join_task = asyncio.create_task(queue.join())
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            done, pending = await asyncio.wait(
                {join_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for pending_task in pending:
                pending_task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if shutdown_task in done:
                async with active_cond:
                    while active > 0:
                        await active_cond.wait()
        else:
            await queue.join()
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def _task_priority(self, task: TaskSpec) -> int:
        gpus_required = int(task.orchestrate.get(task.model_key, {}).get("gpus", 1))
        # Lower values are dequeued first; prioritize larger GPU requests.
        return -gpus_required

    def _allocate(self, task: TaskSpec) -> Allocation:
        model_cfg = task.orchestrate.get(task.model_key, {}) or {}
        gpus_required = int(model_cfg.get("gpus", 1))
        min_free_gb = model_cfg.get("memory_min_gb")
        require_contiguous = bool(model_cfg.get("require_contiguous_gpus", gpus_required > 1))
        gpu_ids = self._resource_manager.reserve_gpus(
            task.task_id,
            count=gpus_required,
            min_free_gb=min_free_gb,
            require_contiguous=require_contiguous,
        )
        try:
            port = self._resource_manager.reserve_port(task.task_id)
        except Exception:
            self._resource_manager.release_gpus(gpu_ids)
            raise
        return Allocation(gpu_ids=gpu_ids, port=port)

    def _release(self, allocation: Allocation) -> None:
        self._resource_manager.release_port(allocation.port)
        self._resource_manager.release_gpus(allocation.gpu_ids)


__all__ = ["Allocation", "TaskRunner", "TaskScheduler"]
