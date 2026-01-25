"""Scheduler skeleton for orchestrator tasks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
import heapq
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
        # Order by (ready_at, priority, seq) so delays don't block ready tasks and
        # larger GPU requests are preferred among ready tasks.
        ready: list[tuple[float, int, int, TaskSpec]] = []
        blocked: list[tuple[float, int, int, TaskSpec]] = []
        sequence = 0
        for task in tasks:
            priority = self._task_priority(task)
            heapq.heappush(ready, (0.0, priority, sequence, task))
            sequence += 1

        remaining = len(ready)
        active = 0
        active_cond = asyncio.Condition()
        slot_available = asyncio.Event()
        slot_available.set()
        resources_changed = asyncio.Event()
        runner_tasks: set[asyncio.Task[None]] = set()
        blocked_cooldown_s = 0.2

        def _sync_slot_available() -> None:
            if active < self._max_parallel:
                slot_available.set()
            else:
                slot_available.clear()

        async def _wait_for_events(
            timeout: float | None = None,
            *,
            wait_for_slot: bool = False,
        ) -> None:
            waiters: list[asyncio.Task[None]] = []
            try:
                if shutdown_event:
                    waiters.append(asyncio.create_task(shutdown_event.wait()))
                waiters.append(asyncio.create_task(resources_changed.wait()))
                if wait_for_slot:
                    waiters.append(asyncio.create_task(slot_available.wait()))
                if timeout is not None:
                    waiters.append(asyncio.create_task(asyncio.sleep(timeout)))
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for waiter in waiters:
                    if not waiter.done():
                        waiter.cancel()
                if waiters:
                    await asyncio.gather(*waiters, return_exceptions=True)

        async def _runner_wrapper(task: TaskSpec, allocation: Allocation) -> None:
            nonlocal active, remaining
            try:
                try:
                    await runner(task, allocation)
                except Exception:
                    pass
            finally:
                self._release(allocation)
                async with active_cond:
                    active -= 1
                    remaining -= 1
                    _sync_slot_available()
                    active_cond.notify_all()
                # Wake the scheduler after we free a slot so it doesn't spin on a stale `active` count.
                resources_changed.set()

        def _launch_runner(task: TaskSpec, allocation: Allocation) -> None:
            nonlocal active
            active += 1
            _sync_slot_available()
            task_runner = asyncio.create_task(_runner_wrapper(task, allocation))
            runner_tasks.add(task_runner)
            task_runner.add_done_callback(runner_tasks.discard)

        async def _drain_active() -> None:
            async with active_cond:
                while active > 0:
                    await active_cond.wait()

        try:
            while True:
                # Always drain/clear resource notifications first. Otherwise, if we're at the
                # concurrency cap, we can keep waking immediately on an already-set Event and
                # starve the event loop (breaking dashboard refresh + signal handling).
                if resources_changed.is_set():
                    for _, priority, seq, task in blocked:
                        heapq.heappush(ready, (0.0, priority, seq, task))
                    blocked.clear()
                    resources_changed.clear()
                if shutdown_event and shutdown_event.is_set():
                    await _drain_active()
                    return
                if remaining == 0 and active == 0:
                    return
                if active >= self._max_parallel:
                    await _wait_for_events(wait_for_slot=True)
                    continue
                now = time.monotonic()
                if blocked:
                    still_blocked = []
                    for retry_at, priority, seq, task in blocked:
                        if retry_at <= now:
                            heapq.heappush(ready, (0.0, priority, seq, task))
                        else:
                            still_blocked.append((retry_at, priority, seq, task))
                    blocked = still_blocked
                if not ready:
                    timeout = None
                    if blocked:
                        next_retry = min(item[0] for item in blocked)
                        timeout = max(0.0, next_retry - time.monotonic())
                    await _wait_for_events(timeout=timeout)
                    continue
                ready_at, priority, seq, task = heapq.heappop(ready)
                now = time.monotonic()
                if ready_at > now:
                    heapq.heappush(ready, (ready_at, priority, seq, task))
                    await _wait_for_events(timeout=ready_at - now)
                    continue
                if shutdown_event and shutdown_event.is_set():
                    await _drain_active()
                    return
                try:
                    allocation = self._allocate(task)
                except ResourceError:
                    blocked.append(
                        (time.monotonic() + blocked_cooldown_s, priority, seq, task)
                    )
                    # Yield to event loop to allow other tasks (e.g., dashboard refresh) to run
                    await asyncio.sleep(0)
                    continue
                if shutdown_event and shutdown_event.is_set():
                    self._release(allocation)
                    await _drain_active()
                    return
                _launch_runner(task, allocation)
        except asyncio.CancelledError:
            for task in runner_tasks:
                task.cancel()
            await asyncio.gather(*runner_tasks, return_exceptions=True)
            raise

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
