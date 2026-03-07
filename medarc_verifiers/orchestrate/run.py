"""Orchestrator runtime loop wiring scheduler, runtime adapters, readiness, and bench."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values

from medarc_verifiers.orchestrate.bench import (
    BenchProcess,
    terminate_benchmark,
)
from medarc_verifiers.orchestrate.bundle import (
    ExecutionAllocation,
    PlannedTaskBundle,
    RuntimeState,
    ensure_run_bundle,
    write_execution_allocation,
    write_runtime_state,
)
from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec, load_job_config
from medarc_verifiers.orchestrate.dashboard import ACTIVE_STATES, OrchestratorDashboard
from medarc_verifiers.orchestrate.resources import ResourceManager
from medarc_verifiers.orchestrate.runtime import LogStreamer, RuntimeAdapter, RuntimeHandle, RuntimeLaunchError
from medarc_verifiers.orchestrate.scheduler import Allocation, TaskScheduler
from medarc_verifiers.orchestrate.state import (
    JobState,
    TaskManifest,
    TaskPaths,
    write_summary,
    write_task_manifest,
    write_task_result,
    write_text,
)
from medarc_verifiers.orchestrate.task_naming import task_root_for_id
from medarc_verifiers.orchestrate.worker import TaskWorker, WorkerCallbacks, WorkerOptions

_COMMAND_TEMPLATE_UV = (
    "uv run medarc-eval bench --config {job_config_path} --api-base-url {base_url} "
    "--run-id {bench_run_id} --on-complete exit"
)
_COMMAND_TEMPLATE_BARE = (
    "medarc-eval bench --config {job_config_path} --api-base-url {base_url} "
    "--run-id {bench_run_id} --on-complete exit"
)
_DEFAULT_BENCH_OUTPUT_DIR = Path("runs") / "raw"

def _shorten(text: str, *, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    suffix = "…"
    keep = max(0, max_len - len(suffix))
    return f"{text[:keep]}{suffix}"


@dataclass(frozen=True)
class OrchestratorOptions:
    run_id: str
    output_root: Path
    readiness_timeout_s: int
    max_parallel: int
    prune_logs_on_success: bool = False


class OrchestratorRunner:
    def __init__(
        self,
        plan: PlanConfig,
        tasks: Iterable[TaskSpec],
        resource_manager: ResourceManager,
        *,
        options: OrchestratorOptions,
        runtime: str = "docker",
        runtime_adapter: RuntimeAdapter | None = None,
        uv_run: bool = True,
        use_dashboard: bool = True,
    ) -> None:
        self._runtime = _normalize_runtime(runtime or plan.runtime or "docker")
        self._plan = plan
        self._tasks = sorted(
            list(tasks),
            key=lambda task: int(
                _get_mapping(task.orchestrate.get(task.model_key), f"orchestrate.{task.model_key}").get("gpus", 1)
            ),
            reverse=True,
        )
        self._resource_manager = resource_manager
        self._options = options
        self._runtime_adapter = runtime_adapter or _build_runtime_adapter(self._runtime)
        self._command_template = _COMMAND_TEMPLATE_UV if uv_run else _COMMAND_TEMPLATE_BARE
        self._dashboard = OrchestratorDashboard(enabled=use_dashboard)
        self._manifests: dict[str, TaskManifest] = {}
        self._active_handles: dict[str, RuntimeHandle] = {}
        self._bench_processes: dict[str, BenchProcess] = {}
        self._log_streamers: dict[str, LogStreamer] = {}
        self._active_runner_tasks: dict[str, asyncio.Task] = {}
        self._shutdown = asyncio.Event()
        self._shutdown_mode: str | None = None
        self._shutdown_requested_at: str | None = None
        self._run_started_at: str | None = None
        self._dashboard_refresh_task: asyncio.Task[None] | None = None
        self._bundle_plan = ensure_run_bundle(
            tasks=self._tasks,
            run_id=self._options.run_id,
            output_root=self._options.output_root,
            mode="local",
            runtime=self._runtime,
        )
        self._init_manifests(self._tasks)

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        scheduler = TaskScheduler(self._resource_manager, max_parallel=self._options.max_parallel)
        self._run_started_at = _utcnow()
        self._dashboard.start()
        self._dashboard.update(self._manifests.values(), caption=self._dashboard_caption())
        self._dashboard_refresh_task = self._start_dashboard_refresh()
        self._dashboard.log(
            f"RUN started run_id={self._options.run_id} tasks={len(self._manifests)} "
            f"runtime={self._runtime} max_parallel={self._options.max_parallel} output={self._options.output_root}"
        )
        write_summary(self._options.output_root / "summary.json", list(self._manifests.values()))
        try:
            loop = asyncio.get_running_loop()
            runner_task = asyncio.create_task(scheduler.run(self._tasks, self._run_task, shutdown_event=self._shutdown))
            _register_signal_handlers(loop, lambda: self._handle_shutdown(runner_task, loop))
            try:
                await runner_task
            except asyncio.CancelledError:
                if self._shutdown_mode != "force":
                    raise
        finally:
            if self._dashboard_refresh_task:
                self._dashboard_refresh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._dashboard_refresh_task
                self._dashboard_refresh_task = None
            self._dashboard.stop()
            if self._shutdown.is_set():
                await self._teardown_active()
                if self._shutdown_mode == "force":
                    self._mark_cancelled()

    async def _run_task(self, task: TaskSpec, allocation: Allocation) -> None:
        current = asyncio.current_task()
        if current:
            self._active_runner_tasks[task.task_id] = current
        manifest = self._get_or_init_manifest(task, allocation)
        bundle = self._bundle_for_task(task)
        paths = TaskPaths(bundle.paths.root)
        write_execution_allocation(
            bundle.paths.allocation_path,
            ExecutionAllocation(
                task_id=task.task_id,
                allocated_gpus=_allocated_gpu_count(allocation, task),
                gpu_ids=list(allocation.gpu_ids),
                server_port=allocation.server_port,
                require_contiguous_gpus=bool(
                    _get_mapping(task.orchestrate.get(task.model_key), f"orchestrate.{task.model_key}").get(
                        "require_contiguous_gpus",
                        bool(allocation.gpu_ids and len(allocation.gpu_ids) > 1),
                    )
                ),
            ),
        )
        attempt = 0
        try:
            while True:
                attempt += 1
                try:
                    await self._run_task_once(task, allocation, manifest, paths)
                    return
                except asyncio.CancelledError:
                    manifest.failure_reason = "cancelled"
                    self._set_state(manifest, paths, JobState.cancelled)
                    raise
                except Exception as exc:
                    manifest.error = str(exc)
                    if attempt == 1 and _is_transient_error(exc):
                        write_text(
                            paths.serve_dir / f"launch_error_attempt{attempt}.txt",
                            f"attempt={attempt}\nstate={manifest.state}\nat={_utcnow()}\nerror={manifest.error}\n",
                        )
                        await asyncio.sleep(5)
                        continue
                    if isinstance(exc, RuntimeLaunchError):
                        manifest.failure_reason = "serve_launch_failed"
                    else:
                        manifest.failure_reason = "unexpected_exception"
                    self._set_state(manifest, paths, JobState.failed)
                    write_task_result(
                        paths,
                        {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error},
                    )
                    raise
        finally:
            self._active_runner_tasks.pop(task.task_id, None)

    async def _run_task_once(
        self, task: TaskSpec, allocation: Allocation, manifest: TaskManifest, paths: TaskPaths
    ) -> None:
        bundle = self._bundle_for_task(task)
        worker = TaskWorker(
            bundle.spec,
            ExecutionAllocation(
                task_id=task.task_id,
                allocated_gpus=_allocated_gpu_count(allocation, task),
                gpu_ids=list(allocation.gpu_ids),
                server_port=allocation.server_port,
                require_contiguous_gpus=bool(
                    _get_mapping(task.orchestrate.get(task.model_key), f"orchestrate.{task.model_key}").get(
                        "require_contiguous_gpus",
                        bool(allocation.gpu_ids and len(allocation.gpu_ids) > 1),
                    )
                ),
            ),
            options=WorkerOptions(
                run_id=self._options.run_id,
                runtime=self._runtime,
                readiness_timeout_s=self._options.readiness_timeout_s,
                command_template=self._command_template,
                env_file=self._plan.env_file,
                prune_logs_on_success=self._options.prune_logs_on_success,
            ),
            runtime_adapter=self._runtime_adapter,
            callbacks=WorkerCallbacks(
                register_handle=lambda task_id, handle: self._active_handles.__setitem__(task_id, handle),
                unregister_handle=lambda task_id: self._active_handles.pop(task_id, None),
                register_bench=lambda task_id, process: self._bench_processes.__setitem__(task_id, process),
                unregister_bench=lambda task_id: self._bench_processes.pop(task_id, None),
                register_log_streamer=lambda task_id, streamer: self._log_streamers.__setitem__(task_id, streamer),
                unregister_log_streamer=lambda task_id: self._log_streamers.pop(task_id, None),
            ),
        )
        await worker.run(manifest=manifest, state_handler=self._set_state, log=self._dashboard.log)

    def _prune_task_logs(self, paths: TaskPaths) -> None:
        for log_path in (paths.container_logs_path, paths.stdout_path, paths.stderr_path):
            try:
                log_path.unlink(missing_ok=True)
            except Exception:
                continue

    def _set_state(self, manifest: TaskManifest, paths: TaskPaths, state: str) -> None:
        prev_state = manifest.state
        prev_state_entered_at = manifest.state_entered_at
        now = _utcnow()
        if state != prev_state:
            manifest.state = state
            manifest.state_entered_at = now
        if state in {JobState.completed, JobState.failed, JobState.cancelled}:
            manifest.completed_at = now
            _update_gpu_accounting(manifest)
        write_task_manifest(paths, manifest)
        write_runtime_state(
            paths.state_path,
            RuntimeState(
                task_id=manifest.task_id,
                state=manifest.state,
                restart_source=manifest.restart_source,
                restart_source_strategy="runtime_state" if manifest.restart_source else "none",
                bench_run_id=manifest.bench_run_id,
                bench_run_dir=manifest.bench_run_dir,
            ),
        )
        write_summary(self._options.output_root / "summary.json", list(self._manifests.values()))
        if state != prev_state:
            self._log_state_transition(manifest, prev_state, state, prev_state_entered_at, now)
        self._dashboard.update(self._manifests.values(), caption=self._dashboard_caption())

    def _get_or_init_manifest(self, task: TaskSpec, allocation: Allocation) -> TaskManifest:
        manifest = self._manifests.get(task.task_id)
        if not manifest:
            manifest = TaskManifest(
                task_id=task.task_id,
                config_path=self._bundle_for_task(task).spec.bundled_eval_config_path,
                model_key=task.model_key,
                model_id=task.model_id,
            )
            self._manifests[task.task_id] = manifest
        manifest.gpu_ids = allocation.gpu_ids
        manifest.port = allocation.server_port
        manifest.allocated_gpu_count = _allocated_gpu_count(allocation, task)
        manifest.effective_gpu_count = int(
            _get_mapping(task.orchestrate.get(task.model_key), f"orchestrate.{task.model_key}").get("gpus", 1)
        )
        if manifest.started_at is None:
            manifest.started_at = _utcnow()
        return manifest

    def _mark_cancelled(self) -> None:
        for task_id, manifest in self._manifests.items():
            if manifest.state in {JobState.completed, JobState.failed, JobState.cancelled, JobState.pending}:
                continue
            manifest.failure_reason = manifest.failure_reason or "cancelled"
            paths = TaskPaths(task_root_for_id(self._options.output_root, task_id))
            self._set_state(manifest, paths, JobState.cancelled)

    async def _teardown_active(self) -> None:
        for handle in list(self._active_handles.values()):
            try:
                await _teardown_runtime(self._runtime_adapter, handle)
            except Exception:
                continue
        self._active_handles.clear()

    async def _force_shutdown(self) -> None:
        for task_id, bench_proc in list(self._bench_processes.items()):
            try:
                await terminate_benchmark(bench_proc)
            except Exception:
                continue
        for task_id, log_streamer in list(self._log_streamers.items()):
            try:
                await asyncio.to_thread(log_streamer.stop)
            except Exception:
                continue
        self._log_streamers.clear()
        await self._teardown_active()

    def _handle_shutdown(self, runner_task: asyncio.Task, loop: asyncio.AbstractEventLoop) -> None:
        if not self._shutdown.is_set():
            self._shutdown_mode = "graceful"
            self._shutdown_requested_at = self._shutdown_requested_at or _utcnow()
            active = self._count_active()
            pending = self._count_pending()
            shutdown_elapsed = _format_elapsed(self._shutdown_requested_at, _utcnow())
            self._dashboard.log(
                "SHUTDOWN graceful requested (press Ctrl+C again to force) "
                f"active={active} pending={pending} shutdown_elapsed={shutdown_elapsed} "
                'note="no new jobs will start"'
            )
            self._shutdown.set()
            self._dashboard.update(self._manifests.values(), caption=self._dashboard_caption())
            return
        if self._shutdown_mode == "force":
            return
        self._shutdown_mode = "force"
        active = self._count_active()
        self._dashboard.log(
            "SHUTDOWN force requested "
            f"active={active} benches={len(self._bench_processes)} handles={len(self._active_handles)}"
        )
        runner_task.cancel()
        for task in list(self._active_runner_tasks.values()):
            task.cancel()
        loop.create_task(self._force_shutdown())
        self._dashboard.update(self._manifests.values(), caption=self._dashboard_caption())

    def _init_manifests(self, tasks: Iterable[TaskSpec]) -> None:
        for task in tasks:
            if task.task_id in self._manifests:
                continue
            self._manifests[task.task_id] = TaskManifest(
                task_id=task.task_id,
                config_path=self._bundle_for_task(task).spec.bundled_eval_config_path,
                model_key=task.model_key,
                model_id=task.model_id,
            )

    def _bundle_for_task(self, task: TaskSpec) -> PlannedTaskBundle:
        return self._bundle_plan.tasks[task.task_id]

    def _count_active(self) -> int:
        return sum(1 for task in self._manifests.values() if task.state in ACTIVE_STATES)

    def _count_pending(self) -> int:
        return sum(1 for task in self._manifests.values() if task.state == JobState.pending)

    def _dashboard_caption(self) -> str | None:
        if not self._run_started_at:
            return None
        uptime = _format_elapsed(self._run_started_at, _utcnow())
        mode = self._shutdown_mode or "running"
        return f"uptime={uptime} mode={mode}"

    def _start_dashboard_refresh(self) -> asyncio.Task[None] | None:
        if not self._dashboard.enabled:
            return None

        async def refresh_loop() -> None:
            refresh_hz = float(getattr(self._dashboard, "refresh_hz", 1.0) or 1.0)
            interval_s = 1.0 / max(0.1, refresh_hz)
            while True:
                await asyncio.sleep(interval_s)
                try:
                    self._dashboard.update(self._manifests.values(), caption=self._dashboard_caption())
                except Exception as exc:  # noqa: BLE001
                    self._dashboard.log(f"RUN dashboard-refresh failed error={exc!r}")

        return asyncio.create_task(refresh_loop())

    def _log_state_transition(
        self,
        manifest: TaskManifest,
        prev_state: str,
        state: str,
        prev_state_entered_at: str | None,
        now: str,
    ) -> None:
        total_elapsed = _format_elapsed(manifest.started_at, now)
        if state == JobState.allocating:
            gpu_text = ",".join(str(gpu) for gpu in manifest.gpu_ids or []) or "-"
            port_text = str(manifest.port) if manifest.port is not None else "-"
            self._dashboard.log(
                f"JOB start task={manifest.task_id} model={manifest.model_key} gpus={gpu_text} port={port_text}"
            )
            return
        if state == JobState.completed:
            exit_code = manifest.bench_exit_code
            exit_text = str(exit_code) if exit_code is not None else "-"
            self._dashboard.log(f"JOB complete task={manifest.task_id} exit={exit_text} total_elapsed={total_elapsed}")
            return
        if state == JobState.failed:
            reason = manifest.failure_reason or "unknown"
            error = f" error={manifest.error!r}" if manifest.error else ""
            self._dashboard.log(
                f"JOB failed task={manifest.task_id} reason={reason} total_elapsed={total_elapsed}{error}"
            )
            return
        if state == JobState.cancelled:
            self._dashboard.log(
                f"JOB cancelled task={manifest.task_id} at_state={prev_state} total_elapsed={total_elapsed}"
            )
            return


def _resolve_tensor_parallel_size(
    model_cfg: dict[str, object],
    *,
    gpus_required: int,
    data_parallel_size: int,
    label: str,
) -> int:
    if data_parallel_size < 1:
        raise RuntimeError(f"{label}.data_parallel_size must be >= 1.")
    tensor_parallel = model_cfg.get("tensor_parallel_size")
    if tensor_parallel is not None:
        resolved = int(tensor_parallel)
    else:
        if gpus_required < 1:
            raise RuntimeError(f"{label}.gpus must be >= 1.")
        if gpus_required % data_parallel_size != 0:
            raise RuntimeError(
                f"{label}.gpus={gpus_required} must be divisible by data_parallel_size={data_parallel_size}."
            )
        resolved = gpus_required // data_parallel_size
    if resolved < 1:
        raise RuntimeError(f"{label}.tensor_parallel_size must be >= 1.")
    return resolved


def _allocated_gpu_count(allocation: Allocation, task: TaskSpec) -> int:
    override = os.environ.get("MEDARC_ALLOCATED_GPU_COUNT")
    if override is not None:
        return int(override)
    if allocation.gpu_ids:
        return len(allocation.gpu_ids)
    model_cfg = _get_mapping(task.orchestrate.get(task.model_key), f"orchestrate.{task.model_key}")
    return int(model_cfg.get("gpus", 1))


def _update_gpu_accounting(manifest: TaskManifest) -> None:
    started = _parse_time(manifest.started_at)
    completed = _parse_time(manifest.completed_at)
    if started is None or completed is None:
        return
    elapsed_hours = max(0.0, (completed - started).total_seconds() / 3600.0)
    if manifest.allocated_gpu_count is not None:
        manifest.allocated_gpu_hours = manifest.allocated_gpu_count * elapsed_hours
    if manifest.effective_gpu_count is not None:
        manifest.effective_gpu_hours = manifest.effective_gpu_count * elapsed_hours


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _format_elapsed(started_at: str | None, completed_at: str | None) -> str:
    start = _parse_time(started_at)
    if not start:
        return "-"
    end = _parse_time(completed_at)
    if not end:
        from datetime import datetime, timezone

        end = datetime.now(timezone.utc)
    elapsed = end - start
    total_seconds = int(elapsed.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _resolve_bench_output_dir(*, job_config_path: Path, repo_root: Path) -> Path:
    payload = dict(load_job_config(job_config_path))
    configured = payload.get("output_dir")
    output_dir = Path(configured) if configured is not None else _DEFAULT_BENCH_OUTPUT_DIR
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    return output_dir.resolve()


async def _discover_bench_run_dir(*, job_config_path: Path, repo_root: Path, timeout_s: float) -> Path | None:
    runs_root = _resolve_bench_output_dir(job_config_path=job_config_path, repo_root=repo_root)
    target_source = str(job_config_path.expanduser().resolve())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(timeout_s, 0.0)
    while True:
        candidate = _find_matching_bench_run_dir(runs_root=runs_root, config_source=target_source)
        if candidate is not None:
            return candidate
        if loop.time() >= deadline:
            return None
        await asyncio.sleep(0.5)


def _find_matching_bench_run_dir(*, runs_root: Path, config_source: str) -> Path | None:
    if not runs_root.exists():
        return None
    latest: tuple[str, Path] | None = None
    for child in runs_root.iterdir():
        manifest_path = child / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if str(payload.get("config_source") or "") != config_source:
            continue
        created_at = str(payload.get("created_at") or "")
        candidate = (created_at, child.resolve())
        if latest is None or candidate > latest:
            latest = candidate
    return latest[1] if latest is not None else None


def _format_restart_source(run_dir: Path, *, repo_root: Path) -> str:
    try:
        return str(run_dir.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(run_dir.resolve())


def _parse_time(value: str | None):
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _get_mapping(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a mapping.")
    return value


def sanitize_container_name(value: str, *, max_len: int = 128) -> str:
    from medarc_verifiers.orchestrate.docker_vllm import sanitize_container_name as _sanitize_container_name

    return _sanitize_container_name(value, max_len=max_len)


async def wait_for_readiness_async(
    base_url: str,
    *,
    model_id: str | None = None,
    timeout_s: float = 1800,
    poll_interval_s: float = 5.0,
):
    from medarc_verifiers.orchestrate.docker_vllm import wait_for_readiness_async as _wait_for_readiness_async

    return await _wait_for_readiness_async(
        base_url,
        model_id=model_id,
        timeout_s=timeout_s,
        poll_interval_s=poll_interval_s,
    )


def write_container_request(path: str, payload: dict[str, object]) -> None:
    from medarc_verifiers.orchestrate.docker_vllm import write_container_request as _write_container_request

    _write_container_request(path, payload)


def _get_optional_mapping(value: object, label: str) -> dict:
    if value is None:
        return {}
    return _get_mapping(value, label)


def _load_env_file(path: object, *, base_dir: Path) -> dict[str, str]:
    if not path:
        return {}
    env_path = Path(str(path)).expanduser()
    if not env_path.is_absolute():
        env_path = (base_dir / env_path).resolve()
    if not env_path.exists():
        raise RuntimeLaunchError(
            f"env_file not found: {env_path} (set orchestrate.vllm-container.env_file relative to {base_dir})"
        )
    values = dotenv_values(env_path)
    return {key: value for key, value in values.items() if value is not None}


async def _teardown_runtime(
    runtime_adapter: RuntimeAdapter,
    handle: RuntimeHandle,
    manifest: TaskManifest | None = None,
) -> None:
    del manifest
    await asyncio.to_thread(runtime_adapter.teardown, handle)


def _register_signal_handlers(loop: asyncio.AbstractEventLoop, handler) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handler)
        except NotImplementedError:
            continue


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if isinstance(exc, RuntimeLaunchError):
        return (
            "port" in message
            or "bind" in message
            or "address already in use" in message
            or "read timed out" in message
            or "timeout" in message
            or "timed out" in message
        )
    return (
        "connection reset" in message or "read timed out" in message or "timeout" in message or "timed out" in message
    )


def _normalize_runtime(value: str) -> str:
    runtime = str(value).strip().lower()
    if runtime not in {"docker", "podman", "pyxis"}:
        raise ValueError(f"Unsupported runtime {value!r}; expected 'docker', 'podman', or 'pyxis'.")
    return runtime


def _build_runtime_adapter(runtime: str) -> RuntimeAdapter:
    if runtime == "docker":
        from medarc_verifiers.orchestrate.docker_vllm import DockerRuntimeAdapter

        return DockerRuntimeAdapter()
    if runtime == "podman":
        from medarc_verifiers.orchestrate.podman_vllm import PodmanRuntimeAdapter

        return PodmanRuntimeAdapter()
    if runtime == "pyxis":
        from medarc_verifiers.orchestrate.pyxis_vllm import PyxisRuntimeAdapter

        return PyxisRuntimeAdapter()
    raise ValueError(f"Unsupported runtime {runtime!r}.")


__all__ = ["OrchestratorOptions", "OrchestratorRunner"]
