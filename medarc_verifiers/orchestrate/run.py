"""Orchestrator runtime loop wiring scheduler, docker, readiness, and bench."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Iterable
import json
import shlex
import signal

from dotenv import dotenv_values

from medarc_verifiers.orchestrate.bench import (
    BenchProcess,
    render_command,
    start_benchmark,
    terminate_benchmark,
    wait_benchmark,
)
from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.dashboard import ACTIVE_STATES, OrchestratorDashboard
from medarc_verifiers.orchestrate.docker_vllm import (
    build_container_args,
    ContainerLogStreamer,
    create_and_start_container,
    DockerLaunchError,
    sanitize_container_name,
    wait_for_readiness_async,
    write_container_request,
)
from medarc_verifiers.orchestrate.resources import ResourceManager
from medarc_verifiers.orchestrate.scheduler import Allocation, TaskScheduler
from medarc_verifiers.orchestrate.state import JobState, TaskManifest, TaskPaths, write_summary, write_task_manifest, write_task_result, write_text


COMMAND_TEMPLATE = (
    "uv run medarc-eval bench --config {job_config_path} --api-base-url {base_url} --on-complete exit"
)

_TASK_DIR_ALLOWED = re.compile(r"[^a-zA-Z0-9_.-]+")


def _shorten(text: str, *, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    suffix = "…"
    keep = max(0, max_len - len(suffix))
    return f"{text[:keep]}{suffix}"


def _sanitize_task_dirname(task_id: str, *, max_len: int = 120) -> str:
    cleaned = _TASK_DIR_ALLOWED.sub("-", task_id).strip("-.")
    if not cleaned:
        cleaned = "task"
    if cleaned != task_id:
        suffix = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]  # noqa: S324
        cleaned = f"{cleaned}-{suffix}"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-.")
    return cleaned or "task"


def _task_root_for_id(output_root: Path, task_id: str) -> Path:
    raw = output_root / task_id
    if raw.exists():
        return raw
    sanitized = output_root / _sanitize_task_dirname(task_id)
    if sanitized.exists():
        return sanitized
    return sanitized


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
        use_dashboard: bool = True,
    ) -> None:
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
        self._dashboard = OrchestratorDashboard(enabled=use_dashboard)
        self._manifests: dict[str, TaskManifest] = {}
        self._active_containers: dict[str, object] = {}
        self._bench_processes: dict[str, BenchProcess] = {}
        self._log_streamers: dict[str, ContainerLogStreamer] = {}
        self._active_runner_tasks: dict[str, asyncio.Task] = {}
        self._shutdown = asyncio.Event()
        self._shutdown_mode: str | None = None
        self._shutdown_requested_at: str | None = None
        self._run_started_at: str | None = None
        self._dashboard_refresh_task: asyncio.Task[None] | None = None
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
            f"max_parallel={self._options.max_parallel} output={self._options.output_root}"
        )
        write_summary(self._options.output_root / "summary.json", list(self._manifests.values()))
        try:
            loop = asyncio.get_running_loop()
            runner_task = asyncio.create_task(
                scheduler.run(self._tasks, self._run_task, shutdown_event=self._shutdown)
            )
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
        paths = TaskPaths(_task_root_for_id(self._options.output_root, task.task_id))
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
                    if isinstance(exc, DockerLaunchError):
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
        self._set_state(manifest, paths, JobState.allocating)

        orchestrate = task.orchestrate
        docker_cfg = _get_mapping(orchestrate.get("vllm-docker"), "orchestrate.vllm-docker")
        model_cfg = _get_mapping(orchestrate.get(task.model_key), f"orchestrate.{task.model_key}")
        container_port = int(docker_cfg.get("container_port", 8000))
        ipc_mode = docker_cfg.get("ipc_mode")
        image = str(docker_cfg.get("image", "")).strip()
        if not image:
            raise RuntimeError(f"Missing orchestrate.vllm-docker.image for {task.job_config_path}")
        manifest.image = image

        tensor_parallel = model_cfg.get("tensor_parallel_size")
        gpus_required = int(model_cfg.get("gpus", 1))
        if gpus_required > 1:
            if not tensor_parallel:
                raise RuntimeError(f"orchestrate.{task.model_key}.tensor_parallel_size is required for multi-GPU.")
            if int(tensor_parallel) != gpus_required:
                raise RuntimeError("gpus must match tensor_parallel_size for multi-GPU models.")
        if gpus_required == 1 and tensor_parallel and int(tensor_parallel) > 1:
            raise RuntimeError("tensor_parallel_size > 1 is invalid for single-GPU models.")

        serve = _get_mapping(model_cfg.get("serve"), f"orchestrate.{task.model_key}.serve")
        container_args = build_container_args(
            task.model_id, tensor_parallel_size=int(tensor_parallel) if tensor_parallel else None, serve=serve
        )
        env: dict[str, str] = {}
        repo_root = Path(__file__).resolve().parents[2]
        if self._plan.env_file is not None:
            env.update(_load_env_file(self._plan.env_file, base_dir=repo_root))
        else:
            default_env = repo_root / ".env"
            if default_env.exists():
                env.update(_load_env_file(default_env, base_dir=repo_root))
        env.update(_load_env_file(docker_cfg.get("env_file"), base_dir=task.job_config_path.parent))
        volumes = docker_cfg.get("volumes", []) or []
        labels = {"orchestrator.run_id": self._options.run_id, "orchestrator.task_id": task.task_id}
        container_name = sanitize_container_name(f"vllm-{self._options.run_id}-{task.task_id}")
        manifest.container_name = container_name

        request_payload = {
            "image": image,
            "name": container_name,
            "container_port": container_port,
            "host_port": allocation.port,
            "ipc_mode": ipc_mode,
            "volumes": volumes,
            "env": sorted(env.keys()),
            "gpu_ids": allocation.gpu_ids,
            "command": container_args,
            "labels": labels,
        }
        write_container_request(str(paths.container_request_path), request_payload)

        self._set_state(manifest, paths, JobState.launching)
        try:
            container = await asyncio.to_thread(
                create_and_start_container,
                image=image,
                name=container_name,
                container_port=container_port,
                host_port=allocation.port,
                env=env,
                volumes=volumes,
                ipc_mode=ipc_mode,
                gpu_ids=allocation.gpu_ids,
                command=container_args,
                labels=labels,
            )
        except DockerLaunchError:
            raise
        except Exception as exc:
            raise DockerLaunchError(str(exc)) from exc
        manifest.container_id = container.id
        self._active_containers[task.task_id] = container

        log_streamer = ContainerLogStreamer(container, str(paths.container_logs_path))
        log_streamer.start()
        self._log_streamers[task.task_id] = log_streamer
        base_url = f"http://127.0.0.1:{allocation.port}/v1"
        completed_successfully = False
        try:
            self._set_state(manifest, paths, JobState.loading)
            readiness = await wait_for_readiness_async(
                base_url,
                model_id=task.model_id,
                timeout_s=self._options.readiness_timeout_s,
            )
            manifest.readiness = readiness.__dict__
            write_text(paths.readiness_path, json.dumps(readiness.__dict__, indent=2))
            if not readiness.ready:
                manifest.failure_reason = "readiness_timeout"
                manifest.error = readiness.last_error
                write_task_result(
                    paths,
                    {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error},
                )
                self._set_state(manifest, paths, JobState.failed)
                return
            loading_elapsed = _format_elapsed(manifest.state_entered_at, _utcnow())
            self._dashboard.log(
                f"JOB ready task={task.task_id} attempts={readiness.attempts} loading_elapsed={loading_elapsed}"
            )

            command_context = {
                "base_url": base_url,
                "host_port": str(allocation.port),
                "model_key": task.model_key,
                "model_id": task.model_id,
                "output_dir": str(paths.bench_dir),
                "run_id": self._options.run_id,
                "task_id": task.task_id,
                "job_config_path": str(task.job_config_path),
            }
            command = render_command(COMMAND_TEMPLATE, command_context)
            restart_source = orchestrate.get("restart")
            if restart_source:
                restart_value = str(restart_source)
                if "--restart" not in command:
                    command.extend(["--restart", restart_value])
            manifest.bench_command = shlex.join(command)
            self._dashboard.log(
                f"JOB bench-start task={task.task_id} cmd={_shorten(manifest.bench_command)}"
            )
            self._set_state(manifest, paths, JobState.running)
            bench_proc = await start_benchmark(
                command,
                cwd=repo_root,
                env=None,
                stdout_path=paths.stdout_path,
                stderr_path=paths.stderr_path,
            )
            self._bench_processes[task.task_id] = bench_proc
            bench_result = await wait_benchmark(bench_proc)
            self._bench_processes.pop(task.task_id, None)
            manifest.bench_exit_code = bench_result.exit_code
            manifest.bench_duration_s = bench_result.duration_s
            if bench_result.terminated:
                self._dashboard.log(
                    f"JOB bench-terminated task={task.task_id} duration={bench_result.duration_s:.1f}s"
                )
            elif bench_result.exit_code == 0:
                self._dashboard.log(
                    f"JOB bench-ok task={task.task_id} duration={bench_result.duration_s:.1f}s"
                )
            else:
                self._dashboard.log(
                    f"JOB bench-failed task={task.task_id} exit={bench_result.exit_code} "
                    f"duration={bench_result.duration_s:.1f}s"
                )

            result_payload = {
                "exit_code": bench_result.exit_code,
                "duration_s": bench_result.duration_s,
                "state": JobState.cancelled if bench_result.terminated else (
                    JobState.completed if bench_result.exit_code == 0 else JobState.failed
                ),
                "command": manifest.bench_command,
                "argv": list(command),
                "terminated": bench_result.terminated,
            }
            write_task_result(paths, result_payload)

            if bench_result.terminated:
                manifest.failure_reason = "bench_terminated"
                self._set_state(manifest, paths, JobState.cancelled)
            elif bench_result.exit_code != 0:
                manifest.failure_reason = "bench_exit_nonzero"
                self._set_state(manifest, paths, JobState.failed)
            else:
                self._set_state(manifest, paths, JobState.completed)
                completed_successfully = True
        finally:
            await _teardown_container(container, manifest)
            log_streamer = self._log_streamers.pop(task.task_id, None)
            if log_streamer:
                await asyncio.to_thread(log_streamer.stop)
            self._active_containers.pop(task.task_id, None)
            if completed_successfully and self._options.prune_logs_on_success:
                self._prune_task_logs(paths)

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
        write_task_manifest(paths, manifest)
        write_summary(self._options.output_root / "summary.json", list(self._manifests.values()))
        if state != prev_state:
            self._log_state_transition(manifest, prev_state, state, prev_state_entered_at, now)
        self._dashboard.update(self._manifests.values(), caption=self._dashboard_caption())

    def _get_or_init_manifest(self, task: TaskSpec, allocation: Allocation) -> TaskManifest:
        manifest = self._manifests.get(task.task_id)
        if not manifest:
            manifest = TaskManifest(
                task_id=task.task_id,
                config_path=str(task.job_config_path),
                model_key=task.model_key,
                model_id=task.model_id,
            )
            self._manifests[task.task_id] = manifest
        manifest.gpu_ids = allocation.gpu_ids
        manifest.port = allocation.port
        if manifest.started_at is None:
            manifest.started_at = _utcnow()
        return manifest

    def _mark_cancelled(self) -> None:
        for task_id, manifest in self._manifests.items():
            if manifest.state in {JobState.completed, JobState.failed, JobState.cancelled, JobState.pending}:
                continue
            manifest.failure_reason = manifest.failure_reason or "cancelled"
            paths = TaskPaths(_task_root_for_id(self._options.output_root, task_id))
            self._set_state(manifest, paths, JobState.cancelled)

    async def _teardown_active(self) -> None:
        for container in list(self._active_containers.values()):
            try:
                await _teardown_container(container)
            except Exception:
                continue
        self._active_containers.clear()

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
                "note=\"no new jobs will start\""
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
            f"active={active} benches={len(self._bench_processes)} containers={len(self._active_containers)}"
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
                config_path=str(task.job_config_path),
                model_key=task.model_key,
                model_id=task.model_id,
            )

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
                f"JOB start task={manifest.task_id} model={manifest.model_key} "
                f"gpus={gpu_text} port={port_text}"
            )
            return
        if state == JobState.completed:
            exit_code = manifest.bench_exit_code
            exit_text = str(exit_code) if exit_code is not None else "-"
            self._dashboard.log(
                f"JOB complete task={manifest.task_id} exit={exit_text} total_elapsed={total_elapsed}"
            )
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


def _load_env_file(path: object, *, base_dir: Path) -> dict[str, str]:
    if not path:
        return {}
    env_path = Path(str(path)).expanduser()
    if not env_path.is_absolute():
        env_path = (base_dir / env_path).resolve()
    if not env_path.exists():
        raise DockerLaunchError(
            f"env_file not found: {env_path} (set orchestrate.vllm-docker.env_file relative to {base_dir})"
        )
    values = dotenv_values(env_path)
    return {key: value for key, value in values.items() if value is not None}


async def _teardown_container(container, manifest: TaskManifest | None = None) -> None:
    try:
        exit_status = await asyncio.to_thread(container.wait, timeout=1)
        if manifest and isinstance(exit_status, dict):
            manifest.container_exit_code = exit_status.get("StatusCode")
    except Exception:
        pass
    await asyncio.to_thread(container.stop, timeout=10)
    await asyncio.to_thread(container.remove, v=True, force=True)


def _register_signal_handlers(loop: asyncio.AbstractEventLoop, handler) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handler)
        except NotImplementedError:
            continue


def _is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if isinstance(exc, DockerLaunchError):
        return (
            "port" in message
            or "bind" in message
            or "address already in use" in message
            or "read timed out" in message
            or "timeout" in message
            or "timed out" in message
        )
    return "connection reset" in message or "read timed out" in message or "timeout" in message or "timed out" in message


__all__ = ["OrchestratorOptions", "OrchestratorRunner"]
