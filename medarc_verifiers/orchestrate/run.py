"""Orchestrator runtime loop wiring scheduler, docker, readiness, and bench."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
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
from medarc_verifiers.orchestrate.dashboard import OrchestratorDashboard
from medarc_verifiers.orchestrate.docker_vllm import (
    build_container_args,
    create_and_start_container,
    DockerLaunchError,
    sanitize_container_name,
    stream_container_logs,
    wait_for_readiness,
    write_container_request,
)
from medarc_verifiers.orchestrate.resources import ResourceManager
from medarc_verifiers.orchestrate.scheduler import Allocation, TaskScheduler
from medarc_verifiers.orchestrate.state import JobState, TaskManifest, TaskPaths, write_summary, write_task_manifest, write_task_result, write_text


DEFAULT_COMMAND_TEMPLATE = (
    "uv run medarc-eval bench --config {job_config_path} --api-base-url {base_url}"
)


@dataclass(frozen=True)
class OrchestratorOptions:
    run_id: str
    output_root: Path
    readiness_timeout_s: int
    max_parallel: int


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
            key=lambda task: int(_get_mapping(task.vllm.get(task.model_key), f"vllm.{task.model_key}").get("gpus", 1)),
            reverse=True,
        )
        self._resource_manager = resource_manager
        self._options = options
        self._dashboard = OrchestratorDashboard() if use_dashboard else None
        self._manifests: dict[str, TaskManifest] = {}
        self._active_containers: dict[str, object] = {}
        self._bench_processes: dict[str, BenchProcess] = {}
        self._shutdown = asyncio.Event()
        self._shutdown_mode: str | None = None

    def run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        scheduler = TaskScheduler(self._resource_manager, max_parallel=self._options.max_parallel)
        if self._dashboard:
            self._dashboard.start()
        try:
            loop = asyncio.get_running_loop()
            runner_task = asyncio.create_task(
                scheduler.run(self._tasks, self._run_task, shutdown_event=self._shutdown)
            )
            _register_signal_handlers(loop, lambda: self._handle_shutdown(runner_task, loop))
            await runner_task
        finally:
            if self._dashboard:
                self._dashboard.stop()
            if self._shutdown.is_set():
                await self._teardown_active()
                if self._shutdown_mode == "force":
                    self._mark_cancelled()

    async def _run_task(self, task: TaskSpec, allocation: Allocation) -> None:
        manifest = self._get_or_init_manifest(task, allocation)
        paths = TaskPaths(self._options.output_root / task.task_id)
        attempt = 0
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

    async def _run_task_once(
        self, task: TaskSpec, allocation: Allocation, manifest: TaskManifest, paths: TaskPaths
    ) -> None:
        self._set_state(manifest, paths, JobState.allocating)

        vllm = task.vllm
        docker_cfg = _get_mapping(vllm.get("docker"), "vllm.docker")
        model_cfg = _get_mapping(vllm.get(task.model_key), f"vllm.{task.model_key}")
        container_port = int(docker_cfg.get("container_port", 8000))
        ipc_mode = docker_cfg.get("ipc_mode")
        image = str(docker_cfg.get("image", "")).strip()
        if not image:
            raise RuntimeError(f"Missing vllm.docker.image for {task.job_config_path}")
        manifest.image = image

        tensor_parallel = model_cfg.get("tensor_parallel_size")
        gpus_required = int(model_cfg.get("gpus", 1))
        if gpus_required > 1:
            if not tensor_parallel:
                raise RuntimeError(f"vllm.{task.model_key}.tensor_parallel_size is required for multi-GPU.")
            if int(tensor_parallel) != gpus_required:
                raise RuntimeError("gpus must match tensor_parallel_size for multi-GPU models.")
        if gpus_required == 1 and tensor_parallel and int(tensor_parallel) > 1:
            raise RuntimeError("tensor_parallel_size > 1 is invalid for single-GPU models.")

        serve = _get_mapping(model_cfg.get("serve"), f"vllm.{task.model_key}.serve")
        container_args = build_container_args(
            task.model_id, tensor_parallel_size=int(tensor_parallel) if tensor_parallel else None, serve=serve
        )
        env = _load_env_file(docker_cfg.get("env_file"), base_dir=task.job_config_path.parent)
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
            container = create_and_start_container(
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

        log_task = asyncio.create_task(
            asyncio.to_thread(stream_container_logs, container, str(paths.container_logs_path))
        )
        base_url = f"http://127.0.0.1:{allocation.port}/v1"
        try:
            self._set_state(manifest, paths, JobState.loading)
            readiness = await asyncio.to_thread(
                wait_for_readiness,
                base_url,
                model_id=task.model_id,
                timeout_s=self._options.readiness_timeout_s,
            )
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

            command_template = self._plan.command_template or DEFAULT_COMMAND_TEMPLATE
            repo_root = Path(__file__).resolve().parents[2]
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
            command = render_command(command_template, command_context)
            manifest.bench_command = shlex.join(command)
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
        finally:
            await _teardown_container(container, manifest)
            log_task.cancel()
            self._active_containers.pop(task.task_id, None)

    def _set_state(self, manifest: TaskManifest, paths: TaskPaths, state: str) -> None:
        manifest.state = state
        if state in {JobState.completed, JobState.failed, JobState.cancelled}:
            manifest.completed_at = _utcnow()
        write_task_manifest(paths, manifest)
        write_summary(self._options.output_root / "summary.json", list(self._manifests.values()))
        if self._dashboard:
            self._dashboard.update(self._manifests.values())

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
            manifest.state = JobState.cancelled
            manifest.completed_at = _utcnow()
        write_summary(self._options.output_root / "summary.json", list(self._manifests.values()))

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
        await self._teardown_active()

    def _handle_shutdown(self, runner_task: asyncio.Task, loop: asyncio.AbstractEventLoop) -> None:
        if not self._shutdown.is_set():
            self._shutdown_mode = "graceful"
            self._shutdown.set()
            return
        if self._shutdown_mode == "force":
            return
        self._shutdown_mode = "force"
        loop.create_task(self._force_shutdown())


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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
        return "port" in message or "bind" in message or "address already in use" in message
    return "connection reset" in message


__all__ = ["OrchestratorOptions", "OrchestratorRunner"]
