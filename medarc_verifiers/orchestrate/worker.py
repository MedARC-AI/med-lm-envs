"""Single-task worker for orchestrator task bundles."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dotenv import dotenv_values

from medarc_verifiers.orchestrate.bench import (
    BenchProcess,
    render_command,
    start_benchmark,
    wait_benchmark,
)
from medarc_verifiers.orchestrate.bundle import (
    ExecutionAllocation,
    ResolvedTaskSpec,
    RuntimeState,
    load_execution_allocation,
    load_runtime_state,
    load_task_spec,
    write_execution_allocation,
    write_runtime_state,
)
from medarc_verifiers.orchestrate.config import load_job_config
from medarc_verifiers.orchestrate.docker_vllm import (
    sanitize_container_name,
    wait_for_readiness_async,
    write_container_request,
)
from medarc_verifiers.orchestrate.runtime import LogStreamer, RuntimeAdapter, RuntimeHandle, RuntimeLaunchError
from medarc_verifiers.orchestrate.state import (
    JobState,
    TaskManifest,
    TaskPaths,
    upsert_summary_entry,
    write_task_manifest,
    write_task_result,
    write_text,
)
from medarc_verifiers.orchestrate.task_naming import bench_run_id
from medarc_verifiers.orchestrate.topology import ResolvedTopology, resolve_task_spec_topology
from medarc_verifiers.orchestrate.vllm_args import build_container_args, normalize_volume_mounts

_COMMAND_TEMPLATE_UV = (
    "uv run medarc-eval bench --config {job_config_path} --api-base-url {base_url} "
    "--run-id {bench_run_id} --on-complete exit"
)
_COMMAND_TEMPLATE_BARE = (
    "medarc-eval bench --config {job_config_path} --api-base-url {base_url} "
    "--run-id {bench_run_id} --on-complete exit"
)
_DEFAULT_BENCH_OUTPUT_DIR = Path("runs") / "raw"

StateHandler = Callable[[TaskManifest, TaskPaths, str], None]
LogFn = Callable[[str], None]
HandleRegistration = Callable[[str, RuntimeHandle], None]
HandleRemoval = Callable[[str], None]
BenchRegistration = Callable[[str, BenchProcess], None]
BenchRemoval = Callable[[str], None]
StreamerRegistration = Callable[[str, LogStreamer], None]
StreamerRemoval = Callable[[str], None]


@dataclass(frozen=True)
class WorkerOptions:
    run_id: str
    runtime: str
    readiness_timeout_s: int
    command_template: str | None = None
    env_file: Path | None = None
    prune_logs_on_success: bool = False
    uv_run: bool = True


@dataclass(frozen=True)
class WorkerCallbacks:
    set_state: StateHandler | None = None
    log: LogFn | None = None
    register_handle: HandleRegistration | None = None
    unregister_handle: HandleRemoval | None = None
    register_bench: BenchRegistration | None = None
    unregister_bench: BenchRemoval | None = None
    register_log_streamer: StreamerRegistration | None = None
    unregister_log_streamer: StreamerRemoval | None = None


class TaskWorker:
    def __init__(
        self,
        task_spec: ResolvedTaskSpec,
        allocation: ExecutionAllocation,
        *,
        options: WorkerOptions,
        runtime_adapter: RuntimeAdapter | None = None,
        callbacks: WorkerCallbacks | None = None,
    ) -> None:
        self._spec = task_spec
        self._allocation = _normalize_allocation(allocation, task_spec=task_spec)
        self._options = options
        self._runtime = _normalize_runtime(options.runtime or task_spec.runtime)
        self._runtime_adapter = runtime_adapter or _build_runtime_adapter(self._runtime)
        self._command_template = options.command_template or (
            _COMMAND_TEMPLATE_UV if options.uv_run else _COMMAND_TEMPLATE_BARE
        )
        self._callbacks = callbacks
        self._active_handle: RuntimeHandle | None = None
        self._bench_process: BenchProcess | None = None
        self._log_streamer: LogStreamer | None = None

    async def run(
        self,
        *,
        manifest: TaskManifest | None = None,
        state_handler: StateHandler | None = None,
        log: LogFn | None = None,
    ) -> TaskManifest:
        paths = TaskPaths(Path(self._spec.output_paths.root))
        manifest = manifest or TaskManifest(
            task_id=self._spec.task_id,
            config_path=self._spec.bundled_eval_config_path,
            model_key=self._spec.model_key,
            model_id=self._spec.model_id,
        )
        manifest.gpu_ids = list(self._allocation.gpu_ids)
        manifest.port = self._allocation.server_port
        manifest.gpus = self._spec.gpus
        manifest.tensor_parallel_size = self._spec.tensor_parallel_size
        manifest.data_parallel_size = self._spec.data_parallel_size
        manifest.allocated_gpus = self._allocation.allocated_gpus
        if manifest.started_at is None:
            manifest.started_at = _utcnow()
        write_execution_allocation(paths.allocation_path, self._allocation)
        state_handler = state_handler or (
            self._callbacks.set_state if self._callbacks is not None and self._callbacks.set_state is not None else _default_state_handler(paths=paths)
        )
        log = log or (self._callbacks.log if self._callbacks is not None and self._callbacks.log is not None else (lambda _: None))

        await self._run_once(manifest=manifest, paths=paths, state_handler=state_handler, log=log)
        return manifest

    async def _run_once(
        self,
        *,
        manifest: TaskManifest,
        paths: TaskPaths,
        state_handler: StateHandler,
        log: LogFn,
    ) -> None:
        topology = _resolve_worker_topology(self._spec, self._allocation)
        _apply_topology_to_manifest(manifest, topology)
        state_handler(manifest, paths, JobState.allocating)
        image = self._spec.container_image.strip()
        if not image:
            raise RuntimeError(f"Missing container image for task {self._spec.task_id}.")
        manifest.image = image

        container_args = build_container_args(
            self._spec.model_id,
            tensor_parallel_size=topology.tensor_parallel_size if topology.tensor_parallel_size > 1 else None,
            data_parallel_size=topology.data_parallel_size if topology.data_parallel_size > 1 else None,
            serve=dict(self._spec.serve_args),
        )
        env = _load_runtime_env(self._spec, allocation=self._allocation, options=self._options)
        volume_mounts = normalize_volume_mounts(self._spec.volume_mounts)
        labels = {"orchestrator.run_id": self._options.run_id, "orchestrator.task_id": self._spec.task_id}
        container_name = sanitize_container_name(f"vllm-{self._options.run_id}-{self._spec.task_id}")
        manifest.container_name = container_name

        request_payload = {
            "runtime": self._runtime,
            "image": image,
            "name": container_name,
            "container_port": self._spec.container_port,
            "host_port": self._allocation.server_port,
            "ipc_mode": self._spec.container_ipc_mode,
            "volumes": volume_mounts,
            "env": sorted(env.keys()),
            "gpu_ids": self._allocation.gpu_ids,
            "allocated_gpus": topology.allocated_gpus,
            "gpus": topology.gpus,
            "tensor_parallel_size": topology.tensor_parallel_size,
            "data_parallel_size": topology.data_parallel_size,
            "vllm_world_size": topology.vllm_world_size,
            "command": container_args,
            "labels": labels,
        }
        write_container_request(str(paths.container_request_path), request_payload)

        state_handler(manifest, paths, JobState.launching)
        try:
            self._active_handle = await asyncio.to_thread(
                self._runtime_adapter.launch,
                task_id=self._spec.task_id,
                model_id=self._spec.model_id,
                container_args=container_args,
                image=image,
                container_port=self._spec.container_port,
                volume_mounts=volume_mounts,
                gpus_required=topology.allocated_gpus,
                gpu_ids=list(self._allocation.gpu_ids),
                server_port=_require_server_port(self._allocation),
                env=env,
                labels=labels,
                name=container_name,
                ipc_mode=self._spec.container_ipc_mode,
                srun_extra_args=list(self._spec.pyxis_srun_extra_args),
            )
        except RuntimeLaunchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeLaunchError(str(exc)) from exc
        manifest.container_id = self._active_handle.identifier
        if self._callbacks is not None and self._callbacks.register_handle is not None:
            self._callbacks.register_handle(self._spec.task_id, self._active_handle)

        self._log_streamer = self._runtime_adapter.stream_logs(self._active_handle, paths.container_logs_path)
        self._log_streamer.start()
        if self._callbacks is not None and self._callbacks.register_log_streamer is not None:
            self._callbacks.register_log_streamer(self._spec.task_id, self._log_streamer)
        completed_successfully = False
        try:
            state_handler(manifest, paths, JobState.loading)
            readiness = await wait_for_readiness_async(
                self._active_handle.base_url,
                model_id=self._spec.model_id,
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
                state_handler(manifest, paths, JobState.failed)
                return
            log(f"JOB ready task={self._spec.task_id} attempts={readiness.attempts}")

            command_context = {
                "base_url": self._active_handle.base_url,
                "bench_run_id": bench_run_id(self._options.run_id, self._spec.task_id),
                "host_port": str(_require_server_port(self._allocation)),
                "model_key": self._spec.model_key,
                "model_id": self._spec.model_id,
                "output_dir": str(paths.bench_dir),
                "run_id": self._options.run_id,
                "task_id": self._spec.task_id,
                "job_config_path": self._spec.bundled_eval_config_path,
            }
            command = render_command(self._command_template, command_context)
            manifest.bench_run_id = command_context["bench_run_id"]
            restart_source = _resolved_restart_source(paths.state_path, self._spec)
            if restart_source:
                manifest.restart_source = restart_source
                if "--restart" not in command:
                    command.extend(["--restart", restart_source])
            manifest.bench_command = shlex.join(command)
            log(f"JOB bench-start task={self._spec.task_id} cmd={_shorten(manifest.bench_command)}")
            state_handler(manifest, paths, JobState.running)
            bench_env = {**os.environ, **dict(self._allocation.runtime_env), "TQDM_DISABLE": "1"}
            self._bench_process = await start_benchmark(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=bench_env,
                stdout_path=paths.stdout_path,
                stderr_path=paths.stderr_path,
            )
            if self._callbacks is not None and self._callbacks.register_bench is not None:
                self._callbacks.register_bench(self._spec.task_id, self._bench_process)
            discovered_run_dir = await _discover_bench_run_dir(
                job_config_path=Path(self._spec.bundled_eval_config_path),
                repo_root=Path(__file__).resolve().parents[2],
                timeout_s=15.0,
            )
            if discovered_run_dir is not None:
                manifest.bench_run_dir = str(discovered_run_dir)
                if restart_source is None:
                    manifest.restart_source = _format_restart_source(
                        discovered_run_dir,
                        repo_root=Path(__file__).resolve().parents[2],
                    )
                    write_runtime_state(
                        paths.state_path,
                        RuntimeState(
                            task_id=self._spec.task_id,
                            state=manifest.state,
                            restart_source=manifest.restart_source,
                            restart_source_strategy="runtime_state",
                            bench_run_id=manifest.bench_run_id,
                            bench_run_dir=manifest.bench_run_dir,
                        ),
                    )
                write_task_manifest(paths, manifest)
            bench_result = await wait_benchmark(self._bench_process)
            if self._callbacks is not None and self._callbacks.unregister_bench is not None:
                self._callbacks.unregister_bench(self._spec.task_id)
            self._bench_process = None
            manifest.bench_exit_code = bench_result.exit_code
            manifest.bench_duration_s = bench_result.duration_s
            result_payload = {
                "exit_code": bench_result.exit_code,
                "duration_s": bench_result.duration_s,
                "state": JobState.cancelled
                if bench_result.terminated
                else (JobState.completed if bench_result.exit_code == 0 else JobState.failed),
                "command": manifest.bench_command,
                "argv": list(command),
                "terminated": bench_result.terminated,
            }
            write_task_result(paths, result_payload)
            if bench_result.terminated:
                manifest.failure_reason = "bench_terminated"
                state_handler(manifest, paths, JobState.cancelled)
            elif bench_result.exit_code != 0:
                manifest.failure_reason = "bench_exit_nonzero"
                state_handler(manifest, paths, JobState.failed)
            else:
                state_handler(manifest, paths, JobState.completed)
                completed_successfully = True
        finally:
            if self._active_handle is not None:
                await _teardown_runtime(self._runtime_adapter, self._active_handle)
                if self._callbacks is not None and self._callbacks.unregister_handle is not None:
                    self._callbacks.unregister_handle(self._spec.task_id)
                self._active_handle = None
            if self._log_streamer is not None:
                await asyncio.to_thread(self._log_streamer.stop)
                if self._callbacks is not None and self._callbacks.unregister_log_streamer is not None:
                    self._callbacks.unregister_log_streamer(self._spec.task_id)
                self._log_streamer = None
            if completed_successfully and self._options.prune_logs_on_success:
                _prune_task_logs(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="medarc-orchestrate worker", description="Run one bundled task.")
    parser.add_argument("--task", type=Path, required=True, help="Path to bundled task.yaml.")
    parser.add_argument("--allocation", type=Path, required=True, help="Path to execution allocation JSON.")
    parser.add_argument("--runtime", choices=("docker", "podman", "pyxis"), required=True)
    parser.add_argument("--run-id", required=True, help="Run identifier used for the inner bench run.")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--readiness-timeout-s", type=int, default=1800)
    parser.add_argument("--prune-logs-on-success", action="store_true")
    parser.add_argument("--no-uv-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task_spec = load_task_spec(args.task.expanduser().resolve())
    allocation = load_execution_allocation(args.allocation.expanduser().resolve())
    if allocation is None:
        raise FileNotFoundError(f"Execution allocation not found: {args.allocation}")
    options = WorkerOptions(
        run_id=args.run_id,
        runtime=args.runtime,
        readiness_timeout_s=args.readiness_timeout_s,
        env_file=args.env_file.expanduser().resolve() if args.env_file is not None else None,
        prune_logs_on_success=bool(args.prune_logs_on_success),
        uv_run=not args.no_uv_run,
    )
    worker = TaskWorker(task_spec, allocation, options=options)
    paths = TaskPaths(Path(task_spec.output_paths.root))
    manifest = TaskManifest(
        task_id=task_spec.task_id,
        config_path=task_spec.bundled_eval_config_path,
        model_key=task_spec.model_key,
        model_id=task_spec.model_id,
    )
    state_handler = _default_state_handler(paths=paths)
    try:
        asyncio.run(worker.run(manifest=manifest))
    except Exception as exc:  # noqa: BLE001
        manifest.error = str(exc)
        manifest.failure_reason = "serve_launch_failed" if isinstance(exc, RuntimeLaunchError) else "unexpected_exception"
        state_handler(manifest, paths, JobState.failed)
        write_task_result(
            paths,
            {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error},
        )
        return 1
    return 0


def _default_state_handler(*, paths: TaskPaths) -> StateHandler:
    summary_path = paths.root.parents[1] / "summary.json"

    def handler(manifest: TaskManifest, task_paths: TaskPaths, state: str) -> None:
        now = _utcnow()
        if state != manifest.state:
            manifest.state = state
            manifest.state_entered_at = now
        if state in {JobState.completed, JobState.failed, JobState.cancelled}:
            manifest.completed_at = now
            _update_gpu_accounting(manifest)
        write_task_manifest(task_paths, manifest)
        write_runtime_state(
            task_paths.state_path,
            RuntimeState(
                task_id=manifest.task_id,
                state=manifest.state,
                restart_source=manifest.restart_source,
                restart_source_strategy="runtime_state" if manifest.restart_source else "none",
                bench_run_id=manifest.bench_run_id,
                bench_run_dir=manifest.bench_run_dir,
            ),
        )
        upsert_summary_entry(summary_path, manifest)

    return handler


def _load_runtime_env(
    spec: ResolvedTaskSpec,
    *,
    allocation: ExecutionAllocation,
    options: WorkerOptions,
) -> dict[str, str]:
    env: dict[str, str] = {}
    repo_root = Path(__file__).resolve().parents[2]
    if options.env_file is not None:
        env.update(_load_env_file(options.env_file, base_dir=repo_root))
    else:
        default_env = repo_root / ".env"
        if default_env.exists():
            env.update(_load_env_file(default_env, base_dir=repo_root))
    if spec.container_env_file:
        env.update(_load_env_file(spec.container_env_file, base_dir=Path(spec.original_job_config_path).parent))
    env.update(dict(allocation.runtime_env))
    return env


def _resolved_restart_source(state_path: Path, spec: ResolvedTaskSpec) -> str | None:
    state = load_runtime_state(state_path)
    if state is not None and state.restart_source:
        return state.restart_source
    return spec.restart_source


def _normalize_allocation(allocation: ExecutionAllocation, *, task_spec: ResolvedTaskSpec) -> ExecutionAllocation:
    allocated_gpus = allocation.allocated_gpus
    if allocated_gpus is None:
        if allocation.gpu_ids:
            allocated_gpus = len(allocation.gpu_ids)
        else:
            allocated_gpus = task_spec.gpus
    server_port = allocation.server_port if allocation.server_port is not None else 8000
    return ExecutionAllocation(
        task_id=allocation.task_id,
        allocated_gpus=allocated_gpus,
        gpu_ids=list(allocation.gpu_ids),
        server_port=server_port,
        require_contiguous_gpus=allocation.require_contiguous_gpus,
        slurm_job_id=allocation.slurm_job_id,
        constraints=dict(allocation.constraints),
        runtime_env=dict(allocation.runtime_env),
    )


def _update_gpu_accounting(manifest: TaskManifest) -> None:
    started = _parse_time(manifest.started_at)
    completed = _parse_time(manifest.completed_at)
    if started is None or completed is None:
        return
    elapsed_hours = max(0.0, (completed - started).total_seconds() / 3600.0)
    if manifest.allocated_gpus is not None:
        manifest.allocated_gpu_hours = manifest.allocated_gpus * elapsed_hours
    if manifest.gpus is not None:
        manifest.gpu_hours = manifest.gpus * elapsed_hours


def _resolve_worker_topology(task_spec: ResolvedTaskSpec, allocation: ExecutionAllocation) -> ResolvedTopology:
    allocated_gpus = allocation.allocated_gpus
    if allocated_gpus is None:
        raise RuntimeError(f"Task {task_spec.task_id} is missing allocated_gpus.")
    return resolve_task_spec_topology(task_spec, allocated_gpus=allocated_gpus)


def _apply_topology_to_manifest(manifest: TaskManifest, topology: ResolvedTopology) -> None:
    manifest.gpus = topology.gpus
    manifest.allocated_gpus = topology.allocated_gpus
    manifest.tensor_parallel_size = topology.tensor_parallel_size
    manifest.data_parallel_size = topology.data_parallel_size
    manifest.vllm_world_size = topology.vllm_world_size


def _prune_task_logs(paths: TaskPaths) -> None:
    for log_path in (paths.container_logs_path, paths.stdout_path, paths.stderr_path):
        try:
            log_path.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            continue


async def _teardown_runtime(runtime_adapter: RuntimeAdapter, handle: RuntimeHandle) -> None:
    await asyncio.to_thread(runtime_adapter.teardown, handle)


def _load_env_file(path: object, *, base_dir: Path) -> dict[str, str]:
    env_path = Path(str(path)).expanduser()
    if not env_path.is_absolute():
        env_path = (base_dir / env_path).resolve()
    if not env_path.exists():
        raise RuntimeLaunchError(f"env_file not found: {env_path}")
    values = dotenv_values(env_path)
    return {key: value for key, value in values.items() if value is not None}


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
        except Exception:  # noqa: BLE001
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


def _parse_time(value: str | None):
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _require_server_port(allocation: ExecutionAllocation) -> int:
    if allocation.server_port is None:
        raise RuntimeError(f"Task {allocation.task_id} is missing server_port in execution allocation.")
    return allocation.server_port


def _shorten(text: str, *, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    suffix = "..."
    keep = max(0, max_len - len(suffix))
    return f"{text[:keep]}{suffix}"


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


__all__ = ["TaskWorker", "WorkerCallbacks", "WorkerOptions", "build_parser", "main"]
