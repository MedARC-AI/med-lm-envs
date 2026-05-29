"""Launch one explicit vLLM benchmark command."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from medarc_verifiers.orchestrate.bench import (
    BenchProcess,
    start_benchmark,
    wait_benchmark,
)
from medarc_verifiers.orchestrate.bundle import (
    RuntimeState,
    write_runtime_state,
)
from medarc_verifiers.orchestrate.docker_vllm import (
    sanitize_container_name,
    wait_for_readiness_async,
    write_container_request,
)
from medarc_verifiers.orchestrate.env import load_explicit_runtime_env
from medarc_verifiers.orchestrate.ranges import parse_index_range
from medarc_verifiers.orchestrate.runtime import (
    LogStreamer,
    RuntimeAdapter,
    RuntimeHandle,
    RuntimeLaunchError,
    RuntimeName,
    build_runtime_adapter,
    normalize_runtime,
)
from medarc_verifiers.orchestrate.state import (
    JobState,
    TaskManifest,
    TaskPaths,
    upsert_summary_entry,
    write_task_manifest,
    write_task_result,
    write_text,
)
from medarc_verifiers.orchestrate.topology import ResolvedTopology, resolve_launch_topology
from medarc_verifiers.orchestrate.vllm_args import (
    build_container_args,
    infer_hf_env_from_volume_mounts,
    normalize_volume_mounts,
)

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
    runtime: RuntimeName
    readiness_timeout_s: int
    env_file: Path | None = None
    prune_logs_on_success: bool = False


@dataclass(frozen=True)
class LaunchInputs:
    task_id: str
    model_id: str
    image: str
    gpus: int
    runtime_dir: Path
    serve_dir: Path
    ready_file: Path
    bench_argv: tuple[str, ...]
    endpoint_id: str | None = None
    container_port: int = 8000
    host_port: int = 8000
    tensor_parallel_size: int = 1
    data_parallel_size: int | None = None
    allocated_gpus: int | None = None
    gpu_ids: tuple[int, ...] = ()
    serve_args: Mapping[str, object] | None = None
    volumes: tuple[str, ...] = ()
    container_env_file: Path | None = None
    container_env_base_dir: Path | None = None
    container_ipc_mode: str | None = None
    pyxis_srun_extra_args: tuple[str, ...] = ()
    container_image_source: str | None = None
    image_dir: Path | None = None
    hf_home: str | None = None
    hub_cache: str | None = None


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
        launch: LaunchInputs,
        *,
        options: WorkerOptions,
        runtime_adapter: RuntimeAdapter | None = None,
        callbacks: WorkerCallbacks | None = None,
    ) -> None:
        self._launch = _normalize_launch_inputs(launch)
        self._options = options
        self._runtime = normalize_runtime(options.runtime)
        self._runtime_adapter = runtime_adapter or build_runtime_adapter(self._runtime)
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
        paths = TaskPaths(self._launch.runtime_dir.parent)
        manifest = manifest or TaskManifest(
            task_id=self._launch.task_id,
            config_path=_bench_config_path(self._launch.bench_argv),
            model_key=self._launch.endpoint_id or self._launch.model_id,
            model_id=self._launch.model_id,
        )
        manifest.gpu_ids = list(self._launch.gpu_ids)
        manifest.port = self._launch.host_port
        manifest.gpus = self._launch.gpus
        manifest.tensor_parallel_size = self._launch.tensor_parallel_size
        manifest.data_parallel_size = self._launch.data_parallel_size
        manifest.allocated_gpus = self._launch.allocated_gpus
        if manifest.started_at is None:
            manifest.started_at = _utcnow()
        state_handler = state_handler or (
            self._callbacks.set_state
            if self._callbacks is not None and self._callbacks.set_state is not None
            else _default_state_handler(paths=paths)
        )
        log = log or (
            self._callbacks.log if self._callbacks is not None and self._callbacks.log is not None else (lambda _: None)
        )

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
        topology = _resolve_worker_topology(self._launch)
        _apply_topology_to_manifest(manifest, topology)
        state_handler(manifest, paths, JobState.allocating)
        image = self._launch.image.strip()
        if not image:
            raise RuntimeError(f"Missing container image for task {self._launch.task_id}.")
        manifest.image = image

        configured_serve_args = dict(self._launch.serve_args or {})
        serve_args = _effective_serve_args(configured_serve_args, topology=topology, image=image)
        if serve_args.get("async_scheduling") is not configured_serve_args.get("async_scheduling"):
            log(
                f"JOB info task={self._launch.task_id} async_scheduling_disabled=true "
                f"reason=vllm_lt_0_20_data_parallel data_parallel_size={topology.data_parallel_size}"
            )
        if serve_args.get("gpu_memory_utilization") != configured_serve_args.get("gpu_memory_utilization"):
            log(
                f"JOB info task={self._launch.task_id} gpu_memory_utilization_adjusted=true "
                f"reason=data_parallel_rank0_overhead data_parallel_size={topology.data_parallel_size} "
                f"base={configured_serve_args.get('gpu_memory_utilization', 0.9)} "
                f"adjusted={serve_args['gpu_memory_utilization']}"
            )

        container_args = build_container_args(
            self._launch.model_id,
            tensor_parallel_size=topology.tensor_parallel_size if topology.tensor_parallel_size > 1 else None,
            data_parallel_size=topology.data_parallel_size if topology.data_parallel_size > 1 else None,
            serve=serve_args,
        )
        _verify_materialized_image(self._launch, image=image, runtime=self._runtime)
        env = load_explicit_runtime_env(
            env_file=self._options.env_file,
            container_env_file=self._launch.container_env_file,
            container_env_base_dir=self._launch.container_env_base_dir,
        )
        if self._launch.hf_home:
            env.setdefault("HF_HOME", str(self._launch.hf_home))
        if self._launch.hub_cache:
            env.setdefault("HUGGINGFACE_HUB_CACHE", str(self._launch.hub_cache))
        volume_mounts = normalize_volume_mounts(list(self._launch.volumes))
        for name, value in infer_hf_env_from_volume_mounts(volume_mounts).items():
            env.setdefault(name, value)
        run_label = _run_label_from_task_root(paths.root)
        labels = {"orchestrator.run_id": run_label, "orchestrator.task_id": self._launch.task_id}
        if self._launch.endpoint_id:
            labels["orchestrator.endpoint_id"] = self._launch.endpoint_id
        container_name = sanitize_container_name(f"vllm-{run_label}-{self._launch.task_id}")
        manifest.container_name = container_name

        request_payload = {
            "runtime": self._runtime,
            "image": image,
            "name": container_name,
            "container_port": self._launch.container_port,
            "host_port": self._launch.host_port,
            "ipc_mode": self._launch.container_ipc_mode,
            "volumes": volume_mounts,
            "env": sorted(env.keys()),
            "gpu_ids": self._launch.gpu_ids,
            "allocated_gpus": topology.allocated_gpus,
            "gpus": topology.gpus,
            "tensor_parallel_size": topology.tensor_parallel_size,
            "data_parallel_size": topology.data_parallel_size,
            "vllm_world_size": topology.vllm_world_size,
            "command": container_args,
            "labels": labels,
        }
        write_container_request(str(paths.container_request_path), request_payload)
        _write_runtime_manifest(paths, self._launch, manifest=manifest, request=request_payload)

        state_handler(manifest, paths, JobState.launching)
        try:
            self._active_handle = await asyncio.to_thread(
                self._runtime_adapter.launch,
                task_id=self._launch.task_id,
                model_id=self._launch.model_id,
                container_args=container_args,
                image=image,
                container_port=self._launch.container_port,
                volume_mounts=volume_mounts,
                gpus_required=topology.allocated_gpus,
                gpu_ids=list(self._launch.gpu_ids),
                server_port=self._launch.host_port,
                env=env,
                labels=labels,
                name=container_name,
                ipc_mode=self._launch.container_ipc_mode,
                srun_extra_args=list(self._launch.pyxis_srun_extra_args),
            )
        except RuntimeLaunchError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise RuntimeLaunchError(str(exc)) from exc
        manifest.container_id = self._active_handle.identifier
        if self._callbacks is not None and self._callbacks.register_handle is not None:
            self._callbacks.register_handle(self._launch.task_id, self._active_handle)

        self._log_streamer = self._runtime_adapter.stream_logs(self._active_handle, paths.container_logs_path)
        self._log_streamer.start()
        if self._callbacks is not None and self._callbacks.register_log_streamer is not None:
            self._callbacks.register_log_streamer(self._launch.task_id, self._log_streamer)
        completed_successfully = False
        try:
            state_handler(manifest, paths, JobState.loading)
            readiness = await wait_for_readiness_async(
                self._active_handle.base_url,
                model_id=self._launch.model_id,
                timeout_s=self._options.readiness_timeout_s,
            )
            manifest.readiness = readiness.__dict__
            write_text(paths.readiness_path, json.dumps(readiness.__dict__, indent=2))
            write_text(self._launch.ready_file, json.dumps(readiness.__dict__, indent=2))
            if not readiness.ready:
                manifest.failure_reason = "readiness_timeout"
                manifest.error = readiness.last_error
                write_task_result(
                    paths,
                    {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error},
                )
                state_handler(manifest, paths, JobState.failed)
                return
            log(f"JOB ready task={self._launch.task_id} attempts={readiness.attempts}")
            quarantined_outputs = _quarantine_malformed_bench_outputs(paths.bench_dir)
            for old_path, archived_path in quarantined_outputs:
                log(f"JOB bench-quarantine task={self._launch.task_id} old={old_path} archived={archived_path}")

            command = list(self._launch.bench_argv)
            manifest.bench_command = shlex.join(command)
            _write_runtime_manifest(paths, self._launch, manifest=manifest, request=request_payload)
            log(f"JOB bench-start task={self._launch.task_id} cmd={_shorten(manifest.bench_command)}")
            state_handler(manifest, paths, JobState.running)
            bench_env = {**os.environ, "TQDM_DISABLE": "1"}
            self._bench_process = await start_benchmark(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=bench_env,
                stdout_path=paths.stdout_path,
                stderr_path=paths.stderr_path,
            )
            if self._callbacks is not None and self._callbacks.register_bench is not None:
                self._callbacks.register_bench(self._launch.task_id, self._bench_process)
            bench_result = await wait_benchmark(self._bench_process)
            if self._callbacks is not None and self._callbacks.unregister_bench is not None:
                self._callbacks.unregister_bench(self._launch.task_id)
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
                    self._callbacks.unregister_handle(self._launch.task_id)
                self._active_handle = None
            if self._log_streamer is not None:
                await asyncio.to_thread(self._log_streamer.stop)
                if self._callbacks is not None and self._callbacks.unregister_log_streamer is not None:
                    self._callbacks.unregister_log_streamer(self._launch.task_id)
                self._log_streamer = None
            if completed_successfully and self._options.prune_logs_on_success:
                _prune_task_logs(paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate launch",
        description="Launch vLLM, wait for readiness, run the benchmark argv after --, then clean up.",
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--model", required=True, help="Model id served by vLLM.")
    parser.add_argument("--endpoint-id", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--runtime", choices=("docker", "podman", "pyxis"), required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--serve-dir", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--container-port", type=int, default=8000)
    parser.add_argument("--host-port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=None)
    parser.add_argument("--serve-args-json", default=None)
    parser.add_argument("--volume", action="append", default=[])
    parser.add_argument("--container-env-file", type=Path, default=None)
    parser.add_argument("--container-env-base-dir", type=Path, default=None)
    parser.add_argument("--container-ipc-mode", default=None)
    parser.add_argument("--pyxis-srun-arg", action="append", default=[])
    parser.add_argument("--container-image-source", default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--hf-home", default=None)
    parser.add_argument("--hub-cache", default=None)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--readiness-timeout-s", type=int, default=1800)
    parser.add_argument("--prune-logs-on-success", action="store_true")
    parser.add_argument("bench_argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    bench_argv = _normalize_bench_argv(args.bench_argv)
    serve_args = _parse_serve_args_json(args.serve_args_json)
    launch = LaunchInputs(
        task_id=args.task_id,
        model_id=args.model,
        endpoint_id=args.endpoint_id,
        image=args.image,
        gpus=args.gpus,
        runtime_dir=args.runtime_dir.expanduser().resolve(),
        serve_dir=args.serve_dir.expanduser().resolve(),
        ready_file=args.ready_file.expanduser().resolve(),
        bench_argv=tuple(bench_argv),
        container_port=args.container_port,
        host_port=args.host_port,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        serve_args=serve_args,
        volumes=tuple(args.volume or ()),
        container_env_file=args.container_env_file.expanduser().resolve()
        if args.container_env_file is not None
        else None,
        container_env_base_dir=args.container_env_base_dir.expanduser().resolve()
        if args.container_env_base_dir is not None
        else None,
        container_ipc_mode=args.container_ipc_mode,
        pyxis_srun_extra_args=tuple(args.pyxis_srun_arg or ()),
        container_image_source=args.container_image_source,
        image_dir=args.image_dir.expanduser().resolve() if args.image_dir is not None else None,
        hf_home=args.hf_home,
        hub_cache=args.hub_cache,
    )
    options = WorkerOptions(
        runtime=normalize_runtime(args.runtime),
        readiness_timeout_s=args.readiness_timeout_s,
        env_file=args.env_file.expanduser().resolve() if args.env_file is not None else None,
        prune_logs_on_success=bool(args.prune_logs_on_success),
    )
    worker = TaskWorker(launch, options=options)
    paths = TaskPaths(launch.runtime_dir.parent)
    manifest = TaskManifest(
        task_id=launch.task_id,
        config_path=_bench_config_path(bench_argv),
        model_key=launch.endpoint_id or launch.model_id,
        model_id=launch.model_id,
    )
    state_handler = _default_state_handler(paths=paths)
    try:
        asyncio.run(worker.run(manifest=manifest))
    except Exception as exc:  # noqa: BLE001
        manifest.error = str(exc)
        manifest.failure_reason = (
            "serve_launch_failed" if isinstance(exc, RuntimeLaunchError) else "unexpected_exception"
        )
        state_handler(manifest, paths, JobState.failed)
        write_task_result(
            paths,
            {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error},
        )
        return 1
    return 0


def _run_label_from_task_root(root: Path) -> str:
    try:
        return root.parents[1].name
    except IndexError:
        return root.parent.name or root.name


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
            ),
        )
        upsert_summary_entry(summary_path, manifest)

    return handler


def _verify_materialized_image(launch: LaunchInputs, *, image: str, runtime: RuntimeName) -> None:
    if runtime != "pyxis" or not launch.container_image_source:
        return
    path = Path(image)
    image_dir = launch.image_dir
    if image_dir is None:
        raise RuntimeLaunchError(f"Task {launch.task_id} uses a materialized image but no image cache root is recorded.")
    try:
        path.expanduser().resolve().relative_to(image_dir.expanduser().resolve())
    except ValueError as exc:
        raise RuntimeLaunchError(f"Materialized image {path} is outside configured image cache {image_dir}.") from exc
    if not path.is_file():
        raise RuntimeLaunchError(f"Materialized Pyxis image not found for task {launch.task_id}: {path}")


def _normalize_launch_inputs(launch: LaunchInputs) -> LaunchInputs:
    expected_root = launch.runtime_dir.parent
    if launch.runtime_dir.name != "runtime" or launch.serve_dir != expected_root / "serve":
        raise ValueError(
            "launch expects task-local paths: --runtime-dir <task>/runtime and --serve-dir <task>/serve."
        )
    allocated_gpus = launch.allocated_gpus
    if allocated_gpus is None:
        if launch.gpu_ids:
            allocated_gpus = len(launch.gpu_ids)
        else:
            allocated_gpus = _infer_allocated_gpu_count_from_environment() or launch.gpus
    return LaunchInputs(
        task_id=launch.task_id,
        model_id=launch.model_id,
        image=launch.image,
        gpus=launch.gpus,
        runtime_dir=launch.runtime_dir,
        serve_dir=launch.serve_dir,
        ready_file=launch.ready_file,
        bench_argv=launch.bench_argv,
        endpoint_id=launch.endpoint_id,
        container_port=launch.container_port,
        host_port=launch.host_port,
        tensor_parallel_size=launch.tensor_parallel_size,
        data_parallel_size=launch.data_parallel_size,
        allocated_gpus=allocated_gpus,
        gpu_ids=tuple(launch.gpu_ids),
        serve_args=dict(launch.serve_args or {}),
        volumes=tuple(launch.volumes),
        container_env_file=launch.container_env_file,
        container_env_base_dir=launch.container_env_base_dir,
        container_ipc_mode=launch.container_ipc_mode,
        pyxis_srun_extra_args=tuple(launch.pyxis_srun_extra_args),
        container_image_source=launch.container_image_source,
        image_dir=launch.image_dir,
        hf_home=launch.hf_home,
        hub_cache=launch.hub_cache,
    )


def _effective_serve_args(
    serve_args: dict[str, object] | Mapping[str, object], *, topology: ResolvedTopology, image: str
) -> dict[str, object]:
    effective = dict(serve_args)
    if (
        topology.data_parallel_size > 1
        and effective.get("async_scheduling") is True
        and _requires_dp_async_scheduling_workaround(image)
    ):
        effective["async_scheduling"] = False
    adjusted_gpu_memory_utilization = _adjust_gpu_memory_utilization_for_dp(
        effective.get("gpu_memory_utilization"),
        data_parallel_size=topology.data_parallel_size,
    )
    if adjusted_gpu_memory_utilization is not None:
        effective["gpu_memory_utilization"] = adjusted_gpu_memory_utilization
    return effective


_VLLM_IMAGE_VERSION_RE = re.compile(r"(?:^|[/:\-_.])v?(\d+)\.(\d+)(?:\.(\d+))?(?:$|[+\-_.])", re.IGNORECASE)


def _requires_dp_async_scheduling_workaround(image: str) -> bool:
    version = _parse_vllm_image_version(image)
    return version is not None and version < (0, 20, 0)


def _parse_vllm_image_version(image: str) -> tuple[int, int, int] | None:
    text = image.strip()
    if "vllm" not in text.lower():
        return None
    for match in _VLLM_IMAGE_VERSION_RE.finditer(text):
        major = int(match.group(1))
        minor = int(match.group(2))
        patch = int(match.group(3) or 0)
        return major, minor, patch
    return None


def _adjust_gpu_memory_utilization_for_dp(
    current_value: object,
    *,
    data_parallel_size: int,
) -> float | None:
    if data_parallel_size <= 1:
        return float(current_value) if current_value is not None else None
    deduction = _dp_gpu_memory_utilization_deduction(data_parallel_size)
    if deduction <= 0:
        return float(current_value) if current_value is not None else None
    base_value = float(current_value) if current_value is not None else 0.9
    adjusted = max(0.5, base_value - deduction)
    return round(adjusted, 3)


def _dp_gpu_memory_utilization_deduction(data_parallel_size: int) -> float:
    return {
        2: 0.01,
        4: 0.03,
        8: 0.05,
    }.get(data_parallel_size, 0.0)


def _infer_allocated_gpu_count_from_environment() -> int | None:
    for key in (
        "MEDARC_ALLOCATED_GPU_COUNT",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_GPUS_ON_NODE",
    ):
        count = _count_visible_gpus(os.environ.get(key))
        if count is not None:
            return count
    return None


def _count_visible_gpus(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "void", "novisibledevices"}:
        return 0
    if text.isdigit():
        return int(text)
    if ":" in text:
        suffix = text.rsplit(":", maxsplit=1)[-1].strip()
        if suffix.isdigit():
            return int(suffix)
    try:
        parsed = parse_index_range(text)
    except ValueError:
        parsed = []
    if parsed:
        return len(parsed)
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if tokens:
        return len(tokens)
    return None


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


def _resolve_worker_topology(launch: LaunchInputs) -> ResolvedTopology:
    allocated_gpus = launch.allocated_gpus
    if allocated_gpus is None:
        raise RuntimeError(f"Task {launch.task_id} is missing allocated_gpus.")
    return resolve_launch_topology(
        task_id=launch.task_id,
        gpus=launch.gpus,
        allocated_gpus=allocated_gpus,
        tensor_parallel_size=launch.tensor_parallel_size,
        data_parallel_size=launch.data_parallel_size,
    )


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


def _write_runtime_manifest(
    paths: TaskPaths,
    launch: LaunchInputs,
    *,
    manifest: TaskManifest,
    request: Mapping[str, object],
) -> None:
    payload = {
        "task_id": launch.task_id,
        "endpoint_id": launch.endpoint_id,
        "model_id": launch.model_id,
        "image": launch.image,
        "container_port": launch.container_port,
        "host_port": launch.host_port,
        "runtime_dir": str(launch.runtime_dir),
        "serve_dir": str(launch.serve_dir),
        "ready_file": str(launch.ready_file),
        "bench_command": manifest.bench_command,
        "bench_argv": list(launch.bench_argv),
        "container_request": dict(request),
    }
    write_text(paths.runtime_dir / "manifest.json", json.dumps(payload, indent=2))


async def _teardown_runtime(runtime_adapter: RuntimeAdapter, handle: RuntimeHandle) -> None:
    await asyncio.to_thread(runtime_adapter.teardown, handle)


def _parse_time(value: str | None):
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _quarantine_malformed_bench_outputs(bench_dir: Path) -> list[tuple[Path, Path]]:
    if not bench_dir.exists():
        return []
    candidates: list[Path] = []
    for candidate in bench_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        try:
            relative_parts = candidate.relative_to(bench_dir).parts
        except ValueError:
            continue
        if len(relative_parts) < 3 or any("__malformed_" in part for part in relative_parts):
            continue
        if not any(candidate.iterdir()):
            continue
        candidates.append(candidate)

    quarantined: list[tuple[Path, Path]] = []
    for candidate in sorted(candidates, key=lambda item: len(item.parts)):
        if not candidate.is_dir():
            continue
        has_metadata = (candidate / "metadata.json").is_file()
        has_results = (candidate / "results.jsonl").is_file()
        if has_metadata and has_results:
            continue
        archived = _archive_malformed_bench_output(candidate)
        quarantined.append((candidate, archived))
    return quarantined


def _archive_malformed_bench_output(path: Path) -> Path:
    candidate = path.with_name(f"{path.name}__malformed_{uuid.uuid4().hex[:8]}")
    while candidate.exists():
        candidate = path.with_name(f"{path.name}__malformed_{uuid.uuid4().hex[:8]}")
    shutil.move(str(path), str(candidate))
    return candidate


def _shorten(text: str, *, max_len: int = 220) -> str:
    if len(text) <= max_len:
        return text
    suffix = "..."
    keep = max(0, max_len - len(suffix))
    return f"{text[:keep]}{suffix}"


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _parse_serve_args_json(value: str | None) -> dict[str, object]:
    if value is None or not value.strip():
        return {}
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("--serve-args-json must decode to a JSON object.")
    return payload


def _normalize_bench_argv(values: list[str]) -> list[str]:
    bench_argv = list(values)
    if bench_argv and bench_argv[0] == "--":
        bench_argv = bench_argv[1:]
    if not bench_argv:
        raise SystemExit("medarc-orchestrate launch requires a benchmark command after --.")
    return bench_argv


def _bench_config_path(bench_argv: tuple[str, ...] | list[str]) -> str:
    values = list(bench_argv)
    for index, value in enumerate(values):
        if value == "--config" and index + 1 < len(values):
            return values[index + 1]
        if value.startswith("--config="):
            return value.split("=", maxsplit=1)[1]
    return ""


__all__ = ["LaunchInputs", "TaskWorker", "WorkerCallbacks", "WorkerOptions", "build_parser", "main"]
