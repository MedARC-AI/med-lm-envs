"""Canonical launch resolution for local and Slurm orchestration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from medarc_verifiers.orchestrate import runtime_probe
from medarc_verifiers.orchestrate.config import (
    PlanConfig,
    TaskSpec,
    expand_tasks,
    load_plan,
    make_plan,
    resolve_default_endpoints_path,
)
from medarc_verifiers.orchestrate.resources import ResourceError, discover_gpus, parse_index_range
from medarc_verifiers.orchestrate.topology import minimum_required_gpus, resolve_topology
from medarc_verifiers.utils.run_naming import generate_run_id

Backend = Literal["local", "slurm"]
Runtime = Literal["docker", "podman", "pyxis"]


@dataclass(frozen=True)
class LaunchSource:
    plan: PlanConfig
    base_dir: Path
    explicit_endpoint_override: Path | None = None


@dataclass(frozen=True)
class LaunchPlan:
    backend: Backend
    plan: PlanConfig
    tasks: list[TaskSpec]
    runtime: Runtime
    run_id: str
    output_root: Path
    endpoint_registry_paths: tuple[Path, ...]
    eval_images_config: Path | None
    gpu_indices: list[int] | None
    port_range: tuple[int, int]
    max_parallel: int
    readiness_timeout_s: int
    uv_run: bool


@dataclass(frozen=True)
class LaunchStatusTarget:
    run_id: str | None
    output_root: Path


@dataclass(frozen=True)
class LaunchCleanupTarget:
    runtime: Runtime
    run_id: str | None


def resolve_launch_plan(args, *, backend: Backend, cwd: Path) -> LaunchPlan:
    source = resolve_plan_source(args, cwd=cwd)
    plan = source.plan
    default_endpoint_path = None if plan.endpoints_path is not None else resolve_default_endpoints_path(cwd)
    runtime = resolve_runtime(getattr(args, "runtime", None) or plan.runtime, backend=backend)
    tasks = expand_tasks(plan, default_endpoints_path=default_endpoint_path)
    run_id, output_root = resolve_output_root(args, plan=plan)
    gpu_indices = derive_allowed_gpu_indices(getattr(args, "gpu_range", None) or plan.gpu_range)
    port_range = parse_port_range_or_default(getattr(args, "port_range", None) or plan.port_range)
    readiness_timeout_s = (
        getattr(args, "readiness_timeout_s", None)
        if getattr(args, "readiness_timeout_s", None) is not None
        else (plan.readiness_timeout_s or 1800)
    )
    max_parallel = resolve_max_parallel(
        args,
        plan=plan,
        tasks=tasks,
        runtime=runtime,
        backend=backend,
        gpu_indices=gpu_indices,
        port_range=port_range,
    )
    uv_run = not bool(getattr(args, "no_uv_run", False)) and plan.uv_run
    return LaunchPlan(
        backend=backend,
        plan=plan,
        tasks=tasks,
        runtime=runtime,
        run_id=run_id,
        output_root=output_root,
        endpoint_registry_paths=tuple(dict.fromkeys(task.endpoints_path for task in tasks if task.endpoints_path)),
        eval_images_config=plan.eval_images_config,
        gpu_indices=gpu_indices,
        port_range=port_range,
        max_parallel=max_parallel,
        readiness_timeout_s=int(readiness_timeout_s),
        uv_run=uv_run,
    )


def resolve_plan_source(args, *, cwd: Path) -> LaunchSource:
    base_dir = cwd.resolve()
    explicit_endpoint_override = getattr(args, "endpoints_path", None)
    if getattr(args, "plan", None) is not None:
        plan_path = args.plan.expanduser().resolve()
        plan = load_plan(plan_path)
        base_dir = plan_path.parent
        if plan.name is None:
            plan.name = plan_path.stem
    else:
        job_configs = list(getattr(args, "job_configs", None) or [])
        name = getattr(args, "name", None)
        if name is None and len(job_configs) == 1:
            name = job_configs[0].expanduser().stem
        plan = make_plan(
            job_configs=job_configs,
            base_dir=base_dir,
            name=name,
            eval_images_config=getattr(args, "eval_images_config", None),
            endpoints_path=explicit_endpoint_override,
        )
    if getattr(args, "eval_images_config", None) is not None:
        plan.eval_images_config = _resolve_path(args.eval_images_config, base_dir=base_dir)
    elif plan.eval_images_config is None:
        plan.eval_images_config = resolve_eval_images_config_path(None, cwd=cwd)
    if explicit_endpoint_override is not None:
        plan.endpoints_path = _resolve_path(explicit_endpoint_override, base_dir=base_dir)
    if getattr(args, "env_file", None) is not None:
        plan.env_file = _resolve_path(args.env_file, base_dir=base_dir)
    if bool(getattr(args, "prune_logs_on_success", False)):
        plan.prune_logs_on_success = True
    return LaunchSource(plan=plan, base_dir=base_dir, explicit_endpoint_override=plan.endpoints_path)


def resolve_runtime(value: str | None, *, backend: Backend) -> Runtime:
    if backend == "slurm":
        return "pyxis"
    if value is not None:
        return _normalize_runtime(value)
    return _resolve_local_auto()


def resolve_output_root(args, *, plan: PlanConfig) -> tuple[str, Path]:
    configured_run_id = getattr(args, "run_id", None) or plan.run_id
    run_id = configured_run_id or generate_run_id(plan.name)
    output_root = getattr(args, "output_dir", None) or plan.output_dir or Path("outputs") / "orchestrate" / run_id
    return run_id, output_root.expanduser().resolve() if output_root.is_absolute() else output_root


def resolve_eval_images_config_path(path: Path | None, *, cwd: Path) -> Path | None:
    if path is not None:
        return path.expanduser().resolve()
    default = (cwd.resolve() / "configs" / "eval_images.toml").resolve()
    return default if default.exists() else None


def resolve_max_parallel(
    args,
    *,
    plan: PlanConfig,
    tasks: Sequence[TaskSpec],
    runtime: Runtime,
    backend: Backend,
    gpu_indices: list[int] | None,
    port_range: tuple[int, int],
) -> int:
    explicit = getattr(args, "max_parallel", None)
    if explicit is not None:
        return int(explicit)
    if plan.max_parallel is not None:
        return int(plan.max_parallel)
    if backend == "slurm":
        return 1
    if runtime == "pyxis":
        return 1
    port_capacity = port_range[1] - port_range[0] + 1
    if gpu_indices is not None:
        return derive_local_max_parallel(tasks, gpu_count=len(gpu_indices), port_capacity=port_capacity)
    try:
        gpu_count = len(discover_gpus())
    except ResourceError as exc:
        raise ValueError("GPU discovery failed; ensure NVML/pynvml is available.") from exc
    return derive_local_max_parallel(tasks, gpu_count=gpu_count, port_capacity=port_capacity)


def derive_local_max_parallel(tasks: Sequence[TaskSpec], *, gpu_count: int, port_capacity: int) -> int:
    remaining = max(0, gpu_count)
    fit = 0
    for required in sorted((minimum_required_gpus(task) for task in tasks), reverse=True):
        if required <= remaining:
            fit += 1
            remaining -= required
    return max(1, min(fit or 1, max(0, port_capacity)))


def derive_allowed_gpu_indices(expr: str | None) -> list[int] | None:
    return parse_index_range(expr) if expr else None


def parse_port_range_or_default(expr: str | None) -> tuple[int, int]:
    if not expr:
        return (8000, 8999)
    start_str, end_str = expr.split("-", maxsplit=1)
    start, end = int(start_str), int(end_str)
    if end < start:
        raise ValueError(f"Port range is invalid: {start}-{end}.")
    return (start, end)


def resolve_status_target(args, *, cwd: Path) -> LaunchStatusTarget:
    plan: PlanConfig | None = None
    if getattr(args, "plan", None) is not None:
        plan = load_plan(args.plan.expanduser().resolve())
        if plan.name is None:
            plan.name = args.plan.expanduser().stem
    run_id = getattr(args, "run_id", None) or (plan.run_id if plan is not None else None)
    output_dir = getattr(args, "output_dir", None) or (plan.output_dir if plan is not None else None)
    if output_dir is None:
        output_dir = Path("outputs") / "orchestrate" / run_id if run_id else Path("outputs") / "orchestrate"
    return LaunchStatusTarget(run_id=run_id, output_root=output_dir)


def resolve_cleanup_target(args, *, cwd: Path) -> LaunchCleanupTarget:
    plan: PlanConfig | None = None
    if getattr(args, "plan", None) is not None:
        plan = load_plan(args.plan.expanduser().resolve())
    runtime_value = getattr(args, "runtime", None) or (plan.runtime if plan is not None else None)
    runtime = _normalize_runtime(runtime_value) if runtime_value is not None else "docker"
    run_id = getattr(args, "run_id", None) or (plan.run_id if plan is not None else None)
    return LaunchCleanupTarget(runtime=runtime, run_id=run_id)


def validate_local_schedule(
    tasks: list[TaskSpec],
    *,
    runtime: Runtime,
    gpu_indices: list[int] | None,
    port_range: tuple[int, int],
    max_parallel: int,
) -> None:
    if runtime == "pyxis" and max_parallel != 1:
        raise ValueError("local --runtime pyxis uses the full outer allocation; max_parallel must be 1.")
    if runtime in {"docker", "podman"}:
        try:
            gpus = discover_gpus()
        except ResourceError as exc:
            raise ValueError("GPU discovery failed; ensure NVML/pynvml is available.") from exc
        discovered_indices = [gpu.index for gpu in gpus]
        if gpu_indices is not None:
            allowed_set = set(gpu_indices)
            allowed_indices = [idx for idx in discovered_indices if idx in allowed_set]
            allowed_desc = ",".join(str(idx) for idx in gpu_indices)
        else:
            allowed_indices = list(discovered_indices)
            allowed_desc = "all"
        for task in tasks:
            model_cfg = task.orchestrate.get("vllm", {}) or {}
            gpus_required = minimum_required_gpus(task)
            resolve_topology(task, allocated_gpus=gpus_required)
            require_contiguous = bool(model_cfg.get("require_contiguous_gpus", gpus_required > 1))
            if gpus_required > len(allowed_indices):
                raise ValueError(
                    f"Task {task.task_id} ({task.job_config_path}) requires gpus={gpus_required}, "
                    f"but only {len(allowed_indices)} available in range {allowed_desc}."
                )
            if (
                gpus_required > 1
                and require_contiguous
                and not _has_contiguous_run(allowed_indices, length=gpus_required)
            ):
                raise ValueError(
                    f"Task {task.task_id} ({task.job_config_path}) requires {gpus_required} contiguous GPUs, "
                    f"but allowed indices {allowed_desc} have no contiguous run."
                )
    start, end = port_range
    port_capacity = end - start + 1
    if port_capacity < max_parallel:
        raise ValueError(f"Port range {start}-{end} has {port_capacity} ports, but max_parallel={max_parallel}.")


def infer_pyxis_allocated_gpu_count() -> int:
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
            if count < 1:
                raise ValueError(f"{key} resolved to {count} GPUs; local --runtime pyxis requires at least one GPU.")
            return count
    raise ValueError(
        "Could not determine the outer GPU allocation for local --runtime pyxis. "
        "Run inside an allocation that sets visible-device or Slurm GPU env vars."
    )


def _resolve_local_auto() -> Runtime:
    attempts: list[str] = []
    probe_order = (
        ("docker", runtime_probe.docker_available),
        ("podman", runtime_probe.podman_available),
        ("pyxis", runtime_probe.pyxis_available_inside_slurm),
    )
    for runtime, probe in probe_order:
        ok, detail = probe()
        attempts.append(f"{runtime}: {detail}")
        if ok:
            return runtime
    raise ValueError("No usable local orchestration runtime found. Tried " + "; ".join(attempts))


def _normalize_runtime(value: str) -> Runtime:
    runtime = str(value).strip().lower()
    if runtime not in {"docker", "podman", "pyxis"}:
        raise ValueError(f"Unsupported runtime {value!r}; expected 'docker', 'podman', or 'pyxis'.")
    return runtime  # type: ignore[return-value]


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _has_contiguous_run(indices: list[int], *, length: int) -> bool:
    if length <= 1:
        return True
    sorted_indices = sorted(indices)
    run = 1
    for idx in range(1, len(sorted_indices)):
        if sorted_indices[idx] == sorted_indices[idx - 1] + 1:
            run += 1
        else:
            run = 1
        if run >= length:
            return True
    return False


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


__all__ = [
    "LaunchCleanupTarget",
    "LaunchPlan",
    "LaunchSource",
    "LaunchStatusTarget",
    "derive_allowed_gpu_indices",
    "derive_local_max_parallel",
    "infer_pyxis_allocated_gpu_count",
    "parse_port_range_or_default",
    "resolve_cleanup_target",
    "resolve_eval_images_config_path",
    "resolve_launch_plan",
    "resolve_output_root",
    "resolve_plan_source",
    "resolve_runtime",
    "resolve_status_target",
    "validate_local_schedule",
]
