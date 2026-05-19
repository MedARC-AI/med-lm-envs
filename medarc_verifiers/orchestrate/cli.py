"""CLI entrypoint for the vLLM orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path

from medarc_verifiers.orchestrate.config import TaskSpec, expand_tasks, load_plan, make_plan
from medarc_verifiers.orchestrate.docker_vllm import cleanup_orphan_containers as cleanup_docker_orphans
from medarc_verifiers.orchestrate.podman_vllm import cleanup_orphan_containers as cleanup_podman_orphans
from medarc_verifiers.orchestrate.resources import (
    PortOnlyResourceManager,
    ResourceError,
    ResourceManager,
    discover_gpus,
    parse_index_range,
)
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.state import filter_tasks_for_resume, load_summary
from medarc_verifiers.orchestrate.topology import minimum_required_gpus, resolve_topology
from medarc_verifiers.utils.run_naming import generate_run_id


def build_record_failure_parser(*, prog: str = "medarc-orchestrate record-failure") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Record a bundled orchestrator task failure before worker start.")
    parser.add_argument("--task-spec", type=Path, required=True, help="Path to bundled task.yaml.")
    parser.add_argument("--allocation", type=Path, required=True, help="Path to execution allocation JSON.")
    parser.add_argument("--reason", required=True, help="Machine-readable failure reason.")
    parser.add_argument("--message", required=True, help="Human-readable failure message.")
    return parser


def _add_local_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path, help="Path to orchestrator plan YAML.")
    source.add_argument(
        "--job-config",
        action="append",
        type=Path,
        dest="job_configs",
        help="Job config to orchestrate. Repeat to launch multiple job configs without a wrapper plan file.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional bundle name when using --job-config directly (used for run-id prefix when --run-id is unset).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Dotenv file shared by runtime launches (overrides plan env_file; defaults to repo .env when present).",
    )
    parser.add_argument(
        "--runtime",
        choices=("docker", "podman", "pyxis"),
        default=None,
        help="Serve runtime backend (defaults to docker when available, otherwise podman, unless plan.runtime is set).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved tasks and exit without running.",
    )
    parser.add_argument("--gpu-range", help="Restrict GPU indices (e.g. 0-3 or 0,2,3).")
    parser.add_argument("--port-range", help="Restrict ports (e.g. 8000-8999).")
    parser.add_argument("--run-id", help="Run identifier (default: timestamp).")
    parser.add_argument("--output-dir", type=Path, help="Override output directory root.")
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum concurrent tasks (defaults to GPU count when unset).",
    )
    parser.add_argument("--readiness-timeout-s", type=int, default=None, help="Readiness timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already marked completed.")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun failed tasks when resuming.")
    parser.add_argument("--status", action="store_true", help="Print current status from summary and exit.")
    parser.add_argument(
        "--kill-orphans",
        action="store_true",
        help="Clean up containers labeled as orchestrator-managed.",
    )
    parser.add_argument(
        "--prune-logs-on-success",
        action="store_true",
        help="Delete per-task serve/bench logs for completed tasks (kept for failures).",
    )
    parser.add_argument(
        "--no-uv-run",
        action="store_true",
        help="Run 'medarc-eval bench' directly instead of via 'uv run' (use when venv is pre-activated).",
    )
    parser.add_argument("--orchestrate-config", type=Path, help="Path to model-serving orchestrate.toml registry.")
    parser.add_argument("--eval-images-config", type=Path, help="Path to eval auxiliary image registry TOML.")
    parser.add_argument("--endpoints-path", type=Path, help="Path to endpoints.toml used for model alias resolution and bench.")
    parser.set_defaults(command="local", handler=_run_local)


def build_local_parser(*, prog: str = "medarc-orchestrate local") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run vLLM orchestration locally or inside an existing allocation.",
    )
    _add_local_arguments(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate",
        description="Run vLLM orchestration with explicit execution modes.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{local,slurm}")
    subparsers.required = True

    local_parser = subparsers.add_parser(
        "local",
        description="Run vLLM orchestration locally or inside an existing allocation.",
        help="Run tasks locally with docker/podman autodetection or an explicit runtime.",
    )
    _add_local_arguments(local_parser)

    from medarc_verifiers.orchestrate.slurm.cli import add_slurm_subparser

    add_slurm_subparser(subparsers)
    return parser


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


def _validate_schedule(
    tasks: list[TaskSpec],
    *,
    runtime: str,
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
            if gpus_required > 1 and require_contiguous and not _has_contiguous_run(allowed_indices, length=gpus_required):
                raise ValueError(
                    f"Task {task.task_id} ({task.job_config_path}) requires {gpus_required} contiguous GPUs, "
                    f"but allowed indices {allowed_desc} have no contiguous run."
                )
    start, end = port_range
    if end < start:
        raise ValueError(f"Port range is invalid: {start}-{end}.")
    port_capacity = end - start + 1
    if port_capacity < max_parallel:
        raise ValueError(f"Port range {start}-{end} has {port_capacity} ports, but max_parallel={max_parallel}.")


def _run_local(args: argparse.Namespace) -> int:
    plan_base_dir = Path.cwd()
    if args.plan is not None:
        plan_path = args.plan.expanduser().resolve()
        plan = load_plan(plan_path)
        plan_base_dir = plan_path.parent
    else:
        plan = make_plan(
            job_configs=args.job_configs or [],
            base_dir=plan_base_dir,
            name=args.name,
            orchestrate_config=args.orchestrate_config,
            eval_images_config=args.eval_images_config,
            endpoints_path=args.endpoints_path,
        )
    if args.orchestrate_config is not None:
        plan.orchestrate_config = args.orchestrate_config.expanduser().resolve()
    if args.eval_images_config is not None:
        plan.eval_images_config = args.eval_images_config.expanduser().resolve()
    if args.endpoints_path is not None:
        plan.endpoints_path = args.endpoints_path.expanduser().resolve()
    runtime = args.runtime or plan.runtime or _default_local_runtime()
    if args.env_file is not None:
        env_file = args.env_file.expanduser()
        if not env_file.is_absolute():
            env_file = plan_base_dir / env_file
        plan.env_file = env_file.resolve()
    configured_run_id = args.run_id or plan.run_id
    if args.kill_orphans or plan.kill_orphans:
        removed = _cleanup_orphans(runtime=runtime, run_id=configured_run_id)
        if removed:
            print("\n".join(removed))
        return 0
    tasks = expand_tasks(plan)
    if configured_run_id:
        run_id = configured_run_id
    else:
        run_id = generate_run_id(plan.name)
    output_root = args.output_dir or plan.output_dir or Path("outputs") / "orchestrate" / run_id
    gpu_range = args.gpu_range or plan.gpu_range
    if gpu_range:
        gpu_indices = parse_index_range(gpu_range)
    else:
        gpu_indices = None
    port_range_expr = args.port_range or plan.port_range
    if port_range_expr:
        start_str, end_str = port_range_expr.split("-", maxsplit=1)
        port_range = (int(start_str), int(end_str))
    else:
        port_range = (8000, 8999)
    if args.max_parallel is not None:
        max_parallel = args.max_parallel
    elif plan.max_parallel is not None:
        max_parallel = plan.max_parallel
    else:
        if runtime == "pyxis":
            max_parallel = 1
        elif gpu_indices is not None:
            max_parallel = max(1, len(gpu_indices))
        else:
            try:
                max_parallel = max(1, len(discover_gpus()))
            except ResourceError:
                max_parallel = 1
    readiness_timeout_s = (
        args.readiness_timeout_s if args.readiness_timeout_s is not None else (plan.readiness_timeout_s or 1800)
    )
    resume = args.resume or plan.resume
    rerun_failed = args.rerun_failed or plan.rerun_failed
    summary_path = output_root / "summary.json"
    if args.status:
        summary = load_summary(summary_path)
        for entry in summary.get("tasks", []):
            print(f"{entry.get('task_id')}\t{entry.get('state')}\t{entry.get('model_id')}")
        return 0
    if resume and summary_path.exists():
        summary = load_summary(summary_path)
        tasks = filter_tasks_for_resume(tasks, summary, rerun_failed=rerun_failed)
    if tasks:
        _validate_schedule(tasks, runtime=runtime, gpu_indices=gpu_indices, port_range=port_range, max_parallel=max_parallel)
    if args.dry_run:
        for task in tasks:
            print(f"{task.task_id}\t{task.model_id}\t{task.job_config_path}")
        return 0
    allocated_gpu_count = _infer_pyxis_allocated_gpu_count() if runtime == "pyxis" else None
    prune_logs_on_success = args.prune_logs_on_success or plan.prune_logs_on_success
    options = OrchestratorOptions(
        run_id=run_id,
        output_root=output_root,
        readiness_timeout_s=readiness_timeout_s,
        max_parallel=max_parallel,
        prune_logs_on_success=prune_logs_on_success,
        allocated_gpu_count=allocated_gpu_count,
    )
    if runtime in {"docker", "podman"}:
        resource_manager = ResourceManager(gpu_indices=gpu_indices, port_range=port_range)
    else:
        resource_manager = PortOnlyResourceManager(port_range=port_range)
    uv_run = not args.no_uv_run and plan.uv_run
    runner = OrchestratorRunner(
        plan, tasks, resource_manager, options=options, runtime=runtime, uv_run=uv_run
    )
    runner.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "worker":
        from medarc_verifiers.orchestrate.worker import main as worker_main

        return worker_main(argv[1:])
    if argv and argv[0] == "record-failure":
        return _run_record_failure(build_record_failure_parser().parse_args(argv[1:]))
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args)


__all__ = ["build_local_parser", "build_parser", "build_record_failure_parser", "main"]


def _run_record_failure(args: argparse.Namespace) -> int:
    from medarc_verifiers.orchestrate.bundle import RuntimeState, load_execution_allocation, load_task_spec, write_runtime_state
    from medarc_verifiers.orchestrate.state import (
        JobState,
        TaskManifest,
        TaskPaths,
        upsert_summary_entry,
        write_task_manifest,
        write_task_result,
    )

    task_spec = load_task_spec(args.task_spec.expanduser().resolve())
    allocation = load_execution_allocation(args.allocation.expanduser().resolve())
    if allocation is None:
        raise FileNotFoundError(f"Execution allocation not found: {args.allocation}")
    paths = TaskPaths(Path(task_spec.output_paths.root))
    manifest = TaskManifest(
        task_id=task_spec.task_id,
        config_path=task_spec.bundled_eval_config_path,
        model_key=task_spec.model_key,
        model_id=task_spec.model_id,
        state=JobState.failed,
        failure_reason=str(args.reason),
        error=str(args.message),
        gpu_ids=list(allocation.gpu_ids),
        port=allocation.server_port,
        gpus=task_spec.gpus,
        tensor_parallel_size=task_spec.tensor_parallel_size,
        data_parallel_size=task_spec.data_parallel_size,
        allocated_gpus=allocation.allocated_gpus,
    )
    manifest.completed_at = manifest.updated_at
    write_task_manifest(paths, manifest)
    write_task_result(paths, {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error})
    write_runtime_state(
        paths.state_path,
        RuntimeState(
            task_id=task_spec.task_id,
            state=JobState.failed,
            restart_source=task_spec.restart_source,
            restart_source_strategy=task_spec.restart_source_strategy,
        ),
    )
    upsert_summary_entry(paths.root.parents[1] / "summary.json", manifest)
    return 0


def _default_local_runtime() -> str:
    if importlib.util.find_spec("docker") is not None:
        return "docker"
    if shutil.which("podman") is not None:
        return "podman"
    return "docker"


def _infer_pyxis_allocated_gpu_count() -> int:
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


def _cleanup_orphans(*, runtime: str, run_id: str | None) -> list[str]:
    if runtime == "podman":
        return cleanup_podman_orphans(run_id=run_id)
    if runtime == "docker":
        return cleanup_docker_orphans(run_id=run_id)
    return []
