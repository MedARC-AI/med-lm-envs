"""CLI entrypoint for the vLLM orchestrator."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from datetime import datetime

from medarc_verifiers.orchestrate.config import TaskSpec, expand_tasks, load_plan
from medarc_verifiers.orchestrate.docker_vllm import cleanup_orphan_containers
from medarc_verifiers.orchestrate.resources import ResourceError, ResourceManager, discover_gpus, parse_index_range
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.state import filter_tasks_for_resume, load_summary


_RUN_ID_ALLOWED = re.compile(r"[^a-zA-Z0-9_.-]+")


def _slug_run_id(value: str, *, fallback: str = "run") -> str:
    cleaned = _RUN_ID_ALLOWED.sub("-", value).strip("-.")
    return cleaned or fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate",
        description="Run vLLM orchestration over job configs.",
    )
    parser.add_argument("--plan", required=True, type=Path, help="Path to orchestrator plan YAML.")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Dotenv file shared by all Docker launches (overrides plan env_file; defaults to repo .env when present).",
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
    gpu_indices: list[int] | None,
    port_range: tuple[int, int],
    max_parallel: int,
) -> None:
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
        model_cfg = task.orchestrate.get(task.model_key, {}) or {}
        gpus_required = int(model_cfg.get("gpus", 1))
        require_contiguous = bool(model_cfg.get("require_contiguous_gpus", gpus_required > 1))
        if gpus_required > len(allowed_indices):
            raise ValueError(
                f"Task {task.task_id} ({task.job_config_path}) requests {gpus_required} GPUs, "
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
        raise ValueError(
            f"Port range {start}-{end} has {port_capacity} ports, but max_parallel={max_parallel}."
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan_path = args.plan.expanduser().resolve()
    plan = load_plan(plan_path)
    if args.env_file is not None:
        env_file = args.env_file.expanduser()
        if not env_file.is_absolute():
            env_file = plan_path.parent / env_file
        plan.env_file = env_file.resolve()
    tasks = expand_tasks(plan)
    configured_run_id = args.run_id or plan.run_id
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    if configured_run_id:
        run_id = configured_run_id
    elif plan.name:
        run_id = f"{_slug_run_id(plan.name)}-{timestamp}"
    else:
        run_id = timestamp
    output_root = args.output_dir or plan.output_dir or Path("outputs") / "orchestrator" / run_id
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
        if gpu_indices is not None:
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
    if args.kill_orphans or plan.kill_orphans:
        removed = cleanup_orphan_containers(run_id=configured_run_id)
        if removed:
            print("\n".join(removed))
        return 0
    if resume and summary_path.exists():
        summary = load_summary(summary_path)
        tasks = filter_tasks_for_resume(tasks, summary, rerun_failed=rerun_failed)
    if tasks:
        _validate_schedule(tasks, gpu_indices=gpu_indices, port_range=port_range, max_parallel=max_parallel)
    if args.dry_run:
        for task in tasks:
            print(f"{task.task_id}\t{task.model_id}\t{task.job_config_path}")
        return 0
    prune_logs_on_success = args.prune_logs_on_success or plan.prune_logs_on_success
    options = OrchestratorOptions(
        run_id=run_id,
        output_root=output_root,
        readiness_timeout_s=readiness_timeout_s,
        max_parallel=max_parallel,
        prune_logs_on_success=prune_logs_on_success,
    )
    runner = OrchestratorRunner(plan, tasks, ResourceManager(gpu_indices=gpu_indices, port_range=port_range), options=options)
    runner.run()
    return 0


__all__ = ["build_parser", "main"]
