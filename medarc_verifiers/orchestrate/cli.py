"""CLI entrypoint for the vLLM orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from medarc_verifiers.orchestrate.docker_vllm import cleanup_orphan_containers as cleanup_docker_orphans
from medarc_verifiers.orchestrate.launch import (
    LaunchPlan,
    infer_pyxis_allocated_gpu_count,
    resolve_cleanup_target,
    resolve_launch_plan,
    resolve_status_target,
    validate_local_schedule,
)
from medarc_verifiers.orchestrate.podman_vllm import cleanup_orphan_containers as cleanup_podman_orphans
from medarc_verifiers.orchestrate.resources import PortOnlyResourceManager, ResourceManager
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.state import filter_tasks_for_resume, load_summary


def build_record_failure_parser(*, prog: str = "medarc-orchestrate record-failure") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Record a bundled orchestrator task failure before worker start."
    )
    parser.add_argument("--task-spec", type=Path, required=True, help="Path to bundled task.yaml.")
    parser.add_argument("--allocation", type=Path, required=True, help="Path to execution allocation JSON.")
    parser.add_argument("--reason", required=True, help="Machine-readable failure reason.")
    parser.add_argument("--message", required=True, help="Human-readable failure message.")
    return parser


def _add_local_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--plan", type=Path, help="Path to orchestrator plan file.")
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
    parser.add_argument("--runtime", choices=("docker", "podman", "pyxis"), default=None, help="Serve runtime backend.")
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
    parser.add_argument("--eval-images-config", type=Path, help="Path to eval auxiliary image registry TOML.")
    parser.add_argument(
        "--endpoints-path",
        type=Path,
        help="Path to endpoint registry TOML with [endpoint.orchestrate] blocks.",
    )
    parser.set_defaults(command="run", handler=_run_launch, backend="local")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate",
        description="Run vLLM orchestration with explicit execution modes.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{run}")
    subparsers.required = True

    from medarc_verifiers.orchestrate.slurm.cli import add_slurm_executor_arguments

    run_parser = subparsers.add_parser(
        "run",
        description="Run vLLM orchestration locally or submit via Slurm.",
        help="Canonical orchestration launcher.",
    )
    _add_local_arguments(run_parser)
    run_parser.add_argument("--backend", choices=("local", "slurm"), default="local")
    add_slurm_executor_arguments(run_parser)
    run_parser.set_defaults(command="run", handler=_run_launch)
    return parser


def _run_launch(args: argparse.Namespace) -> int:
    backend = getattr(args, "backend", "local")
    if backend == "slurm":
        _reject_local_only_slurm_args(args)
    else:
        _reject_slurm_only_local_args(args)
    if args.status:
        target = resolve_status_target(args, cwd=Path.cwd())
        summary = load_summary(target.output_root / "summary.json")
        for entry in summary.get("tasks", []):
            print(f"{entry.get('task_id')}\t{entry.get('state')}\t{entry.get('model_id')}")
        return 0
    if args.kill_orphans:
        target = resolve_cleanup_target(args, cwd=Path.cwd())
        removed = _cleanup_orphans(runtime=target.runtime, run_id=target.run_id)
        if removed:
            print("\n".join(removed))
        return 0
    _require_source(args)
    if backend == "slurm":
        from medarc_verifiers.orchestrate.slurm.cli import run_from_args

        return run_from_args(args)
    launch = resolve_launch_plan(args, backend="local", cwd=Path.cwd())
    return run_local_launch_plan(launch, dry_run=bool(args.dry_run), resume=bool(args.resume), rerun_failed=bool(args.rerun_failed))


def run_local_launch_plan(
    launch: LaunchPlan,
    *,
    dry_run: bool = False,
    resume: bool = False,
    rerun_failed: bool = False,
) -> int:
    tasks = list(launch.tasks)
    resume = resume or launch.plan.resume
    rerun_failed = rerun_failed or launch.plan.rerun_failed
    summary_path = launch.output_root / "summary.json"
    if resume and summary_path.exists():
        summary = load_summary(summary_path)
        tasks = filter_tasks_for_resume(tasks, summary, rerun_failed=rerun_failed)
    if tasks:
        validate_local_schedule(
            tasks,
            runtime=launch.runtime,
            gpu_indices=launch.gpu_indices,
            port_range=launch.port_range,
            max_parallel=launch.max_parallel,
        )
    if dry_run:
        for task in tasks:
            print(f"{task.task_id}\t{task.model_id}\t{task.job_config_path}")
        return 0
    allocated_gpu_count = infer_pyxis_allocated_gpu_count() if launch.runtime == "pyxis" else None
    prune_logs_on_success = launch.plan.prune_logs_on_success
    options = OrchestratorOptions(
        run_id=launch.run_id,
        output_root=launch.output_root,
        readiness_timeout_s=launch.readiness_timeout_s,
        max_parallel=launch.max_parallel,
        prune_logs_on_success=prune_logs_on_success,
        allocated_gpu_count=allocated_gpu_count,
    )
    if launch.runtime in {"docker", "podman"}:
        resource_manager = ResourceManager(gpu_indices=launch.gpu_indices, port_range=launch.port_range)
    else:
        resource_manager = PortOnlyResourceManager(port_range=launch.port_range)
    runner = OrchestratorRunner(
        launch.plan,
        tasks,
        resource_manager,
        options=options,
        runtime=launch.runtime,
        uv_run=launch.uv_run,
    )
    runner.run()
    return 0


def _require_source(args: argparse.Namespace) -> None:
    if args.plan is None and not args.job_configs:
        raise SystemExit("medarc-orchestrate run requires --plan or at least one --job-config.")


def _reject_local_only_slurm_args(args: argparse.Namespace) -> None:
    local_only = {
        "--runtime": getattr(args, "runtime", None),
        "--gpu-range": getattr(args, "gpu_range", None),
        "--port-range": getattr(args, "port_range", None),
        "--max-parallel": getattr(args, "max_parallel", None),
    }
    used = [flag for flag, value in local_only.items() if value is not None]
    if bool(getattr(args, "resume", False)):
        used.append("--resume")
    if bool(getattr(args, "rerun_failed", False)):
        used.append("--rerun-failed")
    if bool(getattr(args, "no_uv_run", False)):
        used.append("--no-uv-run")
    if used:
        raise SystemExit(
            "medarc-orchestrate run --backend slurm does not accept local launch flags: " + ", ".join(used)
        )


def _reject_slurm_only_local_args(args: argparse.Namespace) -> None:
    slurm_only = {
        "--node-gpus": getattr(args, "node_gpus", None),
        "--max-simultaneous-nodes": getattr(args, "max_simultaneous_nodes", None),
        "--cpus-per-gpu": getattr(args, "cpus_per_gpu", None),
        "--time": getattr(args, "time", None),
        "--partition": getattr(args, "partition", None),
        "--account": getattr(args, "account", None),
        "--qos": getattr(args, "qos", None),
        "--nice": getattr(args, "nice", None),
        "--dependency": getattr(args, "dependency", None),
        "--mail-type": getattr(args, "mail_type", None),
        "--mail-user": getattr(args, "mail_user", None),
        "--source-dir": getattr(args, "source_dir", None),
        "--activate-script": getattr(args, "activate_script", None),
    }
    used = [flag for flag, value in slurm_only.items() if value is not None]
    if bool(getattr(args, "run_simultaneously", False)):
        used.append("--run-simultaneously")
    if bool(getattr(args, "test_only", False)):
        used.append("--test-only")
    if getattr(args, "slurm_resume", None) is not None:
        used.append("--slurm-resume")
    if used:
        raise SystemExit("medarc-orchestrate run does not accept Slurm flags without --backend slurm: " + ", ".join(used))


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


__all__ = ["build_parser", "build_record_failure_parser", "main"]


def _run_record_failure(args: argparse.Namespace) -> int:
    from medarc_verifiers.orchestrate.bundle import (
        RuntimeState,
        load_execution_allocation,
        load_task_spec,
        write_runtime_state,
    )
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
    write_task_result(
        paths, {"state": JobState.failed, "failure_reason": manifest.failure_reason, "error": manifest.error}
    )
    write_runtime_state(
        paths.state_path,
        RuntimeState(task_id=task_spec.task_id, state=JobState.failed),
    )
    upsert_summary_entry(paths.root.parents[1] / "summary.json", manifest)
    return 0


def _cleanup_orphans(*, runtime: str, run_id: str | None) -> list[str]:
    if runtime == "podman":
        return cleanup_podman_orphans(run_id=run_id)
    if runtime == "docker":
        return cleanup_docker_orphans(run_id=run_id)
    return []
