"""CLI entrypoint for the vLLM orchestrator."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

from medarc_verifiers.orchestrate.config import expand_tasks, load_plan
from medarc_verifiers.orchestrate.docker_vllm import cleanup_orphan_containers
from medarc_verifiers.orchestrate.resources import ResourceManager, parse_index_range
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner
from medarc_verifiers.orchestrate.state import filter_tasks_for_resume, load_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate",
        description="Run vLLM orchestration over job configs.",
    )
    parser.add_argument("--plan", required=True, type=Path, help="Path to orchestrator plan YAML.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved tasks and exit without running.",
    )
    parser.add_argument("--gpu-range", help="Restrict GPU indices (e.g. 0-3 or 0,2,3).")
    parser.add_argument("--port-range", help="Restrict ports (e.g. 8000-8999).")
    parser.add_argument("--run-id", help="Run identifier (default: timestamp).")
    parser.add_argument("--output-dir", type=Path, help="Override output directory root.")
    parser.add_argument("--max-parallel", type=int, default=1, help="Maximum concurrent tasks.")
    parser.add_argument("--readiness-timeout-s", type=int, default=1800, help="Readiness timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Skip tasks already marked completed.")
    parser.add_argument("--rerun-failed", action="store_true", help="Rerun failed tasks when resuming.")
    parser.add_argument("--status", action="store_true", help="Print current status from summary and exit.")
    parser.add_argument(
        "--kill-orphans",
        action="store_true",
        help="Clean up containers labeled as orchestrator-managed.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan = load_plan(args.plan)
    tasks = expand_tasks(plan)
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_root = args.output_dir or Path("outputs") / "orchestrator" / run_id
    if args.gpu_range:
        gpu_indices = parse_index_range(args.gpu_range)
    else:
        gpu_indices = None
    if args.port_range:
        start_str, end_str = args.port_range.split("-", maxsplit=1)
        port_range = (int(start_str), int(end_str))
    else:
        port_range = (8000, 8999)
    summary_path = output_root / "summary.json"
    if args.status:
        summary = load_summary(summary_path)
        for entry in summary.get("tasks", []):
            print(f"{entry.get('task_id')}\t{entry.get('state')}\t{entry.get('model_id')}")
        return 0
    if args.kill_orphans:
        removed = cleanup_orphan_containers(run_id=run_id if args.run_id else None)
        if removed:
            print("\n".join(removed))
        return 0
    if args.resume and summary_path.exists():
        summary = load_summary(summary_path)
        tasks = filter_tasks_for_resume(tasks, summary, rerun_failed=args.rerun_failed)
    if args.dry_run:
        for task in tasks:
            print(f"{task.task_id}\t{task.model_id}\t{task.job_config_path}")
        return 0
    options = OrchestratorOptions(
        run_id=run_id,
        output_root=output_root,
        readiness_timeout_s=args.readiness_timeout_s,
        max_parallel=args.max_parallel,
    )
    runner = OrchestratorRunner(plan, tasks, ResourceManager(gpu_indices=gpu_indices, port_range=port_range), options=options)
    runner.run()
    return 0


__all__ = ["build_parser", "main"]
