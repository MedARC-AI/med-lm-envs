"""CLI entrypoint for the vLLM orchestrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from medarc_verifiers.orchestrate.docker_vllm import cleanup_orphan_containers as cleanup_docker_orphans
from medarc_verifiers.orchestrate.launch import (
    LaunchRequest,
    resolve_launch_plan,
    resolve_status_target,
)
from medarc_verifiers.orchestrate.podman_vllm import cleanup_orphan_containers as cleanup_podman_orphans
from medarc_verifiers.orchestrate.slurm.submit import SlurmSubmissionOptions, submit_slurm_launch_plan


def build_record_failure_parser(*, prog: str = "medarc-orchestrate record-failure") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description="Record a bundled orchestrator task failure before worker start."
    )
    parser.add_argument("--task-spec", type=Path, required=True, help="Path to bundled task.yaml.")
    parser.add_argument("--allocation", type=Path, required=True, help="Path to execution allocation JSON.")
    parser.add_argument("--reason", required=True, help="Machine-readable failure reason.")
    parser.add_argument("--message", required=True, help="Human-readable failure message.")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate",
        description="Run Slurm/Pyxis vLLM orchestration.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="{run,status,cleanup}")
    subparsers.required = True

    run_parser = subparsers.add_parser("run", description="Submit Slurm orchestration jobs.")
    _add_source_arguments(run_parser)
    run_parser.add_argument("--dry-run", action="store_true", help="Render bundle and print sbatch commands.")
    run_parser.add_argument("--run-id", help="Run identifier.")
    run_parser.add_argument("--output-dir", type=Path, help="Override output directory root.")
    run_parser.add_argument("--readiness-timeout-s", type=int, default=None, help="Readiness timeout in seconds.")
    run_parser.add_argument("--prune-logs-on-success", action="store_true")
    run_parser.add_argument("--eval-images-config", type=Path, help="Path to eval auxiliary image registry TOML.")
    run_parser.add_argument("--endpoints-path", type=Path, help="Path to endpoint registry TOML.")
    _add_slurm_arguments(run_parser)
    run_parser.set_defaults(handler=_run_launch)

    status_parser = subparsers.add_parser("status", description="Print orchestrator status artifacts.")
    status_parser.add_argument("--run-id", help="Run identifier under outputs/orchestrate.")
    status_parser.add_argument("--output-dir", type=Path, help="Run output directory.")
    status_parser.add_argument("--json", action="store_true", help="Print combined status JSON.")
    status_parser.set_defaults(handler=_run_status)

    cleanup_parser = subparsers.add_parser("cleanup", description="Clean local runtime leftovers from tests/dev.")
    cleanup_parser.add_argument("--runtime", choices=("docker", "podman"), required=True)
    cleanup_parser.add_argument("--run-id", help="Only clean containers for this run id.")
    cleanup_parser.set_defaults(handler=_run_cleanup)
    return parser


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument("--plan", type=Path, help="Path to orchestrator plan file.")
    source.add_argument(
        "--job-config",
        action="append",
        type=Path,
        dest="job_configs",
        help="Job config to orchestrate. Repeat to launch multiple job configs without a wrapper plan file.",
    )
    parser.add_argument("--name", default=None, help="Optional bundle name when using --job-config directly.")
    parser.add_argument("--env-file", type=Path, default=None, help="Dotenv file shared by runtime launches.")


def _add_slurm_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-gpus", type=int, default=8, help="Outer Slurm GPU allocation per job.")
    parser.add_argument("--max-simultaneous-nodes", type=int, default=1)
    parser.add_argument("--run-simultaneously", action="store_true")
    parser.add_argument("--cpus-per-gpu", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--nice", type=int, default=None)
    parser.add_argument("--dependency", default=None, help="Base sbatch dependency applied to each chain head.")
    parser.add_argument("--mail-type", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--slurm-resume", action="store_true", default=None)
    parser.add_argument("--source-dir", type=Path, default=None)
    parser.add_argument("--activate-script", type=Path, default=None)


def _run_launch(args: argparse.Namespace) -> int:
    _require_source(args)
    request = LaunchRequest(
        plan=args.plan,
        job_configs=tuple(args.job_configs or ()),
        name=args.name,
        env_file=args.env_file,
        run_id=args.run_id,
        output_dir=args.output_dir,
        readiness_timeout_s=args.readiness_timeout_s,
        prune_logs_on_success=bool(args.prune_logs_on_success),
        eval_images_config=args.eval_images_config,
        endpoints_path=args.endpoints_path,
    )
    launch = resolve_launch_plan(request, cwd=Path.cwd())
    source_dir = (args.source_dir or Path.cwd()).expanduser().resolve()
    activate_script = (
        args.activate_script.expanduser() if args.activate_script is not None else source_dir / ".venv/bin/activate"
    )
    if not activate_script.is_absolute():
        activate_script = source_dir / activate_script
    options = SlurmSubmissionOptions(
        node_gpus=args.node_gpus,
        max_simultaneous_nodes=args.max_simultaneous_nodes,
        run_simultaneously=bool(args.run_simultaneously),
        base_dependency=args.dependency,
        test_only=bool(args.test_only),
        dry_run=bool(args.dry_run),
        source_dir=source_dir,
        activate_script=activate_script.resolve(),
        cpus_per_gpu=args.cpus_per_gpu,
        time=args.time,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        nice=args.nice,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        slurm_resume=args.slurm_resume,
    )
    return submit_slurm_launch_plan(launch, options)


def _run_status(args: argparse.Namespace) -> int:
    target = resolve_status_target(run_id=args.run_id, output_dir=args.output_dir, cwd=Path.cwd())
    payload = _load_combined_status(target.output_root)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for row in payload["tasks"]:
        print(
            "\t".join(
                str(row.get(field) or "")
                for field in (
                    "task_id",
                    "submit_state",
                    "worker_state",
                    "slurm_job_id",
                    "dependency",
                    "model_id",
                    "failure_reason",
                    "error",
                )
            )
        )
    return 0


def _run_cleanup(args: argparse.Namespace) -> int:
    removed = _cleanup_orphans(runtime=args.runtime, run_id=args.run_id)
    if removed:
        print("\n".join(removed))
    return 0


def _load_combined_status(output_root: Path) -> dict[str, object]:
    manifest_path = output_root / "submission_manifest.json"
    summary_path = output_root / "summary.json"
    manifest = _load_json_artifact(manifest_path) if manifest_path.exists() else None
    summary = _load_json_artifact(summary_path) if summary_path.exists() else None
    if manifest is None and summary is None:
        raise SystemExit(
            f"No orchestrator status found at {output_root}: missing submission_manifest.json and summary.json."
        )
    rows: dict[str, dict[str, object]] = {}
    if isinstance(manifest, dict):
        for entry in manifest.get("entries", []) or []:
            if not isinstance(entry, dict):
                continue
            task_id = str(entry.get("task_id") or "")
            row = rows.setdefault(task_id, {"task_id": task_id})
            row.update(
                {
                    "submit_state": entry.get("state"),
                    "slurm_job_id": entry.get("slurm_job_id"),
                    "dependency": entry.get("generated_dependency") or entry.get("base_dependency"),
                }
            )
    if isinstance(summary, dict):
        for entry in summary.get("tasks", []) or []:
            if not isinstance(entry, dict):
                continue
            task_id = str(entry.get("task_id") or "")
            row = rows.setdefault(task_id, {"task_id": task_id})
            row.update(
                {
                    "worker_state": entry.get("state"),
                    "model_id": entry.get("model_id"),
                    "failure_reason": entry.get("failure_reason"),
                    "error": entry.get("error"),
                }
            )
    return {
        "output_root": str(output_root),
        "submission_manifest": manifest,
        "summary": summary,
        "tasks": [rows[key] for key in sorted(rows)],
    }


def _load_json_artifact(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Malformed orchestrator status artifact: {path}") from exc


def _require_source(args: argparse.Namespace) -> None:
    if args.plan is None and not args.job_configs:
        raise SystemExit("medarc-orchestrate run requires --plan or at least one --job-config.")


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
    raise ValueError(f"cleanup --runtime {runtime!r} is not supported.")
