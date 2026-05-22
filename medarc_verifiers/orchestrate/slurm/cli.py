"""CLI entrypoint for Slurm-native orchestration submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from medarc_verifiers.orchestrate.launch import resolve_launch_plan

from .manifest import SlurmBundleManifest, load_bundle_manifest, write_bundle_manifest
from .plan import SlurmCliOverrides, build_submission_plan
from .render import render_bundle
from .submit import mark_dry_run, submit_bundle


def add_slurm_executor_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--node-gpus", type=int, default=None, help="Outer Slurm GPU allocation per job.")
    parser.add_argument(
        "--max-simultaneous-nodes",
        type=int,
        default=None,
        help="Maximum number of Slurm jobs to run concurrently.",
    )
    parser.add_argument(
        "--run-simultaneously",
        action="store_true",
        help="Submit all tasks without generated inter-task dependencies.",
    )
    parser.add_argument("--cpus-per-gpu", type=int, default=None)
    parser.add_argument("--time", default=None)
    parser.add_argument("--partition", default=None)
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--nice", type=int, default=None)
    parser.add_argument("--dependency", default=None, help="Base sbatch dependency applied to each chain head.")
    parser.add_argument("--mail-type", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--test-only", action="store_true", help="Run sbatch --test-only instead of submitting jobs.")
    parser.add_argument(
        "--slurm-resume",
        action="store_true",
        default=None,
        help="Mark jobs requeueable and pass --resume to the inner orchestrator.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Repository root containing the medarc-orchestrate checkout.",
    )
    parser.add_argument(
        "--activate-script",
        type=Path,
        default=None,
        help="Shell activation script sourced before running medarc-orchestrate inside sbatch (defaults to <source-dir>/.venv/bin/activate).",
    )


def run_from_args(args: argparse.Namespace) -> int:
    launch = resolve_launch_plan(args, backend="slurm", cwd=Path.cwd())
    plan = launch.plan
    tasks = launch.tasks
    run_id = launch.run_id
    output_root = launch.output_root.expanduser().resolve()
    node_gpus = args.node_gpus if args.node_gpus is not None else 8
    max_simultaneous_nodes = args.max_simultaneous_nodes if args.max_simultaneous_nodes is not None else 1
    source_dir = (args.source_dir or Path.cwd()).expanduser().resolve()
    if args.activate_script is not None:
        activate_script = args.activate_script.expanduser()
        if not activate_script.is_absolute():
            activate_script = source_dir / activate_script
        activate_script = activate_script.resolve()
    else:
        activate_script = (source_dir / ".venv" / "bin" / "activate").resolve()
    readiness_timeout_s = launch.readiness_timeout_s
    cli_overrides = SlurmCliOverrides(
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
    planned_tasks = build_submission_plan(
        tasks,
        run_id=run_id,
        node_gpus=node_gpus,
        max_simultaneous_nodes=max_simultaneous_nodes,
        run_simultaneously=bool(args.run_simultaneously),
        base_dependency=args.dependency,
        cli_overrides=cli_overrides,
    )

    manifest_path = output_root / "submission_manifest.json"
    existing_manifest = _load_existing_manifest(manifest_path, run_id=run_id)
    manifest = render_bundle(
        planned_tasks=planned_tasks,
        bundle_root=output_root,
        run_id=run_id,
        node_gpus=node_gpus,
        source_dir=source_dir,
        activate_script=activate_script,
        env_file=plan.env_file,
        readiness_timeout_s=readiness_timeout_s,
        prune_logs_on_success=plan.prune_logs_on_success,
        existing_manifest=existing_manifest,
    )
    write_bundle_manifest(manifest_path, manifest)

    if args.dry_run:
        for command in mark_dry_run(manifest_path, manifest):
            print(command)
        return 0

    submit_bundle(manifest_path, manifest, test_only=bool(args.test_only))
    return 0


def _load_existing_manifest(path: Path, *, run_id: str) -> SlurmBundleManifest | None:
    if not path.exists():
        return None
    manifest = load_bundle_manifest(path)
    if manifest.run_id != run_id:
        raise ValueError(f"Existing Slurm manifest at {path} belongs to run_id={manifest.run_id}, not {run_id}.")
    return manifest


__all__ = ["add_slurm_executor_arguments", "run_from_args"]
