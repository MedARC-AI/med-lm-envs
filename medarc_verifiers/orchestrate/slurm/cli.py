"""CLI entrypoint for Slurm-native orchestration submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from medarc_verifiers.orchestrate.config import expand_tasks, load_plan, make_plan
from medarc_verifiers.utils.run_naming import generate_run_id

from .manifest import SlurmBundleManifest, load_bundle_manifest, write_bundle_manifest
from .plan import SlurmCliOverrides, build_submission_plan
from .render import render_bundle
from .submit import mark_dry_run, submit_bundle


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--plan", type=Path, help="Path to orchestrator plan YAML.")
    source.add_argument(
        "--job-config",
        action="append",
        type=Path,
        dest="job_configs",
        help="Job config to orchestrate. Repeat to submit multiple jobs without a wrapper plan file.",
    )
    parser.add_argument("--name", default=None, help="Optional bundle name when using --job-config directly.")
    parser.add_argument("--run-id", help="Submission bundle run identifier.")
    parser.add_argument("--output-dir", type=Path, help="Override the Slurm bundle output directory.")
    parser.add_argument("--env-file", type=Path, default=None, help="Dotenv file passed through to inner orchestrator runs.")
    parser.add_argument("--orchestrate-config", type=Path, help="Path to model-serving orchestrate.toml registry.")
    parser.add_argument("--eval-images-config", type=Path, help="Path to eval auxiliary image registry TOML.")
    parser.add_argument("--endpoints-path", type=Path, help="Path to endpoints.toml used for model alias resolution and bench.")
    parser.add_argument("--readiness-timeout-s", type=int, default=None, help="Inner readiness timeout in seconds.")
    parser.add_argument(
        "--prune-logs-on-success",
        action="store_true",
        help="Delete inner orchestrator logs after successful tasks.",
    )
    parser.add_argument("--node-gpus", type=int, default=8, help="Outer Slurm GPU allocation per job.")
    parser.add_argument(
        "--max-simultaneous-nodes",
        type=int,
        default=1,
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
    parser.add_argument("--dependency", default=None, help="Base sbatch dependency applied to each chain head.")
    parser.add_argument("--mail-type", default=None)
    parser.add_argument("--mail-user", default=None)
    parser.add_argument("--test-only", action="store_true", help="Run sbatch --test-only instead of submitting jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Write scripts and print sbatch commands without submitting.")
    parser.add_argument(
        "--slurm-resume",
        action="store_true",
        default=None,
        help="Mark jobs requeueable and pass --resume to the inner orchestrator.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing the medarc-orchestrate checkout.",
    )
    parser.add_argument(
        "--activate-script",
        type=Path,
        default=None,
        help="Shell activation script sourced before running medarc-orchestrate inside sbatch (defaults to <source-dir>/.venv/bin/activate).",
    )
    parser.set_defaults(command="slurm", handler=run_from_args)


def add_slurm_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "slurm",
        description="Render and submit one sbatch job per orchestrator task.",
        help="Submit one Slurm job per resolved task using pyxis at execution time.",
    )
    _add_arguments(parser)
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-orchestrate slurm",
        description="Render and submit one sbatch job per orchestrator task.",
    )
    _add_arguments(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
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

    if args.env_file is not None:
        env_file = args.env_file.expanduser()
        if not env_file.is_absolute():
            env_file = plan_base_dir / env_file
        plan.env_file = env_file.resolve()

    tasks = expand_tasks(plan)
    configured_run_id = args.run_id or plan.run_id
    if configured_run_id:
        run_id = configured_run_id
    else:
        run_id = generate_run_id(plan.name)

    output_root = args.output_dir or plan.output_dir or Path("outputs") / "orchestrate" / run_id
    output_root = output_root.expanduser().resolve()
    source_dir = args.source_dir.expanduser().resolve()
    if args.activate_script is not None:
        activate_script = args.activate_script.expanduser()
        if not activate_script.is_absolute():
            activate_script = source_dir / activate_script
        activate_script = activate_script.resolve()
    else:
        activate_script = (source_dir / ".venv" / "bin" / "activate").resolve()
    readiness_timeout_s = args.readiness_timeout_s if args.readiness_timeout_s is not None else plan.readiness_timeout_s
    cli_overrides = SlurmCliOverrides(
        cpus_per_gpu=args.cpus_per_gpu,
        time=args.time,
        partition=args.partition,
        account=args.account,
        qos=args.qos,
        mail_type=args.mail_type,
        mail_user=args.mail_user,
        slurm_resume=args.slurm_resume,
    )
    planned_tasks = build_submission_plan(
        tasks,
        run_id=run_id,
        node_gpus=args.node_gpus,
        max_simultaneous_nodes=args.max_simultaneous_nodes,
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
        node_gpus=args.node_gpus,
        source_dir=source_dir,
        activate_script=activate_script,
        env_file=plan.env_file,
        readiness_timeout_s=readiness_timeout_s,
        prune_logs_on_success=bool(args.prune_logs_on_success or plan.prune_logs_on_success),
        existing_manifest=existing_manifest,
    )
    write_bundle_manifest(manifest_path, manifest)

    if args.dry_run:
        for command in mark_dry_run(manifest_path, manifest):
            print(command)
        return 0

    submit_bundle(manifest_path, manifest, test_only=bool(args.test_only))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_from_args(args)


def _load_existing_manifest(path: Path, *, run_id: str) -> SlurmBundleManifest | None:
    if not path.exists():
        return None
    manifest = load_bundle_manifest(path)
    if manifest.run_id != run_id:
        raise ValueError(f"Existing Slurm manifest at {path} belongs to run_id={manifest.run_id}, not {run_id}.")
    return manifest


__all__ = ["add_slurm_subparser", "build_parser", "main", "run_from_args"]
