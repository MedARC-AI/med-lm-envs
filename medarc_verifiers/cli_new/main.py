"""Unified MedARC evaluation CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Sequence

try:
    from rich.console import Console
    from rich.table import Table
except ImportError:  # pragma: no cover - rich is optional at runtime
    Console = None  # type: ignore[assignment]
    Table = None  # type: ignore[assignment]

import yaml
from pydantic import ValidationError

from medarc_verifiers.cli_new._config_loader import ConfigFormatError, load_run_config
from medarc_verifiers.cli_new._job_builder import ResolvedJob, build_jobs
from medarc_verifiers.cli_new._job_executor import ExecutorSettings, JobExecutionResult, execute_jobs
from medarc_verifiers.cli_new._manifest import MANIFEST_FILENAME, ManifestJobEntry, RunManifest, compute_snapshot_checksum
from medarc_verifiers.cli_new._manifest_planner import ManifestPlan, ManifestPlanner
from medarc_verifiers.cli_new._single_run import run_single_mode
from medarc_verifiers.cli_new.process import ProcessOptions, ProcessResult, run_process
from medarc_verifiers.cli_new.process.hf_sync import HFSyncConfig
from medarc_verifiers.cli_new.process.winrate import WinrateConfig
from medarc_verifiers.cli_new.process.winrate_runner import (
    _resolve_source,
    list_models,
    print_winrate_summary_markdown,
    run_winrate,
)
from medarc_verifiers.cli_new.utils.overrides import build_cli_override
from medarc_verifiers.cli_new._schemas import EnvironmentConfigSchema, EnvironmentExportConfig

logger = logging.getLogger(__name__)
HELP_FLAGS = {"-h", "--help"}

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_VAR = "OPENAI_API_KEY"
DEFAULT_ENV_DIR = Path("environments")
DEFAULT_ENV_CONFIG_ROOT = Path("configs") / "envs"
DEFAULT_RUNS_RAW_DIR = Path("runs") / "raw"
DEFAULT_PROCESSED_DIR = Path("runs") / "processed"
BENCH_COMMAND = "bench"
PROCESS_COMMAND = "process"
WINRATE_COMMAND = "winrate"


def build_batch_parser() -> argparse.ArgumentParser:
    """Construct the unified CLI parser."""
    parser = argparse.ArgumentParser(
        prog="medarc-new",
        description="Run MedARC evaluations using unified configuration files.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to a run configuration YAML file.")
    parser.add_argument("--run-id", help="Override the generated run identifier.")
    parser.add_argument("--name", help="Override the human-friendly run name (defaults to the config name).")
    parser.add_argument(
        "--restart",
        help="Seed jobs from a previous run identifier (reuse completed jobs when configs match).",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically resume the newest matching run (default: enabled). "
            "Pass --no-auto-resume to force a fresh run."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-run every job regardless of manifest state.")
    parser.add_argument(
        "--forced",
        action="append",
        help="Re-run jobs for the specified environment(s); repeat or comma-separate values.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the output directory from the configuration.")
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=DEFAULT_ENV_DIR,
        help="Directory containing environments (default: %(default)s).",
    )
    parser.add_argument(
        "--env-config-root",
        type=Path,
        default=DEFAULT_ENV_CONFIG_ROOT,
        help="Directory containing environment YAMLs for auto-discovery (default: %(default)s).",
    )
    parser.add_argument("--endpoints-path", type=Path, help="Override the default endpoints registry path.")
    parser.add_argument(
        "--default-api-key-var",
        default=DEFAULT_API_KEY_VAR,
        help=f"Default API key environment variable (default: {DEFAULT_API_KEY_VAR}).",
    )
    parser.add_argument(
        "--default-api-base-url",
        default=DEFAULT_API_BASE_URL,
        help=f"Default API base URL (default: {DEFAULT_API_BASE_URL}).",
    )
    parser.add_argument(
        "--job-id", action="append", help="Run only the specified job identifier (repeat to select multiple)."
    )
    parser.add_argument(
        "--env-arg", action="append", help="Override an environment argument with KEY=VALUE (repeatable)."
    )
    parser.add_argument("--env-args", help="Override environment arguments with a JSON object.")
    parser.add_argument(
        "--sampling-arg", action="append", help="Override a sampling argument with KEY=VALUE (repeatable)."
    )
    parser.add_argument("--sampling-args", help="Override sampling arguments with a JSON object.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve jobs and report overrides without executing them."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist evaluation outputs (default: enabled).",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload results to the Hugging Face Hub.",
    )
    parser.add_argument("--hf-hub-dataset-name", help="Custom dataset name when uploading to the Hub.")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Override env max_concurrent for all jobs (CLI > model > env > defaults).",
    )
    parser.add_argument("--max-concurrent-generation", type=int, help="Override generation concurrency for all jobs.")
    parser.add_argument("--max-concurrent-scoring", type=int, help="Override scoring concurrency for all jobs.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override request timeout in seconds for all jobs (CLI > model > default).",
    )
    return parser


def build_process_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-new process",
        description="Process MedARC run outputs into Parquet datasets and optional HF uploads.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_RAW_DIR,
        help="Directory containing raw run outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory to store processed parquet files (default: %(default)s).",
    )
    parser.add_argument(
        "--env-config-root",
        type=Path,
        default=DEFAULT_ENV_CONFIG_ROOT,
        help="Directory containing environment YAMLs for export settings (default: %(default)s).",
    )
    parser.add_argument(
        "--status",
        action="append",
        help="Filter runs by manifest status (repeatable).",
    )
    parser.add_argument(
        "--include-prompt-completion",
        dest="include_prompt_completion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include prompt/completion columns (default: env config or false).",
    )
    parser.add_argument("--keep-column", action="append", help="Extra column to keep (repeatable).")
    parser.add_argument("--drop-column", action="append", help="Column to drop (repeatable).")
    parser.add_argument(
        "--combine-rollouts",
        dest="combine_rollouts",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Combine rollout suffixes when deriving base env ids (default: env config or true).",
    )
    parser.add_argument(
        "--no-deduplicate",
        dest="no_deduplicate",
        action="store_true",
        default=False,
        help="Include all runs (don't deduplicate by latest per model+env).",
    )
    parser.add_argument("--exporter-version", default="dev", help="Exporter version tag to embed in outputs.")
    parser.add_argument("--processed-at", help="Override processed_at timestamp (ISO8601).")
    parser.add_argument("--dry-run", action="store_true", help="Plan processing without writing outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing parquet files.")
    parser.add_argument(
        "--process-incomplete",
        dest="process_incomplete",
        action="store_true",
        default=False,
        help="Include runs where run_manifest.json summary has completed < total.",
    )
    parser.add_argument(
        "--append",
        dest="append",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append/merge into existing parquet files when present (default: true). Use --no-append to error unless --overwrite is set.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing datasets (default: %(default)s). Use 1 to disable multiprocessing.",
    )

    parser.add_argument("--hf-repo", help="Hugging Face repo id for dataset sync.")
    parser.add_argument(
        "--hf-merge",
        choices=("append", "update", "replace"),
        default="append",
        help="Merge strategy when syncing to HF (default: %(default)s).",
    )
    parser.add_argument("--hf-branch", help="Target HF branch.")
    parser.add_argument("--hf-token", help="Auth token for HF operations.")
    parser.add_argument(
        "--hf-private",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Push dataset as private (default: false).",
    )

    return parser


def build_winrate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medarc-new winrate",
        description="Compute HELM-style win rates from processed environment parquet files.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed parquet outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for winrates JSON (default: <processed_dir>/../winrate/winrates-<timestamp>.json).",
    )
    parser.add_argument(
        "--output-name",
        help="Base name for winrates JSON (timestamp appended automatically).",
    )
    parser.add_argument(
        "--processed-at",
        help="Timestamp used for default output naming (ISO8601).",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("zero", "neg-inf"),
        default="neg-inf",
        help="Missing reward policy when comparing models (default: %(default)s).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-9,
        help="Tie tolerance epsilon for pairwise comparisons (default: %(default)s).",
    )
    parser.add_argument(
        "--min-common",
        type=int,
        default=0,
        help="Minimum overlapping examples per dataset to retain a pairwise result.",
    )
    parser.add_argument(
        "--weight-policy",
        choices=("equal", "ln", "sqrt", "cap"),
        default="ln",
        help="Dataset weighting policy when aggregating win rates (default: %(default)s).",
    )
    parser.add_argument(
        "--weight-cap",
        type=int,
        default=0,
        help="Cap applied when using --weight-policy=cap (default: %(default)s).",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        help="Only include these model ids in win rate calculation (repeatable).",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        help="Exclude these model ids from win rate calculation (repeatable).",
    )
    parser.add_argument("--hf-repo", help="Hugging Face repo id for dataset download.")
    parser.add_argument("--hf-branch", help="Target HF branch or revision for download.")
    parser.add_argument("--hf-token", help="Auth token for HF operations.")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model ids in the source parquet files (local or HF) and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Unified CLI entry point."""
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        _print_general_help()
        return 0

    if args_list[0] in HELP_FLAGS:
        _print_general_help()
        return 0

    if args_list[0] == BENCH_COMMAND:
        return _run_batch_mode(args_list[1:])
    if args_list[0] == PROCESS_COMMAND:
        return _run_process_mode(args_list[1:])
    if args_list[0] == WINRATE_COMMAND:
        return _run_winrate_mode(args_list[1:])

    return run_single_mode(args_list)


def _run_batch_mode(argv: Sequence[str]) -> int:
    parser = build_batch_parser()
    args = parser.parse_args(argv)

    try:
        args.cli_env_args = build_cli_override(
            json_payload=args.env_args,
            pairs=args.env_arg,
            json_flag="--env-args",
            pair_flag="--env-arg",
        )
        args.cli_sampling_args = build_cli_override(
            json_payload=args.sampling_args,
            pairs=args.sampling_arg,
            json_flag="--sampling-args",
            pair_flag="--sampling-arg",
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.restart:
        args.auto_resume = False
    # Allow in-place regeneration when --run-id matches --regen
    # (previously disallowed). If equality is intended, we'll update the seed run in place.

    try:
        return _execute_batch(args)
    except KeyboardInterrupt:
        logger.warning("Batch run interrupted by user.")
        return 1
    except ConfigFormatError as exc:
        parser.error(str(exc))
    except SystemExit:  # pragma: no cover - argparse already handled messaging
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error: %s", exc)
        return 1


def _run_process_mode(argv: Sequence[str]) -> int:
    parser = build_process_parser()
    args = parser.parse_args(argv)

    try:
        env_export_map = _load_env_export_map(args.env_config_root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load environment export configs: %s", exc)
        env_export_map = {}

    hf_config = HFSyncConfig.from_cli(
        repo=args.hf_repo,
        merge_strategy=args.hf_merge,
        branch=args.hf_branch,
        token=args.hf_token,
        private=args.hf_private,
        dry_run=args.dry_run,
    )

    processed_with_args = {
        "status": args.status or [],
        "include_prompt_completion": args.include_prompt_completion,
        "keep_columns": args.keep_column or [],
        "drop_columns": args.drop_column or [],
        "combine_rollouts": args.combine_rollouts,
        "deduplicate_latest": not args.no_deduplicate,
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "only_complete_runs": not bool(args.process_incomplete),
        "append": args.append,
        "hf_repo": args.hf_repo,
        "hf_merge": args.hf_merge,
        "max_workers": args.max_workers,
    }

    options = ProcessOptions(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        exporter_version=args.exporter_version,
        processed_at=args.processed_at,
        processed_with_args=processed_with_args,
        status_filter=args.status or (),
        include_prompt_completion=args.include_prompt_completion,
        only_complete_runs=not bool(args.process_incomplete),
        keep_columns=args.keep_column or (),
        drop_columns=args.drop_column or (),
        combine_rollouts=args.combine_rollouts,
        deduplicate_latest=not args.no_deduplicate,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        append=args.append,
        hf_config=hf_config,
        max_workers=args.max_workers,
    )

    try:
        result = run_process(options, env_export_map=env_export_map)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Process pipeline failed: %s", exc)
        return 1

    _log_process_result(result)
    return 0


def _run_winrate_mode(argv: Sequence[str]) -> int:
    parser = build_winrate_parser()
    args = parser.parse_args(argv)

    hf_config = HFSyncConfig.from_cli(
        repo=args.hf_repo,
        merge_strategy="append",
        branch=args.hf_branch,
        token=args.hf_token,
        private=False,
        dry_run=False,
    )

    source_dir, datasets, source_desc = _resolve_source(args.processed_dir, hf_config if args.hf_repo else None)
    if not datasets:
        logger.error("No datasets found from %s.", source_desc)
        return 1

    if args.list_models:
        models = list_models(datasets)
        if models:
            print("\n".join(models))
        else:
            logger.info("No models found in datasets from %s.", source_desc)
        return 0

    cfg = WinrateConfig(
        missing_policy=args.missing_policy,
        epsilon=args.epsilon,
        min_common=args.min_common,
        weight_policy=args.weight_policy,
        weight_cap=args.weight_cap,
        include_models=tuple(args.include_model or ()),
        exclude_models=tuple(args.exclude_model or ()),
    )

    try:
        winrate_result = run_winrate(
            processed_dir=source_dir,
            output_path=args.output,
            output_name=args.output_name,
            config=cfg,
            processed_at=args.processed_at,
            hf_config=hf_config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Win rate computation failed: %s", exc)
        return 1

    logger.info("Computed win rates for %d dataset(s): %s", len(winrate_result.datasets), winrate_result.output_path)
    print_winrate_summary_markdown(winrate_result.result)
    return 0


def _execute_batch(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser()
    env_root_override = Path(args.env_config_root).expanduser().resolve() if args.env_config_root else None
    run_config = load_run_config(config_path, env_default_root=env_root_override)

    run_name = args.name or run_config.name
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(run_config.output_dir).expanduser()
    output_dir = output_dir.resolve()
    run_id = args.run_id  # May be None when using --auto-resume discovery

    jobs = build_jobs(run_config)
    if not jobs:
        logger.error("Configuration %s did not produce any jobs.", config_path)
        return 1

    selected_jobs = _filter_jobs(jobs, args.job_id)
    if not selected_jobs:
        logger.error("No jobs matched the provided filters.")
        return 1

    env_args_map, sampling_args_map = _build_effective_args(
        jobs,
        args.cli_env_args or {},
        args.cli_sampling_args or {},
    )
    config_checksum = compute_snapshot_checksum(run_config.model_dump())
    forced_envs = _parse_forced_envs(args.forced)
    forced_envs.update(_collect_rerun_envs(run_config.envs))

    planner = ManifestPlanner(
        output_dir=output_dir,
        run_id=run_id,
        run_name=run_name,
        config_path=config_path,
        config_checksum=config_checksum,
        jobs=jobs,
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        restart_source=args.restart,
        auto_resume=bool(args.auto_resume),
        persist=not bool(args.dry_run),
    )

    try:
        manifest_plan = planner.plan(force_all=bool(args.force), forced_envs=forced_envs)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    runnable_ids = manifest_plan.runnable_job_ids
    selected_ids = {job.job_id for job in selected_jobs}
    planned_jobs = [job for job in jobs if job.job_id in runnable_ids and job.job_id in selected_ids]

    _print_job_plan(
        selected_jobs,
        manifest=manifest_plan.manifest,
        runnable_job_ids=runnable_ids,
        discovered_total=len(jobs),
        dry_run=bool(args.dry_run),
    )

    if not planned_jobs:
        if manifest_plan.reused_job_ids:
            logger.info(
                "All jobs already completed (reused %d job(s) from prior manifests).",
                len(manifest_plan.reused_job_ids),
            )
        else:
            logger.info("No jobs were scheduled after applying filters and resume settings.")

        # Check if all selected jobs are completed (not just filtered out)
        all_completed = all(
            manifest_plan.manifest.job_entry(job.job_id)
            and manifest_plan.manifest.job_entry(job.job_id).status == "completed"
            for job in selected_jobs
        )

        if all_completed and selected_jobs and not args.dry_run and not args.force:
            # Prompt user for action
            choice = _prompt_completed_jobs_action()
            if choice == "new":
                logger.info("Creating a new run with all jobs...")
                # Create a fresh run by disabling auto-resume and forcing a new run_id
                # Recursively call with updated args to create new manifest
                new_args = argparse.Namespace(**vars(args))
                new_args.auto_resume = False
                new_args.run_id = None  # Force generation of new run_id
                new_args.restart = None
                return _execute_batch(new_args)
            elif choice == "rerun":
                logger.info("Rerunning all completed jobs...")
                # Set all selected jobs to runnable
                runnable_ids = {job.job_id for job in selected_jobs}
                planned_jobs = [job for job in jobs if job.job_id in runnable_ids and job.job_id in selected_ids]
                # Continue execution below
            elif choice == "exit":
                logger.info("Exiting without running jobs.")
                _log_summary([], manifest_plan.manifest)
                return 0
            else:  # continue/skip
                logger.info("Continuing without running jobs.")
                _log_summary([], manifest_plan.manifest)
                return 0
        else:
            _log_summary([], manifest_plan.manifest)
            return 0

    if not planned_jobs:
        # After prompting, still no planned jobs (shouldn't happen, but safety check)
        _log_summary([], manifest_plan.manifest)
        return 0

    settings = ExecutorSettings(
        run_id=manifest_plan.manifest.model.run_id or "",
        output_dir=output_dir,
        env_dir=Path(args.env_dir).expanduser(),
        endpoints_path=Path(args.endpoints_path).expanduser() if args.endpoints_path else None,
        default_api_key_var=args.default_api_key_var,
        default_api_base_url=args.default_api_base_url,
        log_level="DEBUG" if args.verbose else "INFO",
        verbose=args.verbose,
        save_results=args.save_results,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=_coerce_optional_str(args.hf_hub_dataset_name),
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        max_concurrent=args.max_concurrent,  # CLI override (None if not provided)
        timeout=args.timeout,
        dry_run=args.dry_run,
        cli_env_args=getattr(args, "cli_env_args", None),
        cli_sampling_args=getattr(args, "cli_sampling_args", None),
    )

    logger.info(
        "Loaded %d job(s); executing %d after filters (%d reusable).",
        len(jobs),
        len(planned_jobs),
        len(manifest_plan.reused_job_ids),
    )

    endpoints_cache: dict[str, Any] = {}
    env_metadata_cache: dict[str, Any] = {}

    results = execute_jobs(
        planned_jobs,
        settings,
        endpoints_cache=endpoints_cache,
        env_metadata_cache=env_metadata_cache,
        manifest=None if args.dry_run else manifest_plan.manifest,
    )

    _log_summary(results, manifest_plan.manifest)

    has_failures = any(result.status == "failed" for result in results if result.status != "skipped")
    return 1 if has_failures else 0


def _build_effective_args(
    jobs: Sequence[ResolvedJob],
    cli_env_args: Mapping[str, Any],
    cli_sampling_args: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    env_map: dict[str, dict[str, Any]] = {}
    sampling_map: dict[str, dict[str, Any]] = {}
    for job in jobs:
        env_args = dict(job.env_args)
        env_args.update(cli_env_args)
        env_map[job.job_id] = env_args
        sampling_args = dict(job.sampling_args)
        sampling_args.update(cli_sampling_args)
        sampling_map[job.job_id] = sampling_args
    return env_map, sampling_map


def _parse_forced_envs(values: Sequence[str] | None) -> set[str]:
    forced: set[str] = set()
    if not values:
        return forced
    for chunk in values:
        if not chunk:
            continue
        for item in chunk.split(","):
            value = item.strip()
            if value:
                forced.add(value.lower())
    return forced


def _collect_rerun_envs(envs: Mapping[str, EnvironmentConfigSchema]) -> set[str]:
    rerun: set[str] = set()
    for env in envs.values():
        if getattr(env, "rerun", False):
            for key in (env.id, env.module, env.matrix_base_id):
                if key:
                    rerun.add(str(key).lower())
    return rerun


def _filter_jobs(jobs: Sequence[ResolvedJob], job_filters: Sequence[str] | None) -> list[ResolvedJob]:
    if not job_filters:
        return list(jobs)
    filters = set(job_filters)
    selected = [job for job in jobs if job.job_id in filters]
    missing = filters - {job.job_id for job in selected}
    if missing:
        logger.warning("Unknown job ids requested: %s", ", ".join(sorted(missing)))
    return selected


def _coerce_optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _prompt_completed_jobs_action() -> str:
    """Prompt user to choose what to do when all jobs are completed.

    Returns:
        "new", "rerun", "continue", or "exit"
    """
    console = Console() if Console is not None else None

    message = "\n[bold yellow]All jobs are already completed.[/bold yellow]\n"
    message += "What would you like to do?\n"
    message += "  [bold cyan]n[/bold cyan] - Create a new run\n"
    message += "  [bold cyan]r[/bold cyan] - Rerun all jobs (ignore completion status)\n"
    message += "  [bold cyan]c[/bold cyan] - Continue without running (default)\n"
    message += "  [bold cyan]e[/bold cyan] - Exit\n"

    if console:
        console.print(message)
    else:
        # Fallback to plain print if rich is not available
        print(
            message.replace("[bold yellow]", "")
            .replace("[/bold yellow]", "")
            .replace("[bold cyan]", "")
            .replace("[/bold cyan]", "")
        )

    try:
        response = input("Choose [n/r/c/e]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # New line after Ctrl+C
        return "exit"

    if response == "n" or response == "new":
        return "new"
    elif response == "r" or response == "rerun":
        return "rerun"
    elif response == "e" or response == "exit":
        return "exit"
    else:
        # Default to continue for any other input (including empty/enter)
        return "continue"


def _log_summary(results: Sequence[JobExecutionResult], manifest: RunManifest | None = None) -> None:
    if manifest is not None:
        summary = manifest.summary
        logger.info(
            "Run complete: %d completed, %d pending, %d failed, %d skipped (total %d).",
            summary.get("completed", 0),
            summary.get("pending", 0),
            summary.get("failed", 0),
            summary.get("skipped", 0),
            summary.get("total", 0),
        )
        return
    total = len(results)
    succeeded = sum(result.status == "succeeded" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    logger.info("Run complete: %d succeeded, %d skipped, %d failed (total %d).", succeeded, skipped, failed, total)


def _print_general_help() -> None:
    message = dedent(
        """\
        Usage:
          medarc-new <ENV> [options]                 # Single run (ENV must be first; use ENV --help for details)
          medarc-new bench --config CONFIG.yaml ...  # Batch run (see: medarc-new bench --help)
          medarc-new process [options]               # Export raw runs to parquet (see: medarc-new process --help)
          medarc-new winrate [options]               # Compute win rates from processed parquet outputs

        First argument must be the environment slug for single runs. Use 'medarc-new bench --help' for batch mode options."""
    )
    print(message)


def _log_process_result(result: ProcessResult) -> None:
    logger.info(
        "Processed %d record(s) into %d environment dataset(s) (%d rows).",
        result.records_processed,
        len(result.env_summaries),
        result.rows_processed,
    )
    for summary in result.env_summaries:
        path_display = summary.output_path if not summary.dry_run else f"(planned) {summary.output_path}"
        logger.info("  %s -> %d rows @ %s", summary.env_id or summary.base_env_id, summary.row_count, path_display)
    if result.hf_summary:
        logger.info(
            "HF sync: repo=%s strategy=%s rows=%d",
            result.hf_summary.repo_id,
            result.hf_summary.strategy,
            result.hf_summary.total_rows,
        )


def _load_env_export_map(root: Path | None) -> dict[str, EnvironmentExportConfig]:
    if root is None:
        return {}
    root = Path(root).expanduser()
    if not root.exists():
        logger.debug("Env config root %s does not exist; skipping export overrides.", root)
        return {}

    if root.is_file():
        files = [root]
    else:
        files = sorted(p for pattern in ("*.yaml", "*.yml") for p in root.rglob(pattern) if p.is_file())

    export_map: dict[str, EnvironmentExportConfig] = {}
    for path in files:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read env config %s: %s", path, exc)
            continue

        if isinstance(payload, list):
            entries = [entry for entry in payload if isinstance(entry, Mapping)]
        elif isinstance(payload, Mapping):
            entries = [payload]
        else:
            continue

        for entry in entries:
            try:
                env_cfg = EnvironmentConfigSchema(**dict(entry))
            except ValidationError:
                continue
            if env_cfg.export is None:
                continue
            keys = {env_cfg.id, env_cfg.matrix_base_id}
            for key in filter(None, keys):
                export_map[key] = env_cfg.export

    return export_map


def _print_job_plan(
    jobs: Sequence[ResolvedJob],
    *,
    manifest: RunManifest | None,
    runnable_job_ids: set[str],
    discovered_total: int,
    dry_run: bool,
) -> None:
    """Render a human-friendly summary of the jobs scheduled for execution."""
    listed_total = len(jobs)
    scheduled_total = sum(1 for job in jobs if job.job_id in runnable_job_ids)
    caption_parts: list[str] = [f"{listed_total} job(s) listed"]
    caption_parts.append(f"{scheduled_total} to {'dry-run' if dry_run else 'run'}")
    if discovered_total != listed_total:
        caption_parts.append(f"{discovered_total} discovered")
    caption = " | ".join(part for part in caption_parts if part)

    if not jobs:
        logger.info("No jobs to display (%s).", caption)
        return

    def _format_label(primary: str | None, secondary: str | None) -> str:
        if primary and secondary and primary != secondary:
            return f"{primary} ({secondary})"
        return primary or secondary or "-"

    def _resolve_status(job_id: str, entry: ManifestJobEntry | None) -> str:
        if job_id in runnable_job_ids:
            return "next"
        if entry and entry.status == "completed":
            return "completed"
        return "pending"

    entries = {}
    if manifest is not None:
        entries = {entry.job_id: entry for entry in manifest.jobs if entry.job_id}

    if Console is None or Table is None:
        lines = []
        for index, job in enumerate(jobs, start=1):
            entry = entries.get(job.job_id)
            model_label = _format_label(job.model.id, job.model.model)
            env_label = _format_label(job.env.id, job.env.module)
            status = _resolve_status(job.job_id, entry)
            text = (
                f"{index:02d}. {job.job_id} | status={status} | name={job.name or '-'} | "
                f"model={model_label} | env={env_label} | examples={job.env.num_examples} | "
                f"rollouts={job.env.rollouts_per_example}"
            )
            lines.append(text)
        logger.info("Planned jobs (%s):\n%s", caption, "\n".join(lines))
        return

    console = Console()
    table = Table(title="Planned Jobs", caption=caption, expand=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Job ID", style="bold cyan", overflow="fold")
    table.add_column("Status", style="yellow")
    table.add_column("Name", style="white", overflow="fold")
    table.add_column("Model", style="magenta", overflow="fold")
    table.add_column("Environment", style="green", overflow="fold")
    table.add_column("Examples", justify="right")
    table.add_column("Rollouts", justify="right")

    for index, job in enumerate(jobs, start=1):
        entry = entries.get(job.job_id)
        model_label = _format_label(job.model.id, job.model.model)
        env_label = _format_label(job.env.id, job.env.module)
        status = _resolve_status(job.job_id, entry)
        table.add_row(
            str(index),
            job.job_id,
            status,
            job.name or "-",
            model_label,
            env_label,
            str(job.env.num_examples),
            str(job.env.rollouts_per_example),
        )

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
