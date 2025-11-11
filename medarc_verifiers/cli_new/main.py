"""Unified MedARC evaluation CLI."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
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
from medarc_verifiers.cli_new._manifest import (
    MANIFEST_FILENAME,
    ManifestJobEntry,
    RunManifest,
    compute_job_checksum,
    compute_snapshot_checksum,
)
from medarc_verifiers.cli_new._single_run import run_single_mode
from medarc_verifiers.cli_new.process import ProcessOptions, ProcessResult, run_process
from medarc_verifiers.cli_new.process.hf_sync import HFSyncConfig
from medarc_verifiers.cli_new.utils.overrides import build_cli_override
from medarc_verifiers.cli_new.utils.shared import slugify
from medarc_verifiers.utils.pathing import from_project_relative
from medarc_verifiers.cli_new._schemas import EnvironmentConfigSchema, EnvironmentExportConfig

logger = logging.getLogger(__name__)
HELP_FLAGS = {"-h", "--help"}

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_VAR = "OPENAI_API_KEY"
DEFAULT_ENV_DIR = Path("environments")
DEFAULT_ENV_CONFIG_ROOT = Path("configs") / "envs"
DEFAULT_RUNS_RAW_DIR = Path("runs") / "raw"
DEFAULT_PROCESSED_DIR = Path("runs") / "processed"
PROCESS_COMMAND = "process"


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
        "--compute-winrates",
        dest="compute_winrates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute HELM-style win rate JSON after parquet export (use --no-compute-winrates to disable).",
    )
    parser.add_argument(
        "--winrate-output",
        type=Path,
        help="Optional output path for winrates JSON (default: <output_dir>/winrates.json).",
    )
    parser.add_argument(
        "--winrate-missing-policy",
        choices=("zero", "neg-inf"),
        default="neg-inf",
        help="Missing reward policy when comparing models (default: %(default)s).",
    )
    parser.add_argument(
        "--winrate-epsilon",
        type=float,
        default=1e-9,
        help="Tie tolerance epsilon for pairwise comparisons (default: %(default)s).",
    )
    parser.add_argument(
        "--winrate-min-common",
        type=int,
        default=0,
        help="Minimum overlapping examples per dataset to retain a pairwise result.",
    )
    parser.add_argument(
        "--winrate-weight-policy",
        choices=("equal", "ln", "sqrt", "cap"),
        default="ln",
        help="Dataset weighting policy when aggregating win rates (default: %(default)s).",
    )
    parser.add_argument(
        "--winrate-weight-cap",
        type=int,
        default=0,
        help="Cap applied when using --winrate-weight-policy=cap (default: %(default)s).",
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


def main(argv: Sequence[str] | None = None) -> int:
    """Unified CLI entry point."""
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        _print_general_help()
        return 0

    if args_list[0] in HELP_FLAGS:
        _print_general_help()
        return 0

    if args_list[0] == "bench":
        return _run_batch_mode(args_list[1:])
    if args_list[0] == PROCESS_COMMAND:
        return _run_process_mode(args_list[1:])

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

    hf_config = None
    if args.hf_repo:
        hf_config = HFSyncConfig(
            repo_id=args.hf_repo,
            merge_strategy=args.hf_merge,
            branch=args.hf_branch,
            private=bool(args.hf_private),
            dry_run=args.dry_run,
            token=args.hf_token,
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
        "hf_repo": args.hf_repo,
        "hf_merge": args.hf_merge,
        "compute_winrates": args.compute_winrates,
        "winrate_output": str(args.winrate_output) if args.winrate_output else None,
        "winrate_missing_policy": args.winrate_missing_policy,
        "winrate_epsilon": args.winrate_epsilon,
        "winrate_min_common": args.winrate_min_common,
        "winrate_weight_policy": args.winrate_weight_policy,
        "winrate_weight_cap": args.winrate_weight_cap,
    }

    options = ProcessOptions(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        exporter_version=args.exporter_version,
        processed_at=args.processed_at,
        processed_with_args=processed_with_args,
        status_filter=args.status or (),
        include_prompt_completion=args.include_prompt_completion,
        keep_columns=args.keep_column or (),
        drop_columns=args.drop_column or (),
        combine_rollouts=args.combine_rollouts,
        deduplicate_latest=not args.no_deduplicate,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        hf_config=hf_config,
        compute_winrates=args.compute_winrates,
        winrate_output=args.winrate_output,
        missing_policy=args.winrate_missing_policy,
        epsilon=args.winrate_epsilon,
        min_common=args.winrate_min_common,
        weight_policy=args.winrate_weight_policy,
        weight_cap=args.winrate_weight_cap,
    )

    try:
        result = run_process(options, env_export_map=env_export_map)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Process pipeline failed: %s", exc)
        return 1

    _log_process_result(result)
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

    try:
        manifest_plan = _prepare_manifest_plan(
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
            force_all=bool(args.force),
            forced_envs=forced_envs,
            dry_run=bool(args.dry_run),
        )
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


@dataclass
class ManifestPlan:
    manifest: RunManifest
    runnable_job_ids: set[str]
    reused_job_ids: set[str]


def _prepare_manifest_plan(
    *,
    output_dir: Path,
    run_id: str | None,
    run_name: str,
    config_path: Path,
    config_checksum: str,
    jobs: Sequence[ResolvedJob],
    env_args_map: Mapping[str, Mapping[str, Any]],
    sampling_args_map: Mapping[str, Mapping[str, Any]],
    restart_source: str | None,
    auto_resume: bool,
    force_all: bool,
    forced_envs: set[str],
    dry_run: bool,
) -> ManifestPlan:
    persist = not dry_run
    checksum = config_checksum

    # Helper to compute run_dir lazily
    def _run_dir_for(rid: str) -> Path:
        return output_dir / rid

    if restart_source:
        restart_path = Path(restart_source).expanduser()
        seed_dir: Path | None = None
        # Determine if restart_source refers to an existing run directory (path or id under output_dir)
        if restart_path.exists() and restart_path.is_dir():
            seed_dir = restart_path.resolve()
        else:
            candidate = (output_dir / restart_source).resolve()
            if candidate.exists() and candidate.is_dir():
                seed_dir = candidate
        if seed_dir and (seed_dir / MANIFEST_FILENAME).exists():
            seed_manifest = RunManifest.load(seed_dir / MANIFEST_FILENAME, persist=persist)
            dest_run_id = seed_manifest.model.run_id
            run_dir = seed_dir
            logger.info(
                "Restart in-place: extending existing run '%s' with any new jobs from current config.", dest_run_id
            )
            # Append any new jobs (or update checksums)
            for job in jobs:
                seed_manifest.ensure_job(
                    job,
                    env_args=env_args_map[job.job_id],
                    sampling_args=sampling_args_map[job.job_id],
                    results_dir=run_dir / job.job_id,
                )
            runnable, reused = _plan_regen_jobs(
                manifest=seed_manifest,
                seed_manifest=seed_manifest,
                jobs=jobs,
                force_all=force_all,
                forced_envs=forced_envs,
            )
            return ManifestPlan(manifest=seed_manifest, runnable_job_ids=runnable, reused_job_ids=reused)
        # Fall back to creating a new restarted run
        if seed_dir is None:
            raise ValueError(f"Run '{restart_source}' does not contain {MANIFEST_FILENAME}.")
        seed_manifest = RunManifest.load(seed_dir / MANIFEST_FILENAME, persist=False)
        dest_run_id = run_id or _generate_run_id(run_name)
        run_dir = _run_dir_for(dest_run_id)
        manifest_path = run_dir / MANIFEST_FILENAME
        if run_dir.exists() and manifest_path.exists() and persist:
            raise ValueError(f"Run directory '{run_dir}' already exists; choose a different --run-id.")
        logger.info("Restarting run '%s' from prior run '%s'.", dest_run_id, restart_source)
        manifest = RunManifest.create(
            run_dir=run_dir,
            run_id=dest_run_id,
            run_name=run_name,
            config_source=config_path,
            config_checksum=checksum,
            jobs=jobs,
            env_args_map=env_args_map,
            sampling_args_map=sampling_args_map,
            persist=persist,
            restart_source=restart_source,
        )
        runnable, reused = _plan_regen_jobs(
            manifest=manifest,
            seed_manifest=seed_manifest,
            jobs=jobs,
            force_all=force_all,
            forced_envs=forced_envs,
        )
        if reused:
            logger.info("Reused %d completed job(s) from '%s'.", len(reused), restart_source)
        return ManifestPlan(manifest=manifest, runnable_job_ids=runnable, reused_job_ids=reused)

    manifest: RunManifest | None = None
    if auto_resume:
        # If a specific run_id is provided, resume it; otherwise discover a candidate
        if run_id:
            run_dir = _run_dir_for(run_id)
            manifest_path = run_dir / MANIFEST_FILENAME
            if run_dir.exists() and manifest_path.exists():
                manifest = RunManifest.load(manifest_path, persist=persist)
                existing_checksum = manifest.model.config_checksum
                if existing_checksum and existing_checksum != checksum:
                    raise ValueError(
                        f"Run '{run_id}' was created from a different configuration. Use --regen {run_id} to seed a new run."
                    )
            elif run_dir.exists():
                raise ValueError(f"Run '{run_id}' is missing {MANIFEST_FILENAME}; cannot auto-resume.")
            else:
                logger.info(
                    "Auto-resume requested for run '%s', but no prior run exists. Starting a fresh run with this id.",
                    run_id,
                )
        else:
            candidate = _find_auto_resume_candidate(output_dir, expected_checksum=checksum)
            if candidate is None:
                logger.info(
                    "Auto-resume enabled but no matching run exists in %s; starting a fresh run. "
                    "Use --no-auto-resume to always start new runs.",
                    output_dir,
                )
            else:
                run_dir = candidate
                manifest_path = run_dir / MANIFEST_FILENAME
                manifest = RunManifest.load(manifest_path, persist=persist)

        if manifest is not None:
            runnable = _plan_auto_resume_jobs(
                manifest=manifest,
                jobs=jobs,
                env_args_map=env_args_map,
                sampling_args_map=sampling_args_map,
                force_all=force_all,
                forced_envs=forced_envs,
            )
            return ManifestPlan(manifest=manifest, runnable_job_ids=runnable, reused_job_ids=set())

    # Fresh run: generate a new run id if not provided
    dest_run_id = run_id or _generate_run_id(run_name)
    run_dir = _run_dir_for(dest_run_id)

    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id=dest_run_id,
        run_name=run_name,
        config_source=config_path,
        config_checksum=checksum,
        jobs=jobs,
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=persist,
        restart_source=None,
    )
    runnable = {job.job_id for job in jobs}
    return ManifestPlan(manifest=manifest, runnable_job_ids=runnable, reused_job_ids=set())


def _find_auto_resume_candidate(output_dir: Path, *, expected_checksum: str) -> Path | None:
    """Pick the best prior run directory to auto-resume for the given checksum.

    Preference order:
    1) Matching config checksum and incomplete (completed < total)
    2) Matching config checksum and most recent updated_at
    Returns the run directory Path or None if no candidates.
    """
    candidates: list[tuple[bool, float, Path]] = []
    for child in sorted(output_dir.iterdir() if output_dir.exists() else [], key=lambda p: p.name):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if payload.get("config_checksum") != expected_checksum:
            continue
        summary = payload.get("summary") or {}
        total = int(summary.get("total", 0))
        completed = int(summary.get("completed", 0))
        incomplete = completed < total if total > 0 else True
        updated_at = payload.get("updated_at") or payload.get("created_at")
        try:
            ts = _parse_iso_ts(updated_at) if isinstance(updated_at, str) else (manifest_path.stat().st_mtime)
        except Exception:  # noqa: BLE001
            ts = manifest_path.stat().st_mtime
        candidates.append((incomplete, float(ts), child))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def _parse_iso_ts(value: str) -> float:
    # Accept timestamps like '2025-11-07T01:23:45Z' or ISO with offset
    try:
        from datetime import datetime

        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def _plan_auto_resume_jobs(
    *,
    manifest: RunManifest,
    jobs: Sequence[ResolvedJob],
    env_args_map: Mapping[str, Mapping[str, Any]],
    sampling_args_map: Mapping[str, Mapping[str, Any]],
    force_all: bool,
    forced_envs: set[str],
) -> set[str]:
    job_lookup = {job.job_id: job for job in jobs}
    runnable: set[str] = set()
    manifest_job_ids = {entry.job_id for entry in manifest.jobs if entry.job_id}
    new_jobs = set(job_lookup) - manifest_job_ids
    if new_jobs:
        logger.info(
            "Auto-resume ignoring %d new job(s) not present in the manifest: %s",
            len(new_jobs),
            ", ".join(sorted(new_jobs)),
        )
    for entry in manifest.jobs:
        job_id = entry.job_id
        if not job_id:
            continue
        job = job_lookup.get(job_id)
        if job is None:
            logger.debug("Manifest contains job '%s' that is absent from the current config; skipping.", job_id)
            continue
        expected_checksum = compute_job_checksum(
            job,
            env_args=env_args_map[job_id],
            sampling_args=sampling_args_map[job_id],
        )
        if entry.checksum != expected_checksum:
            raise ValueError(
                f"Job '{job_id}' arguments changed since the manifest was recorded; use --regen to create a new run."
            )
        env_id = (entry.env_id or job.env.id or job.job_id).lower()
        forced = force_all or env_id in forced_envs
        if forced or entry.status != "completed":
            runnable.add(job_id)
    return runnable


def _plan_regen_jobs(
    *,
    manifest: RunManifest,
    seed_manifest: RunManifest,
    jobs: Sequence[ResolvedJob],
    force_all: bool,
    forced_envs: set[str],
) -> tuple[set[str], set[str]]:
    runnable: set[str] = set()
    reused: set[str] = set()
    for job in jobs:
        entry = manifest.job_entry(job.job_id)
        if entry is None:
            continue
        seed_entry = seed_manifest.job_entry(job.job_id)
        env_id = (entry.env_id or job.env.id or job.job_id).lower()
        forced = force_all or env_id in forced_envs
        if (
            not forced
            and seed_entry is not None
            and seed_entry.status == "completed"
            and seed_entry.checksum == entry.checksum
        ):
            seed_results_dir = seed_entry.results_dir
            resolved_results_dir: Path | str | None = None
            if isinstance(seed_results_dir, str):
                seed_path = Path(seed_results_dir)
                if seed_path.is_absolute():
                    resolved_results_dir = seed_path
                elif seed_path.parts and seed_path.parts[0] == "runs":
                    resolved_results_dir = from_project_relative(seed_path)
                else:
                    resolved_results_dir = (seed_manifest.run_dir / seed_path).resolve()
            elif isinstance(seed_results_dir, Path):
                resolved_results_dir = seed_results_dir
            manifest.record_job_skip(
                job.job_id,
                reason="up_to_date",
                results_dir=resolved_results_dir or seed_results_dir,
                source_entry=seed_entry,
            )
            reused.add(job.job_id)
            continue
        runnable.add(job.job_id)
    return runnable, reused


def _filter_jobs(jobs: Sequence[ResolvedJob], job_filters: Sequence[str] | None) -> list[ResolvedJob]:
    if not job_filters:
        return list(jobs)
    filters = set(job_filters)
    selected = [job for job in jobs if job.job_id in filters]
    missing = filters - {job.job_id for job in selected}
    if missing:
        logger.warning("Unknown job ids requested: %s", ", ".join(sorted(missing)))
    return selected


def _generate_run_id(name: str) -> str:
    base = slugify(name or "run")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{timestamp}"


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
