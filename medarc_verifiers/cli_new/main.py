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

from medarc_verifiers.cli_new._config_loader import ConfigFormatError, load_run_config
from medarc_verifiers.cli_new._job_builder import ResolvedJob, build_jobs
from medarc_verifiers.cli_new._job_executor import ExecutorSettings, JobExecutionResult, execute_jobs
from medarc_verifiers.cli_new._manifest import (
    MANIFEST_FILENAME,
    RunManifest,
    compute_job_checksum,
    compute_snapshot_checksum,
)
from medarc_verifiers.cli_new._single_run import run_single_mode
from medarc_verifiers.cli_new.utils.overrides import build_cli_override

logger = logging.getLogger(__name__)
HELP_FLAGS = {"-h", "--help"}

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_VAR = "OPENAI_API_KEY"
DEFAULT_ENV_DIR = Path("environments")
DEFAULT_MAX_CONCURRENT = 32


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
        "--regen",
        help="Seed jobs from a previous run identifier (reuse completed jobs when configs match).",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Resume a prior run for this configuration. When --run-id is omitted, "
            "the latest matching run is selected automatically."
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
        default=DEFAULT_MAX_CONCURRENT,
        help="Default max concurrency when environments omit a value.",
    )
    parser.add_argument("--max-concurrent-generation", type=int, help="Override generation concurrency for all jobs.")
    parser.add_argument("--max-concurrent-scoring", type=int, help="Override scoring concurrency for all jobs.")
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

    if args.auto_resume and args.regen:
        parser.error("Cannot combine --auto-resume with --regen.")
    if args.regen and args.run_id and args.regen == args.run_id:
        parser.error("--regen target must differ from the destination --run-id.")

    try:
        return _execute_batch(args)
    except ConfigFormatError as exc:
        parser.error(str(exc))
    except SystemExit:  # pragma: no cover - argparse already handled messaging
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error: %s", exc)
        return 1


def _execute_batch(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser()
    run_config = load_run_config(config_path)

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
    config_snapshot = run_config.model_dump()
    forced_envs = _parse_forced_envs(args.forced)

    try:
        manifest_plan = _prepare_manifest_plan(
            output_dir=output_dir,
            run_id=run_id,
            run_name=run_name,
            config_path=config_path,
            config_snapshot=config_snapshot,
            jobs=jobs,
            env_args_map=env_args_map,
            sampling_args_map=sampling_args_map,
            regen_source=args.regen,
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

    if not planned_jobs:
        if manifest_plan.reused_job_ids:
            logger.info(
                "All jobs already completed (reused %d job(s) from prior manifests).",
                len(manifest_plan.reused_job_ids),
            )
        else:
            logger.info("No jobs were scheduled after applying filters and resume settings.")
        _log_summary([], manifest_plan.manifest)
        return 0

    settings = ExecutorSettings(
        run_id=manifest_plan.manifest.payload.get("run_id", ""),
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
        default_max_concurrent=args.max_concurrent,
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
    config_snapshot: Mapping[str, Any],
    jobs: Sequence[ResolvedJob],
    env_args_map: Mapping[str, Mapping[str, Any]],
    sampling_args_map: Mapping[str, Mapping[str, Any]],
    regen_source: str | None,
    auto_resume: bool,
    force_all: bool,
    forced_envs: set[str],
    dry_run: bool,
) -> ManifestPlan:
    persist = not dry_run
    checksum = compute_snapshot_checksum(config_snapshot)

    # Helper to compute run_dir lazily
    def _run_dir_for(rid: str) -> Path:
        return output_dir / rid

    if regen_source:
        seed_dir = output_dir / regen_source
        seed_manifest_path = seed_dir / MANIFEST_FILENAME
        if not seed_manifest_path.exists():
            raise ValueError(f"Run '{regen_source}' does not contain {MANIFEST_FILENAME}.")
        seed_manifest = RunManifest.load(seed_manifest_path, persist=False)
        # Determine destination run_id/run_dir
        dest_run_id = run_id or _generate_run_id(run_name)
        run_dir = _run_dir_for(dest_run_id)
        manifest_path = run_dir / MANIFEST_FILENAME
        if run_dir.exists() and manifest_path.exists() and persist:
            raise ValueError(f"Run directory '{run_dir}' already exists; choose a different --run-id.")
        logger.info("Regenerating run '%s' from prior run '%s'.", run_id, regen_source)
        manifest = RunManifest.create(
            run_dir=run_dir,
            run_id=dest_run_id,
            run_name=run_name,
            config_source=config_path,
            config_snapshot=config_snapshot,
            jobs=jobs,
            env_args_map=env_args_map,
            sampling_args_map=sampling_args_map,
            persist=persist,
            regen_source=regen_source,
        )
        runnable, reused = _plan_regen_jobs(
            manifest=manifest,
            seed_manifest=seed_manifest,
            jobs=jobs,
            force_all=force_all,
            forced_envs=forced_envs,
        )
        if reused:
            logger.info("Reused %d completed job(s) from '%s'.", len(reused), regen_source)
        return ManifestPlan(manifest=manifest, runnable_job_ids=runnable, reused_job_ids=reused)

    if auto_resume:
        # If a specific run_id is provided, resume it; otherwise discover a candidate
        if run_id:
            run_dir = _run_dir_for(run_id)
            manifest_path = run_dir / MANIFEST_FILENAME
            if not run_dir.exists() or not manifest_path.exists():
                raise ValueError(f"Run '{run_id}' is missing {MANIFEST_FILENAME}; cannot auto-resume.")
            manifest = RunManifest.load(manifest_path, persist=persist)
            existing_checksum = manifest.payload.get("config_checksum")
            if existing_checksum and existing_checksum != checksum:
                raise ValueError(
                    f"Run '{run_id}' was created from a different configuration. Use --regen {run_id} to seed a new run."
                )
        else:
            candidate = _find_auto_resume_candidate(output_dir, expected_checksum=checksum)
            if candidate is None:
                raise ValueError(
                    "No existing run found to auto-resume for this configuration. "
                    "Provide --run-id or omit --auto-resume to start a new run."
                )
            run_dir = candidate
            manifest_path = run_dir / MANIFEST_FILENAME
            manifest = RunManifest.load(manifest_path, persist=persist)

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
        config_snapshot=config_snapshot,
        jobs=jobs,
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=persist,
        regen_source=None,
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
    manifest_job_ids = {entry.get("job_id") for entry in manifest.jobs if entry.get("job_id")}
    new_jobs = set(job_lookup) - manifest_job_ids
    if new_jobs:
        logger.info(
            "Auto-resume ignoring %d new job(s) not present in the manifest: %s",
            len(new_jobs),
            ", ".join(sorted(new_jobs)),
        )
    for entry in manifest.jobs:
        job_id = entry.get("job_id")
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
        if entry.get("checksum") != expected_checksum:
            raise ValueError(
                f"Job '{job_id}' arguments changed since the manifest was recorded; use --regen to create a new run."
            )
        env_id = (entry.get("env_id") or job.env.id or job.job_id).lower()
        forced = force_all or env_id in forced_envs
        if forced or entry.get("status") != "completed":
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
        env_id = (entry.get("env_id") or job.env.id or job.job_id).lower()
        forced = force_all or env_id in forced_envs
        if (
            not forced
            and seed_entry is not None
            and seed_entry.get("status") == "completed"
            and seed_entry.get("checksum") == entry.get("checksum")
        ):
            manifest.record_job_skip(
                job.job_id,
                reason="up_to_date",
                results_dir=seed_entry.get("results_dir"),
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
    base = _slugify(name or "run")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{timestamp}"


def _slugify(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value).strip("-") or "run"


def _coerce_optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


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


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
