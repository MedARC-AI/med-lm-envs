from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from verifiers import setup_logging
from verifiers.scripts.install import install_environment
from verifiers.types import ClientConfig, EvalConfig, GenerateOutputs
from verifiers.utils.eval_utils import load_endpoints, run_evaluation

import yaml

from medarc_verifiers.cli._benchmark_utils import (
    ResolvedJob,
    RunConfig,
    build_run_config,
    derive_run_id,
    expand_jobs,
    timestamp,
)
from medarc_verifiers.cli.eval import (
    _coerce_json_mapping,
    _merge_sampling_args,
    _resolve_endpoint_selection,
    build_headers,
    ensure_required_params,
    merge_env_args,
)
from medarc_verifiers.cli.utils.reporting import (
    compute_average,
    compute_metric_averages,
    log_results_summary,
    update_metadata_file,
)
from medarc_verifiers.utils.cli_env_args import EnvParam, gather_env_cli_metadata

logger = logging.getLogger(__name__)
PROGRAM_NAME = "benchmark"
DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.py"


@dataclass(slots=True)
class JobOutcome:
    job_id: str
    job_name: str
    model_id: str
    env_id: str
    status: str
    duration_seconds: float | None = None
    results_path: str | None = None
    error: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Execute a sequence of verifier evaluations across models and environments.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Expand the configuration and print the resolved jobs without running them.",
    )
    parser.add_argument(
        "--install-envs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically install required environments before running (default: True).",
    )
    parser.add_argument(
        "--jobs",
        "-j",
        required=True,
        help="Path to the benchmark configuration YAML file.",
    )
    parser.add_argument(
        "--models",
        "-M",
        help="Override the models source with a YAML file or directory; falls back to the config entry or '<config_dir>/models'.",
    )
    parser.add_argument(
        "--envs",
        "-E",
        help="Override the environments source with a YAML file or directory; falls back to the config entry or '<config_dir>/envs'.",
    )
    parser.add_argument(
        "--output-dir",
        help="Override the output directory; defaults to the configuration value or '<config_dir>/../runs'.",
    )
    parser.add_argument(
        "--run-id",
        help="Explicit run identifier (defaults to the config value or an auto-generated timestamp).",
    )
    parser.add_argument(
        "--resume",
        help="Resume an existing run by identifier.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs even if matching artifacts exist.",
    )
    parser.add_argument(
        "--name",
        help="Logical name for the run (defaults to the configuration value or the jobs filename).",
    )
    parser.add_argument(
        "--env-dir-path",
        help="Override the Verifiers environments lookup directory (defaults to the configuration value or './environments').",
    )
    parser.add_argument(
        "--endpoints-path",
        help="Override the endpoints registry path (defaults to the configuration value or './configs/endpoints.py').",
    )
    return parser


def prepare_run_config(args: argparse.Namespace) -> tuple[RunConfig, Path, Mapping[str, Any]]:
    jobs_path = Path(args.jobs).expanduser().resolve()
    try:
        with jobs_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except FileNotFoundError as exc:  # pragma: no cover - filesystem error
        raise ValueError(f"Jobs file '{jobs_path}' not found.") from exc

    if raw is None:
        data: MutableMapping[str, Any] = {}
    elif isinstance(raw, MutableMapping):
        data = dict(raw)
    elif isinstance(raw, list):
        data = {"jobs": raw}
    else:
        raise ValueError("Jobs file must contain a mapping or a list of job entries.")

    base_dir = jobs_path.parent

    if args.models:
        data["models"] = str(Path(args.models).expanduser())
    elif "models" not in data:
        models_dir = data.pop("models_dir", None)
        if models_dir is not None:
            data["models"] = str(Path(models_dir).expanduser())
        else:
            data["models"] = str(base_dir / "models")

    if args.envs:
        data["envs"] = str(Path(args.envs).expanduser())
    elif "envs" not in data:
        envs_dir = data.pop("envs_dir", None)
        if envs_dir is not None:
            data["envs"] = str(Path(envs_dir).expanduser())
        else:
            data["envs"] = str(base_dir / "envs")

    if "jobs" not in data:
        raise ValueError("Jobs configuration must include a 'jobs' list.")

    if args.name:
        data["name"] = args.name
    else:
        data.setdefault("name", jobs_path.stem)

    if args.output_dir:
        data["output_dir"] = str(Path(args.output_dir).expanduser())
    else:
        output_dir_value = data.get("output_dir")
        if output_dir_value is None:
            default_runs = (base_dir.parent / "runs").resolve()
            data["output_dir"] = str(default_runs)
        else:
            output_path = Path(output_dir_value).expanduser()
            if not output_path.is_absolute():
                output_path = (base_dir / output_path).resolve()
            data["output_dir"] = str(output_path)

    if args.env_dir_path:
        data["env_dir_path"] = str(Path(args.env_dir_path).expanduser())
    if args.endpoints_path:
        data["endpoints_path"] = str(Path(args.endpoints_path).expanduser())
    run_config = build_run_config(data, base_dir)
    config_snapshot = json.loads(json.dumps(data, sort_keys=True, default=str))
    return run_config, jobs_path, config_snapshot


def _print_dry_run(
    *,
    run_config: RunConfig,
    jobs: Sequence[ResolvedJob],
    jobs_path: Path,
    run_id: str,
) -> None:
    print(f"Jobs file: {jobs_path}")
    print(f"Run ID: {run_id}")
    print(f"Output directory: {run_config.output_dir / run_id}")
    print("JOB_ID\tMODEL\tENV")
    for job in jobs:
        print("\t".join([job.job_id, job.model.id, job.env.id]))
    print(f"Total jobs: {len(jobs)}")


async def execute_jobs(
    *,
    run_config: RunConfig,
    jobs: Sequence[ResolvedJob],
    verbose: bool,
    completed_jobs: Mapping[str, JobOutcome] | None = None,
    force: bool = False,
) -> list[JobOutcome]:
    endpoint_cache: dict[str, Mapping[str, Mapping[str, str]]] = {}
    env_metadata_cache: dict[str, Sequence[EnvParam]] = {}
    outcomes: list[JobOutcome] = []
    completed_jobs = completed_jobs or {}
    total_jobs = len(jobs)
    for index, job in enumerate(jobs, start=1):
        job_label = f"{job.job_id} (model={job.model.id}, env={job.env.id})"
        existing = completed_jobs.get(job.job_id)
        if existing and not force:
            logger.info(
                "Job %d/%d %s skipped; already completed.",
                index,
                total_jobs,
                job_label,
            )
            outcomes.append(
                JobOutcome(
                    job_id=existing.job_id,
                    job_name=existing.job_name,
                    model_id=existing.model_id,
                    env_id=existing.env_id,
                    status="skipped",
                    duration_seconds=existing.duration_seconds,
                    results_path=existing.results_path,
                )
            )
            continue
        logger.info("Job %d/%d %s starting.", index, total_jobs, job_label)
        try:
            eval_config = build_eval_config(
                run_config=run_config,
                job=job,
                endpoints_cache=endpoint_cache,
                env_metadata_cache=env_metadata_cache,
                verbose=verbose,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "Job %d/%d %s configuration failed: %s",
                index,
                total_jobs,
                job_label,
                exc,
            )
            outcomes.append(
                JobOutcome(
                    job_id=job.job_id,
                    job_name=job.name,
                    model_id=job.model.id,
                    env_id=job.env.id,
                    status="failed",
                    error=str(exc),
                )
            )
            continue

        start = time.perf_counter()
        try:
            results = await run_evaluation(eval_config)
        except Exception as exc:  # noqa: BLE001
            duration = time.perf_counter() - start
            logger.exception("Job %d/%d %s failed after %.2fs: %s", index, total_jobs, job_label, duration, exc)
            outcomes.append(
                JobOutcome(
                    job_id=job.job_id,
                    job_name=job.name,
                    model_id=job.model.id,
                    env_id=job.env.id,
                    status="failed",
                    duration_seconds=duration,
                    error=str(exc),
                )
            )
            continue

        duration = time.perf_counter() - start
        results_dir = _persist_results(run_config, job, results, duration)
        log_results_summary(
            results=results,
            env_slug=job.env.id,
            judge_name=job.model.id,
            stage="benchmark",
        )
        outcomes.append(
            JobOutcome(
                job_id=job.job_id,
                job_name=job.name,
                model_id=job.model.id,
                env_id=job.env.id,
                status="succeeded",
                duration_seconds=duration,
                results_path=str(results_dir),
            )
        )
        logger.info("Job %d/%d %s completed in %.2fs.", index, total_jobs, job_label, duration)
    return outcomes


def _ensure_root_logging(level: str) -> None:
    """Ensure root logging emits to stderr so CLI messages are visible."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(level.upper())
    logging.getLogger(__name__).setLevel(level.upper())
    httpx_logger = logging.getLogger("httpx")
    httpx_logger.setLevel(logging.WARNING)


def build_eval_config(
    *,
    run_config: RunConfig,
    job: ResolvedJob,
    endpoints_cache: dict[str, Mapping[str, Mapping[str, str]]],
    env_metadata_cache: dict[str, Sequence[EnvParam]] | None = None,
    verbose: bool,
) -> EvalConfig:
    env_metadata_cache = env_metadata_cache or {}
    params = job.model.params
    headers = _normalize_headers(params.headers)
    sampling_args = dict(params.sampling_args)
    if job.sampling_overrides:
        sampling_args.update(job.sampling_overrides)
    merged_sampling = _merge_sampling_args(
        _coerce_json_mapping(sampling_args, "sampling_args"),
        params.max_tokens,
        params.temperature,
    )

    registry_path = params.endpoints_path or run_config.endpoints_path or DEFAULT_ENDPOINTS_PATH
    registry = _load_cached_endpoints(registry_path, endpoints_cache)
    resolved_model, api_key_var, api_base_url = _resolve_endpoint_selection(
        params.model or job.model.id,
        registry,
        params.api_key_var or run_config.default_api_key_var,
        params.api_base_url or run_config.default_api_base_url,
    )
    client_config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=headers or None,
    )

    env_id = job.env.module or job.env.id
    env_args = merge_env_args(params.env_args, dict(job.env.env_args))
    env_args = merge_env_args(job.env_overrides, env_args)
    metadata = _load_env_metadata(env_id, env_metadata_cache)
    ensure_required_params(metadata, {}, env_args)
    max_concurrent = job.env.max_concurrent or 1
    save_every = job.env.save_every if job.env.save_every is not None else -1
    verbose_flag = job.env.verbose if job.env.verbose is not None else verbose

    eval_config = EvalConfig(
        env_id=env_id,
        env_args=env_args,
        env_dir_path=run_config.env_dir_path,
        model=resolved_model,
        client_config=client_config,
        sampling_args=merged_sampling,
        num_examples=job.env.num_examples,
        rollouts_per_example=job.env.rollouts_per_example,
        max_concurrent=max_concurrent,
        interleave_scoring=job.env.interleave_scoring,
        print_results=job.env.print_results,
        verbose=verbose_flag,
        state_columns=job.env.state_columns,
        save_results=True,
        save_every=save_every,
    )
    return eval_config


def _normalize_headers(headers: list[str] | dict[str, str] | None) -> dict[str, str]:
    if headers is None:
        return {}
    if isinstance(headers, dict):
        return {str(key): str(value) for key, value in headers.items()}
    return build_headers(headers)


def _load_cached_endpoints(
    path: str,
    cache: dict[str, Mapping[str, Mapping[str, str]]],
) -> Mapping[str, Mapping[str, str]]:
    if path not in cache:
        cache[path] = load_endpoints(path)
    return cache[path]


def _load_env_metadata(
    env_id: str,
    cache: dict[str, Sequence[EnvParam]],
) -> Sequence[EnvParam]:
    if env_id not in cache:
        cache[env_id] = gather_env_cli_metadata(env_id)
    return cache[env_id]


def _persist_results(
    run_config: RunConfig,
    job: ResolvedJob,
    results: GenerateOutputs,
    duration: float,
) -> Path:
    run_id = run_config.run_id or "benchmark-run"
    root_dir = Path(run_config.output_dir).expanduser().resolve()
    job_dir = root_dir / run_id / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    src_path = Path(results.metadata.path_to_save)
    if src_path.exists():
        for item in src_path.iterdir():
            target = job_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), target)
        with contextlib.suppress(OSError):
            src_path.rmdir()

    avg_reward = compute_average(results.reward)
    if avg_reward is None and results.metadata.avg_reward is not None:
        avg_reward = float(results.metadata.avg_reward)

    metrics_avg = compute_metric_averages(results.metrics)
    if not metrics_avg and results.metadata.avg_metrics:
        metrics_avg = {key: float(value) for key, value in results.metadata.avg_metrics.items()}

    summary = {
        "job_id": job.job_id,
        "job_name": job.name,
        "model_id": job.model.id,
        "env_id": job.env.id,
        "duration_seconds": duration,
        "avg_reward": avg_reward,
        "num_examples": results.metadata.num_examples,
        "rollouts_per_example": results.metadata.rollouts_per_example,
    }
    if metrics_avg:
        summary["avg_metrics"] = metrics_avg
    with (job_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    update_metadata_file(job_dir / "metadata.json", avg_reward, metrics_avg)

    return job_dir


def _compute_config_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(snapshot, sort_keys=True, default=str))


def _normalize_data(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _manifest_job_entry(job: ResolvedJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "job_name": job.name,
        "model_id": job.model.id,
        "env_id": job.env.id,
        "env_overrides": _normalize_data(job.env_overrides),
        "sampling_overrides": _normalize_data(job.sampling_overrides),
        "results_dir": job.job_id,
    }


def _manifest_entries_match(existing: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    keys = (
        "job_name",
        "model_id",
        "env_id",
        "env_overrides",
        "sampling_overrides",
        "results_dir",
    )
    return all(existing.get(key) == candidate.get(key) for key in keys)


def upsert_manifest_jobs(manifest: dict[str, Any], jobs: Sequence[ResolvedJob]) -> tuple[bool, list[str]]:
    entries: list[dict[str, Any]] = list(manifest.get("jobs", []))
    index = {entry["job_id"]: entry for entry in entries if "job_id" in entry}
    changed = False
    new_ids: list[str] = []
    for job in jobs:
        candidate = _manifest_job_entry(job)
        existing = index.get(job.job_id)
        if existing is None:
            entries.append(candidate)
            index[job.job_id] = candidate
            new_ids.append(job.job_id)
            changed = True
        elif not _manifest_entries_match(existing, candidate):
            existing.update(candidate)
            changed = True
    if changed:
        manifest["jobs"] = entries
    elif "jobs" not in manifest:
        manifest["jobs"] = entries
    return changed, new_ids


def update_manifest_metadata(
    manifest: dict[str, Any],
    *,
    run_config: RunConfig,
    config_source: Path | None,
    config_snapshot: Mapping[str, Any],
    resumed: bool,
) -> None:
    snapshot_copy = _normalize_snapshot(config_snapshot)
    manifest["config_source"] = str(config_source) if config_source else manifest.get("config_source")
    manifest["config_checksum"] = _compute_config_checksum(snapshot_copy)
    manifest["config_snapshot"] = snapshot_copy

    run_id = run_config.run_id or manifest.get("run_id")
    run_section = dict(manifest.get("run") or {})
    run_section.update(
        {
            "output_dir": str(run_config.output_dir),
            "run_dir": str(Path(run_config.output_dir) / (run_id or "")),
            "env_dir_path": run_config.env_dir_path,
            "endpoints_path": run_config.endpoints_path,
            "default_api_key_var": run_config.default_api_key_var,
            "default_api_base_url": run_config.default_api_base_url,
        }
    )
    manifest["run"] = run_section
    if run_id is not None:
        manifest["run_id"] = run_id
    manifest["name"] = run_config.name

    now = timestamp()
    manifest["updated_at"] = now
    if resumed:
        manifest["last_resume_at"] = now
    else:
        manifest.setdefault("last_resume_at", None)


def build_run_manifest_payload(
    *,
    run_config: RunConfig,
    jobs: Sequence[ResolvedJob],
    config_source: Path | None,
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = run_config.run_id or derive_run_id(run_config.name)
    snapshot_copy = _normalize_snapshot(config_snapshot)
    created_at = timestamp()
    manifest = {
        "run_id": run_id,
        "name": run_config.name,
        "created_at": created_at,
        "updated_at": created_at,
        "last_resume_at": None,
        "config_source": str(config_source) if config_source else None,
        "config_checksum": _compute_config_checksum(snapshot_copy),
        "run": {
            "output_dir": str(run_config.output_dir),
            "run_dir": str(Path(run_config.output_dir) / run_id),
            "env_dir_path": run_config.env_dir_path,
            "endpoints_path": run_config.endpoints_path,
            "default_api_key_var": run_config.default_api_key_var,
            "default_api_base_url": run_config.default_api_base_url,
        },
        "jobs": [_manifest_job_entry(job) for job in jobs],
        "config_snapshot": snapshot_copy,
    }
    return manifest


def write_run_manifest(
    *,
    run_dir: Path,
    run_config: RunConfig,
    jobs: Sequence[ResolvedJob],
    config_source: Path | None,
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    payload = build_run_manifest_payload(
        run_config=run_config,
        jobs=jobs,
        config_source=config_source,
        config_snapshot=config_snapshot,
    )
    save_run_manifest(run_dir=run_dir, manifest=payload)
    return payload


def save_run_manifest(*, run_dir: Path, manifest: Mapping[str, Any]) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest_path


def load_run_manifest(run_dir: Path) -> Mapping[str, Any] | None:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_completed_job_outcome(run_dir: Path, job: ResolvedJob) -> JobOutcome | None:
    job_dir = run_dir / job.job_id
    summary_path = job_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    job_id = payload.get("job_id") or job.job_id
    job_name = payload.get("job_name") or job.name
    model_id = payload.get("model_id") or job.model.id
    env_id = payload.get("env_id") or job.env.id
    duration = payload.get("duration_seconds")
    return JobOutcome(
        job_id=job_id,
        job_name=job_name,
        model_id=model_id,
        env_id=env_id,
        status="succeeded",
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        results_path=str(job_dir),
    )


def load_run_summary(run_dir: Path) -> dict[str, JobOutcome]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    outcomes: dict[str, JobOutcome] = {}
    for entry in payload.get("jobs", []):
        job_id = entry.get("job_id")
        if not isinstance(job_id, str):
            continue
        duration = entry.get("duration_seconds")
        outcomes[job_id] = JobOutcome(
            job_id=job_id,
            job_name=entry.get("job_name", job_id),
            model_id=entry.get("model_id", ""),
            env_id=entry.get("env_id", ""),
            status=entry.get("status", "unknown"),
            duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
            results_path=entry.get("results_path"),
            error=entry.get("error"),
        )
    return outcomes


def load_completed_job_outcome_by_id(run_dir: Path, entry: Mapping[str, Any]) -> JobOutcome | None:
    job_id = entry.get("job_id")
    if not isinstance(job_id, str):
        return None
    job_dir = run_dir / job_id
    summary_path = job_dir / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    job_name = payload.get("job_name") or entry.get("job_name") or job_id
    model_id = payload.get("model_id") or entry.get("model_id") or ""
    env_id = payload.get("env_id") or entry.get("env_id") or ""
    duration = payload.get("duration_seconds")
    return JobOutcome(
        job_id=job_id,
        job_name=job_name,
        model_id=model_id,
        env_id=env_id,
        status="succeeded",
        duration_seconds=float(duration) if isinstance(duration, (int, float)) else None,
        results_path=str(job_dir),
    )


def reconstruct_manifest_from_run_dir(
    *,
    run_dir: Path,
    run_config: RunConfig,
    run_id: str,
    config_source: Path | None,
    config_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    summary_path = run_dir / "run_summary.json"
    summary_payload: dict[str, Any] | None = None
    if summary_path.exists():
        try:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary_payload = json.load(handle)
        except (OSError, ValueError):
            summary_payload = None

    snapshot_copy = _normalize_snapshot(config_snapshot)
    created_at = summary_payload.get("created_at") if summary_payload else timestamp()
    manifest = {
        "run_id": run_id,
        "name": summary_payload.get("name", run_config.name) if summary_payload else run_config.name,
        "created_at": created_at,
        "updated_at": created_at,
        "last_resume_at": None,
        "config_source": (
            summary_payload.get("config_source")
            if summary_payload and summary_payload.get("config_source") is not None
            else str(config_source) if config_source else None
        ),
        "config_checksum": _compute_config_checksum(snapshot_copy),
        "run": {
            "output_dir": str(run_config.output_dir),
            "run_dir": str(run_dir),
            "env_dir_path": run_config.env_dir_path,
            "endpoints_path": run_config.endpoints_path,
            "default_api_key_var": run_config.default_api_key_var,
            "default_api_base_url": run_config.default_api_base_url,
        },
        "jobs": [],
        "config_snapshot": snapshot_copy,
    }

    job_entries: list[dict[str, Any]] = []
    source_jobs = []
    if summary_payload:
        source_jobs = summary_payload.get("jobs", [])

    def _append_entry(
        *,
        job_id: str,
        job_name: str | None,
        model_id: str | None,
        env_id: str | None,
        results_dir: str,
    ) -> None:
        job_entries.append(
            {
                "job_id": job_id,
                "job_name": job_name or job_id,
                "model_id": model_id or "",
                "env_id": env_id or "",
                "env_overrides": {},
                "sampling_overrides": {},
                "results_dir": results_dir,
            }
        )

    if source_jobs:
        for record in source_jobs:
            job_id = record.get("job_id")
            if not isinstance(job_id, str):
                continue
            _append_entry(
                job_id=job_id,
                job_name=record.get("job_name"),
                model_id=record.get("model_id"),
                env_id=record.get("env_id"),
                results_dir=job_id,
            )
    else:
        for child in sorted(run_dir.iterdir()):
            if not child.is_dir():
                continue
            summary_file = child / "summary.json"
            if not summary_file.exists():
                continue
            try:
                with summary_file.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError):
                continue
            job_id = str(payload.get("job_id") or child.name)
            _append_entry(
                job_id=job_id,
                job_name=payload.get("job_name"),
                model_id=payload.get("model_id"),
                env_id=payload.get("env_id"),
                results_dir=child.name,
            )

    job_entries.sort(key=lambda entry: entry["job_id"])
    manifest["jobs"] = job_entries
    return manifest


def _build_outcome_from_manifest_entry(
    entry: Mapping[str, Any],
    run_dir: Path,
    *,
    status: str = "pending",
) -> JobOutcome:
    job_id = entry.get("job_id")
    job_id_str = str(job_id) if job_id is not None else "unknown"
    job_name = entry.get("job_name") or job_id_str
    model_id = entry.get("model_id") or ""
    env_id = entry.get("env_id") or ""
    results_dir = entry.get("results_dir")
    results_path: str | None = None
    if isinstance(results_dir, str):
        results_path = str(run_dir / results_dir)
    return JobOutcome(
        job_id=job_id_str,
        job_name=str(job_name),
        model_id=str(model_id),
        env_id=str(env_id),
        status=status,
        results_path=results_path,
    )


def merge_run_outcomes(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any] | None,
    current_outcomes: Sequence[JobOutcome],
    completed_jobs: Mapping[str, JobOutcome],
    previous_summary: Mapping[str, JobOutcome],
) -> list[JobOutcome]:
    current_by_id = {outcome.job_id: outcome for outcome in current_outcomes}
    final_outcomes: list[JobOutcome] = []
    manifest_entries: Sequence[Mapping[str, Any]]
    if manifest is not None:
        manifest_entries = [
            entry
            for entry in manifest.get("jobs", [])
            if isinstance(entry, Mapping) and "job_id" in entry
        ]
    else:
        manifest_entries = []

    if manifest_entries:
        manifest_job_ids = {entry["job_id"] for entry in manifest_entries if isinstance(entry["job_id"], str)}
        for entry in manifest_entries:
            job_id = entry.get("job_id")
            if not isinstance(job_id, str):
                continue
            current = current_by_id.get(job_id)
            if current is None:
                fallback = previous_summary.get(job_id)
                if fallback is None:
                    fallback = load_completed_job_outcome_by_id(run_dir, entry)
                if fallback is None:
                    fallback = _build_outcome_from_manifest_entry(entry, run_dir)
                final_outcomes.append(fallback)
                continue
            if current.status == "skipped":
                reuse = (
                    completed_jobs.get(job_id)
                    or previous_summary.get(job_id)
                    or load_completed_job_outcome_by_id(run_dir, entry)
                )
                if reuse is not None:
                    final_outcomes.append(reuse)
                else:
                    final_outcomes.append(_build_outcome_from_manifest_entry(entry, run_dir, status="skipped"))
            else:
                final_outcomes.append(current)
        extra_ids = [job_id for job_id in current_by_id if job_id not in manifest_job_ids]
        for job_id in extra_ids:
            final_outcomes.append(current_by_id[job_id])
        return final_outcomes

    merged: dict[str, JobOutcome] = dict(previous_summary)
    for job_id, outcome in current_by_id.items():
        if outcome.status == "skipped":
            merged[job_id] = completed_jobs.get(job_id) or previous_summary.get(job_id) or outcome
        else:
            merged[job_id] = outcome
    return list(merged.values())


def write_summary(
    *,
    summary_path: Path,
    run_config: RunConfig,
    config_source: str,
    outcomes: Sequence[JobOutcome],
) -> None:
    payload = {
        "run_id": run_config.run_id,
        "name": run_config.name,
        "created_at": timestamp(),
        "config_source": config_source,
        "output_dir": str(summary_path.parent),
        "jobs": [
            {
                "job_id": outcome.job_id,
                "job_name": outcome.job_name,
                "model_id": outcome.model_id,
                "env_id": outcome.env_id,
                "status": outcome.status,
                "duration_seconds": outcome.duration_seconds,
                "results_path": outcome.results_path,
                "error": outcome.error,
            }
            for outcome in outcomes
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    _ensure_root_logging(log_level)

    try:
        run_config, jobs_path, config_snapshot = prepare_run_config(args)
    except ValueError as exc:
        logger.error("Failed to prepare run configuration: %s", exc)
        return 1

    resume_run_id = args.resume
    if resume_run_id and args.run_id:
        logger.error("Cannot combine --resume with --run-id.")
        return 1

    run_id = resume_run_id or args.run_id or run_config.run_id or derive_run_id(run_config.name)
    jobs = expand_jobs(run_config)

    if args.dry_run:
        _print_dry_run(run_config=run_config, jobs=jobs, jobs_path=jobs_path, run_id=run_id)
        return 0

    if not jobs:
        logger.warning("No jobs were resolved from the configuration.")
        return 0

    if args.install_envs:
        env_ids = sorted({job.env.id for job in jobs})
        if env_ids:
            logger.info(
                "Installing %d environment(s) from '%s' before benchmarking.",
                len(env_ids),
                run_config.env_dir_path,
            )
            for env_id in env_ids:
                logger.info("Installing environment '%s'.", env_id)
                install_environment(env=env_id, path=run_config.env_dir_path, from_repo=False, branch="main")

    run_config.run_id = run_id
    manifest: dict[str, Any] | None = None
    completed_jobs_map: dict[str, JobOutcome] = {}
    outcomes: list[JobOutcome] = []
    run_dir = run_config.output_dir / run_id
    try:
        resume_mode = resume_run_id is not None
        if resume_mode:
            if not run_dir.exists():
                logger.error("Run directory '%s' does not exist; cannot resume.", run_dir)
                return 1
            manifest_data = load_run_manifest(run_dir)
            if manifest_data is None:
                logger.warning("Run %s lacks a manifest; creating a fresh snapshot before resuming.", run_id)
                manifest = reconstruct_manifest_from_run_dir(
                    run_dir=run_dir,
                    run_config=run_config,
                    run_id=run_id,
                    config_source=jobs_path,
                    config_snapshot=config_snapshot,
                )
            else:
                manifest = dict(manifest_data)
            _, new_job_ids = upsert_manifest_jobs(manifest, jobs)
            update_manifest_metadata(
                manifest,
                run_config=run_config,
                config_source=jobs_path,
                config_snapshot=config_snapshot,
                resumed=True,
            )
            save_run_manifest(run_dir=run_dir, manifest=manifest)
            manifest_job_ids = {
                entry["job_id"]
                for entry in manifest.get("jobs", [])
                if isinstance(entry, Mapping) and "job_id" in entry
            }
            current_job_ids = {job.job_id for job in jobs}
            removed_job_ids = sorted(manifest_job_ids - current_job_ids)
            if removed_job_ids:
                logger.info(
                    "Configuration now excludes %d previously scheduled job(s): %s",
                    len(removed_job_ids),
                    ", ".join(removed_job_ids),
                )
            if new_job_ids:
                logger.info("Discovered %d new job(s) for resume: %s", len(new_job_ids), ", ".join(sorted(new_job_ids)))
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(jobs_path, run_dir / jobs_path.name)
            manifest = write_run_manifest(
                run_dir=run_dir,
                run_config=run_config,
                jobs=jobs,
                config_source=jobs_path,
                config_snapshot=config_snapshot,
            )

        logger.info("Starting run %s with %d job(s).", run_id, len(jobs))
        if args.force:
            logger.info("Force mode enabled; completed jobs will be re-run.")

        completed_jobs_map = {}
        for job in jobs:
            existing = load_completed_job_outcome(run_dir, job)
            if existing:
                completed_jobs_map[job.job_id] = existing

        outcomes = asyncio.run(
            execute_jobs(
                run_config=run_config,
                jobs=jobs,
                verbose=args.verbose,
                completed_jobs=completed_jobs_map,
                force=args.force,
            )
        )
    except KeyboardInterrupt:
        logger.error("Benchmark interrupted by user.")
        return 1
    except Exception:  # noqa: BLE001
        logger.exception("Benchmark command failed.")
        return 1

    previous_summary = load_run_summary(run_dir)
    final_outcomes = merge_run_outcomes(
        run_dir=run_dir,
        manifest=manifest,
        current_outcomes=outcomes,
        completed_jobs=completed_jobs_map,
        previous_summary=previous_summary,
    )

    summary_path = run_dir / "run_summary.json"
    write_summary(
        summary_path=summary_path,
        run_config=run_config,
        config_source=str(jobs_path),
        outcomes=final_outcomes,
    )
    failures = [outcome for outcome in final_outcomes if outcome.status == "failed"]
    if failures:
        logger.error("Run %s completed with %d failure(s).", run_id, len(failures))
        return 1
    logger.info("Run %s completed successfully.", run_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
