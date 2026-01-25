"""Job execution utilities for the unified CLI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Literal, Mapping, Sequence
from pydantic import BaseModel, field_validator

from verifiers.types import GenerateOutputs
from verifiers.utils.eval_utils import run_evaluation

from medarc_verifiers.cli.utils.reporting import compute_average, compute_metric_averages
from medarc_verifiers.cli._eval_builder import build_client_config, build_eval_config
from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._manifest import RunManifest
from medarc_verifiers.cli._schemas import ModelConfigSchema
from medarc_verifiers.cli.utils.endpoint_utils import (
    EndpointRegistry,
    EndpointRegistryCache,
    EnvMetadataCache,
    load_endpoint_registry,
    load_env_metadata,
)
from medarc_verifiers.cli.utils.shared import DEFAULT_BATCH_MAX_CONCURRENT, ensure_root_logging, resolve_env_identifier

try:
    from rich import print as rich_print  # type: ignore
except ImportError:  # pragma: no cover - rich is optional
    rich_print = None

logger = logging.getLogger(__name__)


class ExecutorSettings(BaseModel):
    """Run-level options controlling how jobs are executed."""

    run_id: str
    output_dir: Path
    env_dir: Path
    endpoints_path: Path | None = None
    default_api_key_var: str
    default_api_base_url: str
    api_base_url_override: str | None = None
    log_level: str = "INFO"
    verbose: bool = False
    save_results: bool = True
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None
    max_concurrent_generation: int | None = None
    max_concurrent_scoring: int | None = None
    max_concurrent: int | None = None  # CLI override for max_concurrent
    timeout: float | None = None
    sleep: float = 0.0
    dry_run: bool = False
    cli_env_args: dict[str, Any] | None = None
    cli_sampling_args: dict[str, Any] | None = None

    @field_validator("output_dir", "env_dir", mode="before")
    @classmethod
    def _expand_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser()

    @field_validator("endpoints_path", mode="before")
    @classmethod
    def _expand_optional_path(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()


class JobExecutionResult(BaseModel):
    """Outcome emitted for each executed job."""

    job_id: str
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None = None
    duration_seconds: float | None = None
    output_path: Path | None = None
    result: Any | None = None


def execute_jobs(
    jobs: Sequence[ResolvedJob],
    settings: ExecutorSettings,
    *,
    endpoints_cache: EndpointRegistryCache | None = None,
    env_metadata_cache: EnvMetadataCache | None = None,
    manifest: RunManifest | None = None,
) -> list[JobExecutionResult]:
    """Execute a sequence of resolved jobs."""
    ensure_root_logging(settings.log_level)
    logger.info("Starting run '%s' with %d job(s).", settings.run_id, len(jobs))

    run_dir = settings.output_dir / settings.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    job_statuses: dict[str, str] = {job.job_id: "pending" for job in jobs}
    results: list[JobExecutionResult] = []
    interrupted = False

    for index, job in enumerate(jobs):
        is_last_job = index == len(jobs) - 1
        env_identifier = resolve_env_identifier(job.env)
        model_identifier = job.model.id or job.model.model or job.job_id
        job_label = f"{job.job_id} (env={env_identifier}, model={model_identifier})"
        logger.info("Job %d/%d starting: %s", index + 1, len(jobs), job_label)
        job_dir = (run_dir / job.job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        job_statuses[job.job_id] = "running"

        if settings.dry_run:
            logger.info("Dry run enabled; skipping execution for job '%s'.", job.job_id)
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="skipped",
                    output_path=job_dir,
                )
            )
            job_statuses[job.job_id] = "skipped"
            _log_job_progress_window(jobs, index, job_statuses, event="dry-run skip")
            continue

        if manifest is not None:
            manifest.record_job_start(job.job_id)

        try:
            endpoints = _load_endpoints_for_model(job.model, settings, cache=endpoints_cache)
            resolved_model, client_config, prime_sampling_overrides = build_client_config(
                job.model,
                endpoints=endpoints,
                default_api_key_var=settings.default_api_key_var,
                default_api_base_url=settings.default_api_base_url,
                api_base_url_override=settings.api_base_url_override,
                timeout_override=settings.timeout,
                headers=job.model.headers,
            )
            # Merge Prime Inference overrides with job sampling args (job args take precedence)
            merged_sampling_args = {**prime_sampling_overrides, **job.sampling_args}
            eval_config = build_eval_config(
                job_label=job.job_id,
                model_cfg=job.model,
                env_cfg=job.env,
                env_args=job.env_args,
                sampling_args=merged_sampling_args,
                cli_env_args=settings.cli_env_args,
                cli_sampling_args=settings.cli_sampling_args,
                resolved_model=resolved_model,
                client_config=client_config,
                env_dir=settings.env_dir,
                max_concurrent_override=settings.max_concurrent,
                max_concurrent_generation=settings.max_concurrent_generation,
                max_concurrent_scoring=settings.max_concurrent_scoring,
                default_max_concurrent=DEFAULT_BATCH_MAX_CONCURRENT,
                save_results=settings.save_results,
                save_to_hf_hub=settings.save_to_hf_hub,
                hf_hub_dataset_name=settings.hf_hub_dataset_name,
                verbose=settings.verbose,
                env_metadata_cache=env_metadata_cache,
                env_metadata_loader=load_env_metadata,
                enforce_required_env_args=True,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted while preparing job %s.", job_label)
            if manifest is not None:
                manifest.record_job_failure(job.job_id, error="interrupted by user")
            interruption_message = f"{job_label} interrupted by user"
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="failed",
                    error=interruption_message,
                    output_path=job_dir,
                )
            )
            job_statuses[job.job_id] = "interrupted"
            _log_job_progress_window(jobs, index, job_statuses, event="interruption", note="during preparation")
            interrupted = True
            break
        except Exception as exc:  # noqa: BLE001
            error_message = f"{job_label} preparation failed: {exc}"
            logger.exception("%s", error_message)
            if manifest is not None:
                manifest.record_job_failure(job.job_id, error=str(exc))
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="failed",
                    error=error_message,
                    output_path=job_dir,
                )
            )
            job_statuses[job.job_id] = "failed"
            _log_job_progress_window(jobs, index, job_statuses, event="failure", note="during preparation")
            _maybe_sleep_between_jobs(job, settings, is_last=is_last_job)
            continue

        start = perf_counter()
        try:
            eval_result = asyncio.run(run_evaluation(eval_config))
        except KeyboardInterrupt:
            duration = perf_counter() - start
            logger.warning("Job %s interrupted by user after %.2fs.", job_label, duration)
            if manifest is not None:
                manifest.record_job_failure(job.job_id, error="interrupted by user", duration_seconds=duration)
            interruption_message = f"{job_label} interrupted by user"
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="failed",
                    error=interruption_message,
                    duration_seconds=duration,
                    output_path=job_dir,
                )
            )
            job_statuses[job.job_id] = "interrupted"
            _log_job_progress_window(jobs, index, job_statuses, event="interruption")
            interrupted = True
            break
        except Exception as exc:  # noqa: BLE001
            duration = perf_counter() - start
            error_message = f"{job_label} failed after {duration:.2f}s: {exc}"
            logger.exception("%s", error_message)
            if manifest is not None:
                manifest.record_job_failure(job.job_id, error=str(exc), duration_seconds=duration)
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="failed",
                    error=error_message,
                    duration_seconds=duration,
                    output_path=job_dir,
                )
            )
            job_statuses[job.job_id] = "failed"
            _log_job_progress_window(jobs, index, job_statuses, event="failure")
            _maybe_sleep_between_jobs(job, settings, is_last=is_last_job)
            continue

        duration = perf_counter() - start
        logger.info("Job '%s' completed in %.2fs.", job.job_id, duration)

        artifacts = _materialize_results(job_dir, run_dir, eval_result)
        avg_reward = _extract_avg_reward(eval_result)
        metrics_avg = compute_metric_averages(_safe_get(eval_result, "metrics", {}))
        metadata = _safe_get(eval_result, "metadata", None)
        num_examples = _safe_get(metadata, "num_examples", None)
        rollouts_per_example = _safe_get(metadata, "rollouts_per_example", None)

        if manifest is not None:
            manifest.record_job_completion(
                job.job_id,
                duration_seconds=duration,
                results_dir=job_dir,
                artifacts=artifacts,
                avg_reward=avg_reward,
                metrics=metrics_avg,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
            )

        results.append(
            JobExecutionResult(
                job_id=job.job_id,
                status="succeeded",
                duration_seconds=duration,
                output_path=job_dir,
                result=eval_result,
            )
        )
        job_statuses[job.job_id] = "completed"
        _log_job_progress_window(jobs, index, job_statuses, event="completion")
        _maybe_sleep_between_jobs(job, settings, is_last=is_last_job)

    if interrupted:
        logger.warning("Execution interrupted by user; %d job(s) left pending.", len(jobs) - len(results))

    return results


def _load_endpoints_for_model(
    model_cfg: ModelConfigSchema,
    settings: ExecutorSettings,
    *,
    cache: EndpointRegistryCache | None,
) -> EndpointRegistry:
    """Load the endpoint registry to use for a model."""
    registry_path = model_cfg.endpoints_path or settings.endpoints_path
    if registry_path is None:
        return {}
    return load_endpoint_registry(registry_path, cache=cache)


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Retrieve attribute or dict key, allowing newer dict-style GenerateOutputs."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _materialize_results(job_dir: Path, run_dir: Path, results: GenerateOutputs) -> list[str]:
    """Move evaluation artifacts into the job directory and report their paths."""
    artifacts: list[str] = []
    metadata = _safe_get(results, "metadata", None)
    raw_path = _safe_get(metadata, "path_to_save", None)
    src_path = Path(raw_path) if raw_path else job_dir
    try:
        resolved_src = src_path.resolve()
    except OSError:
        resolved_src = src_path

    if src_path.exists() and resolved_src != job_dir:
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

    for path in sorted(job_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            artifacts.append(str(path.relative_to(run_dir)))
        except ValueError:
            artifacts.append(str(path))
    return artifacts


def _extract_avg_reward(results: GenerateOutputs) -> float | None:
    """Compute the average reward from the evaluation payload."""
    rewards = _safe_get(results, "reward", None)
    avg = compute_average(rewards)
    if avg is not None:
        return avg
    metadata = _safe_get(results, "metadata", None)
    metadata_avg = _safe_get(metadata, "avg_reward", None)
    if metadata_avg is not None:
        return float(metadata_avg)
    return None


def _log_job_progress_window(
    jobs: Sequence[ResolvedJob],
    center_index: int,
    job_statuses: Mapping[str, str],
    *,
    event: str,
    note: str | None = None,
) -> None:
    if not jobs:
        return
    start = max(0, center_index - 1)
    end = min(len(jobs), center_index + 2)
    lines: list[str] = []
    header = "Segment | Job ID | Status | Model | Env | Name"
    divider = "-" * len(header)
    lines.append(header)
    lines.append(divider)
    for idx in range(start, end):
        job = jobs[idx]
        segment = "current" if idx == center_index else ("previous" if idx < center_index else "next")
        status = job_statuses.get(job.job_id, "pending")
        model_label = job.model.id or job.model.model or "-"
        try:
            env_label = resolve_env_identifier(job.env)
        except ValueError:
            env_label = job.env.id or job.job_id
        lines.append(
            f"{segment:8} | {job.job_id:20} | {status:10} | {model_label:15} | {env_label:20} | {job.name or '-'}"
        )
    label = f"Job progress after {event}"
    if note:
        label = f"{label} ({note})"
    logger.info("%s:\n%s", label, "\n".join(lines))


def _maybe_sleep_between_jobs(job: ResolvedJob, settings: ExecutorSettings, *, is_last: bool) -> None:
    """Optionally pause between jobs to spread out environment runs."""
    if settings.dry_run or is_last:
        return
    delay = job.sleep if job.sleep is not None else settings.sleep
    if delay is None or delay <= 0:
        return
    if rich_print:
        rich_print(f"[cyan]Sleeping {delay:.2f} second(s) before next job...[/cyan]")
    logger.info("Sleeping %.2f second(s) before next job...", delay)
    sleep(delay)


__all__ = ["ExecutorSettings", "JobExecutionResult", "execute_jobs"]
