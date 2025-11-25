"""Job execution utilities for the unified CLI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Sequence, Mapping
from pydantic import BaseModel, field_validator

from verifiers.types import ClientConfig, EvalConfig, GenerateOutputs
from verifiers.utils.eval_utils import run_evaluation

from medarc_verifiers.cli.utils.reporting import compute_average, compute_metric_averages
from medarc_verifiers.cli_new._job_builder import ResolvedJob
from medarc_verifiers.cli_new._manifest import RunManifest
from medarc_verifiers.cli_new._schemas import ModelConfigSchema
from medarc_verifiers.cli_new.utils.endpoint_utils import (
    EndpointRegistry,
    EndpointRegistryCache,
    EnvMetadataCache,
    load_endpoint_registry,
    load_env_metadata,
    resolve_model_endpoint,
)
from medarc_verifiers.cli_new.utils.env_args import validate_env_args_or_raise
from medarc_verifiers.cli_new.utils.shared import (
    DEFAULT_BATCH_MAX_CONCURRENT,
    ensure_root_logging,
    normalize_headers,
    resolve_env_identifier,
)
from medarc_verifiers.utils import sanitize_sampling_args_for_openai

logger = logging.getLogger(__name__)


class ExecutorSettings(BaseModel):
    """Run-level options controlling how jobs are executed."""

    run_id: str
    output_dir: Path
    env_dir: Path
    endpoints_path: Path | None = None
    default_api_key_var: str
    default_api_base_url: str
    log_level: str = "INFO"
    verbose: bool = False
    save_results: bool = True
    save_to_hf_hub: bool = False
    hf_hub_dataset_name: str | None = None
    max_concurrent_generation: int | None = None
    max_concurrent_scoring: int | None = None
    max_concurrent: int | None = None  # CLI override for max_concurrent
    timeout: float | None = None
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
            eval_config = _build_eval_config(
                job,
                settings=settings,
                endpoints_cache=endpoints_cache,
                env_metadata_cache=env_metadata_cache,
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

    if interrupted:
        logger.warning("Execution interrupted by user; %d job(s) left pending.", len(jobs) - len(results))

    return results


def _build_eval_config(
    job: ResolvedJob,
    *,
    settings: ExecutorSettings,
    endpoints_cache: EndpointRegistryCache | None,
    env_metadata_cache: EnvMetadataCache | None,
) -> EvalConfig:
    """Construct the EvalConfig for a given job."""
    model_cfg = job.model
    env_cfg = job.env

    headers = normalize_headers(model_cfg.headers)
    endpoints = _load_endpoints_for_model(model_cfg, settings, cache=endpoints_cache)
    model_alias = model_cfg.model or model_cfg.id
    if not model_alias:
        raise ValueError("Model entries must define 'id' or 'model'.")

    default_key_var = model_cfg.api_key_var or settings.default_api_key_var
    default_base_url = model_cfg.api_base_url or settings.default_api_base_url
    resolved_model, api_key_var, api_base_url = resolve_model_endpoint(
        model_alias,
        endpoints,
        default_key_var=default_key_var,
        default_base_url=default_base_url,
    )

    client_kwargs: dict[str, Any] = {
        "api_key_var": api_key_var,
        "api_base_url": api_base_url,
        "extra_headers": headers or None,
    }
    timeout = settings.timeout if settings.timeout is not None else model_cfg.timeout
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    if model_cfg.max_connections is not None:
        client_kwargs["max_connections"] = model_cfg.max_connections
    if model_cfg.max_keepalive_connections is not None:
        client_kwargs["max_keepalive_connections"] = model_cfg.max_keepalive_connections
    if model_cfg.max_retries is not None:
        client_kwargs["max_retries"] = model_cfg.max_retries
    client_config = ClientConfig(**client_kwargs)

    env_id = resolve_env_identifier(env_cfg)
    env_args = dict(job.env_args)

    # Apply CLI overrides (Layer 5: highest precedence)
    # This is the final merge in the precedence chain:
    # env.env_args → model.env_args → model.env_overrides → job.env_args → CLI
    if settings.cli_env_args:
        # Log what CLI is overriding for debugging
        if settings.verbose:
            overridden_keys = set(env_args) & set(settings.cli_env_args)
            new_keys = set(settings.cli_env_args) - set(env_args)
            if overridden_keys:
                logger.debug(
                    "CLI overriding env_args for job '%s': %s",
                    job.job_id,
                    {k: f"{env_args[k]} → {settings.cli_env_args[k]}" for k in overridden_keys},
                )
            if new_keys:
                logger.debug("CLI adding new env_args for job '%s': %s", job.job_id, list(new_keys))
        env_args.update(settings.cli_env_args)

    # Phase 2 validation: Enforce required parameters now that CLI overrides are merged
    # This is stricter than load-time validation (Phase 1 in _config_loader.py)
    try:
        metadata = load_env_metadata(env_id, cache=env_metadata_cache)
    except ImportError as exc:
        logger.warning("Skipping env_args validation for '%s': %s", env_id, exc)
    else:
        if metadata:
            # Now enforce required parameters (unlike Phase 1 which was lenient)
            validate_env_args_or_raise(
                env_id,
                env_args,
                metadata,
                enforce_required=True,  # Strict validation before execution
            )

    # Resolve max_concurrent with proper precedence:
    # 1. CLI --max-concurrent (settings.max_concurrent)
    # 2. Model config max_concurrent (model_cfg.max_concurrent)
    # 3. Environment config max_concurrent (env_cfg.max_concurrent)
    # 4. DEFAULT_BATCH_MAX_CONCURRENT constant
    max_concurrent = (
        settings.max_concurrent or model_cfg.max_concurrent or env_cfg.max_concurrent or DEFAULT_BATCH_MAX_CONCURRENT
    )
    if env_cfg.verbose is None:
        verbose_flag = settings.verbose
    else:
        verbose_flag = env_cfg.verbose

    save_every = env_cfg.save_every if env_cfg.save_every is not None else -1

    sampling_args = dict(job.sampling_args)
    if settings.cli_sampling_args:
        sampling_args.update(settings.cli_sampling_args)
    # Route non-OpenAI kwargs (e.g., top_k, min_p) into extra_body to avoid client errors
    sampling_args = sanitize_sampling_args_for_openai(sampling_args)
    state_columns = list(env_cfg.state_columns) if env_cfg.state_columns else None

    return EvalConfig(
        env_id=env_id,
        env_args=env_args,
        env_dir_path=str(settings.env_dir),
        model=resolved_model,
        client_config=client_config,
        sampling_args=sampling_args,
        num_examples=env_cfg.num_examples,
        rollouts_per_example=env_cfg.rollouts_per_example,
        max_concurrent=max_concurrent,
        max_concurrent_generation=settings.max_concurrent_generation,
        max_concurrent_scoring=settings.max_concurrent_scoring,
        interleave_scoring=env_cfg.interleave_scoring,
        print_results=env_cfg.print_results,
        verbose=verbose_flag,
        state_columns=state_columns,
        save_results=settings.save_results,
        save_every=save_every,
        save_to_hf_hub=settings.save_to_hf_hub,
        hf_hub_dataset_name=settings.hf_hub_dataset_name,
    )


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


__all__ = ["ExecutorSettings", "JobExecutionResult", "execute_jobs"]
