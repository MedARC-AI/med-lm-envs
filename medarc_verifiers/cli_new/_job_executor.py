"""Job execution utilities for the unified CLI."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Sequence
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
    default_max_concurrent: int = DEFAULT_BATCH_MAX_CONCURRENT
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

    results: list[JobExecutionResult] = []
    for index, job in enumerate(jobs, start=1):
        env_identifier = resolve_env_identifier(job.env)
        model_identifier = job.model.id or job.model.model or job.job_id
        job_label = f"{job.job_id} env={env_identifier} model={model_identifier}"
        logger.info("Job %d/%d starting: %s", index, len(jobs), job_label)
        job_dir = (run_dir / job.job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)

        if settings.dry_run:
            logger.info("Dry run enabled; skipping execution for job '%s'.", job.job_id)
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="skipped",
                    output_path=job_dir,
                )
            )
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
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to prepare evaluation config for job '%s': %s", job.job_id, exc)
            if manifest is not None:
                manifest.record_job_failure(job.job_id, error=str(exc))
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="failed",
                    error=str(exc),
                    output_path=job_dir,
                )
            )
            continue

        start = perf_counter()
        try:
            eval_result = asyncio.run(run_evaluation(eval_config))
        except Exception as exc:  # noqa: BLE001
            duration = perf_counter() - start
            logger.exception("Job '%s' failed after %.2fs: %s", job.job_id, duration, exc)
            if manifest is not None:
                manifest.record_job_failure(job.job_id, error=str(exc), duration_seconds=duration)
            results.append(
                JobExecutionResult(
                    job_id=job.job_id,
                    status="failed",
                    error=str(exc),
                    duration_seconds=duration,
                    output_path=job_dir,
                )
            )
            continue

        duration = perf_counter() - start
        logger.info("Job '%s' completed in %.2fs.", job.job_id, duration)

        artifacts = _materialize_results(job_dir, run_dir, eval_result)
        avg_reward = _extract_avg_reward(eval_result)
        metrics_avg = compute_metric_averages(eval_result.metrics)
        num_examples = getattr(eval_result.metadata, "num_examples", None)
        rollouts_per_example = getattr(eval_result.metadata, "rollouts_per_example", None)

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
    if model_cfg.timeout is not None:
        client_kwargs["timeout"] = model_cfg.timeout
    if model_cfg.max_connections is not None:
        client_kwargs["max_connections"] = model_cfg.max_connections
    if model_cfg.max_keepalive_connections is not None:
        client_kwargs["max_keepalive_connections"] = model_cfg.max_keepalive_connections
    if model_cfg.max_retries is not None:
        client_kwargs["max_retries"] = model_cfg.max_retries
    client_config = ClientConfig(**client_kwargs)

    env_id = resolve_env_identifier(env_cfg)
    env_args = dict(job.env_args)
    if settings.cli_env_args:
        env_args.update(settings.cli_env_args)

    try:
        metadata = load_env_metadata(env_id, cache=env_metadata_cache)
    except ImportError as exc:
        logger.warning("Skipping env_args validation for '%s': %s", env_id, exc)
    else:
        if metadata:
            validate_env_args_or_raise(env_id, env_args, metadata, enforce_required=True)

    max_concurrent = env_cfg.max_concurrent or settings.default_max_concurrent
    if env_cfg.verbose is None:
        verbose_flag = settings.verbose
    else:
        verbose_flag = env_cfg.verbose

    save_every = env_cfg.save_every if env_cfg.save_every is not None else -1

    sampling_args = dict(job.sampling_args)
    if settings.cli_sampling_args:
        sampling_args.update(settings.cli_sampling_args)
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


def _materialize_results(job_dir: Path, run_dir: Path, results: GenerateOutputs) -> list[str]:
    """Move evaluation artifacts into the job directory and report their paths."""
    artifacts: list[str] = []
    raw_path = getattr(results.metadata, "path_to_save", None)
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
    avg = compute_average(results.reward)
    if avg is not None:
        return avg
    metadata_avg = getattr(results.metadata, "avg_reward", None)
    if metadata_avg is not None:
        return float(metadata_avg)
    return None


__all__ = ["ExecutorSettings", "JobExecutionResult", "execute_jobs"]
