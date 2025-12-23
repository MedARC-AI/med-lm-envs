"""Resolve validated run configurations into executable job definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ._schemas import EnvironmentConfigSchema, ModelConfigSchema, RunConfigSchema
from .utils.env_args import merge_env_args
from .utils.shared import compute_checksum, slugify


@dataclass(slots=True)
class ResolvedJob:
    """Executable job produced from a run configuration."""

    job_id: str
    name: str
    model: ModelConfigSchema
    env: EnvironmentConfigSchema
    env_args: dict[str, Any]
    sampling_args: dict[str, Any]
    sleep: float | None = None


def build_jobs(config: RunConfigSchema) -> list[ResolvedJob]:
    """Expand a validated run configuration into concrete jobs."""
    matrix_index = _build_matrix_index(config.envs.values())
    models: dict[str, ModelConfigSchema] = config.models
    resolved: list[ResolvedJob] = []
    used_ids: set[str] = set()

    for job_cfg in config.jobs:
        model_id, model = _resolve_model(job_cfg.model, models)
        if model.id is None:
            model = model.model_copy(update={"id": model_id})
            models[model_id] = model
        env_targets = _coerce_iterable(job_cfg.env)
        for env_target in env_targets:
            for env_id in _resolve_env_ids(env_target, config.envs, matrix_index):
                env = config.envs[env_id]
                if env.id is None:
                    env = env.model_copy(update={"id": env_id})
                    config.envs[env_id] = env
                env_args = _compose_env_args(env, model, job_cfg.env_args)
                sampling_args = _compose_sampling_args(model.sampling_args, job_cfg.sampling_args)
                name = job_cfg.name or f"{model_id}-{env.id}"
                job_id = _build_job_id(
                    model_id=model_id,
                    env_id=env.id,
                    job_name=job_cfg.name,
                    env_overrides=job_cfg.env_args,
                    sampling_overrides=job_cfg.sampling_args,
                    used_ids=used_ids,
                )
                used_ids.add(job_id)
                resolved.append(
                    ResolvedJob(
                        job_id=job_id,
                        name=name,
                        model=model,
                        env=env,
                        env_args=env_args,
                        sampling_args=sampling_args,
                        sleep=job_cfg.sleep,
                    )
                )

    return resolved


def _resolve_model(
    model_ref: str | dict[str, Any],
    models: dict[str, ModelConfigSchema],
) -> tuple[str, ModelConfigSchema]:
    if isinstance(model_ref, str):
        model = models.get(model_ref)
        if model is None:
            raise ValueError(f"Job references unknown model '{model_ref}'.")
        return model_ref, model

    inline = ModelConfigSchema(**model_ref)
    if not inline.id:
        raise ValueError("Inline model definitions must include an 'id'.")
    existing = models.get(inline.id)
    if existing is not None and existing != inline:
        raise ValueError(f"Conflicting inline model definition for id '{inline.id}'.")
    models[inline.id] = inline
    return inline.id, inline


def _resolve_env_ids(
    env_ref: str,
    envs: dict[str, EnvironmentConfigSchema],
    matrix_index: dict[str, list[str]],
) -> list[str]:
    candidates: list[str] = []
    if env_ref in envs:
        candidates.append(env_ref)
    if env_ref in matrix_index:
        candidates.extend(matrix_index[env_ref])
    if not candidates:
        raise ValueError(f"Job references unknown environment '{env_ref}'.")
    # Preserve order while removing duplicates
    unique: list[str] = []
    seen: set[str] = set()
    for env_id in candidates:
        if env_id not in seen:
            unique.append(env_id)
            seen.add(env_id)
    return unique


def _resolve_env_override(model: ModelConfigSchema, env: EnvironmentConfigSchema) -> dict[str, Any] | None:
    """Resolve env-specific overrides from model config.

    Tries in order:
    1. env.id (exact match for the environment identifier)
    2. env.matrix_base_id (for matrix-expanded variants like 'medqa-seed-1')
    3. env.module (fallback for module-based lookup)

    Returns the override dict if found, None otherwise.
    """
    for key in (env.id, env.matrix_base_id, env.module):
        if key and key in model.env_overrides:
            return model.env_overrides[key]
    return None


def _compose_env_args(
    env: EnvironmentConfigSchema,
    model: ModelConfigSchema,
    job_env_args: dict[str, Any],
) -> dict[str, Any]:
    """Compose env_args up to job overrides (CLI is applied later)."""
    return merge_env_args(
        None,
        sources=[
            env.env_args,
            model.env_args,
            _resolve_env_override(model, env) or {},
            job_env_args,
        ],
    )


def _compose_sampling_args(
    model_sampling: dict[str, Any],
    job_sampling: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(model_sampling)
    merged.update(job_sampling)
    return merged


def _build_matrix_index(envs: Iterable[EnvironmentConfigSchema]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for env in envs:
        base_id = env.matrix_base_id
        if base_id:
            index.setdefault(base_id, []).append(env.id)
    return index


def _coerce_iterable(value: str | list[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def _build_job_id(
    *,
    model_id: str,
    env_id: str,
    job_name: str | None,
    env_overrides: dict[str, Any],
    sampling_overrides: dict[str, Any],
    used_ids: set[str],
) -> str:
    segments = [slugify(model_id), slugify(env_id)]
    if job_name:
        segments.append(slugify(job_name))
    base = "-".join(filter(None, segments)) or "job"
    job_id = base
    if job_id not in used_ids:
        return job_id

    payload = {
        "model_id": model_id,
        "env_id": env_id,
        "job_name": job_name,
        "env_overrides": env_overrides,
        "sampling_overrides": sampling_overrides,
    }
    fingerprint = compute_checksum(payload)[:10]
    job_id = f"{base}-{fingerprint}"
    suffix = 1
    while job_id in used_ids:
        suffix += 1
        job_id = f"{base}-{fingerprint}{suffix}"
    return job_id


__all__ = ["ResolvedJob", "build_jobs"]
