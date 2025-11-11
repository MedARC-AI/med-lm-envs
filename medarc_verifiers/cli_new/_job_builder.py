"""Resolve validated run configurations into executable job definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ._schemas import EnvironmentConfigSchema, ModelConfigSchema, RunConfigSchema
from .utils.shared import compute_checksum, merge_dicts_with_precedence, slugify


@dataclass(slots=True)
class ResolvedJob:
    """Executable job produced from a run configuration."""

    job_id: str
    name: str
    model: ModelConfigSchema
    env: EnvironmentConfigSchema
    env_args: dict[str, Any]
    sampling_args: dict[str, Any]


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
    """Compose env_args following the precedence chain (lowest to highest):

    Precedence order (later sources override earlier ones):
    1. Environment config env_args (base defaults from env YAML)
    2. Model config env_args (global model settings applied to all envs)
    3. Model config env_overrides[env_id] (env-specific model settings)
    4. Job config env_args (per-job overrides from jobs section)
    5. CLI --env-arg/--env-args (applied later in executor, NOT here)

    Example:
        env.env_args = {"shuffle": False, "seed": 42}
        model.env_args = {"seed": 123}  # Overrides env default
        model.env_overrides["medqa"] = {"shuffle": True}  # Env-specific
        job_env_args = {"workers": 4}  # Job-specific addition
        → Result: {"shuffle": True, "seed": 123, "workers": 4}

    Note: CLI overrides (layer 5) are applied in _job_executor._build_eval_config,
    not here, to enable --restart functionality and manifest job reuse.
    """
    # Merge layers 1-4 with later ones taking precedence
    return merge_dicts_with_precedence(
        env.env_args,  # Layer 1: Base env defaults
        model.env_args,  # Layer 2: Global model settings
        _resolve_env_override(model, env),  # Layer 3: Env-specific overrides
        job_env_args,  # Layer 4: Job-level overrides
        # Layer 5: CLI overrides (applied later in executor)
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
