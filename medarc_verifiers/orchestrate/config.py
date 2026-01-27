"""Configuration loader and task expansion for the vLLM orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationError


class PlanConfig(BaseModel):
    """Schema for the orchestrator plan file."""

    name: str | None = None
    job_configs: list[Path] = Field(..., min_length=1)
    env_file: Path | None = None
    gpu_range: str | None = None
    port_range: str | None = None
    run_id: str | None = None
    output_dir: Path | None = None
    max_parallel: int | None = None
    readiness_timeout_s: int | None = None
    resume: bool = False
    rerun_failed: bool = False
    kill_orphans: bool = False
    prune_logs_on_success: bool = False


@dataclass(frozen=True)
class TaskSpec:
    """Resolved task tuple for one job config + model key."""

    task_id: str
    job_config_path: Path
    model_key: str
    model_id: str
    orchestrate: Mapping[str, Any]


class ConfigFormatError(ValueError):
    """Raised when a configuration file cannot be interpreted as a mapping."""


def load_plan(path: Path) -> PlanConfig:
    resolved = path.expanduser().resolve()
    payload = _load_mapping(resolved)
    try:
        plan = PlanConfig(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid plan file: {resolved}") from exc
    base_dir = resolved.parent
    resolved_job_configs: list[Path] = []
    for cfg in plan.job_configs:
        cfg_path = Path(cfg).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = base_dir / cfg_path
        resolved_job_configs.append(cfg_path.resolve())
    plan.job_configs = resolved_job_configs
    if plan.env_file is not None:
        env_file = Path(plan.env_file).expanduser()
        if not env_file.is_absolute():
            env_file = base_dir / env_file
        plan.env_file = env_file.resolve()
    if plan.output_dir is not None:
        output_dir = Path(plan.output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
        plan.output_dir = output_dir.resolve()
    return plan


def load_job_config(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    return _load_mapping(resolved)


def expand_tasks(plan: PlanConfig) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for job_path in plan.job_configs:
        resolved_job_path = job_path.expanduser().resolve()
        job_cfg = load_job_config(resolved_job_path)
        model_key, model_entry = _extract_single_model(job_cfg, source=resolved_job_path)
        orchestrate_cfg = _extract_orchestrate_config(job_cfg, model_key=model_key, source=resolved_job_path)
        model_id = str(model_entry.get("model", "")).strip()
        if not model_id:
            raise ValueError(f"Job config {resolved_job_path} is missing models.{model_key}.model.")
        task_id = f"{resolved_job_path.stem}:{model_key}"
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_config_path=resolved_job_path,
                model_key=model_key,
                model_id=model_id,
                orchestrate=orchestrate_cfg,
            )
        )
    return tasks


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported config format: {path} (expected .yaml/.yml/.json)")
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - OmegaConf error types vary
        raise ConfigFormatError(f"Failed to load config: {path}") from exc
    if not isinstance(data, Mapping):
        raise ConfigFormatError(f"Config must be a mapping at top level: {path}")
    return data


def _extract_single_model(payload: Mapping[str, Any], *, source: Path) -> tuple[str, Mapping[str, Any]]:
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ValueError(f"Job config {source} must define a models mapping.")
    keys = list(models.keys())
    if len(keys) != 1:
        raise ValueError(f"Job config {source} must define exactly one model; found {len(keys)}.")
    model_key = str(keys[0])
    model_entry = models.get(model_key)
    if not isinstance(model_entry, Mapping):
        raise ValueError(f"Job config {source} models.{model_key} must be a mapping.")
    return model_key, model_entry


def _extract_orchestrate_config(
    payload: Mapping[str, Any], *, model_key: str, source: Path
) -> Mapping[str, Any]:
    orchestrate = payload.get("orchestrate")
    if not isinstance(orchestrate, Mapping):
        raise ValueError(f"Job config {source} must define a top-level orchestrate mapping.")
    if "vllm-docker" not in orchestrate:
        raise ValueError(f"Job config {source} must define orchestrate.vllm-docker settings.")
    if model_key not in orchestrate:
        raise ValueError(f"Job config {source} must define orchestrate.{model_key} settings.")
    return orchestrate


__all__ = ["ConfigFormatError", "PlanConfig", "TaskSpec", "expand_tasks", "load_job_config", "load_plan"]
