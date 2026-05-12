"""Configuration loader and task expansion for the vLLM orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import tomllib

from omegaconf import OmegaConf
from pydantic import BaseModel, Field, ValidationError


_ORCHESTRATE_NON_MODEL_KEYS = {"restart", "vllm-container", "pyxis"}


class PlanConfig(BaseModel):
    """Schema for the orchestrator plan file."""

    name: str | None = None
    job_configs: list[Path] = Field(..., min_length=1)
    env_file: Path | None = None
    runtime: str | None = None
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
    uv_run: bool = True


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
    if resolved.suffix != ".toml":
        raise ValueError(f"Unsupported job config format: {resolved} (expected .toml)")
    return _load_mapping(resolved)


def expand_tasks(plan: PlanConfig) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for job_path in plan.job_configs:
        resolved_job_path = job_path.expanduser().resolve()
        job_cfg = load_job_config(resolved_job_path)
        model_key, model_entry = _extract_task_model(job_cfg, source=resolved_job_path)
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
    if path.suffix not in {".yaml", ".yml", ".json", ".toml"}:
        raise ValueError(f"Unsupported config format: {path} (expected .yaml/.yml/.json/.toml)")
    try:
        if path.suffix == ".toml":
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        else:
            data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - OmegaConf error types vary
        raise ConfigFormatError(f"Failed to load config: {path}") from exc
    if not isinstance(data, Mapping):
        raise ConfigFormatError(f"Config must be a mapping at top level: {path}")
    return data


def _extract_task_model(payload: Mapping[str, Any], *, source: Path) -> tuple[str, Mapping[str, Any]]:
    model_id = str(payload.get("model", "")).strip()
    if not model_id:
        raise ValueError(f"Job config {source} must define a top-level model.")
    orchestrate, table_name = _extract_orchestrate_root(payload, source=source)
    model_keys = [str(key) for key, value in orchestrate.items() if key not in _ORCHESTRATE_NON_MODEL_KEYS]
    if len(model_keys) != 1:
        raise ValueError(
            f"Job config {source} must define exactly one {table_name} model settings table; found {len(model_keys)}."
        )
    return model_keys[0], {"model": model_id}


def _extract_orchestrate_config(payload: Mapping[str, Any], *, model_key: str, source: Path) -> Mapping[str, Any]:
    orchestrate, table_name = _extract_orchestrate_root(payload, source=source)
    if "vllm-container" not in orchestrate:
        raise ValueError(f"Job config {source} must define {table_name}.vllm-container settings.")
    if model_key not in orchestrate:
        raise ValueError(f"Job config {source} must define {table_name}.{model_key} settings.")
    return orchestrate


def _extract_orchestrate_root(payload: Mapping[str, Any], *, source: Path) -> tuple[Mapping[str, Any], str]:
    medarc = payload.get("medarc")
    if medarc is not None:
        if not isinstance(medarc, Mapping):
            raise ValueError(f"Job config {source} medarc must be a mapping.")
        medarc_orchestrate = medarc.get("orchestrate")
        if medarc_orchestrate is not None:
            if not isinstance(medarc_orchestrate, Mapping):
                raise ValueError(f"Job config {source} medarc.orchestrate must be a mapping.")
            return medarc_orchestrate, "medarc.orchestrate"

    raise ValueError(f"Job config {source} must define a [medarc.orchestrate] mapping.")


__all__ = ["ConfigFormatError", "PlanConfig", "TaskSpec", "expand_tasks", "load_job_config", "load_plan"]
