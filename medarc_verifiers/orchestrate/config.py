"""Configuration loader and task expansion for the vLLM orchestrator."""

from __future__ import annotations

import hashlib
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from medarc_verifiers.cli.eval_identity import slug_component
from medarc_verifiers.cli.upstream_eval import EvalConfigOverrides, build_eval_identity_payload, load_toml_eval_configs


class PlanConfig(BaseModel):
    """Schema for the orchestrator plan file."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    orchestrate_config: Path | None = None
    eval_images_config: Path | None = None
    endpoints_path: Path | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    """Resolved registry metadata persisted into task bundles."""

    path: str | None
    checksum: str | None
    schema_version: int | None
    matched: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """Resolved task tuple for one TOML job config."""

    task_id: str
    job_config_path: Path
    model_key: str
    model_id: str
    orchestrate: Mapping[str, Any]
    slurm: Mapping[str, Any] = field(default_factory=dict)
    eval_images: list[Mapping[str, Any]] = field(default_factory=list)
    eval_ids: list[str] = field(default_factory=list)
    env_ids: list[str] = field(default_factory=list)
    endpoints_path: Path | None = None
    matched_model: Mapping[str, Any] = field(default_factory=dict)
    orchestrate_registry: RegistrySnapshot = field(default_factory=lambda: RegistrySnapshot(None, None, None))
    eval_images_registry: RegistrySnapshot = field(default_factory=lambda: RegistrySnapshot(None, None, None))


class ConfigFormatError(ValueError):
    """Raised when a configuration file cannot be interpreted as a mapping."""


def load_plan(path: Path) -> PlanConfig:
    resolved = path.expanduser().resolve()
    payload = _load_mapping_any(resolved)
    try:
        plan = PlanConfig(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid plan file: {resolved}") from exc
    _resolve_plan_paths(plan, base_dir=resolved.parent)
    return plan


def make_plan(
    *,
    job_configs: list[Path],
    base_dir: Path | None = None,
    name: str | None = None,
    orchestrate_config: Path | None = None,
    eval_images_config: Path | None = None,
    endpoints_path: Path | None = None,
) -> PlanConfig:
    plan = PlanConfig(
        job_configs=job_configs,
        name=name,
        orchestrate_config=orchestrate_config,
        eval_images_config=eval_images_config,
        endpoints_path=endpoints_path,
    )
    _resolve_plan_paths(plan, base_dir=(base_dir or Path.cwd()).resolve())
    return plan


def load_job_config(path: Path) -> Mapping[str, Any]:
    """Load a public orchestrated eval config. Public job configs are TOML-only."""

    resolved = path.expanduser().resolve()
    if resolved.suffix != ".toml":
        raise ValueError(f"Unsupported job config format: {resolved} (expected .toml)")
    return _load_toml_mapping(resolved)


def load_orchestrate_config(path: Path | None = None) -> Mapping[str, Any]:
    resolved = _default_orchestrate_config_path() if path is None else path.expanduser().resolve()
    payload = _load_toml_mapping(resolved)
    _validate_orchestrate_registry(payload, source=resolved)
    return payload


def load_eval_images_config(path: Path | None) -> Mapping[str, Any] | None:
    resolved = _resolve_eval_images_config_path(path)
    if resolved is None:
        return None
    payload = _load_toml_mapping(resolved)
    _validate_eval_images_registry(payload, source=resolved)
    return payload


def _resolve_eval_images_config_path(path: Path | None) -> Path | None:
    if path is not None:
        return path.expanduser().resolve()
    default = (Path("configs") / "eval_images.toml").resolve()
    return default if default.exists() else None


def expand_tasks(plan: PlanConfig) -> list[TaskSpec]:
    orchestrate_path = plan.orchestrate_config or _default_orchestrate_config_path()
    orchestrate_payload = load_orchestrate_config(orchestrate_path)
    eval_images_path = _resolve_eval_images_config_path(plan.eval_images_config)
    eval_images_payload = load_eval_images_config(eval_images_path)
    endpoints_path = plan.endpoints_path
    tasks: list[TaskSpec] = []
    seen_identities: set[tuple[str, str, str]] = set()

    for job_path in plan.job_configs:
        resolved_job_path = job_path.expanduser().resolve()
        load_job_config(resolved_job_path)
        raw_eval_configs = load_toml_eval_configs(resolved_job_path)
        if not raw_eval_configs:
            raise ValueError(f"Job config {resolved_job_path} did not produce any evals.")
        raw_eval_configs = [_absolutize_eval_config_paths(raw, base_dir=resolved_job_path.parent) for raw in raw_eval_configs]
        _reject_model_ablation(raw_eval_configs, source=resolved_job_path)
        overrides = EvalConfigOverrides(endpoints_path=endpoints_path) if endpoints_path is not None else None
        identity_payloads = [build_eval_identity_payload(raw, overrides=overrides) for raw in raw_eval_configs]
        model_ids = {str(payload["model"]) for payload in identity_payloads}
        if len(model_ids) != 1:
            raise ValueError(
                f"Job config {resolved_job_path} resolves to multiple effective models {sorted(model_ids)}; "
                "orchestration requires exactly one model per task."
            )
        effective_model_id = next(iter(model_ids))
        matched_model, matched_by = _match_model_entry(orchestrate_payload, effective_model_id, source=orchestrate_path)
        model_id = str(matched_model["id"])
        model_key = slug_component(model_id)
        task_id = f"{resolved_job_path.stem}:{model_key}"
        eval_ids = _resolved_eval_ids(identity_payloads)
        env_ids = sorted({str(payload["env_id"]) for payload in identity_payloads})
        for payload in identity_payloads:
            identity = (model_id, str(payload["env_id"]), str(payload.get("variant_id") or payload.get("name") or "base"))
            if identity in seen_identities:
                raise ValueError(
                    "Duplicate orchestrated eval identity across tasks: "
                    f"model={identity[0]!r}, env_id={identity[1]!r}, variant_id={identity[2]!r}."
                )
            seen_identities.add(identity)
        selected_eval_images = _select_eval_images(eval_images_payload, eval_ids=eval_ids, env_ids=env_ids)
        orchestrate_snapshot = RegistrySnapshot(
            path=str(orchestrate_path.expanduser().resolve()),
            checksum=_sha256_file(orchestrate_path.expanduser().resolve()),
            schema_version=int(orchestrate_payload["schema_version"]),
            matched=dict(matched_model),
        )
        eval_images_snapshot = RegistrySnapshot(
            path=str(eval_images_path) if eval_images_path is not None else None,
            checksum=_sha256_file(eval_images_path) if eval_images_path is not None else None,
            schema_version=int(eval_images_payload["schema_version"]) if eval_images_payload is not None else None,
            matched={"eval_image": selected_eval_images},
        )
        vllm = dict(_required_mapping(matched_model.get("vllm"), f"model {model_id} vllm"))
        container = dict(_required_mapping(matched_model.get("container"), f"model {model_id} container"))
        orchestrate = {
            "vllm": vllm,
            "container": container,
            "pyxis": dict(matched_model.get("pyxis") or {}),
        }
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_config_path=resolved_job_path,
                model_key=model_key,
                model_id=model_id,
                orchestrate=orchestrate,
                slurm=dict(matched_model.get("slurm") or {}),
                eval_images=selected_eval_images,
                eval_ids=eval_ids,
                env_ids=env_ids,
                endpoints_path=endpoints_path,
                matched_model={**dict(matched_model), "matched_by": matched_by, "effective_model_id": effective_model_id},
                orchestrate_registry=orchestrate_snapshot,
                eval_images_registry=eval_images_snapshot,
            )
        )
    return tasks


def _resolve_plan_paths(plan: PlanConfig, *, base_dir: Path) -> None:
    plan.job_configs = [_resolve_path(path, base_dir=base_dir) for path in plan.job_configs]
    for field_name in ("env_file", "output_dir", "orchestrate_config", "eval_images_config", "endpoints_path"):
        value = getattr(plan, field_name)
        if value is not None:
            setattr(plan, field_name, _resolve_path(value, base_dir=base_dir))


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _default_orchestrate_config_path() -> Path:
    return (Path("configs") / "orchestrate.toml").resolve()


def _load_mapping_any(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    if path.suffix == ".toml":
        return _load_toml_mapping(path)
    if path.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported config format: {path} (expected .yaml/.yml/.json/.toml)")
    try:
        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    except Exception as exc:  # pragma: no cover - OmegaConf error types vary
        raise ConfigFormatError(f"Failed to load config: {path}") from exc
    if not isinstance(data, Mapping):
        raise ConfigFormatError(f"Config must be a mapping at top level: {path}")
    return data


def _load_toml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - tomllib error types vary
        raise ConfigFormatError(f"Failed to load config: {path}") from exc
    if not isinstance(data, Mapping):
        raise ConfigFormatError(f"Config must be a mapping at top level: {path}")
    return data


def _validate_orchestrate_registry(payload: Mapping[str, Any], *, source: Path) -> None:
    _reject_unknown_keys(payload, {"schema_version", "model"}, label=str(source))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Orchestrate registry {source} must set schema_version = 1.")
    entries = payload.get("model")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Orchestrate registry {source} must define one or more [[model]] entries.")
    ids: set[str] = set()
    aliases: set[str] = set()
    slugs: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Each [[model]] entry in {source} must be a table.")
        _reject_unknown_keys(entry, {"id", "aliases", "vllm", "container", "pyxis", "slurm"}, label=f"{source} [[model]]")
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"Each [[model]] entry in {source} must include non-empty id.")
        if model_id in ids:
            raise ValueError(f"Duplicate model id in {source}: {model_id}")
        if model_id in aliases:
            raise ValueError(f"Model id collides with an alias in {source}: {model_id}")
        ids.add(model_id)
        alias_values = entry.get("aliases", []) or []
        if not isinstance(alias_values, list) or any(not isinstance(alias, str) or not alias for alias in alias_values):
            raise ValueError(f"Model {model_id} aliases in {source} must be a list of non-empty strings.")
        for alias in alias_values:
            if alias in aliases or alias in ids:
                raise ValueError(f"Duplicate or colliding model alias in {source}: {alias}")
            aliases.add(alias)
        slug = slug_component(model_id)
        previous = slugs.get(slug)
        if previous is not None:
            raise ValueError(f"Model ids {previous!r} and {model_id!r} derive duplicate slug {slug!r} in {source}.")
        slugs[slug] = model_id
        vllm = _required_mapping(entry.get("vllm"), f"model {model_id} vllm")
        container = _required_mapping(entry.get("container"), f"model {model_id} container")
        for required_key in ("gpus", "tensor_parallel_size"):
            if required_key not in vllm:
                raise ValueError(f"Model {model_id} in {source} must set [model.vllm].{required_key}.")
        if not str(container.get("image", "")).strip():
            raise ValueError(f"Model {model_id} in {source} must set [model.container].image.")
        _reject_unknown_keys(vllm, {"gpus", "tensor_parallel_size", "data_parallel_size", "require_contiguous_gpus", "memory_min_gb", "serve"}, label=f"{source} {model_id}.vllm")
        _reject_unknown_keys(container, {"image", "container_port", "volumes", "ipc_mode", "env_file"}, label=f"{source} {model_id}.container")
        if "serve" in vllm:
            _reject_unknown_keys(
                _required_mapping(vllm.get("serve"), f"model {model_id} vllm.serve"),
                {
                    "dtype",
                    "max_model_len",
                    "gpu_memory_utilization",
                    "max_num_seqs",
                    "max_num_batched_tokens",
                    "tokenizer_mode",
                    "config_format",
                    "load_format",
                    "reasoning_parser",
                    "reasoning_parser_plugin",
                    "tool_call_parser",
                    "tool_parser_plugin",
                    "mamba_ssm_cache_dtype",
                    "quantization",
                    "chat_template",
                    "async_scheduling",
                    "enable_prefix_caching",
                    "enable_chunked_prefill",
                    "trust_remote_code",
                    "enable_expert_parallel",
                    "enable_auto_tool_choice",
                    "language_model_only",
                    "limit_mm_per_prompt",
                },
                label=f"{source} {model_id}.vllm.serve",
            )
        if "pyxis" in entry:
            _reject_unknown_keys(_required_mapping(entry.get("pyxis"), f"model {model_id} pyxis"), {"srun_extra_args"}, label=f"{source} {model_id}.pyxis")
        if "slurm" in entry:
            _reject_unknown_keys(
                _required_mapping(entry.get("slurm"), f"model {model_id} slurm"),
                {"job_name", "cpus_per_gpu", "time", "partition", "account", "qos", "mail_type", "mail_user", "slurm_resume"},
                label=f"{source} {model_id}.slurm",
            )


def _validate_eval_images_registry(payload: Mapping[str, Any], *, source: Path) -> None:
    _reject_unknown_keys(payload, {"schema_version", "eval_image"}, label=str(source))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Eval image registry {source} must set schema_version = 1.")
    entries = payload.get("eval_image", []) or []
    if not isinstance(entries, list):
        raise ValueError(f"Eval image registry {source} must use [[eval_image]] entries.")
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Each [[eval_image]] entry in {source} must be a table.")
        _reject_unknown_keys(
            entry,
            {"id", "evals", "envs", "env_ids", "runtime", "image", "command", "srun_args", "env", "readiness"},
            label=f"{source} [[eval_image]]",
        )
        if "readiness" in entry:
            _reject_unknown_keys(
                _required_mapping(entry.get("readiness"), "eval_image readiness"),
                {"enabled", "url", "timeout_s", "interval_s"},
                label=f"{source} [[eval_image]].readiness",
            )
        image_id = entry.get("id")
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError(f"Each [[eval_image]] entry in {source} must include non-empty id.")
        if image_id in ids:
            raise ValueError(f"Duplicate eval_image id in {source}: {image_id}")
        ids.add(image_id)
        selectors = list(entry.get("evals", []) or []) + list(entry.get("envs", []) or []) + list(entry.get("env_ids", []) or [])
        if not selectors:
            raise ValueError(f"Eval image {image_id} in {source} must define at least one selector: evals, envs, or env_ids.")
        for field_name in ("evals", "envs", "env_ids", "command", "srun_args"):
            if field_name in entry and (not isinstance(entry.get(field_name), list) or any(not isinstance(item, str) for item in entry.get(field_name, []))):
                raise ValueError(f"Eval image {image_id} field {field_name} in {source} must be a list of strings.")
        runtime = entry.get("runtime")
        if runtime != "pyxis":
            raise ValueError(f"Eval image {image_id} in {source} must set runtime = 'pyxis'.")
        if not isinstance(entry.get("image"), str) or not str(entry.get("image")).strip():
            raise ValueError(f"Eval image {image_id} in {source} must set non-empty image.")
        if not entry.get("command"):
            raise ValueError(f"Eval image {image_id} in {source} must set non-empty command.")


def _match_model_entry(payload: Mapping[str, Any], model_id: str, *, source: Path) -> tuple[Mapping[str, Any], str]:
    matches: list[tuple[Mapping[str, Any], str]] = []
    for entry in payload.get("model", []) or []:
        entry_id = str(entry["id"])
        if model_id == entry_id:
            matches.append((entry, "id"))
        elif model_id in {str(alias) for alias in entry.get("aliases", []) or []}:
            matches.append((entry, "alias"))
    if not matches:
        raise ValueError(f"No [[model]] entry in {source} matches effective model {model_id!r}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple [[model]] entries in {source} match effective model {model_id!r}.")
    return matches[0]


def _absolutize_eval_config_paths(raw: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    normalized = dict(raw)
    for key in ("endpoints_path", "env_dir_path"):
        value = normalized.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            normalized[key] = str((base_dir / path).resolve())
    return normalized


def _reject_model_ablation(raw_eval_configs: list[Mapping[str, Any]], *, source: Path) -> None:
    raw_models = {str(raw["model"]) for raw in raw_eval_configs if raw.get("model") is not None}
    raw_endpoints = {str(raw["endpoint_id"]) for raw in raw_eval_configs if raw.get("endpoint_id") is not None}
    if len(raw_models) > 1:
        raise ValueError(f"Job config {source} ablates model, which is not supported by orchestration.")
    if len(raw_endpoints) > 1:
        raise ValueError(f"Job config {source} ablates endpoint_id, which is not supported by orchestration.")
    if raw_models and raw_endpoints:
        raise ValueError(f"Job config {source} mixes model and endpoint_id across evals, which is not supported.")


def _resolved_eval_ids(identity_payloads: list[Mapping[str, Any]]) -> list[str]:
    values: list[str] = []
    for payload in identity_payloads:
        env_id = str(payload["env_id"])
        variant = payload.get("variant_id") or payload.get("name")
        values.append(f"{env_id}:{variant}" if variant else env_id)
    return sorted(set(values))


def _select_eval_images(
    payload: Mapping[str, Any] | None,
    *,
    eval_ids: list[str],
    env_ids: list[str],
) -> list[Mapping[str, Any]]:
    if payload is None:
        return []
    selected: list[Mapping[str, Any]] = []
    eval_set = set(eval_ids)
    env_set = set(env_ids)
    for entry in payload.get("eval_image", []) or []:
        entry_evals = {str(item) for item in entry.get("evals", []) or []}
        entry_envs = {str(item) for item in (entry.get("envs", []) or []) + (entry.get("env_ids", []) or [])}
        if (entry_evals and entry_evals & eval_set) or (entry_envs and entry_envs & env_set):
            selected.append(dict(entry))
    return selected


def _required_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str], *, label: str) -> None:
    unknown = sorted(str(key) for key in payload.keys() if str(key) not in allowed)
    if unknown:
        raise ValueError(f"Unknown fields in {label}: {unknown}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return deepcopy(dict(value))


__all__ = [
    "ConfigFormatError",
    "PlanConfig",
    "RegistrySnapshot",
    "TaskSpec",
    "expand_tasks",
    "load_eval_images_config",
    "load_job_config",
    "load_orchestrate_config",
    "load_plan",
    "make_plan",
]
