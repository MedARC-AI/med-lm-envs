"""Configuration loader and task expansion for the vLLM orchestrator."""

from __future__ import annotations

import hashlib
import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from medarc_verifiers.cli.eval_identity import BASE_VARIANT_ID, plan_eval_paths, slug_component
from medarc_verifiers.cli.upstream_eval import EvalConfigOverrides, build_eval_identity_payload, load_toml_eval_configs

DEFAULT_CONTAINER_CONFIG: Mapping[str, Any] = {
    "image": "vllm/vllm-openai:latest",
    "container_port": 8000,
    "ipc_mode": "host",
}
DEFAULT_SLURM_CONFIG: Mapping[str, Any] = {
    "qos": "low",
    "nice": 500,
    "slurm_resume": True,
}
DEFAULT_PYXIS_CONFIG: Mapping[str, Any] = {
    "srun_extra_args": ["--overlap"],
}
DEFAULT_VLLM_SERVE_CONFIG: Mapping[str, Any] = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 32768,
    "async_scheduling": True,
    "enable_prefix_caching": True,
    "enable_auto_tool_choice": True,
}


class PlanConfig(BaseModel):
    """Schema for the orchestrator plan file."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str | None = None
    job_configs: list[Path] = Field(..., min_length=1)
    env_file: Path | None = None
    run_id: str | None = None
    output_dir: Path | None = None
    readiness_timeout_s: int | None = None
    prune_logs_on_success: bool = False
    eval_images_config: Path | None = None
    endpoints_path: Path | None = None


@dataclass(frozen=True)
class RegistrySnapshot:
    """Resolved registry metadata persisted into task bundles."""

    path: str | None
    checksum: str | None
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
    if resolved.suffix != ".toml":
        raise ValueError(f"Unsupported plan format: {resolved} (expected .toml)")
    payload = _load_toml_mapping(resolved)
    try:
        plan = PlanConfig(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid plan file: {resolved}") from exc
    return _resolve_plan_paths(plan, base_dir=resolved.parent)


def make_plan(
    *,
    job_configs: list[Path],
    base_dir: Path | None = None,
    name: str | None = None,
    eval_images_config: Path | None = None,
    endpoints_path: Path | None = None,
) -> PlanConfig:
    plan = PlanConfig(
        job_configs=job_configs,
        name=name,
        eval_images_config=eval_images_config,
        endpoints_path=endpoints_path,
    )
    return _resolve_plan_paths(plan, base_dir=(base_dir or Path.cwd()).resolve())


def load_job_config(path: Path) -> Mapping[str, Any]:
    """Load a public orchestrated eval config. Public job configs are TOML-only."""

    resolved = path.expanduser().resolve()
    if resolved.suffix != ".toml":
        raise ValueError(f"Unsupported job config format: {resolved} (expected .toml)")
    return _load_toml_mapping(resolved)


def load_endpoint_orchestration_registry(path: Path | None = None) -> Mapping[str, Any]:
    if path is None:
        raise ValueError(
            "Orchestration requires an endpoint registry containing [endpoint.orchestrate]; "
            "pass --endpoints-path or create configs/medmarks-endpoints.toml or configs/endpoints.toml."
        )
    resolved = path.expanduser().resolve()
    payload = _load_toml_mapping(resolved)
    registry = _extract_endpoint_orchestration_registry(payload)
    registry = _apply_orchestrate_defaults(registry)
    _validate_endpoint_orchestration_registry(registry, source=resolved)
    return registry


def load_eval_images_config(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    payload = _load_toml_mapping(resolved)
    _validate_eval_images_registry(payload, source=resolved)
    return payload


def expand_tasks(plan: PlanConfig, *, default_endpoints_path: Path | None = None) -> list[TaskSpec]:
    eval_images_path = plan.eval_images_config.expanduser().resolve() if plan.eval_images_config is not None else None
    eval_images_payload = load_eval_images_config(eval_images_path)
    tasks: list[TaskSpec] = []
    seen_identities: set[tuple[str, str, str]] = set()
    endpoint_registry_cache: dict[Path, Mapping[str, Any]] = {}

    for job_path in plan.job_configs:
        resolved_job_path = job_path.expanduser().resolve()
        job_payload = load_job_config(resolved_job_path)
        raw_eval_configs = load_toml_eval_configs(resolved_job_path)
        if not raw_eval_configs:
            raise ValueError(f"Job config {resolved_job_path} did not produce any evals.")
        raw_eval_configs = [
            _absolutize_eval_config_paths(raw, base_dir=resolved_job_path.parent) for raw in raw_eval_configs
        ]
        endpoints_path = _resolve_task_endpoints_path(
            plan,
            job_payload=job_payload,
            base_dir=resolved_job_path.parent,
            default_endpoints_path=default_endpoints_path,
        )
        orchestrate_payload = endpoint_registry_cache.get(endpoints_path)
        if orchestrate_payload is None:
            orchestrate_payload = load_endpoint_orchestration_registry(endpoints_path)
            endpoint_registry_cache[endpoints_path] = orchestrate_payload
        _reject_model_ablation(raw_eval_configs, source=resolved_job_path)
        overrides = EvalConfigOverrides(endpoints_path=endpoints_path)
        identity_payloads = [build_eval_identity_payload(raw, overrides=overrides) for raw in raw_eval_configs]
        path_plans = plan_eval_paths(identity_payloads, output_root=".")
        model_ids = {plan.identity.model_id for plan in path_plans}
        if len(model_ids) != 1:
            raise ValueError(
                f"Job config {resolved_job_path} resolves to multiple effective models {sorted(model_ids)}; "
                "orchestration requires exactly one model per task."
            )
        effective_model_id = next(iter(model_ids))
        endpoint_ids = {
            endpoint_id
            for payload in identity_payloads
            if isinstance((endpoint_id := payload.get("endpoint_id")), str) and endpoint_id
        }
        if not endpoint_ids:
            raise ValueError(
                f"Job config {resolved_job_path} must resolve to an endpoint_id; model-only orchestration configs "
                "are not supported."
            )
        if len(endpoint_ids) > 1:
            raise ValueError(
                f"Job config {resolved_job_path} resolves to multiple endpoint ids {sorted(endpoint_ids)}; "
                "orchestration requires exactly one model endpoint per task."
            )
        effective_endpoint_id = next(iter(endpoint_ids))
        matched_model = _match_endpoint_orchestration_entry(
            orchestrate_payload,
            effective_model_id,
            endpoint_id=effective_endpoint_id,
            source=endpoints_path,
        )
        model_id = _orchestrate_model_id(matched_model)
        model_key = slug_component(model_id)
        task_id = f"{resolved_job_path.stem}:{model_key}"
        eval_ids = _resolved_eval_ids(path_plans)
        env_ids = sorted({plan.identity.env_id for plan in path_plans})
        for plan_item in path_plans:
            identity = (
                model_id,
                plan_item.identity.env_id,
                plan_item.identity.variant_id,
            )
            if identity in seen_identities:
                raise ValueError(
                    "Duplicate orchestrated eval identity across tasks: "
                    f"model={identity[0]!r}, env_id={identity[1]!r}, variant_id={identity[2]!r}."
                )
            seen_identities.add(identity)
        selected_eval_images = _select_eval_images(eval_images_payload, eval_ids=eval_ids, env_ids=env_ids)
        orchestrate_snapshot = RegistrySnapshot(
            path=str(endpoints_path),
            checksum=_sha256_file(endpoints_path),
            matched=dict(matched_model),
        )
        eval_images_snapshot = RegistrySnapshot(
            path=str(eval_images_path) if eval_images_path is not None else None,
            checksum=_sha256_file(eval_images_path) if eval_images_path is not None else None,
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
                matched_model={
                    **dict(matched_model),
                    "effective_model_id": effective_model_id,
                },
                orchestrate_registry=orchestrate_snapshot,
                eval_images_registry=eval_images_snapshot,
            )
        )
    return tasks


def _resolve_plan_paths(plan: PlanConfig, *, base_dir: Path) -> PlanConfig:
    updates: dict[str, object] = {"job_configs": [_resolve_path(path, base_dir=base_dir) for path in plan.job_configs]}
    for field_name in ("env_file", "output_dir", "eval_images_config", "endpoints_path"):
        value = getattr(plan, field_name)
        if value is not None:
            updates[field_name] = _resolve_path(value, base_dir=base_dir)
    return plan.model_copy(update=updates)


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def resolve_default_endpoints_path(cwd: Path | None = None) -> Path | None:
    root = (cwd or Path.cwd()).expanduser().resolve()
    medmarks = (root / "configs" / "medmarks-endpoints.toml").resolve()
    if medmarks.exists():
        return medmarks
    endpoints = (root / "configs" / "endpoints.toml").resolve()
    if endpoints.exists():
        return endpoints
    return None


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


def _validate_endpoint_orchestration_registry(payload: Mapping[str, Any], *, source: Path) -> None:
    _reject_unknown_keys(payload, {"model"}, label=str(source))
    entries = payload.get("model")
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"Endpoint registry {source} must define one or more [[endpoint]] entries with [endpoint.orchestrate]."
        )
    endpoint_ids: set[str] = set()
    model_ids: set[str] = set()
    slugs: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"Each orchestratable endpoint entry in {source} must be a table.")
        _reject_unknown_keys(
            entry,
            {"endpoint_id", "model", "vllm", "container", "pyxis", "slurm"},
            label=f"{source} [[endpoint]].orchestrate",
        )
        model_id = _orchestrate_model_id(entry)
        endpoint_id = _orchestrate_endpoint_id(entry)
        if endpoint_id in endpoint_ids:
            raise ValueError(f"Duplicate endpoint_id in {source}: {endpoint_id}")
        endpoint_ids.add(endpoint_id)
        model_ids.add(model_id)
        slug = slug_component(model_id)
        previous = slugs.get(slug)
        if previous is not None and previous != model_id:
            raise ValueError(f"Models {previous!r} and {model_id!r} derive duplicate slug {slug!r} in {source}.")
        slugs[slug] = model_id
        vllm = _required_mapping(entry.get("vllm"), f"model {model_id} vllm")
        container = _required_mapping(entry.get("container"), f"model {model_id} container")
        if "gpus" not in vllm:
            raise ValueError(f"Endpoint {endpoint_id} in {source} must set [endpoint.orchestrate.vllm].gpus.")
        if not str(container.get("image", "")).strip():
            raise ValueError(f"Endpoint {endpoint_id} in {source} must set [endpoint.orchestrate.container].image.")
        _reject_unknown_keys(
            vllm,
            {"gpus", "tensor_parallel_size", "data_parallel_size", "require_contiguous_gpus", "memory_min_gb", "serve"},
            label=f"{source} {endpoint_id}.orchestrate.vllm",
        )
        _reject_unknown_keys(
            container,
            {"image", "container_port", "volumes", "ipc_mode", "env_file"},
            label=f"{source} {endpoint_id}.orchestrate.container",
        )
        if "serve" in vllm:
            _reject_unknown_keys(
                _required_mapping(vllm.get("serve"), f"endpoint {endpoint_id} vllm.serve"),
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
                label=f"{source} {endpoint_id}.orchestrate.vllm.serve",
            )
        if "pyxis" in entry:
            _reject_unknown_keys(
                _required_mapping(entry.get("pyxis"), f"endpoint {endpoint_id} pyxis"),
                {"srun_extra_args"},
                label=f"{source} {endpoint_id}.orchestrate.pyxis",
            )
        if "slurm" in entry:
            _reject_unknown_keys(
                _required_mapping(entry.get("slurm"), f"endpoint {endpoint_id} slurm"),
                {
                    "job_name",
                    "cpus_per_gpu",
                    "time",
                    "partition",
                    "account",
                    "qos",
                    "nice",
                    "mail_type",
                    "mail_user",
                    "slurm_resume",
                },
                label=f"{source} {endpoint_id}.orchestrate.slurm",
            )


def _extract_endpoint_orchestration_registry(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    entries = payload.get("endpoint", [])
    if not isinstance(entries, list):
        raise ValueError("Endpoint registry must use [[endpoint]] entries.")
    models: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"Each [[endpoint]] entry must be a table (index {index}).")
        orchestrate = entry.get("orchestrate")
        if orchestrate is None:
            continue
        orchestrate_mapping = dict(_required_mapping(orchestrate, f"[[endpoint]] index {index} orchestrate"))
        endpoint_id = _endpoint_entry_id(entry, source=f"[[endpoint]] index {index}")
        model_id = _endpoint_entry_model_id(entry, endpoint_id=endpoint_id)
        models.append({"endpoint_id": endpoint_id, "model": model_id, **orchestrate_mapping})
    return {"model": models}


def _apply_orchestrate_defaults(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = deepcopy(payload)
    models = normalized.get("model")
    if not isinstance(models, list):
        return normalized
    for entry in models:
        if not isinstance(entry, dict):
            continue
        for section_name, section_default in (
            ("container", DEFAULT_CONTAINER_CONFIG),
            ("pyxis", DEFAULT_PYXIS_CONFIG),
            ("slurm", DEFAULT_SLURM_CONFIG),
        ):
            section = entry.get(section_name)
            merged_section = dict(section_default)
            if isinstance(section, Mapping):
                merged_section.update(section)
            elif section is not None:
                entry[section_name] = section
                continue
            entry[section_name] = merged_section
        vllm = entry.get("vllm")
        if isinstance(vllm, dict):
            if vllm.get("tensor_parallel_size") is None and vllm.get("gpus") is not None:
                vllm["tensor_parallel_size"] = vllm["gpus"]
            serve = vllm.get("serve")
            merged_serve = dict(DEFAULT_VLLM_SERVE_CONFIG)
            if isinstance(serve, Mapping):
                merged_serve.update(serve)
            elif serve is not None:
                continue
            vllm["serve"] = merged_serve
    return normalized


def _validate_eval_images_registry(payload: Mapping[str, Any], *, source: Path) -> None:
    _reject_unknown_keys(payload, {"eval_image"}, label=str(source))
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
        selectors = (
            list(entry.get("evals", []) or [])
            + list(entry.get("envs", []) or [])
            + list(entry.get("env_ids", []) or [])
        )
        if not selectors:
            raise ValueError(
                f"Eval image {image_id} in {source} must define at least one selector: evals, envs, or env_ids."
            )
        for field_name in ("evals", "envs", "env_ids", "command", "srun_args"):
            if field_name in entry and (
                not isinstance(entry.get(field_name), list)
                or any(not isinstance(item, str) for item in entry.get(field_name, []))
            ):
                raise ValueError(f"Eval image {image_id} field {field_name} in {source} must be a list of strings.")
        runtime = entry.get("runtime")
        if runtime != "pyxis":
            raise ValueError(f"Eval image {image_id} in {source} must set runtime = 'pyxis'.")
        if not isinstance(entry.get("image"), str) or not str(entry.get("image")).strip():
            raise ValueError(f"Eval image {image_id} in {source} must set non-empty image.")
        if not entry.get("command"):
            raise ValueError(f"Eval image {image_id} in {source} must set non-empty command.")


def _orchestrate_model_id(entry: Mapping[str, Any]) -> str:
    model_id = entry.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("Each orchestratable endpoint entry must include non-empty model.")
    return model_id


def _orchestrate_endpoint_id(entry: Mapping[str, Any]) -> str:
    endpoint_id = entry.get("endpoint_id")
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        raise ValueError("Each orchestratable endpoint entry must include non-empty endpoint_id.")
    return endpoint_id


def _endpoint_entry_id(entry: Mapping[str, Any], *, source: str) -> str:
    endpoint_id = entry.get("endpoint_id")
    if not isinstance(endpoint_id, str) or not endpoint_id.strip():
        raise ValueError(f"{source} with orchestration config must include non-empty endpoint_id.")
    return endpoint_id


def _endpoint_entry_model_id(entry: Mapping[str, Any], *, endpoint_id: str) -> str:
    model_id = entry.get("model")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError(f"Endpoint {endpoint_id!r} with orchestration config must include non-empty model.")
    return model_id


def _match_endpoint_orchestration_entry(
    payload: Mapping[str, Any],
    model_id: str,
    *,
    endpoint_id: str,
    source: Path,
) -> Mapping[str, Any]:
    for entry in payload.get("model", []) or []:
        if _orchestrate_endpoint_id(entry) == endpoint_id:
            entry_model = _orchestrate_model_id(entry)
            if entry_model != model_id:
                raise ValueError(
                    f"Endpoint {endpoint_id!r} in {source} resolves to model {entry_model!r}, "
                    f"but eval config resolves to {model_id!r}."
                )
            return entry
    known = sorted(_orchestrate_endpoint_id(entry) for entry in payload.get("model", []) or [])
    raise ValueError(f"No [[endpoint]] entry in {source} matches endpoint_id {endpoint_id!r}. Known IDs: {known}.")


def _resolve_task_endpoints_path(
    plan: PlanConfig,
    *,
    job_payload: Mapping[str, Any],
    base_dir: Path,
    default_endpoints_path: Path | None = None,
) -> Path:
    if plan.endpoints_path is not None:
        return plan.endpoints_path
    raw_path = job_payload.get("endpoints_path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        return path.resolve()
    if default_endpoints_path is not None:
        return default_endpoints_path.expanduser().resolve()
    raise ValueError(
        "Orchestration requires an endpoint registry containing [endpoint.orchestrate]; "
        "set endpoints_path in the job config, pass --endpoints-path, or create "
        "configs/medmarks-endpoints.toml or configs/endpoints.toml."
    )


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


def _resolved_eval_ids(path_plans: list[Any]) -> list[str]:
    values: list[str] = []
    for plan_item in path_plans:
        env_id = plan_item.identity.env_id
        variant = plan_item.identity.variant_id
        values.append(env_id if variant == BASE_VARIANT_ID else f"{env_id}:{variant}")
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
    "load_endpoint_orchestration_registry",
    "load_job_config",
    "load_plan",
    "make_plan",
]
