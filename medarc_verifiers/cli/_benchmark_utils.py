"""Helper utilities for resolving benchmark runs.

Job identifiers are deterministic and follow the pattern '<model>-<env>[-<name>][-<fingerprint>]'.
The optional fingerprint is appended only when multiple jobs share the same base identifiers.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml


@dataclass(slots=True)
class ModelParams:
    model: str | None = None
    headers: list[str] | dict[str, str] | None = None
    sampling_args: dict[str, Any] = field(default_factory=dict)
    max_tokens: int | None = None
    temperature: float | None = None
    api_key_var: str | None = None
    api_base_url: str | None = None
    endpoints_path: str | None = None
    env_args: dict[str, Any] = field(default_factory=dict)
    env_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(slots=True)
class ModelConfig:
    id: str
    params: ModelParams = field(default_factory=ModelParams)


@dataclass(slots=True)
class EnvironmentConfig:
    id: str
    module: str | None = None
    num_examples: int = 5
    rollouts_per_example: int = 1
    max_concurrent: int | None = None
    env_args: dict[str, Any] = field(default_factory=dict)
    interleave_scoring: bool = True
    state_columns: list[str] | None = None
    save_every: int | None = None
    print_results: bool = False
    verbose: bool | None = None


@dataclass(slots=True)
class JobConfig:
    model: str
    env: str
    env_args: dict[str, Any] = field(default_factory=dict)
    sampling_args: dict[str, Any] = field(default_factory=dict)
    name: str | None = None


@dataclass(slots=True)
class RunConfig:
    name: str
    run_id: str | None
    output_dir: Path
    env_dir_path: str
    endpoints_path: str | None
    default_api_key_var: str
    default_api_base_url: str
    models: dict[str, ModelConfig]
    envs: dict[str, EnvironmentConfig]
    jobs: list[JobConfig]


@dataclass(slots=True)
class ResolvedJob:
    job_id: str
    name: str
    model: ModelConfig
    env: EnvironmentConfig
    env_overrides: dict[str, Any] = field(default_factory=dict)
    sampling_overrides: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate.resolve()
    return (Path.cwd() / path).resolve()


def _ensure_mapping(value: Any, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MutableMapping):
        raise ValueError(f"{context} must be a mapping.")
    return dict(value)


def _resolve_optional_path(value: str | None, base_dir: Path) -> str | None:
    if value is None:
        return None
    return str(_resolve_path(value, base_dir))


def _coerce_optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # noqa: FBT003
        raise ValueError(f"{context} must be an integer.") from exc


def _coerce_optional_str(value: Any, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string.")
    return value


def _coerce_optional_float(value: Any, context: str) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # noqa: FBT003
        raise ValueError(f"{context} must be a number.") from exc


def _ensure_headers(value: Any, context: str) -> list[str] | dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, MutableMapping):
        return {str(k): str(v) for k, v in value.items()}
    raise ValueError(f"{context} must be a list or mapping.")


def _ensure_env_overrides(value: Any, context: str) -> dict[str, dict[str, Any]]:
    mapping = _ensure_mapping(value, context)
    overrides: dict[str, dict[str, Any]] = {}
    for key, override in mapping.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{context} keys must be non-empty strings.")
        overrides[key] = _ensure_mapping(override, f"{context} for environment '{key}'")
    return overrides


def _ensure_optional_str_list(value: Any, context: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list of strings.")
    return [str(item) for item in value]


def _sanitize_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


def _load_mapping_entries(source: Any, base_dir: Path, context: str) -> list[dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, Path):
        return _load_mapping_entries(str(source), base_dir, context)
    if isinstance(source, str):
        resolved_path = _resolve_path(source, base_dir)
        if not resolved_path.exists():
            raise FileNotFoundError(f"{context} path '{resolved_path}' does not exist.")
        if resolved_path.is_dir():
            entries: list[dict[str, Any]] = []
            for child in sorted(resolved_path.iterdir()):
                if child.is_file() and child.suffix.lower() in {".yaml", ".yml"}:
                    entries.extend(_load_mapping_entries(str(child), base_dir, context))
            return entries
        loaded = _load_yaml(resolved_path)
        return _load_mapping_entries(loaded, base_dir, context)
    if isinstance(source, MutableMapping):
        return [dict(source)]
    if isinstance(source, list):
        resolved: list[dict[str, Any]] = []
        for item in source:
            resolved.extend(_load_mapping_entries(item, base_dir, context))
        return resolved
    raise ValueError(f"{context} must be a list of mappings or paths to YAML files.")


def _load_job_entries(source: Any, base_dir: Path) -> list[dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, Path):
        return _load_job_entries(str(source), base_dir)
    if isinstance(source, str):
        resolved_path = _resolve_path(source, base_dir)
        if not resolved_path.exists():
            raise FileNotFoundError(f"jobs path '{resolved_path}' does not exist.")
        if resolved_path.is_dir():
            entries: list[dict[str, Any]] = []
            for child in sorted(resolved_path.iterdir()):
                if child.is_file() and child.suffix.lower() in {".yaml", ".yml"}:
                    entries.extend(_load_job_entries(str(child), base_dir))
            return entries
        loaded = _load_yaml(resolved_path)
        return _load_job_entries(loaded, base_dir)
    if isinstance(source, MutableMapping):
        return [dict(source)]
    if isinstance(source, list):
        resolved: list[dict[str, Any]] = []
        for item in source:
            resolved.extend(_load_job_entries(item, base_dir))
        return resolved
    raise ValueError("jobs must be a list of mappings or paths to YAML files.")


def load_run_config(config_path: Path) -> RunConfig:
    raw = _load_yaml(config_path)
    if not isinstance(raw, MutableMapping):
        raise ValueError("Run configuration must decode to a mapping.")
    return build_run_config(raw, config_path.parent)


def build_run_config(data: Mapping[str, Any], base_dir: Path) -> RunConfig:
    data = dict(data)
    models_entries = _load_mapping_entries(data.get("models"), base_dir, "models")
    if not models_entries:
        raise ValueError("Run configuration must define at least one model.")
    models: dict[str, ModelConfig] = {}
    for entry in models_entries:
        model_id = entry.get("id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("Model entries require a non-empty string 'id'.")
        params_mapping = _ensure_mapping(entry.get("params"), f"model '{model_id}' params")
        model_params = ModelParams(
            model=_coerce_optional_str(params_mapping.get("model"), f"model '{model_id}' params.model"),
            headers=_ensure_headers(params_mapping.get("headers"), f"model '{model_id}' params.headers"),
            sampling_args=_ensure_mapping(params_mapping.get("sampling_args"), f"model '{model_id}' sampling_args"),
            max_tokens=_coerce_optional_int(params_mapping.get("max_tokens"), f"model '{model_id}' params.max_tokens"),
            temperature=_coerce_optional_float(
                params_mapping.get("temperature"), f"model '{model_id}' params.temperature"
            ),
            api_key_var=_coerce_optional_str(
                params_mapping.get("api_key_var"), f"model '{model_id}' params.api_key_var"
            ),
            api_base_url=_coerce_optional_str(
                params_mapping.get("api_base_url"), f"model '{model_id}' params.api_base_url"
            ),
            endpoints_path=_resolve_optional_path(params_mapping.get("endpoints_path"), base_dir),
            env_args=_ensure_mapping(params_mapping.get("env_args"), f"model '{model_id}' env_args"),
            env_overrides=_ensure_env_overrides(
                params_mapping.get("env_overrides"),
                f"model '{model_id}' env_overrides",
            ),
        )
        models[model_id] = ModelConfig(id=model_id, params=model_params)

    env_entries = _load_mapping_entries(data.get("envs"), base_dir, "envs")
    if not env_entries:
        raise ValueError("Run configuration must define at least one environment.")
    envs: dict[str, EnvironmentConfig] = {}
    for entry in env_entries:
        env_id = entry.get("id")
        if not isinstance(env_id, str) or not env_id:
            raise ValueError("Environment entries require a non-empty string 'id'.")
        num_examples = int(entry.get("num_examples", 5))
        if num_examples < 1 and num_examples != -1:
            raise ValueError(f"environment '{env_id}' num_examples must be >= 1 or -1 for all examples.")
        rollouts_per_example = int(entry.get("rollouts_per_example", 1))
        if rollouts_per_example < 1:
            raise ValueError(f"environment '{env_id}' rollouts_per_example must be >= 1.")
        max_concurrent = _coerce_optional_int(entry.get("max_concurrent"), f"environment '{env_id}' max_concurrent")
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError(f"environment '{env_id}' max_concurrent must be >= 1 when provided.")
        interleave_val = entry.get("interleave_scoring", True)
        if not isinstance(interleave_val, bool):
            raise ValueError(f"environment '{env_id}' interleave_scoring must be a boolean.")
        print_results_val = entry.get("print_results", False)
        if not isinstance(print_results_val, bool):
            raise ValueError(f"environment '{env_id}' print_results must be a boolean.")
        verbose_val = entry.get("verbose")
        if verbose_val is not None and not isinstance(verbose_val, bool):
            raise ValueError(f"environment '{env_id}' verbose must be a boolean when provided.")
        env_config = EnvironmentConfig(
            id=env_id,
            module=_coerce_optional_str(entry.get("module"), f"environment '{env_id}' module"),
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=max_concurrent,
            env_args=_ensure_mapping(entry.get("env_args"), f"environment '{env_id}' env_args"),
            interleave_scoring=interleave_val,
            state_columns=_ensure_optional_str_list(
                entry.get("state_columns"), f"environment '{env_id}' state_columns"
            ),
            save_every=_coerce_optional_int(entry.get("save_every"), f"environment '{env_id}' save_every"),
            print_results=print_results_val,
            verbose=verbose_val,
        )
        envs[env_id] = env_config

    jobs_entries = _load_job_entries(data.get("jobs"), base_dir)
    jobs: list[JobConfig] = []
    if not jobs_entries:
        for model_id in models:
            for env_id in envs:
                jobs.append(JobConfig(model=model_id, env=env_id))
    else:
        for entry in jobs_entries:
            model_id = entry.get("model")
            if not isinstance(model_id, str) or not model_id:
                raise ValueError("Job entries require a non-empty 'model'.")
            job_name = _coerce_optional_str(entry.get("name"), f"job ({model_id}) name")
            env_values = entry.get("envs")
            if env_values is not None:
                if not isinstance(env_values, list):
                    raise ValueError(f"Job for model '{model_id}' must provide 'envs' as a list.")
                env_ids = [str(value) for value in env_values]
            else:
                env_value = entry.get("env")
                if env_value is None:
                    env_ids = list(envs.keys())
                else:
                    env_ids = [str(env_value)]
            if model_id not in models:
                raise ValueError(f"Job references unknown model '{model_id}'.")
            shared_env_args = _ensure_mapping(entry.get("env_args"), f"job ({model_id}) env_args")
            shared_sampling_args = _ensure_mapping(entry.get("sampling_args"), f"job ({model_id}) sampling_args")
            for env_id in env_ids:
                if env_id not in envs:
                    raise ValueError(f"Job references unknown environment '{env_id}'.")
                jobs.append(
                    JobConfig(
                        model=model_id,
                        env=env_id,
                        env_args=dict(shared_env_args),
                        sampling_args=dict(shared_sampling_args),
                        name=job_name,
                    )
                )

    output_dir_raw = data.get("output_dir", "runs")
    if not isinstance(output_dir_raw, str):
        raise ValueError("output_dir must be a string path.")
    output_dir_path = Path(output_dir_raw).expanduser()
    if not output_dir_path.is_absolute():
        output_dir_path = (base_dir / output_dir_path).resolve()

    env_dir_raw = data.get("env_dir_path", "./environments")
    env_dir_str = _coerce_optional_str(env_dir_raw, "env_dir_path") or "./environments"
    env_dir_path = _resolve_optional_path(env_dir_str, base_dir) or "./environments"

    endpoints_path = _resolve_optional_path(
        _coerce_optional_str(data.get("endpoints_path"), "endpoints_path"), base_dir
    )

    run_id = data.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise ValueError("run_id must be a string when provided.")

    name_value = data.get("name", "benchmark")
    if not isinstance(name_value, str):
        raise ValueError("name must be a string.")
    default_api_key = _coerce_optional_str(data.get("default_api_key_var"), "default_api_key_var") or "OPENAI_API_KEY"
    default_api_base_url = (
        _coerce_optional_str(
            data.get("default_api_base_url"),
            "default_api_base_url",
        )
        or "https://api.openai.com/v1"
    )

    run_config = RunConfig(
        name=name_value,
        run_id=run_id,
        output_dir=output_dir_path,
        env_dir_path=env_dir_path,
        endpoints_path=endpoints_path,
        default_api_key_var=default_api_key,
        default_api_base_url=default_api_base_url,
        models=models,
        envs=envs,
        jobs=jobs,
    )
    return run_config


def derive_run_id(name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = "".join(char.lower() if char.isalnum() else "-" for char in name).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)
    return f"{slug or 'run'}-{timestamp}"


def expand_jobs(config: RunConfig) -> list[ResolvedJob]:
    resolved: list[ResolvedJob] = []
    used_ids: set[str] = set()
    for job in config.jobs:
        model = config.models[job.model]
        env = config.envs[job.env]
        env_overrides = dict(job.env_args)
        sampling_overrides = dict(job.sampling_args)
        display_name = job.name or f"{model.id}-{env.id}"
        job_id = _build_job_id(
            model_id=model.id,
            env_id=env.id,
            job_name=job.name,
            env_overrides=env_overrides,
            sampling_overrides=sampling_overrides,
            used_ids=used_ids,
        )
        used_ids.add(job_id)
        resolved.append(
            ResolvedJob(
                job_id=job_id,
                model=model,
                env=env,
                env_overrides=env_overrides,
                sampling_overrides=sampling_overrides,
                name=display_name,
            )
        )
    return resolved


def _build_job_id(
    *,
    model_id: str,
    env_id: str,
    job_name: str | None,
    env_overrides: dict[str, Any],
    sampling_overrides: dict[str, Any],
    used_ids: set[str],
) -> str:
    segments = [_sanitize_slug(model_id), _sanitize_slug(env_id)]
    if job_name:
        segments.append(_sanitize_slug(job_name))
    base_slug = "-".join(part for part in segments if part)
    if not base_slug:
        base_slug = "job"
    slug = base_slug
    if slug in used_ids:
        payload = {
            "model_id": model_id,
            "env_id": env_id,
            "job_name": job_name,
            "env_overrides": env_overrides,
            "sampling_overrides": sampling_overrides,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:10]
        slug = f"{base_slug}-{fingerprint}"
        counter = 1
        while slug in used_ids:
            counter += 1
            slug = f"{base_slug}-{fingerprint}{counter}"
    return slug


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "EnvironmentConfig",
    "JobConfig",
    "ModelConfig",
    "ModelParams",
    "ResolvedJob",
    "RunConfig",
    "build_run_config",
    "derive_run_id",
    "expand_jobs",
    "load_run_config",
    "timestamp",
]
