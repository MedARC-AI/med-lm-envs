"""Config loader utilities bridging OmegaConf YAML files and Pydantic schemas."""

from __future__ import annotations

import logging
from itertools import product
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any

from omegaconf import OmegaConf

from ._schemas import EnvironmentConfigSchema, RunConfigSchema
from .utils.endpoint_utils import EnvMetadataCache, load_env_metadata
from .utils.env_args import validate_env_arg_values

logger = logging.getLogger(__name__)


class ConfigFormatError(ValueError):
    """Raised when a configuration file cannot be interpreted as a mapping."""


def _load_raw_config(path: Path) -> Any:
    """Load and resolve an OmegaConf configuration file."""
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def _convert_legacy_root_list(entries: list[Any], *, base_dir: Path) -> dict[str, Any]:
    """Support legacy list-based roots used by early configs."""
    if not entries:
        raise ConfigFormatError("Legacy list-based configs cannot be empty.")

    merged: dict[str, Any] = {}
    for index, item in enumerate(entries):
        if not isinstance(item, Mapping):
            raise ConfigFormatError(
                f"Legacy config entry {index} must be a mapping, got {type(item).__name__}."
            )
        merged.update(item)

    expected_keys = {"name", "models", "envs", "jobs"}
    if not expected_keys.intersection(merged):
        raise ConfigFormatError("Legacy list root must define models/envs/jobs entries.")

    return merged


def load_run_config(path: str | Path) -> RunConfigSchema:
    """Load a run configuration file into the top-level schema."""
    resolved_path = Path(path).expanduser().resolve()
    data = _load_raw_config(resolved_path)

    if isinstance(data, list):
        data = _convert_legacy_root_list(data, base_dir=resolved_path.parent)

    if not isinstance(data, dict):
        msg = f"Configuration root must be a mapping, got {type(data).__name__}."
        raise ConfigFormatError(msg)

    data = _apply_legacy_adapter(data, base_dir=resolved_path.parent)

    run_config = RunConfigSchema(**data)
    expanded_envs = _expand_env_matrices(run_config.envs)
    _validate_env_args(expanded_envs.values())
    return run_config.model_copy(update={"envs": expanded_envs})


def _expand_env_matrices(envs: dict[str, EnvironmentConfigSchema]) -> dict[str, EnvironmentConfigSchema]:
    scalar_fields = {
        name
        for name in EnvironmentConfigSchema.model_fields
        if name
        not in {
            "id",
            "module",
            "env_args",
            "matrix",
            "matrix_exclude",
            "matrix_id_format",
            "matrix_base_id",
        }
    }
    expanded: dict[str, EnvironmentConfigSchema] = {}
    for env_id, env in envs.items():
        env_with_id = env if env.id else env.model_copy(update={"id": env_id})
        for variant in _expand_single_environment(env_with_id, scalar_fields):
            if variant.id in expanded:
                raise ValueError(f"environment '{variant.id}' defined multiple times after expansion.")
            expanded[variant.id] = variant
    return expanded


def _expand_single_environment(
    env: EnvironmentConfigSchema,
    scalar_fields: Iterable[str],
) -> list[EnvironmentConfigSchema]:
    if not env.matrix:
        return [
            env.model_copy(
                update={
                    "env_args": dict(env.env_args),
                    "matrix": None,
                    "matrix_exclude": None,
                    "matrix_id_format": None,
                }
            )
        ]

    matrix = env.matrix
    base_id = env.id
    if not base_id:
        raise ValueError("environment entries must specify an id.")
    reserved_keys = {
        "id",
        "module",
        "env_args",
        "matrix",
        "matrix_exclude",
        "matrix_id_format",
        "matrix_base_id",
        "state_columns",
    }
    for key in matrix:
        if key in reserved_keys:
            raise ValueError(f"environment '{base_id}' matrix cannot vary '{key}'.")

    exclude_patterns = env.matrix_exclude or []
    for pattern in exclude_patterns:
        invalid_keys = set(pattern) - set(matrix)
        if invalid_keys:
            invalid = ", ".join(sorted(invalid_keys))
            raise ValueError(
                f"environment '{base_id}' matrix_exclude entry references unknown keys: {invalid}."
            )

    matrix_keys = list(matrix.keys())
    matrix_values = [matrix[key] for key in matrix_keys]
    variants: list[EnvironmentConfigSchema] = []
    seen_ids: set[str] = set()

    base_env_args = dict(env.env_args)
    module_name = env.module or env.id

    combos: Iterable[tuple[Any, ...]]
    if matrix_keys:
        combos = product(*matrix_values)
    else:
        combos = [()]

    for combo_values in combos:
        combo = dict(zip(matrix_keys, combo_values))
        if any(_matches_matrix_pattern(combo, pattern) for pattern in exclude_patterns):
            continue

        env_args = dict(base_env_args)
        updates: dict[str, Any] = {}
        for key, value in combo.items():
            if value is None:
                continue
            if key in scalar_fields:
                updates[key] = value
            else:
                env_args[key] = value

        variant_id = _build_matrix_variant_id(base_id, combo, env.matrix_id_format)
        if variant_id in seen_ids:
            raise ValueError(f"environment '{base_id}' matrix generated duplicate id '{variant_id}'.")
        seen_ids.add(variant_id)

        variant_data = env.model_dump()
        variant_data.update(updates)
        variant_data["id"] = variant_id
        variant_data["env_args"] = env_args
        variant_data["module"] = module_name
        variant_data["matrix"] = None
        variant_data["matrix_exclude"] = None
        variant_data["matrix_id_format"] = None
        variant_data["matrix_base_id"] = base_id

        variants.append(EnvironmentConfigSchema(**variant_data))

    if not variants:
        raise ValueError(f"environment '{base_id}' matrix produced no variants after exclusions.")

    return variants


def _apply_legacy_adapter(data: dict[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Normalize legacy configuration shapes before schema validation."""

    adapted = dict(data)

    if "models" in adapted:
        adapted["models"] = _normalize_models_field(adapted["models"], base_dir=base_dir)

    if "envs" in adapted:
        adapted["envs"] = _normalize_envs_field(adapted["envs"], base_dir=base_dir)

    if "jobs" in adapted:
        adapted["jobs"] = _normalize_jobs_field(adapted["jobs"], base_dir=base_dir)

    return adapted


def _normalize_models_field(value: Any, *, base_dir: Path) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping) and all(isinstance(v, Mapping) for v in value.values()):
        normalized: dict[str, Any] = {}
        for key, entry in value.items():
            ingested = _ingest_model_entries(entry, base_dir=base_dir, context=f"models['{key}']", default_id=str(key))
            _ensure_no_duplicates(normalized, ingested, entry_type="model")
        return normalized

    entries = _collect_model_entries(value, base_dir=base_dir, context="models")
    normalized: dict[str, Any] = {}
    for entry in entries:
        adapted = _adapt_model_entry(entry)
        if not isinstance(adapted, dict):
            raise ValueError("Model entries must be mappings.")
        model_id = adapted.get("id")
        if not model_id:
            raise ValueError("Legacy model entries must include an 'id'.")
        if model_id in normalized:
            raise ValueError(f"Duplicate model id '{model_id}' in configuration.")
        normalized[str(model_id)] = adapted
    return normalized


def _normalize_envs_field(value: Any, *, base_dir: Path) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping) and all(isinstance(v, Mapping) for v in value.values()):
        normalized: dict[str, Any] = {}
        for key, entry in value.items():
            ingested = _ingest_env_entries(entry, base_dir=base_dir, context=f"envs['{key}']", default_id=str(key))
            _ensure_no_duplicates(normalized, ingested, entry_type="environment")
        return normalized

    entries = _collect_env_entries(value, base_dir=base_dir, context="envs")
    normalized: dict[str, Any] = {}
    for entry in entries:
        adapted = _adapt_env_entry(entry)
        if not isinstance(adapted, dict):
            raise ValueError("Environment entries must be mappings.")
        env_id = adapted.get("id")
        if not env_id:
            raise ValueError("Legacy environment entries must include an 'id'.")
        if env_id in normalized:
            raise ValueError(f"Duplicate environment id '{env_id}' in configuration.")
        normalized[str(env_id)] = adapted
    return normalized


def _normalize_jobs_field(value: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    entries = _collect_job_entries(value, base_dir=base_dir)
    return [_adapt_job_entry(entry) for entry in entries]


def _ensure_no_duplicates(
    target: dict[str, Any],
    incoming: dict[str, Any],
    *,
    entry_type: str,
) -> None:
    for key, value in incoming.items():
        if key in target:
            raise ValueError(f"Duplicate {entry_type} id '{key}' in configuration.")
        target[key] = value


def _ingest_model_entries(
    entry: Any,
    *,
    base_dir: Path,
    context: str,
    default_id: str | None = None,
) -> dict[str, Any]:
    accumulated: dict[str, Any] = {}
    if isinstance(entry, Mapping):
        adapted = _adapt_model_entry(dict(entry))
        if isinstance(adapted, dict) and default_id and not adapted.get("id"):
            adapted["id"] = default_id
        model_id = adapted.get("id")
        if not model_id:
            raise ValueError(f"{context} entries must include an 'id'.")
        accumulated[str(model_id)] = adapted
        return accumulated

    sub_entries = _collect_model_entries(entry, base_dir=base_dir, context=context)
    for sub_entry in sub_entries:
        adapted = _adapt_model_entry(sub_entry)
        if isinstance(adapted, dict) and default_id and not adapted.get("id"):
            adapted["id"] = default_id
        model_id = adapted.get("id")
        if not model_id:
            raise ValueError(f"{context} entries must include an 'id'.")
        accumulated[str(model_id)] = adapted
    return accumulated


def _ingest_env_entries(
    entry: Any,
    *,
    base_dir: Path,
    context: str,
    default_id: str | None = None,
) -> dict[str, Any]:
    accumulated: dict[str, Any] = {}
    if isinstance(entry, Mapping):
        adapted = _adapt_env_entry(dict(entry))
        if isinstance(adapted, dict) and default_id and not adapted.get("id"):
            adapted["id"] = default_id
        env_id = adapted.get("id")
        if not env_id:
            raise ValueError(f"{context} entries must include an 'id'.")
        accumulated[str(env_id)] = adapted
        return accumulated

    sub_entries = _collect_env_entries(entry, base_dir=base_dir, context=context)
    for sub_entry in sub_entries:
        adapted = _adapt_env_entry(sub_entry)
        if isinstance(adapted, dict) and default_id and not adapted.get("id"):
            adapted["id"] = default_id
        env_id = adapted.get("id")
        if not env_id:
            raise ValueError(f"{context} entries must include an 'id'.")
        accumulated[str(env_id)] = adapted
    return accumulated


def _collect_model_entries(source: Any, *, base_dir: Path, context: str) -> list[dict[str, Any]]:
    return _collect_entries(source, base_dir=base_dir, context=context, entry_description="models")


def _collect_env_entries(source: Any, *, base_dir: Path, context: str) -> list[dict[str, Any]]:
    return _collect_entries(source, base_dir=base_dir, context=context, entry_description="envs")


def _collect_job_entries(source: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    return _collect_entries(source, base_dir=base_dir, context="jobs", entry_description="jobs")


def _collect_entries(
    source: Any,
    *,
    base_dir: Path,
    context: str,
    entry_description: str,
) -> list[dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, Mapping):
        return [dict(source)]
    if isinstance(source, (str, Path)):
        return _collect_entries_from_path(source, base_dir=base_dir, context=context, entry_description=entry_description)
    if isinstance(source, list):
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(source):
            item_context = f"{context}[{index}]"
            if isinstance(item, Mapping):
                entries.append(dict(item))
            elif isinstance(item, (str, Path)):
                entries.extend(
                    _collect_entries_from_path(
                        item,
                        base_dir=base_dir,
                        context=item_context,
                        entry_description=entry_description,
                    )
                )
            else:
                raise ValueError(f"{item_context} must be a mapping or path.")
        return entries
    raise ValueError(f"{context} must be provided as a mapping, list, or path.")


def _collect_entries_from_path(
    source: str | Path,
    *,
    base_dir: Path,
    context: str,
    entry_description: str,
) -> list[dict[str, Any]]:
    path = _resolve_include_path(source, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"{context} path '{path}' does not exist.")
    if path.is_dir():
        if entry_description not in {"envs", "jobs"}:
            msg = (
                f"{context} path '{path}' must be a file. Directory includes are only supported"
                " for envs and jobs."
            )
            raise ValueError(msg)
        entries: list[dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.is_file() and child.suffix.lower() in {".yaml", ".yml"}:
                entries.extend(
                    _collect_entries_from_path(
                        child,
                        base_dir=child.parent,
                        context=f"{context}/{child.name}",
                        entry_description=entry_description,
                    )
                )
        return entries

    loaded = _load_raw_config(path)
    if isinstance(loaded, Mapping):
        if not loaded:
            return []
        if not all(isinstance(v, Mapping) for v in loaded.values()):
            msg = (
                f"{context} included {entry_description} must be a mapping of id→mapping or a"
                " list of mappings."
            )
            raise ValueError(msg)
        entries: list[dict[str, Any]] = []
        for key, value in loaded.items():
            entry = dict(value)
            entry.setdefault("id", str(key))
            entries.append(entry)
        return entries
    if isinstance(loaded, list):
        entries: list[dict[str, Any]] = []
        for index, item in enumerate(loaded):
            if not isinstance(item, Mapping):
                raise ValueError(f"{context}[{index}] in included {entry_description} must be a mapping.")
            entries.append(dict(item))
        return entries
    if loaded is None:
        return []
    raise ValueError(
        f"{context} included {entry_description} must be a mapping of id→mapping or a list of mappings."
    )


def _resolve_include_path(source: str | Path, *, base_dir: Path) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _adapt_model_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry

    normalized = dict(entry)
    params = normalized.pop("params", None)
    if isinstance(params, dict):
        merged = {**params, **normalized}
        merged.setdefault("id", normalized.get("id"))
        normalized = merged

    env_args = normalized.get("env_args")
    if env_args is None:
        normalized["env_args"] = {}
    elif isinstance(env_args, dict):
        normalized["env_args"] = dict(env_args)
    else:
        raise ValueError("model env_args must be a mapping when provided.")

    env_overrides = normalized.get("env_overrides")
    if env_overrides is None:
        normalized["env_overrides"] = {}
    elif isinstance(env_overrides, dict):
        normalized["env_overrides"] = {str(key): dict(value) for key, value in env_overrides.items()}
    else:
        raise ValueError("model env_overrides must be a mapping when provided.")

    return normalized


def _adapt_env_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry

    normalized = dict(entry)
    env_args = normalized.get("env_args")
    if env_args is None:
        normalized["env_args"] = {}
    elif isinstance(env_args, dict):
        normalized["env_args"] = dict(env_args)
    else:
        raise ValueError("environment env_args must be a mapping when provided.")

    return normalized


def _adapt_job_entry(entry: Any) -> Any:
    if not isinstance(entry, dict):
        return entry

    normalized = dict(entry)
    for key in ("env_args", "sampling_args"):
        value = normalized.get(key)
        if value is None:
            normalized[key] = {}
        elif isinstance(value, dict):
            normalized[key] = dict(value)
        else:
            raise ValueError(f"job {key} must be a mapping when provided.")

    return normalized


def _build_matrix_variant_id(
    base_id: str,
    combo: dict[str, Any],
    id_format: str | None,
) -> str:
    format_values = {key: _format_matrix_value(value) for key, value in combo.items()}
    format_values["base"] = base_id

    if id_format:
        try:
            variant_id = id_format.format(**format_values)
        except KeyError as exc:  # noqa: F841
            missing = exc.args[0]
            raise ValueError(
                f"environment '{base_id}' matrix_id_format references unknown key '{missing}'."
            ) from exc
    else:
        suffix_parts = [
            f"{key}-{_format_matrix_value(value)}"
            for key, value in combo.items()
            if value is not None
        ]
        variant_id = base_id if not suffix_parts else f"{base_id}-{'-'.join(suffix_parts)}"

    if not isinstance(variant_id, str) or not variant_id:
        raise ValueError(f"environment '{base_id}' matrix generated an invalid id '{variant_id!r}'.")

    return variant_id


def _format_matrix_value(value: Any) -> str:
    if value is None:
        return "base"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _matches_matrix_pattern(combo: dict[str, Any], pattern: dict[str, Any]) -> bool:
    return all(combo.get(key) == value for key, value in pattern.items())


def _validate_env_args(envs: Iterable[EnvironmentConfigSchema]) -> None:
    cache: EnvMetadataCache = {}
    for env in envs:
        env_module = env.module or env.matrix_base_id or env.id
        try:
            metadata = load_env_metadata(env_module, cache=cache)
        except ImportError as exc:
            logger.warning("Cannot validate env_args for '%s': %s", env.id, exc)
            continue
        if not metadata:
            continue

        param_map = {param.name: param for param in metadata if getattr(param, "supports_cli", True)}
        unknown = sorted(set(env.env_args) - set(param_map))
        if unknown:
            valid_params = ", ".join(sorted(param_map)) or "<none>"
            msg = (
                f"Environment '{env.id}' env_args contain unknown parameters: {', '.join(unknown)}."
                f" Valid parameters: {valid_params}."
            )
            raise ValueError(msg)

        validate_env_arg_values(env.id, env.env_args, metadata)


__all__ = ["ConfigFormatError", "load_run_config"]
