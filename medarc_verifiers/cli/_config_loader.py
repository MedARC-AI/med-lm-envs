"""Config loader utilities bridging OmegaConf YAML files and Pydantic schemas."""

from __future__ import annotations

import logging
from itertools import product
from collections.abc import Iterable, Mapping
from typing import Any, Callable
from pathlib import Path

from omegaconf import OmegaConf

from ._schemas import EnvironmentConfigSchema, RunConfigSchema, RESERVED_MATRIX_KEYS
from .utils.endpoint_utils import EnvMetadataCache, load_env_metadata
from .utils.env_args import validate_env_args_or_raise

logger = logging.getLogger(__name__)
DEFAULT_ENV_FILE_SUFFIXES = (".yaml", ".yml")

# Scalar fields (non-env_args) that may be overridden by matrix combos.
SCALAR_FIELD_NAMES = {
    name for name in EnvironmentConfigSchema.model_fields if name not in RESERVED_MATRIX_KEYS and name != "env_args"
}


class ConfigFormatError(ValueError):
    """Raised when a configuration file cannot be interpreted as a mapping."""


def _load_raw_config(path: Path) -> Any:
    """Load and resolve an OmegaConf configuration file."""
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


def load_run_config(path: str | Path, *, env_default_root: str | Path | None = None) -> RunConfigSchema:
    """Load a run configuration file into the top-level schema."""
    # Loader responsibilities:
    # 1. Read and resolve OmegaConf input (supporting includes and defaults).
    # 2. Normalize models/envs/jobs into canonical mappings or lists.
    # 3. Let Pydantic schemas handle structural validation and coercion.
    # 4. Expand environment matrices into concrete variants.
    # 5. Perform lightweight env_args validation using environment metadata.
    resolved_path = Path(path).expanduser().resolve()
    env_default_root_path = Path(env_default_root).expanduser().resolve() if env_default_root is not None else None
    data = _load_raw_config(resolved_path)

    if not isinstance(data, dict):
        msg = f"Configuration root must be a mapping, got {type(data).__name__}."
        raise ConfigFormatError(msg)

    if "envs" not in data or data["envs"] in (None, [], {}):
        if env_default_root_path is None:
            raise ConfigFormatError(
                "Configuration must define 'envs' or --env-config-root must supply a discovery directory."
            )
        data = dict(data)
        data["envs"] = str(env_default_root_path)

    data = _normalize_config_fields(data, base_dir=resolved_path.parent, env_default_root=env_default_root_path)

    run_config = RunConfigSchema(**data)
    expanded_envs = _expand_env_matrices(run_config.envs)
    _validate_env_args(expanded_envs.values())
    return run_config.model_copy(update={"envs": expanded_envs})


def _expand_env_matrices(envs: dict[str, EnvironmentConfigSchema]) -> dict[str, EnvironmentConfigSchema]:
    scalar_fields = SCALAR_FIELD_NAMES
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

    matrix_keys = list(matrix.keys())
    matrix_values = [matrix[key] for key in matrix_keys]
    variants: list[EnvironmentConfigSchema] = []
    seen_ids: set[str] = set()

    base_env_args = dict(env.env_args)
    module_name = env.module or env.id  # prefer explicit module override when present

    exclude_patterns = env.matrix_exclude or []

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


def _normalize_config_fields(
    data: Mapping[str, Any], *, base_dir: Path, env_default_root: Path | None
) -> dict[str, Any]:
    """Apply include expansion and shape normalization before schema validation."""

    normalized = dict(data)

    if "models" in normalized:
        normalized["models"] = _normalize_models_field(normalized["models"], base_dir=base_dir)

    if "envs" in normalized:
        normalized["envs"] = _normalize_envs_field(
            normalized["envs"],
            base_dir=base_dir,
            env_default_root=env_default_root,
        )

    if "jobs" in normalized:
        normalized["jobs"] = _normalize_jobs_field(normalized["jobs"], base_dir=base_dir)

    return normalized


def _normalize_models_field(value: Any, *, base_dir: Path) -> dict[str, Any]:
    return _normalize_section(
        value,
        base_dir=base_dir,
        context="models",
        entry_description="models",
        default_id_from_key=True,
        allow_duplicate_ids=False,
        env_default_root=None,
    )


def _normalize_envs_field(value: Any, *, base_dir: Path, env_default_root: Path | None) -> dict[str, Any]:
    # Env configs intentionally allow duplicate "id" entries so multiple blocks can
    # share a common base id (e.g., m_arc + rollout variants). We de-duplicate only
    # the internal map key while preserving each entry's explicit "id".
    return _normalize_section(
        value,
        base_dir=base_dir,
        context="envs",
        entry_description="envs",
        default_id_from_key=True,
        allow_duplicate_ids=True,
        duplicate_key_fn=_make_duplicate_key,
        env_default_root=env_default_root,
    )


def _make_duplicate_key(base: str, count: int, existing: Mapping[str, Any]) -> str:
    suffix = count
    while True:
        candidate = f"{base}__dup__{suffix}"
        if candidate not in existing:
            return candidate
        suffix += 1


def _normalize_jobs_field(value: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    entries = _collect_job_entries(value, base_dir=base_dir)
    return [_adapt_job_entry(entry) for entry in entries]


def _collect_model_entries(source: Any, *, base_dir: Path, context: str) -> list[dict[str, Any]]:
    return _collect_entries(
        source, base_dir=base_dir, context=context, entry_description="models", env_default_root=None
    )


def _collect_env_entries(
    source: Any, *, base_dir: Path, context: str, env_default_root: Path | None
) -> list[dict[str, Any]]:
    return _collect_entries(
        source,
        base_dir=base_dir,
        context=context,
        entry_description="envs",
        env_default_root=env_default_root,
    )


def _collect_job_entries(source: Any, *, base_dir: Path) -> list[dict[str, Any]]:
    return _collect_entries(source, base_dir=base_dir, context="jobs", entry_description="jobs", env_default_root=None)


def _normalize_section(
    value: Any,
    *,
    base_dir: Path,
    context: str,
    entry_description: str,
    default_id_from_key: bool,
    allow_duplicate_ids: bool,
    duplicate_key_fn: Callable[[str, int, Mapping[str, Any]], str] | None = None,
    env_default_root: Path | None,
) -> dict[str, Any]:
    """Normalize section entries (models/envs) with shared include handling."""
    if value is None:
        return {}

    normalized: dict[str, Any] = {}

    def _add_entry(entry: Mapping[str, Any], *, key_hint: str | None = None, count_map: dict[str, int] | None = None) -> None:
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} entries must be mappings.")
        adapted = dict(entry)
        item_id = adapted.get("id") or key_hint
        if not item_id:
            raise ValueError(f"{context} entries must include an 'id'.")
        key = str(item_id)
        if count_map is not None:
            count_map.setdefault(key, 1)
        if key in normalized:
            if not allow_duplicate_ids:
                raise ValueError(f"Duplicate {entry_description.rstrip('s')} id '{key}' in configuration.")
            if duplicate_key_fn is None:
                raise ValueError(f"Duplicate {entry_description.rstrip('s')} id '{key}' in configuration.")
            # Env entries can intentionally repeat ids to group variants under a common
            # base id; we only de-duplicate the internal map key, not the entry's id.
            counter = 2
            if count_map is not None:
                counter = count_map.get(key, 1) + 1
                count_map[key] = counter
            key = duplicate_key_fn(key, counter, normalized)
        normalized[key] = adapted

    if isinstance(value, Mapping) and all(isinstance(v, Mapping) for v in value.values()):
        for key, entry in value.items():
            adapted = dict(entry)
            if default_id_from_key and "id" not in adapted:
                adapted["id"] = str(key)
            _add_entry(adapted)
        return normalized

    entries = _collect_entries(
        value,
        base_dir=base_dir,
        context=context,
        entry_description=entry_description,
        env_default_root=env_default_root,
    )
    duplicate_counts: dict[str, int] = {}
    for entry in entries:
        _add_entry(entry, count_map=duplicate_counts)
    return normalized


def _collect_entries(
    source: Any,
    *,
    base_dir: Path,
    context: str,
    entry_description: str,
    env_default_root: Path | None,
) -> list[dict[str, Any]]:
    if source is None:
        return []
    if isinstance(source, Mapping):
        return [dict(source)]
    if isinstance(source, (str, Path)):
        return _collect_entries_from_path(
            source,
            base_dir=base_dir,
            context=context,
            entry_description=entry_description,
            env_default_root=env_default_root,
        )
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
                        env_default_root=env_default_root,
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
    env_default_root: Path | None,
) -> list[dict[str, Any]]:
    path = _resolve_include_path(source, base_dir=base_dir)
    if not path.exists() and entry_description == "envs":
        fallback = _resolve_default_env_path(source, base_dir=base_dir, env_default_root=env_default_root)
        if fallback is not None:
            path = fallback
    if not path.exists():
        raise FileNotFoundError(f"{context} path '{path}' does not exist.")
    if path.is_dir():
        if entry_description not in {"envs", "jobs"}:
            msg = f"{context} path '{path}' must be a file. Directory includes are only supported for envs and jobs."
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
                        env_default_root=env_default_root,
                    )
                )
        return entries

    loaded = _load_raw_config(path)
    if isinstance(loaded, Mapping):
        if not loaded:
            return []
        if not all(isinstance(v, Mapping) for v in loaded.values()):
            msg = f"{context} included {entry_description} must be a mapping of id→mapping or a list of mappings."
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
    raise ValueError(f"{context} included {entry_description} must be a mapping of id→mapping or a list of mappings.")


def _resolve_include_path(source: str | Path, *, base_dir: Path) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _resolve_default_env_path(source: str | Path, *, base_dir: Path, env_default_root: Path | None) -> Path | None:
    raw_source = Path(source)
    if raw_source.is_absolute() or env_default_root is None:
        return None

    normalized = env_default_root if env_default_root.is_absolute() else env_default_root.resolve()
    candidates = _candidate_env_paths(normalized, raw_source)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_env_paths(root: Path, relative_entry: Path) -> list[Path]:
    base = root / relative_entry
    candidates = [base]
    if not relative_entry.suffix:
        for suffix in DEFAULT_ENV_FILE_SUFFIXES:
            candidates.append((root / relative_entry).with_suffix(suffix))
    return [candidate.resolve() for candidate in candidates]


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
            raise ValueError(f"environment '{base_id}' matrix_id_format references unknown key '{missing}'.") from exc
    else:
        suffix_parts = [f"{key}-{_format_matrix_value(value)}" for key, value in combo.items() if value is not None]
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
    """Validate env_args at config load time (lenient - no required param enforcement).

    This is the first of two validation phases:
    Phase 1 (here): Check for unknown parameters and type mismatches
                    Do NOT enforce required parameters (allow partial configs)
    Phase 2 (executor): Enforce required parameters after CLI overrides applied

    Why two phases?
    - Matrix expansion can create variants with different required params
    - Users might load a config with 100 jobs but only run 5 with --job-id
    - Failing at load time for jobs we won't run would be frustrating

    This phase catches obvious mistakes (typos, wrong types) early while deferring
    required parameter checks until execution when we know what will actually run.
    """
    cache: EnvMetadataCache = {}
    for env in envs:
        env_module = env.module or env.matrix_base_id or env.id
        if not env_module:
            continue
        try:
            metadata = load_env_metadata(env_module, cache=cache)
        except ImportError as exc:
            logger.warning("Skipping env_args validation for '%s': %s", env_module, exc)
            continue
        # Phase 1 validation: unknown/type checks only; do not enforce requireds at load time.
        validate_env_args_or_raise(
            env_module,
            env.env_args,
            metadata=metadata,
            metadata_cache=cache,
            allow_unknown=False,
            enforce_required=False,  # Deferred to execution time
        )


__all__ = ["ConfigFormatError", "load_run_config"]
