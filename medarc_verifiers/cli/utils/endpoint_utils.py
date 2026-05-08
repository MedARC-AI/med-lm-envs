"""Utilities for loading and caching endpoint registries and environment metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, MutableMapping, Sequence, cast

from verifiers.types import Endpoints
from verifiers.utils.eval_utils import load_endpoints, resolve_endpoints_file
from verifiers.utils.import_utils import load_toml

from medarc_verifiers.cli.utils.env_args import EnvParam, gather_env_cli_metadata

logger = logging.getLogger(__name__)

EndpointRegistry = Endpoints
EndpointRegistryCache = MutableMapping[str, Endpoints]
EnvMetadataCache = MutableMapping[str, Sequence[EnvParam]]

_GLOBAL_ENDPOINT_CACHE: dict[str, Endpoints] = {}
_GLOBAL_ENV_METADATA_CACHE: dict[str, Sequence[EnvParam]] = {}


def _normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def load_endpoint_registry(
    path: str | Path,
    *,
    cache: EndpointRegistryCache | None = None,
) -> EndpointRegistry:
    """Load the endpoint registry, memoizing results for subsequent calls."""
    normalized = _normalize_path(path)
    store = cache if cache is not None else _GLOBAL_ENDPOINT_CACHE

    if normalized not in store:
        logger.debug("Loading endpoint registry from '%s'.", normalized)
        store[normalized] = load_endpoints(normalized)
    else:
        logger.debug("Using cached endpoint registry for '%s'.", normalized)

    return store[normalized]


def load_endpoint_sampling_profiles(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load MedARC endpoint-level sampling defaults from a TOML registry."""
    resolved = resolve_endpoints_file(str(path))
    if resolved is None or not resolved.exists() or resolved.suffix != ".toml":
        return {}

    with resolved.open("rb") as handle:
        raw_toml = load_toml(handle)
    if not isinstance(raw_toml, dict):
        raise ValueError(f"Expected top-level TOML table in endpoint registry {resolved}")

    raw_entries = raw_toml.get("endpoint", [])
    if not isinstance(raw_entries, list):
        raise ValueError(f"Expected [[endpoint]] array-of-tables in endpoint registry {resolved}")

    profiles: dict[str, list[dict[str, Any]]] = {}
    for index, raw_entry in enumerate(raw_entries):
        entry_source = f"{resolved} ([[endpoint]] index {index})"
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Each [[endpoint]] entry must be a table in {entry_source}")

        endpoint_id = raw_entry.get("endpoint_id")
        if not isinstance(endpoint_id, str) or not endpoint_id:
            if "sampling_args" in raw_entry:
                raise ValueError(
                    f"Endpoint profile with sampling_args must include non-empty string endpoint_id in {entry_source}"
                )
            continue

        raw_sampling_args = raw_entry.get("sampling_args", {})
        if isinstance(raw_sampling_args, list):
            raise ValueError(
                f"Endpoint '{endpoint_id}' sampling_args must be a table in {entry_source}; "
                "use [endpoint.sampling_args] or an inline table, not [[endpoint.sampling_args]]."
            )
        if not isinstance(raw_sampling_args, dict):
            raise ValueError(f"Endpoint '{endpoint_id}' sampling_args must be a table in {entry_source}")

        profiles.setdefault(endpoint_id, []).append(dict(cast(dict[str, Any], raw_sampling_args)))

    return profiles


def load_env_metadata(
    env_id: str,
    *,
    cache: EnvMetadataCache | None = None,
) -> Sequence[EnvParam]:
    """Retrieve environment CLI metadata with caching."""
    store = cache if cache is not None else _GLOBAL_ENV_METADATA_CACHE

    if env_id not in store:
        logger.debug("Gathering environment CLI metadata for '%s'.", env_id)
        store[env_id] = gather_env_cli_metadata(env_id)
    else:
        logger.debug("Using cached environment CLI metadata for '%s'.", env_id)

    return store[env_id]


def resolve_model_endpoint(
    model: str,
    endpoints: EndpointRegistry,
    *,
    default_key_var: str,
    default_base_url: str,
) -> tuple[str, str, str]:
    """Resolve model aliases and infer endpoint configuration."""
    if model in endpoints:
        variants = endpoints[model]
        if not variants:
            logger.warning(
                "Model '%s' has no endpoint variants configured; using CLI-specified API config.",
                model,
            )
            return model, default_key_var, default_base_url

        if len(variants) > 1:
            logger.debug("Endpoint id '%s' has %d variants configured.", model, len(variants))

        entry = variants[0]
        resolved_model = entry.get("model", model)
        api_key_var = entry.get("key", default_key_var)
        api_base_url = entry.get("url", default_base_url)
        logger.debug(
            "Resolved model '%s' using endpoint registry entry '%s'.",
            model,
            resolved_model,
        )
        return resolved_model, api_key_var, api_base_url

    logger.debug(
        "Model '%s' not found in endpoint registry; using CLI-specified API config.",
        model,
    )
    return model, default_key_var, default_base_url


__all__ = [
    "EndpointRegistry",
    "EndpointRegistryCache",
    "EnvMetadataCache",
    "load_endpoint_registry",
    "load_endpoint_sampling_profiles",
    "load_env_metadata",
    "resolve_model_endpoint",
]
