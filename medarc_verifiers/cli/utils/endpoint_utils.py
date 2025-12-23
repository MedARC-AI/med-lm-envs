"""Utilities for loading and caching endpoint registries and environment metadata."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from verifiers.utils.eval_utils import load_endpoints

from medarc_verifiers.cli.utils.env_args import EnvParam, gather_env_cli_metadata

logger = logging.getLogger(__name__)

EndpointRegistry = Mapping[str, Mapping[str, str]]
EndpointRegistryCache = MutableMapping[str, EndpointRegistry]
EnvMetadataCache = MutableMapping[str, Sequence[EnvParam]]

_GLOBAL_ENDPOINT_CACHE: dict[str, EndpointRegistry] = {}
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
        entry = endpoints[model]
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
    "load_env_metadata",
    "resolve_model_endpoint",
]
