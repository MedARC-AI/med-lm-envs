"""Shared helper utilities for the unified CLI implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from verifiers import setup_logging

from .endpoint_utils import resolve_model_endpoint
from .env_args import (
    HEADER_SEPARATOR,
    MissingEnvParamError,
    build_headers,
    ensure_required_params,
)
STATE_COLUMNS_SEPARATOR = ","
_LOGGING_INITIALIZED = False
def coerce_json_mapping(value: Any, *, flag: str) -> dict[str, Any]:
    """Ensure a decoded JSON value is a mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        msg = f"{flag} must be a JSON object."
        raise ValueError(msg)
    return dict(value)


def merge_sampling_args(
    sampling_args: Mapping[str, Any] | None,
    *,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None = None,
    top_k: int | None = None,
    n: int | None = None,
) -> dict[str, Any]:
    """Merge scalar sampling overrides with an arbitrary mapping."""
    merged: dict[str, Any] = dict(sampling_args or {})
    if max_tokens is not None and "max_tokens" not in merged:
        merged["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged:
        merged["temperature"] = temperature
    if top_p is not None and "top_p" not in merged:
        merged["top_p"] = top_p
    if top_k is not None and "top_k" not in merged:
        merged["top_k"] = top_k
    if n is not None and "n" not in merged:
        merged["n"] = n
    return merged


def flatten_state_columns(values: Iterable[Sequence[str]] | None) -> list[str]:
    """Flatten repeated state column entries into a single list."""
    if not values:
        return []
    flattened: list[str] = []
    for group in values:
        flattened.extend(group)
    return flattened


def resolve_endpoint_selection(
    model: str,
    endpoints: Mapping[str, Mapping[str, str]],
    *,
    default_key_var: str,
    default_base_url: str,
) -> tuple[str, str, str]:
    """Resolve endpoint configuration for a model alias."""
    return resolve_model_endpoint(
        model,
        endpoints,
        default_key_var=default_key_var,
        default_base_url=default_base_url,
    )


def merge_env_args(explicit: Mapping[str, Any], json_args: Mapping[str, Any]) -> dict[str, Any]:
    """Merge JSON-provided env args with CLI overrides (explicit wins)."""
    merged: dict[str, Any] = dict(json_args)
    for key, value in explicit.items():
        if key in merged and merged[key] != value:
            logging.getLogger(__name__).debug(
                "CLI option '%s' overriding JSON value '%s' with '%s'.",
                key,
                merged[key],
                value,
            )
        merged[key] = value
    return merged


def ensure_root_logging(level: str) -> None:
    """Configure root logging once while allowing level updates."""
    global _LOGGING_INITIALIZED
    root_logger = logging.getLogger()
    if not _LOGGING_INITIALIZED:
        setup_logging(level)
        _LOGGING_INITIALIZED = True
    else:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)


def asdict_sanitized(obj: Any) -> Any:
    """Convert arbitrary dataclass-backed objects into JSON-friendly structures."""
    return _sanitize(obj)


def _sanitize(value: Any) -> Any:
    if is_dataclass(value):
        return _sanitize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _sanitize(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


__all__ = [
    "MissingEnvParamError",
    "HEADER_SEPARATOR",
    "STATE_COLUMNS_SEPARATOR",
    "build_headers",
    "coerce_json_mapping",
    "merge_sampling_args",
    "flatten_state_columns",
    "resolve_endpoint_selection",
    "merge_env_args",
    "ensure_required_params",
    "ensure_root_logging",
    "asdict_sanitized",
]
