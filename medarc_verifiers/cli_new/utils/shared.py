"""Shared helper utilities for the unified CLI implementation."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable as _Iterable

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


def slugify(value: str) -> str:
    """Create a filesystem/ID friendly slug from an arbitrary string.

    Preserves alphanumerics, ``-`` and ``_``; replaces everything else with ``-``
    and trims leading/trailing separators. Returns ``"run"`` when empty.
    """
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in value).strip("-")
    return slug or "run"


def compute_checksum(payload: Any) -> str:
    """Compute a deterministic SHA256 checksum for arbitrary JSON-like payloads.

    Uses json.dumps with sorted keys and compact separators; falls back to ``str`` for unknown types.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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


def build_headers_with_file(
    header_values: _Iterable[str] | None,
    header_file: Path | None,
) -> dict[str, str]:
    """Build headers from CLI values plus optional file.

    The file is newline-delimited ``Name: Value`` entries. Values from the file
    override duplicates from the CLI list, matching existing single-run behavior.
    """
    headers = build_headers(header_values)
    if header_file is None:
        return headers
    try:
        contents = Path(header_file).expanduser().read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - I/O guarded by parser in tests
        raise ValueError(f"Failed to read header file '{header_file}': {exc}") from exc
    file_headers = build_headers(line.strip() for line in contents.splitlines() if line.strip())
    headers.update(file_headers)
    return headers


def resolve_env_identifier(env_cfg: Any) -> str:
    """Resolve the import identifier to use for an environment config.

    Expects an object with attributes ``module`` and ``id``; raises ValueError if neither is set.
    """
    module = getattr(env_cfg, "module", None)
    eid = getattr(env_cfg, "id", None)
    if module:
        return module
    if eid:
        return eid
    raise ValueError("Environment entries must define 'id' or 'module'.")


def resolve_env_identifier_or(env_cfg: Any, fallback: str) -> str:
    """Resolve environment identifier or return provided fallback when missing."""
    try:
        return resolve_env_identifier(env_cfg)
    except ValueError:
        return fallback


__all__ = [
    "slugify",
    "compute_checksum",
    "MissingEnvParamError",
    "HEADER_SEPARATOR",
    "STATE_COLUMNS_SEPARATOR",
    "build_headers",
    "build_headers_with_file",
    "resolve_env_identifier",
    "resolve_env_identifier_or",
    "coerce_json_mapping",
    "merge_sampling_args",
    "flatten_state_columns",
    "resolve_endpoint_selection",
    "merge_env_args",
    "ensure_required_params",
    "ensure_root_logging",
    "asdict_sanitized",
]
