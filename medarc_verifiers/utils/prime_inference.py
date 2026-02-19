"""Helpers for Prime Inference API integration."""

from __future__ import annotations

import os
from typing import Any

PRIME_INFERENCE_URL = "https://api.pinference.ai/api/v1"


def _resolve_include_usage(include_usage: bool | None, is_prime_inference: bool) -> bool:
    """Resolve the effective include_usage value.

    Priority:
    1. Explicit include_usage parameter (if not None)
    2. MEDARC_INCLUDE_USAGE environment variable (if set)
    3. Auto-detect based on whether base_url is Prime Inference
    """
    if include_usage is not None:
        return include_usage

    env_value = os.environ.get("MEDARC_INCLUDE_USAGE")
    if env_value is not None:
        return env_value.lower() in ("1", "true", "yes")

    return is_prime_inference


def prime_inference_overrides(
    base_url: str | None,
    *,
    include_usage: bool | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return Prime Inference-specific headers and sampling_args overrides.

    Args:
        base_url: The API base URL. If it matches Prime Inference, overrides are returned.
        include_usage: Whether to include usage reporting. If None, auto-detects
            based on MEDARC_INCLUDE_USAGE env var or Prime Inference URL.

    Returns:
        A tuple of (extra_headers, sampling_args_overrides).
        - extra_headers: {"X-Prime-Team-ID": ...} if PRIME_TEAM_ID is set, else {}
        - sampling_args_overrides: {"extra_body": {"usage": {"include": True}}} if enabled, else {}
    """
    is_prime = base_url == PRIME_INFERENCE_URL
    extra_headers: dict[str, str] = {}
    sampling_overrides: dict[str, Any] = {}

    if is_prime:
        prime_team_id = os.environ.get("PRIME_TEAM_ID")
        if prime_team_id:
            extra_headers["X-Prime-Team-ID"] = prime_team_id

    effective_include_usage = _resolve_include_usage(include_usage, is_prime)
    if effective_include_usage:
        sampling_overrides["extra_body"] = {"usage": {"include": True}}

    return extra_headers, sampling_overrides
