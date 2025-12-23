"""Shared env args merge + validation helpers."""

from __future__ import annotations

import logging
from typing import Any, Mapping

from medarc_verifiers.cli_new.utils.env_args import validate_env_args_or_raise

logger = logging.getLogger(__name__)


def merge_env_args(
    *,
    env_defaults: Mapping[str, Any],
    model_defaults: Mapping[str, Any],
    model_env_override: Mapping[str, Any] | None,
    job_overrides: Mapping[str, Any],
    cli_overrides: Mapping[str, Any] | None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Merge env args with precedence env→model→model_override→job→CLI."""
    merged: dict[str, Any] = {}
    for source in (env_defaults, model_defaults, model_env_override or {}, job_overrides, cli_overrides or {}):
        if not source:
            continue
        if source is cli_overrides and verbose:
            overridden_keys = set(merged) & set(source)
            new_keys = set(source) - set(merged)
            if overridden_keys:
                logger.debug(
                    "CLI overriding env_args: %s",
                    {k: f"{merged[k]} → {source[k]}" for k in overridden_keys},
                )
            if new_keys:
                logger.debug("CLI adding env_args: %s", list(new_keys))
        merged.update(source)
    return merged


def validate_env_args(
    *,
    env_id: str,
    env_args: Mapping[str, Any],
    metadata_loader,
    metadata_cache,
    allow_unknown: bool,
    enforce_required: bool,
) -> None:
    """Validate env args against loader metadata with configurable strictness."""
    try:
        metadata = metadata_loader(env_id, cache=metadata_cache)
    except ImportError as exc:
        logger.warning("Skipping env_args validation for '%s': %s", env_id, exc)
        return
    except TypeError:
        raise
    # When no metadata is available, skip validation
    if metadata is None:
        return
    validate_env_args_or_raise(
        env_id,
        env_args,
        metadata,
        allow_unknown=allow_unknown,
        enforce_required=enforce_required,
    )


__all__ = ["merge_env_args", "validate_env_args"]
