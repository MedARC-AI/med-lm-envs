"""Rollout folding helpers for exporter process pipeline.

This module provides utilities to:
- Derive a base environment id and rollout index from a manifest env id that may
    include a rollout suffix (e.g., "env-a-rollout7" or "env-a-r7").
- Extract a rollout index from arbitrary strings such as results directory names
    when manifests do not encode rollout suffixes in env ids.
"""

from __future__ import annotations

import re
from typing import Tuple

ROLL_OUT_SUFFIX_PATTERN = re.compile(r"-(?:r|rollout)(?P<index>\d+)$")


def derive_base_env_id(env_id: str | None, *, combine_rollouts: bool = True) -> tuple[str, int]:
    """Return the base env id and rollout index.

    If `combine_rollouts` is False, the original env_id is returned with index 0.
    """
    if not env_id:
        return "", 0
    if not combine_rollouts:
        return env_id, 0
    match = ROLL_OUT_SUFFIX_PATTERN.search(env_id)
    if not match:
        return env_id, 0
    start = match.start()
    base_env_id = env_id[:start] or env_id
    try:
        rollout_index = int(match.group("index"))
    except (TypeError, ValueError):
        rollout_index = 0
    return base_env_id, rollout_index


def extract_rollout_index(value: str | None) -> int:
    """Extract a rollout index from an arbitrary string, if present.

    This is useful when the manifest env id doesn't include a rollout suffix, but
    the results directory (or other identifier) does, e.g., "...-rollout1618".
    """
    if not value:
        return 0
    match = ROLL_OUT_SUFFIX_PATTERN.search(value)
    if not match:
        return 0
    try:
        return int(match.group("index"))
    except (TypeError, ValueError):
        return 0


__all__: Tuple[str, ...] = ("derive_base_env_id", "extract_rollout_index")
