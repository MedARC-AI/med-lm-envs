"""
HELM-style reward normalization helpers.

Many of our judge prompts emit per-dimension scores on a 1–5 scale (inclusive).
This module provides a shared normalization routine to convert those scores into
an averaged reward in [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def normalize_helm_reward(
    scores: Mapping[str, Mapping[str, Any]],
    *,
    dimensions: Sequence[str],
    min_score: float = 1.0,
    max_score: float = 5.0,
) -> float:
    """Normalize per-dimension judge scores into a single reward in [0.0, 1.0].

    - Per-dimension scores are clamped to [min_score, max_score].
    - Each dimension is mapped to [0, 1] via (score - min_score) / (max_score - min_score).
    - Missing/unparseable scores are ignored, and the result is averaged over the
      dimensions that had valid scores.
    """

    if not dimensions:
        return 0.0
    denom = max_score - min_score
    if denom <= 0:
        return 0.0

    accumulated = 0.0
    count = 0
    for dimension in dimensions:
        score = _coerce_score(scores.get(dimension, {}).get("score"))
        if score is None:
            continue
        clamped = max(min_score, min(max_score, score))
        accumulated += (clamped - min_score) / denom
        count += 1

    if count == 0:
        return 0.0
    return max(0.0, min(1.0, accumulated / count))
