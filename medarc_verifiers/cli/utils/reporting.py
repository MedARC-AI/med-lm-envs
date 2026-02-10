from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from verifiers.types import GenerateOutputs

from medarc_verifiers.cli.utils.json_io import write_json

logger = logging.getLogger(__name__)


def log_results_summary(
    *,
    results: GenerateOutputs,
    env_slug: str,
    judge_name: str,
    stage: str,
    reward_limit: int = 25,
) -> None:
    """Emit a concise summary of rewards and key metrics for a run."""
    metadata = _safe_get(results, "metadata", {}) or {}
    avg_reward = _safe_get(metadata, "avg_reward", None)
    rollouts = _safe_get(metadata, "rollouts_per_example", None)
    examples = _safe_get(metadata, "num_examples", None)

    if avg_reward is None:
        logger.info(
            "[%s] %s / %s: examples=%s, rollouts_per_example=%s",
            stage,
            env_slug,
            judge_name,
            examples,
            rollouts,
        )
    else:
        logger.info(
            "[%s] %s / %s: avg_reward=%.4f, examples=%s, rollouts_per_example=%s",
            stage,
            env_slug,
            judge_name,
            float(avg_reward),
            examples,
            rollouts,
        )

    rewards = _safe_get(results, "reward", None)
    per_rollout: list[list[float]] = []
    if isinstance(rollouts, int) and rollouts > 0 and rewards:
        block = len(rewards) // rollouts
        for idx in range(rollouts):
            start = idx * block
            end = start + block
            per_rollout.append(rewards[start:end])
    for idx, sequence in enumerate(per_rollout, start=1):
        display = sequence[:reward_limit]
        suffix = ""
        if len(sequence) > reward_limit:
            suffix = f" (showing first {reward_limit} of {len(sequence)})"
        logger.info("  r%d rewards: %s%s", idx, [round(val, 3) for val in display], suffix)

    pass_rate = _summarize_metric(_safe_get(results, "metrics", {}) or {}, "pass_rate")
    if pass_rate is not None:
        logger.info("  pass_rate avg: %.4f", pass_rate)


def compute_average(values: Sequence[float] | Iterable[float] | None) -> float | None:
    """Compute the arithmetic mean for a sequence of numeric values."""
    if not values:
        return None
    total = 0.0
    count = 0
    for value in values:
        if value is None:
            continue
        total += float(value)
        count += 1
    if count == 0:
        return None
    return total / count


def compute_metric_averages(metrics: Mapping[str, Sequence[float] | Iterable[float]] | None) -> dict[str, float]:
    """Average every metric list present in the evaluation payload."""
    if not metrics:
        return {}
    summary: dict[str, float] = {}
    for key, values in metrics.items():
        avg = compute_average(values)
        if avg is not None:
            summary[key] = avg
    return summary


def update_metadata_file(path: Path, avg_reward: float | None, metrics_avg: Mapping[str, float]) -> None:
    """Patch persisted metadata with up-to-date averages if the file exists."""
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return

    changed = False
    if avg_reward is not None and payload.get("avg_reward") != avg_reward:
        payload["avg_reward"] = avg_reward
        changed = True
    if metrics_avg:
        current_metrics = payload.get("avg_metrics")
        if current_metrics != metrics_avg:
            payload["avg_metrics"] = metrics_avg
            changed = True
    if changed:
        write_json(path, payload)


def _summarize_metric(metrics: Mapping[str, Iterable[float]], key: str) -> float | None:
    values = metrics.get(key)
    if not values:
        return None
    values_list = list(values)
    if not values_list:
        return None
    return sum(values_list) / len(values_list)


def _safe_get(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
