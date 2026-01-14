"""HELM-style win rate helpers for the process pipeline."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import polars as pl
from polars import DataFrame as PLDataFrame, LazyFrame as PLLazyFrame

logger = logging.getLogger(__name__)

MODEL_COL = "model_id"
REWARD_COL = "reward"
EXAMPLE_ID_COL = "example_id"


@dataclass(slots=True)
class WinrateConfig:
    """Configuration switches for win rate calculations."""

    missing_policy: str = "neg-inf"  # "zero" or "neg-inf"
    epsilon: float = 1e-9
    min_common: int = 0
    weight_policy: str = "ln"  # "equal", "ln", "sqrt", or "cap"
    weight_cap: int = 0
    include_models: tuple[str, ...] = ()
    exclude_models: tuple[str, ...] = ()
    partial_datasets: str = "strict"  # "strict" or "include"


@dataclass(slots=True)
class DatasetStats:
    """Per-dataset win rate statistics."""

    pairwise: dict[tuple[str, str], tuple[float | None, int]]
    n_questions: int
    models: list[str]
    avg_reward_per_model: dict[str, float | None]


@dataclass(slots=True)
class ModelCentricResult:
    """Final JSON-ready payload organized by model."""

    models: dict[str, dict[str, Any]]
    datasets: dict[str, dict[str, Any]]


def read_dataset_lazy(
    parquet_path: Path | str | Sequence[Path | str | PLDataFrame | PLLazyFrame] | PLDataFrame | PLLazyFrame,
) -> pl.LazyFrame:
    """Read required columns lazily and normalize reward values."""
    if isinstance(parquet_path, PLLazyFrame):
        lf = parquet_path
    elif isinstance(parquet_path, PLDataFrame):
        lf = parquet_path.lazy()
    else:
        if isinstance(parquet_path, (list, tuple)):
            if parquet_path and isinstance(parquet_path[0], (PLDataFrame, PLLazyFrame)):
                frames: list[PLLazyFrame] = []
                for item in parquet_path:
                    if isinstance(item, PLLazyFrame):
                        frames.append(item)
                    elif isinstance(item, PLDataFrame):
                        frames.append(item.lazy())
                lf = pl.concat(frames, how="vertical") if len(frames) > 1 else frames[0]
            else:
                paths = [str(Path(p)) for p in parquet_path]
                lf = pl.scan_parquet(paths)
        else:
            lf = pl.scan_parquet([str(Path(parquet_path))])
    cols = lf.collect_schema().names()
    col_set = set(cols)
    required = {EXAMPLE_ID_COL, REWARD_COL, MODEL_COL}
    missing = required - col_set
    if missing:
        raise ValueError(
            f"Parquet missing required columns {missing} at {parquet_path}. "
            "Expected at least example_id, model_id, reward."
        )

    has_rollout = "rollout_index" in cols
    select_exprs = [pl.col(EXAMPLE_ID_COL), pl.col(MODEL_COL), pl.col(REWARD_COL)]
    if has_rollout:
        select_exprs.append(pl.col("rollout_index"))
    else:
        select_exprs.append(pl.lit(0).alias("rollout_index"))

    lf = lf.select(select_exprs)
    lf = lf.with_columns(pl.col(REWARD_COL).cast(pl.Float64))
    lf = lf.with_columns(
        pl.when(pl.col(REWARD_COL).is_nan()).then(pl.lit(None)).otherwise(pl.col(REWARD_COL)).alias(REWARD_COL)
    )
    return lf


def average_rollouts(lf: pl.LazyFrame) -> tuple[pl.DataFrame, int]:
    """Average rewards across rollouts per (example_id, model)."""
    df_avg = (
        lf.group_by([EXAMPLE_ID_COL, MODEL_COL])  # type: ignore[arg-type]
        .agg(pl.col(REWARD_COL).mean().alias("reward_mean"))
        .collect()
    )
    if EXAMPLE_ID_COL not in df_avg.columns or df_avg.is_empty():
        return df_avg, 0
    n_questions = int(df_avg.select(pl.col(EXAMPLE_ID_COL).n_unique()).item())
    return df_avg, n_questions


def to_wide(df_avg: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Pivot to wide format with one column per model."""
    if df_avg.is_empty():
        return df_avg, []
    df_wide = df_avg.pivot(values="reward_mean", index=EXAMPLE_ID_COL, on=MODEL_COL)
    model_cols = [c for c in df_wide.columns if c != EXAMPLE_ID_COL]
    return df_wide, model_cols


def _pairwise_win_rate_series(
    a: pl.Series,
    b: pl.Series,
    *,
    missing_policy: str = "neg-inf",
    epsilon: float = 1e-9,
    min_common: int = 0,
) -> tuple[float | None, int]:
    """Calculate win rate for two reward series."""
    a = a.cast(pl.Float64)
    b = b.cast(pl.Float64)
    a_is_null = a.is_null()
    b_is_null = b.is_null()
    a_not_null = ~a_is_null
    b_not_null = ~b_is_null
    used = ~(a_is_null & b_is_null)
    n_used = int(used.sum() or 0)
    if n_used == 0 or n_used < min_common:
        return None, 0

    fill_val = float("-inf") if missing_policy == "neg-inf" else 0.0
    a2 = a.set(a_is_null & b_not_null, fill_val)
    b2 = b.set(b_is_null & a_not_null, fill_val)

    a_used = a2.filter(used)
    b_used = b2.filter(used)
    diff = a_used - b_used
    greater = diff > epsilon
    less = diff < -epsilon
    tie = ~(greater | less)
    comp = greater.cast(pl.Float64) + tie.cast(pl.Float64) * 0.5
    win_rate = comp.mean()
    if win_rate is None:
        return None, 0
    return float(win_rate), n_used


def pairwise_win_rate(
    df_wide: pl.DataFrame,
    model_a: str,
    model_b: str,
    *,
    missing_policy: str = "neg-inf",
    epsilon: float = 1e-9,
    min_common: int = 0,
) -> tuple[float | None, int]:
    """Calculate win rate of model_a vs model_b for a dataset."""
    if model_a not in df_wide.columns or model_b not in df_wide.columns:
        return None, 0
    a = df_wide.get_column(model_a)
    b = df_wide.get_column(model_b)
    return _pairwise_win_rate_series(
        a,
        b,
        missing_policy=missing_policy,
        epsilon=epsilon,
        min_common=min_common,
    )


def weight_of(nq: int, policy: str, cap: int) -> float:
    """Dataset weighting policy."""
    if policy == "equal":
        return 1.0
    if policy == "ln":
        return math.log(nq) if nq > 0 else 0.0
    if policy == "sqrt":
        return math.sqrt(nq) if nq > 0 else 0.0
    if policy == "cap":
        return float(min(max(nq, 0), max(cap, 0)))
    return 1.0


def dataset_model_mean_winrates(
    pair_results: dict[tuple[str, str], tuple[float | None, int]],
    models: Sequence[str],
) -> dict[str, float]:
    """Compute mean win rate per model vs all opponents for a single dataset."""
    means: dict[str, float] = {}
    deduped_models = list(dict.fromkeys(models))
    if len(deduped_models) <= 1:
        return means
    for m in deduped_models:
        wrs: list[float] = []
        for o in deduped_models:
            if o == m:
                continue
            a, b = sorted([m, o])
            wr, n_used = pair_results.get((a, b), (None, 0))
            if wr is None or n_used <= 0:
                continue
            wrs.append(float(wr) if m == a else 1.0 - float(wr))
        if wrs:
            means[m] = sum(wrs) / len(wrs)
    return means


def compute_winrates(
    datasets: Sequence[tuple[str, Path | str | Sequence[Path | str]]],
    config: WinrateConfig | None = None,
    *,
    known_models: Sequence[str] | None = None,
) -> ModelCentricResult:
    """Compute win rates for a list of datasets."""
    cfg = config or WinrateConfig()
    include_set = _normalize_model_ids(cfg.include_models)
    exclude_set = _normalize_model_ids(cfg.exclude_models)
    target_models = sorted(include_set - exclude_set) if include_set else []
    partial_datasets = str(cfg.partial_datasets or "strict").lower()
    known_model_set = _normalize_model_ids(known_models)

    if include_set and not target_models:
        _raise_user_error("No models remain after applying include/exclude filters.")
    if partial_datasets not in {"strict", "include"}:
        _raise_user_error(f"Unsupported partial_datasets policy: {cfg.partial_datasets}")

    if known_model_set:
        unknown_includes = include_set - known_model_set
        if unknown_includes:
            _raise_user_error(f"Unknown include model ids: {sorted(unknown_includes)}")
        unknown_excludes = exclude_set - known_model_set
        if unknown_excludes:
            _warn_user(f"Unknown exclude model ids ignored: {sorted(unknown_excludes)}")

    per_dataset_pairwise: dict[str, dict[tuple[str, str], tuple[float | None, int]]] = {}
    per_dataset_model_means: dict[str, dict[str, float]] = {}
    avg_rewards_by_dataset: dict[str, dict[str, float | None]] = {}
    n_questions_by_ds: dict[str, int] = {}
    models_by_ds: dict[str, list[str]] = {}
    seen_models: set[str] = set()

    dataset_iter: Iterable[tuple[str, Path | str]] = datasets
    try:
        from rich.progress import track

        dataset_iter = track(datasets, description="Computing win rates", transient=True)
    except Exception:
        dataset_iter = datasets

    for dataset_name, parquet_path in dataset_iter:
        stats, models_present = _process_dataset(
            dataset_name,
            parquet_path,
            cfg,
            include_set=include_set,
            exclude_set=exclude_set,
            target_models=target_models,
            partial_datasets=partial_datasets,
        )
        seen_models.update(models_present)
        if not stats:
            continue
        per_dataset_pairwise[dataset_name] = stats.pairwise
        per_dataset_model_means[dataset_name] = dataset_model_mean_winrates(stats.pairwise, stats.models)
        avg_rewards_by_dataset[dataset_name] = stats.avg_reward_per_model
        n_questions_by_ds[dataset_name] = stats.n_questions
        models_by_ds[dataset_name] = stats.models

    if not known_model_set:
        if include_set:
            unknown_includes = include_set - seen_models
            if unknown_includes:
                _raise_user_error(f"Unknown include model ids: {sorted(unknown_includes)}")
        if exclude_set:
            unknown_excludes = exclude_set - seen_models
            if unknown_excludes:
                _warn_user(f"Unknown exclude model ids ignored: {sorted(unknown_excludes)}")

    return build_model_centric_result(
        per_dataset_pairwise=per_dataset_pairwise,
        per_dataset_model_means=per_dataset_model_means,
        avg_rewards_by_dataset=avg_rewards_by_dataset,
        n_questions_by_ds=n_questions_by_ds,
        models_by_ds=models_by_ds,
        config=cfg,
    )


def build_model_centric_result(
    *,
    per_dataset_pairwise: Mapping[str, Mapping[tuple[str, str], tuple[float | None, int]]],
    per_dataset_model_means: Mapping[str, Mapping[str, float]],
    avg_rewards_by_dataset: Mapping[str, Mapping[str, float | None]],
    n_questions_by_ds: Mapping[str, int],
    models_by_ds: Mapping[str, Sequence[str]],
    config: WinrateConfig,
) -> ModelCentricResult:
    """Aggregate per-dataset stats into the final model-centric payload."""
    model_means = _aggregate_model_means(per_dataset_model_means, n_questions_by_ds, config)
    avg_rewards_by_model = compute_avg_rewards_per_model(avg_rewards_by_dataset)
    all_models: set[str] = set(avg_rewards_by_model.keys())
    for models in models_by_ds.values():
        all_models.update(models)
    all_models = set(_filter_models(sorted(all_models), config.include_models, config.exclude_models))
    models_out: dict[str, dict[str, Any]] = {}
    for model in sorted(all_models):
        mean_entry = model_means.get(model)
        mean_payload = {
            "simple_mean": mean_entry["simple_mean"] if mean_entry else None,
            "weighted_mean": mean_entry["weighted_mean"] if mean_entry else None,
            "n_datasets": mean_entry["n_datasets"] if mean_entry else 0,
        }
        opponents = _collect_opponents(model, models_by_ds)
        vs_payload = {}
        for opponent in opponents:
            stats = compute_per_opponent_stats(
                per_dataset_pairwise=per_dataset_pairwise,
                model=model,
                opponent=opponent,
                n_questions_by_ds=n_questions_by_ds,
                weight_policy=config.weight_policy,
                weight_cap=config.weight_cap,
            )
            if stats:
                vs_payload[opponent] = stats
        avg_rewards = avg_rewards_by_model.get(model, {})
        models_out[model] = {
            "mean_winrate": mean_payload,
            "vs": dict(sorted(vs_payload.items())),
            "avg_reward_per_dataset": dict(sorted(avg_rewards.items())),
        }
    dataset_payload = _build_dataset_rewards(avg_rewards_by_dataset, n_questions_by_ds)
    return ModelCentricResult(models=models_out, datasets=dataset_payload)


def _build_dataset_rewards(
    avg_rewards_by_dataset: Mapping[str, Mapping[str, float | None]],
    n_questions_by_ds: Mapping[str, int],
) -> dict[str, dict[str, Any]]:
    """Organize average rewards by dataset for easier downstream use."""
    datasets: dict[str, dict[str, Any]] = {}
    for dataset, rewards in sorted(avg_rewards_by_dataset.items()):
        datasets[dataset] = {
            "avg_reward_per_model": dict(sorted(rewards.items())),
            "n_questions": int(n_questions_by_ds.get(dataset, 0)),
        }
    return datasets


def compute_avg_rewards_per_model(
    per_dataset_rewards: Mapping[str, Mapping[str, float | None]],
) -> dict[str, dict[str, float | None]]:
    """Flip {dataset -> model -> avg} into {model -> dataset -> avg}."""
    per_model: dict[str, dict[str, float | None]] = {}
    for dataset, rewards in per_dataset_rewards.items():
        for model, avg in rewards.items():
            per_model.setdefault(model, {})[dataset] = avg
    return per_model


def compute_per_opponent_stats(
    *,
    per_dataset_pairwise: Mapping[str, Mapping[tuple[str, str], tuple[float | None, int]]],
    model: str,
    opponent: str,
    n_questions_by_ds: Mapping[str, int],
    weight_policy: str,
    weight_cap: int,
) -> dict[str, Any] | None:
    """Build the `vs` entry for a model-opponent pair."""
    per_dataset_wr: dict[str, float] = {}
    weights: list[float] = []
    values: list[float] = []
    key = tuple(sorted((model, opponent)))
    for dataset in sorted(per_dataset_pairwise.keys()):
        wr, n_used = per_dataset_pairwise[dataset].get(key, (None, 0))
        if wr is None or n_used <= 0:
            continue
        adjusted = float(wr) if model == key[0] else 1.0 - float(wr)
        per_dataset_wr[dataset] = adjusted
        nq = max(n_questions_by_ds.get(dataset, 0), 0)
        weight = weight_of(nq, weight_policy, weight_cap)
        weights.append(weight)
        values.append(adjusted)
    if not per_dataset_wr:
        return None
    simple = sum(per_dataset_wr.values()) / len(per_dataset_wr)
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_total = sum(weights)
    weighted_mean = weighted_sum / weight_total if weight_total > 0 else simple
    return {
        "mean_winrate": {
            "simple_mean": simple,
            "weighted_mean": weighted_mean,
        },
        "per_dataset": per_dataset_wr,
        "n_datasets": len(per_dataset_wr),
    }


def _collect_opponents(model: str, models_by_ds: Mapping[str, Sequence[str]]) -> list[str]:
    opponents: set[str] = set()
    for models in models_by_ds.values():
        if model in models:
            opponents.update(m for m in models if m != model)
    return sorted(opponents)


def _aggregate_model_means(
    per_dataset_model_means: Mapping[str, Mapping[str, float]],
    n_questions_by_ds: Mapping[str, int],
    config: WinrateConfig,
) -> dict[str, dict[str, Any]]:
    """Aggregate per-model mean win rates across datasets."""
    all_models: set[str] = set()
    for ds_means in per_dataset_model_means.values():
        all_models.update(ds_means.keys())
    out: dict[str, dict[str, Any]] = {}
    for model in sorted(all_models):
        vals: list[tuple[str, float, float]] = []
        for dataset, means in per_dataset_model_means.items():
            if model not in means:
                continue
            nq = max(n_questions_by_ds.get(dataset, 0), 0)
            weight = weight_of(nq, config.weight_policy, config.weight_cap)
            vals.append((dataset, means[model], weight))
        if not vals:
            continue
        simple = sum(v for _, v, _ in vals) / len(vals)
        weight_total = sum(w for _, _, w in vals)
        weighted = sum(v * w for _, v, w in vals) / weight_total if weight_total > 0 else simple
        out[model] = {
            "simple_mean": simple,
            "weighted_mean": weighted,
            "n_datasets": len(vals),
        }
    return out


def _process_dataset(
    dataset_name: str,
    parquet_path: Path | str | Sequence[Path | str] | PLDataFrame | PLLazyFrame,
    config: WinrateConfig,
    *,
    include_set: set[str],
    exclude_set: set[str],
    target_models: Sequence[str],
    partial_datasets: str,
) -> tuple[DatasetStats | None, list[str]]:
    """Read and process a dataset, raising on failure and honoring selection policies."""
    try:
        lf = read_dataset_lazy(parquet_path)
        df_avg, n_questions = average_rollouts(lf)
        models_present = _models_present(df_avg)

        if include_set:
            missing_required = sorted(set(target_models) - set(models_present))
            if missing_required and partial_datasets == "strict":
                _emit_note(
                    f"Dropping dataset {dataset_name} (missing include models: {missing_required})."
                )
                return None, models_present

        if include_set:
            models_filtered = [m for m in target_models if m in models_present]
        else:
            models_filtered = [m for m in models_present if m not in exclude_set]
        if models_filtered:
            df_filtered = df_avg.filter(pl.col(MODEL_COL).is_in(models_filtered))
        else:
            df_filtered = df_avg.head(0)

        df_wide, _ = to_wide(df_filtered)
        if include_set:
            missing_cols = [m for m in target_models if m not in df_wide.columns]
            if missing_cols:
                df_wide = df_wide.with_columns([pl.lit(None).alias(m) for m in missing_cols])
            models = list(target_models)
        else:
            models = list(models_filtered)
        if models and EXAMPLE_ID_COL in df_wide.columns:
            df_wide = df_wide.select([EXAMPLE_ID_COL, *models])

        pairwise: dict[tuple[str, str], tuple[float | None, int]] = {}
        model_series = {m: df_wide.get_column(m) for m in models}
        for a, b in combinations(models, 2):
            wr, n_used = _pairwise_win_rate_series(
                model_series[a],
                model_series[b],
                missing_policy=config.missing_policy,
                epsilon=config.epsilon,
                min_common=config.min_common,
            )
            pairwise[(a, b)] = (wr, n_used)
        avg_reward_per_model = _mean_reward_per_model(df_avg, allowed=models)
        return (
            DatasetStats(
                pairwise=pairwise,
                n_questions=n_questions,
                models=models,
                avg_reward_per_model=avg_reward_per_model,
            ),
            models_present,
        )
    except Exception as exc:  # noqa: BLE001
        message = (
            f"Failed to process dataset {dataset_name} at {_format_parquet_source(parquet_path)}: {exc}"
        )
        _raise_user_error(message, exc)


def _mean_reward_per_model(df_avg: pl.DataFrame, allowed: Sequence[str] | None = None) -> dict[str, float | None]:
    """Average reward_mean per model inside a dataset."""
    if df_avg.is_empty() or MODEL_COL not in df_avg.columns:
        return {}
    allowed_set = {str(m) for m in allowed or []}
    grouped = (
        df_avg.group_by(MODEL_COL)  # type: ignore[arg-type]
        .agg(pl.col("reward_mean").mean().alias("avg_reward"))
        .sort(MODEL_COL)
    )
    rewards: dict[str, float | None] = {}
    for model, avg_reward in grouped.iter_rows():
        if allowed_set and str(model) not in allowed_set:
            continue
        rewards[str(model)] = None if avg_reward is None else float(avg_reward)
    return rewards


def _models_present(df_avg: pl.DataFrame) -> list[str]:
    if df_avg.is_empty() or MODEL_COL not in df_avg.columns:
        return []
    values = df_avg.get_column(MODEL_COL).unique()
    return sorted(str(value) for value in values if value is not None)


def _normalize_model_ids(values: Sequence[str] | None) -> set[str]:
    return {str(value).strip() for value in values or [] if str(value).strip()}


def _filter_models(
    models: Sequence[str],
    include: Sequence[str] | None,
    exclude: Sequence[str] | None,
) -> list[str]:
    include_set = _normalize_model_ids(include)
    exclude_set = _normalize_model_ids(exclude)
    filtered: list[str] = []
    for model in models:
        if exclude_set and model in exclude_set:
            continue
        if include_set and model not in include_set:
            continue
        filtered.append(model)
    return filtered


def _format_parquet_source(
    parquet_path: Path | str | Sequence[Path | str] | PLDataFrame | PLLazyFrame,
) -> str:
    if isinstance(parquet_path, (PLDataFrame, PLLazyFrame)):
        return "<in-memory frame>"
    if isinstance(parquet_path, (list, tuple)):
        parts: list[str] = []
        for item in parquet_path:
            if isinstance(item, (PLDataFrame, PLLazyFrame)):
                parts.append("<in-memory frame>")
            else:
                parts.append(str(Path(item)))
        return ", ".join(parts)
    return str(Path(parquet_path))


def _emit_note(message: str) -> None:
    console = _get_console()
    if console:
        console.print(f"[yellow]Note:[/] {message}")
    else:
        print(f"Note: {message}")


def _warn_user(message: str) -> None:
    console = _get_console()
    if console:
        console.print(f"[yellow]Warning:[/] {message}")
    else:
        print(f"Warning: {message}")


def _raise_user_error(message: str, exc: Exception | None = None) -> None:
    console = _get_console()
    if console:
        console.print(f"[red]Error:[/] {message}")
    else:
        print(f"Error: {message}")
    if exc is not None:
        raise ValueError(message) from exc
    raise ValueError(message)


def _get_console():  # type: ignore[no-untyped-def]
    return _cached_console()


@lru_cache(maxsize=1)
def _cached_console():  # type: ignore[no-untyped-def]
    try:
        from rich.console import Console
    except Exception:
        return None
    return Console()


def to_json(result: ModelCentricResult) -> dict[str, Any]:
    """Return a JSON-serializable dict."""
    return {"models": result.models, "datasets": result.datasets}


def write_json(payload: Mapping[str, Any], path: Path | str) -> None:
    """Write the JSON payload to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = [
    "WinrateConfig",
    "DatasetStats",
    "ModelCentricResult",
    "read_dataset_lazy",
    "average_rollouts",
    "to_wide",
    "pairwise_win_rate",
    "weight_of",
    "dataset_model_mean_winrates",
    "compute_winrates",
    "build_model_centric_result",
    "compute_avg_rewards_per_model",
    "compute_per_opponent_stats",
    "to_json",
    "write_json",
]
