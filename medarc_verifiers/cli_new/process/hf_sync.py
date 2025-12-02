"""Hugging Face dataset sync helpers for exporter process pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import polars as pl
import pyarrow.parquet as pq
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from medarc_verifiers.cli_new.process.writer import EnvWriteSummary

logger = logging.getLogger(__name__)

ROW_KEY = ("job_run_id", "example_id", "rollout_index")
ENTITY_KEY = ("base_env_id", "example_id", "rollout_index")


@dataclass(slots=True)
class HFSyncConfig:
    repo_id: str | None
    merge_strategy: str = "append"  # append|update|replace
    branch: str | None = None
    private: bool = False
    dry_run: bool = False
    token: str | None = None


@dataclass(slots=True)
class SplitMergeStats:
    env_id: str
    base_env_id: str
    new_rows: int
    existing_rows: int
    merged_rows: int
    action: str


@dataclass(slots=True)
class HFMergeSummary:
    repo_id: str
    strategy: str
    total_rows: int
    splits: Sequence[SplitMergeStats]


def sync_to_hub(
    env_summaries: Sequence[EnvWriteSummary],
    config: HFSyncConfig,
) -> HFMergeSummary | None:
    """Merge local parquet exports into a HF dataset according to the given strategy."""
    if not config.repo_id:
        logger.debug("HF sync skipped: no repo_id provided.")
        return None
    if not env_summaries:
        logger.debug("HF sync skipped: no environment summaries available.")
        return None
    if all(summary.dry_run for summary in env_summaries):
        logger.debug("HF sync skipped: only dry-run summaries available.")
        return None

    new_splits = _load_local_splits(env_summaries)
    if not new_splits:
        logger.debug("HF sync skipped: no local splits to sync.")
        return None
    base_lookup = {
        summary.env_id: summary.base_env_id for summary in env_summaries if not summary.dry_run and summary.env_id
    }

    try:
        remote_splits = _load_remote_dataset(config) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load remote dataset %s: %s", config.repo_id, exc)
        remote_splits = {}

    merged_splits, split_stats = _merge_splits(
        new_splits,
        remote_splits,
        strategy=config.merge_strategy,
        base_lookup=base_lookup,
    )

    summary = HFMergeSummary(
        repo_id=config.repo_id,
        strategy=config.merge_strategy,
        total_rows=sum(stat.merged_rows for stat in split_stats),
        splits=split_stats,
    )

    if config.dry_run:
        logger.debug("HF sync dry-run; skipping push.")
        return summary

    dataset_dict = DatasetDict(merged_splits)
    _push_dataset(dataset_dict, config, summary)
    return summary


def _load_local_splits(env_summaries: Sequence[EnvWriteSummary]) -> dict[str, Dataset]:
    splits: dict[str, Dataset] = {}
    for summary in env_summaries:
        if summary.dry_run:
            continue
        path = summary.output_path
        if not path.exists():
            logger.debug("Skipping HF split %s: parquet path %s missing.", summary.env_id, path)
            continue
        table = pq.read_table(path)
        dataset = Dataset.from_pandas(table.to_pandas(), preserve_index=False)
        splits[summary.env_id] = dataset
    return splits


def _merge_splits(
    new_splits: Mapping[str, Dataset],
    existing_splits: Mapping[str, Dataset],
    *,
    strategy: str,
    base_lookup: Mapping[str, str] | None = None,
) -> tuple[dict[str, Dataset], list[SplitMergeStats]]:
    merged: dict[str, Dataset] = {}
    stats: list[SplitMergeStats] = []
    all_keys = set(existing_splits) | set(new_splits)

    for key in sorted(all_keys):
        new_ds = new_splits.get(key)
        old_ds = existing_splits.get(key)
        merged_ds = _merge_split(new_ds, old_ds, strategy=strategy)
        merged[key] = merged_ds
        stats.append(
            SplitMergeStats(
                env_id=key,
                base_env_id=(base_lookup.get(key) if base_lookup else None) or key,
                new_rows=len(new_ds) if new_ds is not None else 0,
                existing_rows=len(old_ds) if old_ds is not None else 0,
                merged_rows=len(merged_ds),
                action=strategy if new_ds is not None else "carry",
            )
        )
    return merged, stats


def _merge_split(
    new_ds: Dataset | None,
    old_ds: Dataset | None,
    *,
    strategy: str,
) -> Dataset:
    normalized_strategy = strategy.lower()
    if old_ds is None:
        if new_ds is None:
            raise ValueError("At least one dataset must be provided for merge.")
        return new_ds
    if new_ds is None:
        return old_ds
    if normalized_strategy == "replace":
        return new_ds

    combined = concatenate_datasets([old_ds, new_ds])
    pl_df = pl.from_pandas(combined.to_pandas())

    if normalized_strategy == "append":
        deduped = _dedupe_rows(pl_df, ROW_KEY)
        return Dataset.from_pandas(deduped.to_pandas(), preserve_index=False)
    if normalized_strategy == "update":
        deduped = _dedupe_rows(
            pl_df,
            ENTITY_KEY,
            sort_order=[
                ("started_at", True),
                ("job_run_id", True),
            ],
        )
        return Dataset.from_pandas(deduped.to_pandas(), preserve_index=False)

    raise ValueError(f"Unsupported merge strategy '{strategy}'.")


def _dedupe_rows(
    df: pl.DataFrame,
    keys: Iterable[str],
    sort_order: Sequence[tuple[str, bool]] | None = None,
) -> pl.DataFrame:
    key_list = [key for key in keys if key in df.columns]
    if not key_list:
        logger.debug("Dedupe skipped: required keys missing.")
        return df

    if sort_order:
        columns: list[str] = []
        descending: list[bool] = []
        for column, desc in sort_order:
            if column in df.columns:
                columns.append(column)
                descending.append(desc)
        if columns:
            df = df.sort(columns, descending=descending)
    df = df.unique(subset=key_list, keep="first")
    return df


def _load_remote_dataset(config: HFSyncConfig) -> Mapping[str, Dataset]:
    try:
        dataset_dict = load_dataset(
            config.repo_id,
            revision=config.branch,
            token=config.token,
        )
    except FileNotFoundError:
        return {}
    if isinstance(dataset_dict, DatasetDict):
        return dict(dataset_dict.items())
    # Single split scenario
    return {"default": dataset_dict}  # type: ignore[return-value]


def load_remote_dataset(config: HFSyncConfig) -> Mapping[str, Dataset] | None:
    """Public helper to load a HF dataset as split -> Dataset mapping."""
    return _load_remote_dataset(config)


def _push_dataset(dataset_dict: DatasetDict, config: HFSyncConfig, summary: HFMergeSummary) -> None:
    commit_message = (
        f"medarc process: {len(summary.splits)} envs, {summary.total_rows} rows, strategy={summary.strategy}"
    )
    dataset_dict.push_to_hub(
        config.repo_id,
        private=config.private,
        branch=config.branch,
        token=config.token,
        commit_message=commit_message,
    )


__all__ = [
    "HFMergeSummary",
    "HFSyncConfig",
    "load_remote_dataset",
    "SplitMergeStats",
    "sync_to_hub",
]
