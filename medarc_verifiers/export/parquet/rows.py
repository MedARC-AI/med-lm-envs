"""Row enrichment helpers for the JSONL → Parquet export pipeline."""

from __future__ import annotations

import logging
import re
from typing import Any, Collection, Mapping

from verifiers.scripts.tui import RunInfo, load_run_results

from medarc_verifiers.export.parquet.metadata import NormalizedMetadata

logger = logging.getLogger(__name__)

ROLL_OUT_SUFFIX_PATTERN = re.compile(r"-r(?P<index>\d+)$")


def load_enriched_rows(
    metadata: NormalizedMetadata,
    *,
    drop_fields: Collection[str] | None = None,
) -> list[dict[str, Any]]:
    """Load results.jsonl and attach run-level metadata to each row."""
    record = metadata.record
    if not record.has_results:
        logger.debug("Run %s missing results.jsonl; skipping.", record.job_id)
        return []

    run_info = RunInfo(
        env_id=metadata.base_env_id or record.manifest_env_id or "",
        model=metadata.model or record.model_id or "",
        run_id=record.job_id,
        path=record.results_dir,
        metadata=dict(metadata.raw_metadata),
    )
    try:
        rows = load_run_results(run_info)
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to load results for %s: %s", record.results_path, exc)
        return []

    rollout_index = _compute_rollout_index(record.manifest_env_id)
    intra_rollout_fn = _build_intra_rollout_resolver(
        len(rows),
        metadata.num_examples,
        metadata.rollouts_per_example,
    )

    enriched_rows: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping):
            logger.debug("Skipping non-mapping row at position %d in %s.", position, record.results_path)
            continue
        enriched = dict(row)
        if drop_fields:
            for field in drop_fields:
                enriched.pop(field, None)
        # Remove known heavy or redundant columns that are handled elsewhere.
        enriched.pop("info", None)
        enriched.pop("sampling_args", None)
        enriched.update(metadata.env_columns)
        enriched.update(
            {
                "env_id": metadata.base_env_id,
                "job_id": record.manifest.job_run_id,
                "model": metadata.model or record.model_id,
                "rollout_index": rollout_index or intra_rollout_fn(position),
                "position": position,
                "status": record.status,
                "error": record.error,
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def _compute_rollout_index(manifest_env_id: str | None) -> int:
    if not manifest_env_id:
        return 0
    match = ROLL_OUT_SUFFIX_PATTERN.search(manifest_env_id)
    if not match:
        return 0
    try:
        return int(match.group("index"))
    except (TypeError, ValueError):
        return 0


def _build_intra_rollout_resolver(
    results_count: int,
    num_examples: int | None,
    rollouts_per_example: int | None,
):
    should_compute = (
        isinstance(num_examples, int)
        and num_examples > 0
        and isinstance(rollouts_per_example, int)
        and rollouts_per_example > 1
        and results_count >= num_examples
    )
    if not should_compute:
        return lambda _: None

    def _resolver(position: int) -> int:
        return position // num_examples

    return _resolver


__all__ = ["load_enriched_rows"]
