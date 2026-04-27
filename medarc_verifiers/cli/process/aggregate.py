"""Aggregation helpers for exporter process pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from medarc_verifiers.cli.process.metadata import RunIdentity
from medarc_verifiers.cli.process.rollout import extract_rollout_index

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AggregatedEnvRows:
    """Container describing all rows for a single environment."""

    env_id: str
    base_env_id: str
    model_id: str | None
    rows: list[Mapping[str, Any]]
    column_names: tuple[str, ...]
    job_run_ids: tuple[str, ...]


def aggregate_rows_by_env(
    rows: Iterable[Mapping[str, Any]],
    *,
    identities: Iterable[RunIdentity] | None = None,
) -> list[AggregatedEnvRows]:
    """Group enriched rows by (model_id, base_env_id), capturing unioned schemas."""
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    identity_list = list(identities or ())
    fake_rollout_groups = {
        (identity.model_id, identity.output_env_id) for identity in identity_list if identity.rollout_index is not None
    }

    for row in rows:
        base_env_id = str(row.get("base_env_id") or row.get("env_id") or "")
        env_id = str(row.get("env_id") or base_env_id)
        model_id = str(row.get("model_id") or "unknown")
        group_key = (model_id, base_env_id or env_id)
        if not group_key[1]:  # no env identifier
            logger.debug("Skipping row without env identifiers.")
            continue

        if group_key not in groups:
            groups[group_key] = {
                "env_id": env_id if env_id else base_env_id,
                "base_env_id": base_env_id,
                "model_id": model_id,
                "rows": [],
                "column_names": set(),
                "job_run_ids": set(),
            }

        group = groups[group_key]
        if not group["env_id"] and env_id:
            group["env_id"] = env_id
        if not group["base_env_id"] and base_env_id:
            group["base_env_id"] = base_env_id
        if not group["model_id"] and model_id:
            group["model_id"] = model_id
        group["rows"].append(row)
        group["column_names"].update(row.keys())
        job_run_id = row.get("job_run_id")
        if job_run_id:
            group["job_run_ids"].add(str(job_run_id))

    aggregated: list[AggregatedEnvRows] = []
    for key in sorted(groups):
        group = groups[key]
        # Preserve rollout_index as assigned during row loading. Only normalize rollout_index values when
        # processing "fake rollouts" that are created by running separate jobs with rollout suffixes
        # (e.g., env-a-rollout7) and then combining them under a shared base_env_id.
        normalized_rows: list[Mapping[str, Any]] = list(group["rows"])  # shallow copy
        if key in fake_rollout_groups:
            _ensure_rollout_index_from_identities(
                normalized_rows,
                identities=identity_list,
                model_id=group["model_id"],
                base_env_id=group["base_env_id"] or key[1],
            )
            _normalize_rollout_indices(normalized_rows)
        elif _group_uses_rollout_suffixes(normalized_rows, base_env_id=group["base_env_id"] or key[1]):
            _ensure_rollout_index_from_suffix(normalized_rows, base_env_id=group["base_env_id"] or key[1])
            _normalize_rollout_indices(normalized_rows)
        candidate_env_id = group["env_id"] or group["base_env_id"] or ""
        aggregated.append(
            AggregatedEnvRows(
                env_id=candidate_env_id,
                base_env_id=group["base_env_id"] or key[1],
                model_id=group["model_id"],
                rows=normalized_rows,
                column_names=tuple(sorted(group["column_names"])),
                job_run_ids=tuple(sorted(group["job_run_ids"])),
            )
        )
    return aggregated


def _ensure_rollout_index_from_identities(
    rows: list[Mapping[str, Any]],
    *,
    identities: list[RunIdentity],
    model_id: str,
    base_env_id: str,
) -> None:
    rollout_by_manifest_env: dict[str, int] = {}
    for identity in identities:
        if identity.model_id != model_id or identity.output_env_id != base_env_id:
            continue
        if identity.rollout_index is None:
            continue
        rollout_by_manifest_env[identity.manifest_env_id] = identity.rollout_index

    if not rollout_by_manifest_env:
        return

    for row in rows:
        value = row.get("rollout_index")
        if _coerce_rollout_index(value) is not None:
            continue
        manifest_env_id = row.get("manifest_env_id")
        if not isinstance(manifest_env_id, str):
            continue
        resolved = rollout_by_manifest_env.get(manifest_env_id)
        if resolved is None:
            continue
        try:
            row["rollout_index"] = resolved
        except TypeError:
            continue


def _group_uses_rollout_suffixes(rows: list[Mapping[str, Any]], *, base_env_id: str) -> bool:
    for row in rows:
        manifest_env_id = row.get("manifest_env_id")
        if not isinstance(manifest_env_id, str) or not manifest_env_id:
            continue
        row_base_env_id = str(row.get("base_env_id") or base_env_id or "")
        if row_base_env_id and manifest_env_id != row_base_env_id:
            return True
    return False


def _ensure_rollout_index_from_suffix(rows: list[Mapping[str, Any]], *, base_env_id: str) -> None:
    for row in rows:
        value = row.get("rollout_index")
        if _coerce_rollout_index(value) is not None:
            continue
        manifest_env_id = row.get("manifest_env_id")
        if not isinstance(manifest_env_id, str) or not manifest_env_id:
            continue
        row_base_env_id = str(row.get("base_env_id") or base_env_id or "")
        if not row_base_env_id or manifest_env_id == row_base_env_id:
            continue
        derived_index = extract_rollout_index(manifest_env_id)
        if derived_index <= 0:
            continue
        try:
            row["rollout_index"] = derived_index
        except TypeError:
            continue


def _coerce_rollout_index(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_rollout_indices(rows: list[Mapping[str, Any]]) -> None:
    values: list[int] = []
    for row in rows:
        coerced = _coerce_rollout_index(row.get("rollout_index"))
        if coerced is None:
            continue
        values.append(coerced)
    if not values:
        return
    mapping = {val: idx for idx, val in enumerate(sorted(set(values)))}
    for row in rows:
        coerced = _coerce_rollout_index(row.get("rollout_index"))
        if coerced is None:
            continue
        normalized = mapping.get(coerced)
        if normalized is None:
            continue
        try:
            row["rollout_index"] = normalized
        except TypeError:
            # Ignore non-mutable mappings.
            continue


__all__ = ["AggregatedEnvRows", "aggregate_rows_by_env"]
