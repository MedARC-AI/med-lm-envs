"""Manifest-driven discovery helpers for the JSONL → Parquet exporter."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

from pydantic import ValidationError

from medarc_verifiers.export.parquet.schemas import (
    ManifestJob,
    RunManifestModel,
    RunSummaryModel,
    SummaryEntry,
)

logger = logging.getLogger(__name__)

DEFAULT_STATUS = "unknown"


@dataclass(frozen=True, slots=True)
class RunManifestInfo:
    """Run-level metadata captured from run_manifest.json."""

    job_run_id: str
    run_name: str | None
    manifest_path: Path
    run_dir: Path
    created_at: str | None
    updated_at: str | None
    config_source: str | None
    config_checksum: str | None
    config_snapshot: Mapping[str, Any] | None
    run_summary_path: Path


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Resolved job entry enriched with filesystem paths and status information."""

    manifest: RunManifestInfo
    job_id: str
    job_name: str | None
    model_id: str | None
    manifest_env_id: str | None
    env_overrides: Mapping[str, Any]
    sampling_overrides: Mapping[str, Any]
    results_dir_name: str
    results_dir: Path
    metadata_path: Path
    results_path: Path
    summary_path: Path
    has_metadata: bool
    has_results: bool
    has_summary: bool
    status: str
    duration_seconds: float | None
    error: str | None


def discover_run_records(
    runs_dir: Path | str,
    *,
    filter_status: Sequence[str] | None = None,
) -> list[RunRecord]:
    """Return all discovered run records within the provided runs directory."""
    return list(
        iter_run_records(
            runs_dir=runs_dir,
            filter_status=filter_status,
        )
    )


def iter_run_records(
    runs_dir: Path | str,
    *,
    filter_status: Sequence[str] | None = None,
) -> Iterator[RunRecord]:
    """Yield run records for each job entry found under the runs directory."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        logger.debug("Runs directory %s does not exist; nothing to export.", runs_path)
        return

    normalized_status = _normalize_status_filter(filter_status)

    try:
        run_dirs = sorted(path for path in runs_path.iterdir() if path.is_dir())
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to list runs directory %s: %s", runs_path, exc)
        return

    for run_dir in run_dirs:
        manifest_info, job_entries = _load_manifest(run_dir)
        if manifest_info is None:
            continue
        summary_map = _load_run_summary(run_dir)
        for job_entry in job_entries:
            summary_entry = summary_map.get(job_entry.job_id)
            record = _build_run_record(manifest_info, job_entry, summary_entry)
            if record is None:
                continue
            if normalized_status and record.status not in normalized_status:
                continue
            yield record


def _build_run_record(
    manifest: RunManifestInfo,
    job_entry: ManifestJob,
    summary_entry: SummaryEntry | None,
) -> RunRecord | None:
    job_id = job_entry.job_id
    if not job_id:
        logger.debug("Skipping job entry without a valid job_id in %s", manifest.manifest_path)
        return None
    results_dir_name = job_entry.results_dir or job_id
    results_dir = manifest.run_dir / results_dir_name
    metadata_path = results_dir / "metadata.json"
    results_path = results_dir / "results.jsonl"
    summary_path = results_dir / "summary.json"

    status = (summary_entry.status or DEFAULT_STATUS).lower() if summary_entry else DEFAULT_STATUS
    duration_seconds = summary_entry.duration_seconds if summary_entry else None
    error = summary_entry.error if summary_entry else None

    return RunRecord(
        manifest=manifest,
        job_id=job_id,
        job_name=job_entry.job_name,
        model_id=job_entry.model_id,
        manifest_env_id=job_entry.env_id,
        env_overrides=job_entry.env_overrides,
        sampling_overrides=job_entry.sampling_overrides,
        results_dir_name=results_dir_name,
        results_dir=results_dir,
        metadata_path=metadata_path,
        results_path=results_path,
        summary_path=summary_path,
        has_metadata=metadata_path.exists(),
        has_results=results_path.exists(),
        has_summary=summary_path.exists(),
        status=status,
        duration_seconds=duration_seconds,
        error=error,
    )


def _load_manifest(run_dir: Path) -> tuple[RunManifestInfo | None, Sequence[ManifestJob]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        logger.debug("Skipping %s: no run_manifest.json present.", run_dir)
        return None, ()
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse manifest %s: %s", manifest_path, exc)
        return None, ()

    try:
        manifest_model = RunManifestModel.model_validate(manifest_payload)
    except ValidationError as exc:
        logger.warning("Manifest schema validation failed for %s: %s", manifest_path, exc)
        return None, ()

    job_run_id = manifest_model.run_id or run_dir.name

    manifest_info = RunManifestInfo(
        job_run_id=job_run_id,
        run_name=manifest_model.name,
        manifest_path=manifest_path,
        run_dir=run_dir,
        created_at=manifest_model.created_at,
        updated_at=manifest_model.updated_at,
        config_source=manifest_model.config_source,
        config_checksum=manifest_model.config_checksum,
        config_snapshot=manifest_model.config_snapshot,
        run_summary_path=run_dir / "run_summary.json",
    )

    if not manifest_model.jobs:
        logger.debug("Manifest %s has no jobs array.", manifest_path)
        return manifest_info, ()
    return manifest_info, manifest_model.jobs


def _load_run_summary(run_dir: Path) -> Mapping[str, SummaryEntry]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse run summary %s: %s", summary_path, exc)
        return {}
    try:
        summary_model = RunSummaryModel.model_validate(payload)
    except ValidationError as exc:
        logger.warning("Run summary schema validation failed for %s: %s", summary_path, exc)
        return {}
    summary: Dict[str, SummaryEntry] = {}
    for entry in summary_model.jobs:
        if entry.job_id:
            summary[entry.job_id] = entry
    return summary


def _normalize_status_filter(statuses: Sequence[str] | None) -> tuple[str, ...]:
    if not statuses:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for status in statuses:
        value = status.strip().lower()
        if not value or value in seen:
            continue
        normalized.append(value)
        seen.add(value)
    return tuple(normalized)


__all__ = [
    "RunManifestInfo",
    "RunRecord",
    "discover_run_records",
    "iter_run_records",
]
