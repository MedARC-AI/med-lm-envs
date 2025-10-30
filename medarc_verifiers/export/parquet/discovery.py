"""Manifest-driven discovery helpers for the JSONL → Parquet exporter."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

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
            record = _build_run_record(manifest_info, job_entry, summary_map)
            if record is None:
                continue
            if normalized_status and record.status not in normalized_status:
                continue
            yield record


def _build_run_record(
    manifest: RunManifestInfo,
    job_entry: Mapping[str, Any],
    summary_map: Mapping[str, Mapping[str, Any]],
) -> RunRecord | None:
    job_id = job_entry.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        logger.debug(
            "Skipping job entry without a valid job_id in %s",
            manifest.manifest_path,
        )
        return None
    results_dir_name = _coerce_results_dir(job_entry)
    results_dir = manifest.run_dir / results_dir_name
    metadata_path = results_dir / "metadata.json"
    results_path = results_dir / "results.jsonl"
    summary_path = results_dir / "summary.json"
    summary_entry = summary_map.get(job_id, {})

    status = summary_entry.get("status", DEFAULT_STATUS)
    if not isinstance(status, str):
        status = DEFAULT_STATUS
    status = status.lower()

    duration_seconds = _coerce_optional_float(summary_entry.get("duration_seconds"))
    error_value = summary_entry.get("error")
    error = str(error_value) if error_value is not None else None

    return RunRecord(
        manifest=manifest,
        job_id=job_id,
        job_name=_coerce_optional_str(job_entry.get("job_name")),
        model_id=_coerce_optional_str(job_entry.get("model_id")),
        manifest_env_id=_coerce_optional_str(job_entry.get("env_id")),
        env_overrides=_ensure_mapping(job_entry.get("env_overrides"), "env_overrides"),
        sampling_overrides=_ensure_mapping(
            job_entry.get("sampling_overrides"), "sampling_overrides"
        ),
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


def _load_manifest(run_dir: Path) -> tuple[RunManifestInfo | None, Sequence[Mapping[str, Any]]]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        logger.debug("Skipping %s: no run_manifest.json present.", run_dir)
        return None, ()
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse manifest %s: %s", manifest_path, exc)
        return None, ()

    job_run_id = manifest_payload.get("run_id") or run_dir.name
    run_name = manifest_payload.get("name") if isinstance(manifest_payload.get("name"), str) else None
    created_at = manifest_payload.get("created_at") if isinstance(manifest_payload.get("created_at"), str) else None
    updated_at = manifest_payload.get("updated_at") if isinstance(manifest_payload.get("updated_at"), str) else None
    config_source = manifest_payload.get("config_source") if isinstance(manifest_payload.get("config_source"), str) else None
    config_checksum = manifest_payload.get("config_checksum") if isinstance(manifest_payload.get("config_checksum"), str) else None
    config_snapshot = manifest_payload.get("config_snapshot")
    if isinstance(config_snapshot, MutableMapping):
        config_snapshot = dict(config_snapshot)
    else:
        config_snapshot = None

    manifest_info = RunManifestInfo(
        job_run_id=job_run_id,
        run_name=run_name,
        manifest_path=manifest_path,
        run_dir=run_dir,
        created_at=created_at,
        updated_at=updated_at,
        config_source=config_source,
        config_checksum=config_checksum,
        config_snapshot=config_snapshot,
        run_summary_path=run_dir / "run_summary.json",
    )

    jobs = manifest_payload.get("jobs")
    if not isinstance(jobs, Sequence):
        logger.debug("Manifest %s has no jobs array.", manifest_path)
        return manifest_info, ()
    return manifest_info, jobs


def _load_run_summary(run_dir: Path) -> Mapping[str, Mapping[str, Any]]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse run summary %s: %s", summary_path, exc)
        return {}
    entries = payload.get("jobs")
    if not isinstance(entries, Iterable):
        return {}
    summary: Dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        job_id = entry.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            continue
        summary[job_id] = entry
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


def _ensure_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    logger.debug("Expected mapping for %s but received %r; defaulting to empty mapping.", context, value)
    return {}


def _coerce_results_dir(job_entry: Mapping[str, Any]) -> str:
    results_dir = job_entry.get("results_dir")
    if isinstance(results_dir, str) and results_dir:
        return results_dir
    job_id = job_entry.get("job_id")
    if isinstance(job_id, str) and job_id:
        return job_id
    return "results"


def _coerce_optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _coerce_optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


__all__ = [
    "RunManifestInfo",
    "RunRecord",
    "discover_run_records",
    "iter_run_records",
]
