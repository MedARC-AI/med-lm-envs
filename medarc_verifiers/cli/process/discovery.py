"""Discovery helpers for the exporter process subcommand."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from pydantic import ValidationError

from medarc_verifiers.cli._manifest import (
    MANIFEST_FILENAME,
    ManifestJobEntry,
    RunManifestModel,
    _require_manifest_v2,
)
from medarc_verifiers.utils.pathing import from_project_relative

logger = logging.getLogger(__name__)

DEFAULT_STATUS = "unknown"
_COMPLETED_STATUSES = {"completed", "succeeded", "success"}


@dataclass(frozen=True, slots=True)
class RunManifestInfo:
    """Metadata describing a run directory and its manifest."""

    job_run_id: str
    run_name: str | None
    summary_completed: int
    summary_total: int
    summary_total_known: bool
    manifest_path: Path
    run_dir: Path
    created_at: str | None
    updated_at: str | None
    config_source: str | None
    config_checksum: str | None
    run_summary_path: Path
    models: Mapping[str, Any] = field(default_factory=dict)
    env_templates: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Resolved job entry enriched with filesystem paths and status info."""

    manifest: RunManifestInfo
    job_id: str
    job_name: str | None
    model_id: str | None
    manifest_env_id: str | None
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
    reason: str | None
    started_at: str | None
    ended_at: str | None
    num_examples: int | None
    rollouts_per_example: int | None
    env_args: Mapping[str, Any]
    sampling_args: Mapping[str, Any]
    env_config: Mapping[str, Any] | None
    model_config: Mapping[str, Any] | None


def discover_run_records(
    runs_dir: Path | str,
    *,
    filter_status: Sequence[str] | None = None,
    only_complete_runs: bool = False,
) -> list[RunRecord]:
    """Return all discovered run records within the provided runs directory."""
    return list(iter_run_records(runs_dir, filter_status=filter_status, only_complete_runs=only_complete_runs))


def iter_run_records(
    runs_dir: Path | str,
    *,
    filter_status: Sequence[str] | None = None,
    only_complete_runs: bool = False,
) -> Iterator[RunRecord]:
    """Yield run records for each job entry found under the runs directory."""
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        logger.debug("Runs directory %s does not exist; nothing to process.", runs_path)
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
        if (
            only_complete_runs
            and manifest_info.summary_total_known
            and manifest_info.summary_completed != manifest_info.summary_total
        ):
            # Skip entire run if not fully completed
            continue
        summary_map = _load_run_summary(run_dir)
        for job_entry in job_entries:
            summary_entry = summary_map.get(job_entry.job_id or "")
            record = _build_run_record(manifest_info, job_entry, summary_entry)
            if record is None:
                continue
            if normalized_status and record.status not in normalized_status:
                continue
            yield record


def _build_run_record(
    manifest: RunManifestInfo,
    job_entry: ManifestJobEntry,
    summary_entry: Mapping[str, Any] | None,
) -> RunRecord | None:
    job_id = job_entry.job_id
    if not job_id:
        logger.debug("Skipping job entry without a valid job_id in %s", manifest.manifest_path)
        return None

    results_dir_name, results_dir = _resolve_results_dir(job_entry.results_dir, job_id, manifest.run_dir)
    metadata_path = results_dir / "metadata.json"
    results_path = results_dir / "results.jsonl"
    summary_path = results_dir / "summary.json"

    status = DEFAULT_STATUS
    duration_seconds = None
    reason: str | None = None

    if summary_entry:
        status = (str(summary_entry.get("status", DEFAULT_STATUS)) or DEFAULT_STATUS).lower()
        duration_seconds = summary_entry.get("duration_seconds")
        reason = summary_entry.get("error")
    elif job_entry.status:
        status = job_entry.status.lower()
        reason = job_entry.reason

    model_config = _ensure_mapping(manifest.models.get(job_entry.model_id) if manifest.models else {})
    env_template = _ensure_mapping(
        manifest.env_templates.get(job_entry.env_template_id) if manifest.env_templates else {}
    )
    env_config = dict(env_template)
    if "module" not in env_config and job_entry.env_id:
        env_config["module"] = job_entry.env_id
    env_config["id"] = job_entry.env_variant_id
    env_config["env_args"] = job_entry.env_args
    env_args = _ensure_mapping(job_entry.env_args)
    sampling_args = _ensure_mapping(job_entry.sampling_args or model_config.get("sampling_args"))

    return RunRecord(
        manifest=manifest,
        job_id=job_id,
        job_name=job_entry.job_name,
        model_id=job_entry.model_id,
        manifest_env_id=job_entry.env_id,
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
        reason=reason or job_entry.reason,
        started_at=job_entry.started_at,
        ended_at=job_entry.ended_at,
        num_examples=job_entry.num_examples,
        rollouts_per_example=job_entry.rollouts_per_example,
        env_args=env_args,
        sampling_args=sampling_args,
        env_config=env_config,
        model_config=model_config,
    )


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _resolve_results_dir(
    stored_value: str | None,
    job_id: str,
    run_dir: Path,
) -> tuple[str, Path]:
    """Interpret manifest results_dir values which may be job-relative, rooted at runs/, or absolute."""
    name = stored_value or job_id
    candidate = Path(name)
    if candidate.is_absolute():
        return name, candidate
    if candidate.parts and candidate.parts[0] == "runs":
        resolved = from_project_relative(candidate)
        return name, resolved
    return name, (run_dir / candidate).resolve()


def _load_manifest(run_dir: Path) -> tuple[RunManifestInfo | None, Sequence[ManifestJobEntry]]:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        logger.debug("Skipping %s: no %s present.", run_dir, MANIFEST_FILENAME)
        return None, ()
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse manifest %s: %s", manifest_path, exc)
        return None, ()

    _require_manifest_v2(manifest_payload, path=manifest_path)

    try:
        manifest_model = RunManifestModel.model_validate(manifest_payload)
    except ValidationError as exc:
        logger.warning("Manifest schema validation failed for %s: %s", manifest_path, exc)
        return None, ()

    job_run_id = manifest_model.run_id or run_dir.name
    summary_payload = manifest_model.summary or {}
    try:
        completed_count = int(summary_payload.get("completed", 0))
    except Exception:
        completed_count = 0
    total_known = False
    if "total" in summary_payload:
        try:
            total_count = int(summary_payload.get("total", 0))
        except Exception:
            total_count = 0
        total_known = total_count > 0 or not manifest_model.jobs
    else:
        total_count = 0
    if total_count == 0 and manifest_model.jobs:
        total_count = len(manifest_model.jobs)
        total_known = True

    manifest_info = RunManifestInfo(
        job_run_id=job_run_id,
        run_name=manifest_model.name,
        summary_completed=completed_count,
        summary_total=total_count,
        summary_total_known=total_known,
        manifest_path=manifest_path,
        run_dir=run_dir,
        created_at=manifest_model.created_at,
        updated_at=manifest_model.updated_at,
        config_source=manifest_model.config_source,
        config_checksum=manifest_model.config_checksum,
        run_summary_path=run_dir / "run_summary.json",
        models=manifest_model.models or {},
        env_templates=manifest_model.env_templates or {},
    )

    if not manifest_model.jobs:
        logger.debug("Manifest %s has no jobs array.", manifest_path)
        return manifest_info, ()
    return manifest_info, manifest_model.jobs


def _load_run_summary(run_dir: Path) -> Mapping[str, Mapping[str, Any]]:
    summary_path = run_dir / "run_summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse run summary %s: %s", summary_path, exc)
        return {}
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return {}
    summary: Dict[str, Mapping[str, Any]] = {}
    for entry in jobs:
        job_id = entry.get("job_id") if isinstance(entry, Mapping) else None
        if not job_id:
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


__all__ = [
    "RunManifestInfo",
    "RunRecord",
    "discover_run_records",
    "iter_run_records",
]
