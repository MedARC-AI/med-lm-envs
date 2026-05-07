"""Discovery helpers for the exporter process subcommand."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Sequence

from pydantic import ValidationError

from medarc_verifiers.cli.eval_identity import (
    MEDARC_EVAL_METADATA_FILENAME,
    MEDARC_VARIANT_ID_KEY,
)
from medarc_verifiers.cli._manifest import (
    MANIFEST_FILENAME,
    ManifestJobEntry,
    RunManifestModel,
    _require_manifest_v3,
)

logger = logging.getLogger(__name__)

DEFAULT_STATUS = "unknown"
RESULTS_FILENAME = "results.jsonl"
METADATA_FILENAME = "metadata.json"


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
    version: int = 3
    artifacts_root: str = "."
    models: Mapping[str, Any] = field(default_factory=dict)
    env_templates: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Resolved job entry enriched with filesystem paths and status info."""

    manifest: RunManifestInfo
    job_id: str
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
    avg_reward: float | None
    num_examples: int | None
    rollouts_per_example: int | None
    row_count: int | None
    env_args: Mapping[str, Any]
    sampling_args: Mapping[str, Any]
    env_config: Mapping[str, Any] | None
    model_config: Mapping[str, Any] | None


def discover_run_records(
    runs_dir: Path | str,
    *,
    filter_status: Sequence[str] | None = None,
) -> list[RunRecord]:
    """Return all discovered run records within the provided runs directory."""
    return list(iter_run_records(runs_dir, filter_status=filter_status))


def iter_run_records(
    runs_dir: Path | str,
    *,
    filter_status: Sequence[str] | None = None,
) -> Iterator[RunRecord]:
    """Yield run records for each job entry found under the runs directory."""
    runs_path = Path(runs_dir)
    normalized_status = _normalize_status_filter(filter_status)
    emitted_results_dirs: set[Path] = set()

    if runs_path.exists():
        try:
            run_dirs = sorted(path for path in runs_path.iterdir() if path.is_dir())
        except OSError as exc:  # noqa: FBT003
            logger.warning("Failed to list runs directory %s: %s", runs_path, exc)
            run_dirs = []

        for run_dir in run_dirs:
            manifest_info, job_entries = _load_manifest(run_dir)
            if manifest_info is None:
                continue
            summary_map = _load_run_summary(run_dir)
            for job_entry in job_entries:
                summary_entry = summary_map.get(job_entry.job_id or "")
                record = _build_run_record(manifest_info, job_entry, summary_entry)
                if record is None:
                    continue
                emitted_results_dirs.add(_dedupe_key(record.results_dir))
                if normalized_status and record.status not in normalized_status:
                    continue
                yield record
    else:
        logger.debug("Runs directory %s does not exist; checking eval output roots.", runs_path)

    for evals_root in _candidate_evals_roots(runs_path):
        for record in _iter_eval_output_records(evals_root):
            results_key = _dedupe_key(record.results_dir)
            if results_key in emitted_results_dirs:
                continue
            emitted_results_dirs.add(results_key)
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

    results_dir_name, results_dir = _resolve_results_dir(
        job_entry.results_relpath,
        manifest.artifacts_root,
        job_id,
        manifest.run_dir,
    )
    results_dir_name, results_dir = _fallback_results_dir_if_missing(
        results_dir_name,
        results_dir,
        manifest.run_dir,
        job_id,
    )
    metadata_path = results_dir / METADATA_FILENAME
    results_path = results_dir / RESULTS_FILENAME
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
        avg_reward=job_entry.avg_reward,
        num_examples=job_entry.num_examples,
        rollouts_per_example=job_entry.rollouts_per_example,
        row_count=job_entry.row_count,
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
    stored_results_relpath: str | None,
    artifacts_root: str | None,
    job_id: str,
    run_dir: Path,
) -> tuple[str, Path]:
    """Resolve a job's results directory from v3 manifest artifact fields."""
    if stored_results_relpath:
        rel = Path(stored_results_relpath)
        base = run_dir / str(artifacts_root or ".")
        candidate_file = (base / rel).resolve()
        # v3 stores results_relpath to results.jsonl; derive the containing directory.
        candidate_dir = candidate_file.parent if candidate_file.name == RESULTS_FILENAME else candidate_file
        return candidate_dir.name, candidate_dir

    # Backward-compatible fallback for malformed v3 payloads missing relpaths.
    fallback = (run_dir / job_id).resolve()
    return job_id, fallback


def _fallback_results_dir_if_missing(
    results_dir_name: str,
    results_dir: Path,
    run_dir: Path,
    job_id: str,
) -> tuple[str, Path]:
    metadata_path = results_dir / METADATA_FILENAME
    results_path = results_dir / RESULTS_FILENAME
    if metadata_path.exists() or results_path.exists():
        return results_dir_name, results_dir
    fallback = (run_dir / job_id).resolve()
    fallback_metadata = fallback / METADATA_FILENAME
    fallback_results = fallback / RESULTS_FILENAME
    if fallback_metadata.exists() or fallback_results.exists():
        logger.warning(
            "Manifest results path missing for job '%s'; falling back to run-relative directory '%s'.",
            job_id,
            fallback,
        )
        return job_id, fallback
    return results_dir_name, results_dir


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

    _require_manifest_v3(manifest_payload, path=manifest_path)

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
        version=int(manifest_model.version),
        artifacts_root=str(getattr(manifest_model, "artifacts_root", ".") or "."),
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


def _candidate_evals_roots(runs_path: Path) -> tuple[Path, ...]:
    candidates: list[Path] = []
    if runs_path.name == "evals":
        candidates.append(runs_path)
    candidates.append(runs_path / "evals")
    candidates.append(runs_path.parent / "evals")

    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        key = _dedupe_key(candidate)
        if key in seen or not candidate.exists() or not candidate.is_dir():
            continue
        seen.add(key)
        roots.append(candidate)
    return tuple(roots)


def _iter_eval_output_records(evals_root: Path) -> Iterator[RunRecord]:
    """Yield synthetic run records for upstream eval output directories."""
    try:
        results_paths = sorted(evals_root.rglob(RESULTS_FILENAME))
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to scan eval outputs under %s: %s", evals_root, exc)
        return

    helper_entries = _load_model_helper_entries(evals_root)
    seen: set[Path] = set()
    for results_path in results_paths:
        results_dir = results_path.parent
        if results_path.name == RESULTS_FILENAME and results_dir.name == "__pycache__":
            continue
        key = _dedupe_key(results_dir)
        if key in seen:
            continue
        seen.add(key)
        metadata_path = results_dir / METADATA_FILENAME
        if not metadata_path.exists():
            continue
        record = _build_eval_output_record(evals_root, results_dir, helper_entries.get(_dedupe_key(results_dir)))
        if record is not None:
            yield record


def _build_eval_output_record(
    evals_root: Path,
    results_dir: Path,
    helper_entry: Mapping[str, Any] | None = None,
) -> RunRecord | None:
    metadata_path = results_dir / METADATA_FILENAME
    metadata_payload = _read_metadata_payload(metadata_path)
    if metadata_payload is None:
        return None

    layout = _infer_eval_output_layout(evals_root, results_dir, metadata_payload, helper_entry)
    updated_at = _path_timestamp(metadata_path)
    job_run_id = layout["job_run_id"]
    job_id = layout["job_id"]
    model_id = layout["model_id"]
    env_id = layout["env_id"]

    manifest = RunManifestInfo(
        job_run_id=job_run_id,
        run_name=job_run_id,
        summary_completed=1,
        summary_total=1,
        summary_total_known=True,
        manifest_path=metadata_path,
        run_dir=results_dir,
        created_at=updated_at,
        updated_at=updated_at,
        config_source=None,
        config_checksum=None,
        run_summary_path=results_dir / "summary.json",
        models={model_id: {"sampling_args": _mapping_or_empty(metadata_payload.get("sampling_args"))}},
        env_templates={env_id: {"module": env_id}},
    )

    env_args = _mapping_or_empty(metadata_payload.get("env_args"))
    sampling_args = _mapping_or_empty(metadata_payload.get("sampling_args"))
    row_count = _count_results_rows(results_dir / RESULTS_FILENAME)
    return RunRecord(
        manifest=manifest,
        job_id=job_id,
        model_id=model_id,
        manifest_env_id=env_id,
        results_dir_name=results_dir.name,
        results_dir=results_dir,
        metadata_path=metadata_path,
        results_path=results_dir / RESULTS_FILENAME,
        summary_path=results_dir / "summary.json",
        has_metadata=True,
        has_results=True,
        has_summary=(results_dir / "summary.json").exists(),
        status="completed",
        duration_seconds=None,
        reason=None,
        started_at=None,
        ended_at=None,
        avg_reward=_float_or_none(metadata_payload.get("avg_reward")),
        num_examples=_int_or_none(metadata_payload.get("num_examples")),
        rollouts_per_example=_int_or_none(metadata_payload.get("rollouts_per_example")),
        row_count=row_count,
        env_args=env_args,
        sampling_args=sampling_args,
        env_config={
            "id": env_id,
            "module": env_id,
            "variant_id": layout.get("variant_id"),
        },
        model_config={"sampling_args": sampling_args},
    )


def _infer_eval_output_layout(
    evals_root: Path,
    results_dir: Path,
    metadata_payload: Mapping[str, Any],
    helper_entry: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    try:
        parts = results_dir.relative_to(evals_root).parts
    except ValueError:
        parts = results_dir.parts

    metadata_env_id = _string_or_none(metadata_payload.get("env_id"))
    metadata_model = _string_or_none(metadata_payload.get("model"))
    helper_env_id = _string_or_none(helper_entry.get("env_id") if helper_entry else None)
    helper_variant_id = _string_or_none(helper_entry.get("variant_id") if helper_entry else None)
    parent_name = results_dir.parent.name
    if "--" in parent_name and len(parts) >= 2:
        env_from_parent, model_from_parent = parent_name.split("--", 1)
        env_id = helper_env_id or metadata_env_id or env_from_parent
        model_id = metadata_model or model_from_parent
        job_run_id = results_dir.name
        variant_id = helper_variant_id
    else:
        model_id = metadata_model or (parts[0] if len(parts) >= 1 else "unknown")
        env_id = helper_env_id or metadata_env_id or (parts[1] if len(parts) >= 2 else results_dir.name)
        variant_id = helper_variant_id or (
            parts[2] if len(parts) >= 3 else _string_or_none(metadata_payload.get(MEDARC_VARIANT_ID_KEY))
        )
        job_run_id = "::".join(part for part in (model_id, env_id, variant_id) if part)

    return {
        "job_run_id": job_run_id,
        "job_id": results_dir.name,
        "model_id": model_id,
        "env_id": env_id,
        "variant_id": variant_id or "",
    }


def _load_model_helper_entries(evals_root: Path) -> dict[Path, Mapping[str, Any]]:
    entries: dict[Path, Mapping[str, Any]] = {}
    try:
        helper_paths = sorted(evals_root.glob(f"*/{MEDARC_EVAL_METADATA_FILENAME}"))
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to scan eval metadata helpers under %s: %s", evals_root, exc)
        return entries

    for helper_path in helper_paths:
        payload = _read_metadata_payload(helper_path)
        if payload is None:
            continue
        raw_outputs = payload.get("outputs")
        if not isinstance(raw_outputs, Mapping):
            continue
        model_dir = helper_path.parent
        for key, raw_entry in raw_outputs.items():
            if not isinstance(raw_entry, Mapping):
                continue
            raw_results_path = raw_entry.get("results_path") or key
            if not isinstance(raw_results_path, str) or not raw_results_path:
                continue
            relative_results_path = Path(raw_results_path)
            if relative_results_path.is_absolute():
                continue
            results_dir = (model_dir / relative_results_path).resolve()
            try:
                results_dir.relative_to(model_dir.resolve())
            except ValueError:
                continue
            if not (results_dir / METADATA_FILENAME).exists() or not (results_dir / RESULTS_FILENAME).exists():
                continue
            entries[_dedupe_key(results_dir)] = raw_entry
    return entries


def _read_metadata_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # noqa: FBT003
        logger.warning("Failed to parse eval metadata %s: %s", path, exc)
        return None
    if not isinstance(payload, Mapping):
        logger.warning("Invalid eval metadata payload type for %s: expected JSON object.", path)
        return None
    return dict(payload)


def _dedupe_key(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _path_timestamp(path: Path) -> str:
    try:
        timestamp = path.stat().st_mtime
    except OSError:
        return ""
    return datetime.fromtimestamp(timestamp, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _count_results_rows(path: Path) -> int | None:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    count += 1
    except OSError:
        return None
    return count


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
