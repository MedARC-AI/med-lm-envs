"""Discovery helpers for the exporter process subcommand."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from medarc_verifiers.cli.eval_identity import MEDARC_VARIANT_ID_KEY

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
    if not runs_path.exists():
        logger.debug("Runs directory %s does not exist.", runs_path)
        return

    for record in _iter_eval_output_records(runs_path):
        if normalized_status and record.status not in normalized_status:
            continue
        yield record


def _iter_eval_output_records(evals_root: Path) -> Iterator[RunRecord]:
    """Yield synthetic run records for upstream eval output directories."""
    try:
        results_paths = sorted(evals_root.rglob(RESULTS_FILENAME))
    except OSError as exc:  # noqa: FBT003
        logger.warning("Failed to scan eval outputs under %s: %s", evals_root, exc)
        return

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
        record = _build_eval_output_record(evals_root, results_dir)
        if record is not None:
            yield record


def _build_eval_output_record(
    evals_root: Path,
    results_dir: Path,
) -> RunRecord | None:
    metadata_path = results_dir / METADATA_FILENAME
    metadata_payload = _read_metadata_payload(metadata_path)
    if metadata_payload is None:
        return None

    layout = _infer_eval_output_layout(evals_root, results_dir, metadata_payload)
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
) -> dict[str, str]:
    try:
        parts = results_dir.relative_to(evals_root).parts
    except ValueError:
        parts = results_dir.parts
    if "bench" in parts:
        bench_index = parts.index("bench")
        bench_parts = parts[bench_index + 1 :]
        if bench_parts:
            parts = bench_parts

    metadata_env_id = _string_or_none(metadata_payload.get("env_id"))
    metadata_model = _string_or_none(metadata_payload.get("model"))
    if len(parts) == 2 and "--" in parts[0]:
        env_from_parent, model_from_parent = parts[0].split("--", 1)
        env_id = metadata_env_id or env_from_parent
        model_id = metadata_model or model_from_parent
        job_run_id = results_dir.name
        variant_id = None
    else:
        model_id = metadata_model or (parts[0] if len(parts) >= 1 else "unknown")
        env_id = metadata_env_id or (parts[1] if len(parts) >= 2 else results_dir.name)
        variant_id = parts[2] if len(parts) >= 3 else _string_or_none(metadata_payload.get(MEDARC_VARIANT_ID_KEY))
        job_run_id = "::".join(part for part in (model_id, env_id, variant_id) if part)

    return {
        "job_run_id": job_run_id,
        "job_id": results_dir.name,
        "model_id": model_id,
        "env_id": env_id,
        "variant_id": variant_id or "",
    }


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
