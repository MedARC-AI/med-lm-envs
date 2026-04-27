"""Utilities for manifest validation and migration."""

from __future__ import annotations

import os
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from medarc_verifiers.cli._manifest import MANIFEST_FILENAME, RunManifestModel, SUPPORTED_MANIFEST_VERSIONS

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ManifestValidationIssue:
    run_id: str
    job_id: str
    kind: str
    message: str


@dataclass(slots=True)
class ManifestValidationResult:
    manifests_checked: int
    jobs_checked: int
    issues: list[ManifestValidationIssue]

    @property
    def has_errors(self) -> bool:
        return any(issue.kind == "error" for issue in self.issues)


def validate_manifests_in_runs(runs_dir: Path | str, *, strict: bool = False) -> ManifestValidationResult:
    runs_path = Path(runs_dir)
    issues: list[ManifestValidationIssue] = []
    manifests_checked = 0
    jobs_checked = 0
    if not runs_path.exists():
        return ManifestValidationResult(manifests_checked=0, jobs_checked=0, issues=[])

    run_dirs = sorted(path for path in runs_path.iterdir() if path.is_dir())
    logger.info("Scanning manifests under %s...", runs_path)

    manifest_run_dirs = [run_dir for run_dir in run_dirs if (run_dir / MANIFEST_FILENAME).exists()]
    if not manifest_run_dirs:
        return ManifestValidationResult(manifests_checked=0, jobs_checked=0, issues=[])

    max_workers = min(len(manifest_run_dirs), max(1, (os.cpu_count() or 4) * 4))
    if max_workers <= 1:
        results = [_validate_run_dir(run_dir, strict=strict) for run_dir in manifest_run_dirs]
    else:
        results = list(_validate_run_dirs_parallel(manifest_run_dirs, strict=strict, max_workers=max_workers))

    for result in results:
        manifests_checked += result.manifests_checked
        jobs_checked += result.jobs_checked
        issues.extend(result.issues)

    issues.sort(key=lambda item: (item.run_id, item.job_id, item.kind, item.message))
    return ManifestValidationResult(manifests_checked=manifests_checked, jobs_checked=jobs_checked, issues=issues)


def _validate_run_dirs_parallel(
    run_dirs: Sequence[Path],
    *,
    strict: bool,
    max_workers: int,
) -> list[ManifestValidationResult]:
    results: list[ManifestValidationResult] = []
    progress, task_id = _create_manifest_scan_progress(len(run_dirs))
    executor: ThreadPoolExecutor | None = None
    futures = []
    try:
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = [executor.submit(_validate_run_dir, run_dir, strict=strict) for run_dir in run_dirs]
        if progress is not None and task_id is not None:
            with progress:
                for future in as_completed(futures):
                    results.append(future.result())
                    progress.update(task_id, advance=1)
        else:
            for future in as_completed(futures):
                results.append(future.result())
    except KeyboardInterrupt:
        logger.warning("Manifest scanning interrupted; cancelling validation workers.")
        for future in futures:
            future.cancel()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            executor = None
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    return results


def _create_manifest_scan_progress(total: int) -> tuple[object | None, object | None]:
    if total <= 0 or not sys.stderr.isatty():
        return None, None
    try:
        from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            transient=True,
        )
        task_id = progress.add_task("Scanning manifests", total=total)
        return progress, task_id
    except Exception:
        return None, None


def _validate_run_dir(run_dir: Path, *, strict: bool) -> ManifestValidationResult:
    issues: list[ManifestValidationIssue] = []
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return ManifestValidationResult(manifests_checked=0, jobs_checked=0, issues=[])

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return ManifestValidationResult(
            manifests_checked=1,
            jobs_checked=0,
            issues=[
                ManifestValidationIssue(
                    run_id=run_dir.name,
                    job_id="",
                    kind="error",
                    message=f"Failed to parse manifest: {exc}",
                )
            ],
        )

    version = payload.get("version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        return ManifestValidationResult(
            manifests_checked=1,
            jobs_checked=0,
            issues=[
                ManifestValidationIssue(
                    run_id=run_dir.name,
                    job_id="",
                    kind="error",
                    message=f"Unsupported manifest version: {version}",
                )
            ],
        )

    model = RunManifestModel.model_validate(payload)
    artifacts_root = str(getattr(model, "artifacts_root", ".") or ".")
    jobs_checked = 0

    for entry in model.jobs:
        jobs_checked += 1
        results_path, metadata_path, used_fallback = _resolve_job_artifact_paths(
            run_dir=run_dir,
            artifacts_root=artifacts_root,
            job_id=entry.job_id,
            results_relpath=entry.results_relpath,
            metadata_relpath=entry.metadata_relpath,
        )
        if used_fallback:
            issues.append(
                ManifestValidationIssue(
                    run_id=model.run_id,
                    job_id=entry.job_id,
                    kind="warning",
                    message="Manifest artifact path missing; fallback to run-relative job directory would be used.",
                )
            )
        if not results_path.exists():
            kind = "error" if strict else "warning"
            issues.append(
                ManifestValidationIssue(
                    run_id=model.run_id,
                    job_id=entry.job_id,
                    kind=kind,
                    message=f"Missing results.jsonl at {results_path}",
                )
            )
        if results_path.exists():
            for message in _quick_validate_results_jsonl(
                results_path,
                num_examples=entry.num_examples,
                rollouts_per_example=entry.rollouts_per_example,
            ):
                kind = "error" if strict else "warning"
                issues.append(
                    ManifestValidationIssue(
                        run_id=model.run_id,
                        job_id=entry.job_id,
                        kind=kind,
                        message=message,
                    )
                )
        if entry.metadata_relpath and not metadata_path.exists():
            kind = "error" if strict else "warning"
            issues.append(
                ManifestValidationIssue(
                    run_id=model.run_id,
                    job_id=entry.job_id,
                    kind=kind,
                    message=f"Missing metadata.json at {metadata_path}",
                )
            )

    return ManifestValidationResult(manifests_checked=1, jobs_checked=jobs_checked, issues=issues)


def _resolve_job_artifact_paths(
    *,
    run_dir: Path,
    artifacts_root: str,
    job_id: str,
    results_relpath: str | None,
    metadata_relpath: str | None,
) -> tuple[Path, Path, bool]:
    used_fallback = False
    if results_relpath:
        root = (run_dir / artifacts_root).resolve()
        results_path = (root / results_relpath).resolve()
        metadata_path = (
            root / (metadata_relpath or f"{Path(results_relpath).parent.as_posix()}/metadata.json")
        ).resolve()
    else:
        base_dir = (run_dir / job_id).resolve()
        results_path = base_dir / "results.jsonl"
        metadata_path = base_dir / "metadata.json"
    if not results_path.exists() and (run_dir / job_id / "results.jsonl").exists():
        used_fallback = True
        results_path = (run_dir / job_id / "results.jsonl").resolve()
        metadata_path = (run_dir / job_id / "metadata.json").resolve()
    return results_path, metadata_path, used_fallback


def _quick_validate_results_jsonl(
    path: Path,
    *,
    num_examples: int | None,
    rollouts_per_example: int | None,
) -> list[str]:
    first_line = _read_first_nonempty_line(path)
    last_line = _read_last_nonempty_line(path)
    if first_line is None or last_line is None:
        return [f"results.jsonl at {path} is empty"]

    issues: list[str] = []
    first_payload = _decode_probe_line(first_line, path=path, position="first", issues=issues)
    last_payload = _decode_probe_line(last_line, path=path, position="last", issues=issues)
    if first_payload is None or last_payload is None:
        return issues

    for position, payload in (("first", first_payload), ("last", last_payload)):
        if "example_id" not in payload:
            issues.append(f"{position} JSONL row in {path} is missing example_id")
    _validate_rollout_index(
        first_payload,
        path=path,
        position="first",
        rollouts_per_example=rollouts_per_example,
        issues=issues,
    )
    _validate_rollout_index(
        last_payload,
        path=path,
        position="last",
        rollouts_per_example=rollouts_per_example,
        issues=issues,
    )

    return issues


def _decode_probe_line(
    raw_line: str,
    *,
    path: Path,
    position: str,
    issues: list[str],
) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        issues.append(f"failed to parse {position} JSONL row in {path}: {exc.msg}")
        return None
    if not isinstance(payload, Mapping):
        issues.append(f"{position} JSONL row in {path} is not a JSON object")
        return None
    return payload


def _read_first_nonempty_line(path: Path) -> str | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            candidate = line.strip()
            if candidate:
                return candidate
    return None


def _read_last_nonempty_line(path: Path) -> str | None:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        file_size = handle.tell()
        if file_size <= 0:
            return None

        chunk_size = 8192
        buffer = b""
        position = file_size
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = buffer.splitlines()
            for raw_line in reversed(lines):
                candidate = raw_line.strip()
                if candidate:
                    return candidate.decode("utf-8")
        return None


def _validate_rollout_index(
    payload: Mapping[str, Any],
    *,
    path: Path,
    position: str,
    rollouts_per_example: int | None,
    issues: list[str],
) -> None:
    rollout_index = _coerce_int(payload.get("rollout_index"))
    if rollout_index is None:
        return
    if rollout_index < 0:
        issues.append(f"{position} JSONL row in {path} has negative rollout_index={payload.get('rollout_index')!r}")
        return
    if rollouts_per_example and rollout_index >= rollouts_per_example:
        issues.append(
            f"{position} JSONL row in {path} has out-of-range rollout_index={payload.get('rollout_index')!r}; "
            f"expected < {rollouts_per_example}"
        )


def _coerce_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def format_validation_issues(issues: Sequence[ManifestValidationIssue]) -> list[str]:
    lines: list[str] = []
    for issue in issues:
        prefix = issue.kind.upper()
        target = f"run={issue.run_id}"
        if issue.job_id:
            target += f" job={issue.job_id}"
        lines.append(f"[{prefix}] {target}: {issue.message}")
    return lines


__all__ = [
    "ManifestValidationIssue",
    "ManifestValidationResult",
    "validate_manifests_in_runs",
    "format_validation_issues",
]
