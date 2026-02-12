"""Utilities for manifest validation and migration."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from medarc_verifiers.cli._manifest import MANIFEST_FILENAME, RunManifestModel, SUPPORTED_MANIFEST_VERSIONS
from medarc_verifiers.cli.utils.shared import count_jsonl_rows

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

    for run_dir in sorted(path for path in runs_path.iterdir() if path.is_dir()):
        manifest_path = run_dir / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        manifests_checked += 1
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            issues.append(
                ManifestValidationIssue(
                    run_id=run_dir.name,
                    job_id="",
                    kind="error",
                    message=f"Failed to parse manifest: {exc}",
                )
            )
            continue

        version = payload.get("version")
        if version not in SUPPORTED_MANIFEST_VERSIONS:
            issues.append(
                ManifestValidationIssue(
                    run_id=run_dir.name,
                    job_id="",
                    kind="error",
                    message=f"Unsupported manifest version: {version}",
                )
            )
            continue
        model = RunManifestModel.model_validate(payload)
        artifacts_root = str(getattr(model, "artifacts_root", ".") or ".")

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
            if entry.row_count is not None and results_path.exists():
                row_count = count_jsonl_rows(results_path)
                if row_count is not None and int(row_count) != int(entry.row_count):
                    kind = "error" if strict else "warning"
                    issues.append(
                        ManifestValidationIssue(
                            run_id=model.run_id,
                            job_id=entry.job_id,
                            kind=kind,
                            message=f"row_count mismatch: manifest={entry.row_count} actual={row_count}",
                        )
                    )
            # metadata is optional; only flag when declared explicitly in v3.
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
    return ManifestValidationResult(manifests_checked=manifests_checked, jobs_checked=jobs_checked, issues=issues)


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
        metadata_path = (root / (metadata_relpath or f"{Path(results_relpath).parent.as_posix()}/metadata.json")).resolve()
    else:
        base_dir = (run_dir / job_id).resolve()
        results_path = base_dir / "results.jsonl"
        metadata_path = base_dir / "metadata.json"
    if not results_path.exists() and (run_dir / job_id / "results.jsonl").exists():
        used_fallback = True
        results_path = (run_dir / job_id / "results.jsonl").resolve()
        metadata_path = (run_dir / job_id / "metadata.json").resolve()
    return results_path, metadata_path, used_fallback


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
