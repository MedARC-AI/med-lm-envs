"""sbatch submission helpers for Slurm orchestration bundles."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from .manifest import SlurmBundleManifest, SlurmTaskEntry, write_bundle_manifest

_JOB_ID_RE = re.compile(r"(\d+)")


def mark_dry_run(path: Path, manifest: SlurmBundleManifest) -> list[str]:
    commands: list[str] = []
    for entry in manifest.entries:
        if entry.state == "submitted" and entry.slurm_job_id:
            continue
        if entry.state != "submitted":
            entry.state = "dry-run"
        dependency = _combine_dependency(entry.base_dependency, entry.generated_dependency)
        commands.append(_render_sbatch_command(entry.script_path, dependency=dependency, test_only=False))
    write_bundle_manifest(path, manifest)
    return commands


def submit_bundle(path: Path, manifest: SlurmBundleManifest, *, test_only: bool = False) -> SlurmBundleManifest:
    entry_map = manifest.entry_map()
    for entry in manifest.entries:
        if entry.state == "submitted" and entry.slurm_job_id:
            continue
        generated_dependency = _actual_generated_dependency(entry, entry_map=entry_map)
        entry.generated_dependency = generated_dependency
        dependency = _combine_dependency(entry.base_dependency, generated_dependency)
        command = _sbatch_command(entry.script_path, dependency=dependency, test_only=test_only)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"sbatch failed for {entry.task_id}")
        if test_only:
            entry.state = "dry-run"
        else:
            entry.slurm_job_id = _parse_job_id(completed.stdout)
            entry.state = "submitted"
        write_bundle_manifest(path, manifest)
    return manifest


def _actual_generated_dependency(entry: SlurmTaskEntry, *, entry_map: dict[str, SlurmTaskEntry]) -> str | None:
    if entry.predecessor_task_id is None:
        return None
    predecessor = entry_map[entry.predecessor_task_id]
    if not predecessor.slurm_job_id:
        raise RuntimeError(f"Missing Slurm job id for predecessor task {entry.predecessor_task_id}.")
    return f"afterany:{predecessor.slurm_job_id}"


def _combine_dependency(base_dependency: str | None, generated_dependency: str | None) -> str | None:
    parts = [part for part in (base_dependency, generated_dependency) if part]
    if not parts:
        return None
    return ",".join(parts)


def _sbatch_command(script_path: str, *, dependency: str | None, test_only: bool) -> list[str]:
    command = ["sbatch"]
    if not test_only:
        command.append("--parsable")
    if test_only:
        command.append("--test-only")
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(script_path)
    return command


def _render_sbatch_command(script_path: str, *, dependency: str | None, test_only: bool) -> str:
    return " ".join(shlex.quote(arg) for arg in _sbatch_command(script_path, dependency=dependency, test_only=test_only))


def _parse_job_id(output: str) -> str:
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    token = first_line.split(";", maxsplit=1)[0]
    match = _JOB_ID_RE.search(token)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from output: {output!r}")
    return match.group(1)


__all__ = ["mark_dry_run", "submit_bundle"]
