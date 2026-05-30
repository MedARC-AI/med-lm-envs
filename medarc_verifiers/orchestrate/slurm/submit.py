"""sbatch submission helpers for Slurm orchestration bundles."""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from medarc_verifiers.orchestrate.launch import LaunchPlan

from .manifest import SlurmBundleManifest, write_bundle_manifest
from .manifest import load_bundle_manifest
from .plan import build_submission_plan
from .render import render_bundle

_JOB_ID_RE = re.compile(r"(\d+)")


@dataclass(frozen=True)
class SlurmSubmissionOptions:
    base_dependency: str | None = None
    test_only: bool = False
    dry_run: bool = False
    source_dir: Path | None = None
    activate_script: Path | None = None
    cpus_per_gpu: int | None = None
    time: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nice: int | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    slurm_resume: bool | None = None
    signal: str | None = None


def submit_slurm_launch_plan(launch: LaunchPlan, options: SlurmSubmissionOptions) -> int:
    output_root = launch.output_root.expanduser().resolve()
    source_dir = (options.source_dir or Path.cwd()).expanduser().resolve()
    activate_script = options.activate_script or (source_dir / ".venv" / "bin" / "activate")
    activate_script = activate_script.expanduser()
    if not activate_script.is_absolute():
        activate_script = source_dir / activate_script
    planned_tasks = build_submission_plan(
        launch.tasks,
        base_dependency=options.base_dependency,
        submission_options=options,
    )
    manifest_path = output_root / "slurm_manifest.json"
    existing_manifest = _load_existing_manifest(manifest_path, run_id=launch.run_id)
    manifest = render_bundle(
        planned_tasks=planned_tasks,
        bundle_root=output_root,
        run_id=launch.run_id,
        source_dir=source_dir,
        activate_script=activate_script.resolve(),
        env_file=launch.env_file,
        readiness_timeout_s=launch.readiness_timeout_s,
        prune_logs_on_success=launch.prune_logs_on_success,
        construct=launch.prepare,
        teardown=launch.teardown,
        existing_manifest=existing_manifest,
    )
    write_bundle_manifest(manifest_path, manifest)

    if options.dry_run:
        for command in mark_dry_run(manifest_path, manifest):
            print(command)
        return 0

    if manifest.lifecycle_entries:
        submit_lifecycle_bundle(manifest_path, manifest, test_only=options.test_only)
    else:
        submit_bundle(manifest_path, manifest, test_only=options.test_only)
    return 0


def _load_existing_manifest(path: Path, *, run_id: str) -> SlurmBundleManifest | None:
    if not path.exists():
        return None
    manifest = load_bundle_manifest(path)
    if manifest.run_id != run_id:
        raise ValueError(f"Existing Slurm manifest at {path} belongs to run_id={manifest.run_id}, not {run_id}.")
    return manifest


def mark_dry_run(path: Path, manifest: SlurmBundleManifest) -> list[str]:
    commands: list[str] = []
    lifecycle = manifest.lifecycle_entry_map()
    for entry in manifest.entries:
        prepare = lifecycle.get((entry.task_id, "prepare"))
        teardown = lifecycle.get((entry.task_id, "teardown"))
        if prepare is not None and not (prepare.state == "submitted" and prepare.slurm_job_id):
            if prepare.state != "submitted":
                prepare.state = "dry-run"
            commands.append(
                _render_sbatch_command(
                    prepare.script_path,
                    dependency=prepare.base_dependency,
                    account=prepare.account,
                    test_only=False,
                )
            )
            entry.generated_dependency = f"afterok:${{{entry.task_id}:prepare}}"
        if entry.state == "submitted" and entry.slurm_job_id:
            eval_dependency = None
        else:
            if entry.state != "submitted":
                entry.state = "dry-run"
            dependency = _combine_dependency(entry.base_dependency, entry.generated_dependency)
            commands.append(
                _render_sbatch_command(entry.script_path, dependency=dependency, account=entry.account, test_only=False)
            )
            eval_dependency = f"afterany:${{{entry.task_id}:eval}}"
        if teardown is not None and not (teardown.state == "submitted" and teardown.slurm_job_id):
            if teardown.state != "submitted":
                teardown.state = "dry-run"
            teardown.generated_dependency = eval_dependency or teardown.generated_dependency
            commands.append(
                _render_sbatch_command(
                    teardown.script_path,
                    dependency=teardown.generated_dependency,
                    account=teardown.account,
                    test_only=False,
                )
            )
    write_bundle_manifest(path, manifest)
    return commands


def submit_bundle(path: Path, manifest: SlurmBundleManifest, *, test_only: bool = False) -> SlurmBundleManifest:
    for entry in manifest.entries:
        if entry.state == "submitted" and entry.slurm_job_id:
            continue
        dependency = _combine_dependency(entry.base_dependency, entry.generated_dependency)
        command = _sbatch_command(entry.script_path, dependency=dependency, account=entry.account, test_only=test_only)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip() or completed.stdout.strip() or f"sbatch failed for {entry.task_id}"
            )
        if test_only:
            entry.state = "dry-run"
        else:
            entry.slurm_job_id = _parse_job_id(completed.stdout)
            entry.state = "submitted"
        write_bundle_manifest(path, manifest)
    return manifest


def submit_lifecycle_bundle(path: Path, manifest: SlurmBundleManifest, *, test_only: bool = False) -> SlurmBundleManifest:
    lifecycle = manifest.lifecycle_entry_map()
    for entry in manifest.entries:
        prepare = lifecycle.get((entry.task_id, "prepare"))
        teardown = lifecycle.get((entry.task_id, "teardown"))
        prepare_job_id: str | None = None
        if prepare is not None:
            if not (prepare.state == "submitted" and prepare.slurm_job_id):
                prepare_job_id = _submit_entry(
                    prepare,
                    dependency=prepare.base_dependency,
                    account=prepare.account,
                    test_only=test_only,
                    task_id=entry.task_id,
                    path=path,
                    manifest=manifest,
                )
            else:
                prepare_job_id = prepare.slurm_job_id
            entry.generated_dependency = f"afterok:{prepare_job_id}" if prepare_job_id else entry.generated_dependency
            entry.base_dependency = None
            write_bundle_manifest(path, manifest)
        if not (entry.state == "submitted" and entry.slurm_job_id):
            eval_dependency = _combine_dependency(entry.base_dependency, entry.generated_dependency)
            eval_job_id = _submit_entry(
                entry,
                dependency=eval_dependency,
                account=entry.account,
                test_only=test_only,
                task_id=entry.task_id,
                path=path,
                manifest=manifest,
            )
        else:
            eval_job_id = entry.slurm_job_id
        if teardown is not None and not (teardown.state == "submitted" and teardown.slurm_job_id):
            teardown.generated_dependency = f"afterany:{eval_job_id}" if eval_job_id else teardown.generated_dependency
            write_bundle_manifest(path, manifest)
            _submit_entry(
                teardown,
                dependency=teardown.generated_dependency,
                account=teardown.account,
                test_only=test_only,
                task_id=entry.task_id,
                path=path,
                manifest=manifest,
            )
    return manifest


def _submit_entry(
    entry,
    *,
    dependency: str | None,
    account: str | None,
    test_only: bool,
    task_id: str,
    path: Path,
    manifest: SlurmBundleManifest,
) -> str | None:
    command = _sbatch_command(entry.script_path, dependency=dependency, account=account, test_only=test_only)
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"sbatch failed for {task_id}")
    if test_only:
        entry.state = "dry-run"
        job_id = None
    else:
        job_id = _parse_job_id(completed.stdout)
        entry.slurm_job_id = job_id
        entry.state = "submitted"
    write_bundle_manifest(path, manifest)
    return job_id


def _combine_dependency(base_dependency: str | None, generated_dependency: str | None) -> str | None:
    parts = [part for part in (base_dependency, generated_dependency) if part]
    if not parts:
        return None
    return ",".join(parts)


def _sbatch_command(script_path: str, *, dependency: str | None, account: str | None, test_only: bool) -> list[str]:
    command = ["sbatch"]
    if account:
        command.extend(["--account", account])
    if not test_only:
        command.append("--parsable")
    if test_only:
        command.append("--test-only")
    if dependency:
        command.append(f"--dependency={dependency}")
    command.append(script_path)
    return command


def _render_sbatch_command(script_path: str, *, dependency: str | None, account: str | None, test_only: bool) -> str:
    return " ".join(
        shlex.quote(arg)
        for arg in _sbatch_command(script_path, dependency=dependency, account=account, test_only=test_only)
    )


def _parse_job_id(output: str) -> str:
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    token = first_line.split(";", maxsplit=1)[0]
    match = _JOB_ID_RE.search(token)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from output: {output!r}")
    return match.group(1)


__all__ = ["SlurmSubmissionOptions", "mark_dry_run", "submit_bundle", "submit_lifecycle_bundle", "submit_slurm_launch_plan"]
