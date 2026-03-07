"""Artifact rendering for Slurm submission bundles."""

from __future__ import annotations

from pathlib import Path
import shlex

from medarc_verifiers.orchestrate.bundle import ExecutionAllocation, PlannedTaskBundle, ensure_run_bundle

from .manifest import SlurmBundleManifest, SlurmTaskEntry
from .plan import PlannedSlurmTask, placeholder_dependency


def render_bundle(
    *,
    planned_tasks: list[PlannedSlurmTask],
    bundle_root: Path,
    run_id: str,
    node_gpus: int,
    source_dir: Path,
    activate_script: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    existing_manifest: SlurmBundleManifest | None = None,
) -> SlurmBundleManifest:
    bundle_root.mkdir(parents=True, exist_ok=True)
    existing_entries = existing_manifest.entry_map() if existing_manifest else {}
    task_order = {task.task.task_id: task.submission_order for task in planned_tasks}
    eval_config_overrides = {
        task.task.task_id: {
            "orchestrate": {
                task.task.model_key: {
                    "gpus": task.effective_gpus,
                    "tensor_parallel_size": task.tp_size,
                    "data_parallel_size": task.dp_size,
                }
            }
        }
        for task in planned_tasks
    }
    allocation_defaults = {
        task.task.task_id: ExecutionAllocation(
            task_id=task.task.task_id,
            allocated_gpus=node_gpus,
            require_contiguous_gpus=node_gpus > 1,
            slurm_job_id=(existing_entries.get(task.task.task_id).slurm_job_id if existing_entries.get(task.task.task_id) else None),
            constraints={"scheduler": "slurm", "node_gpus": node_gpus},
        )
        for task in planned_tasks
    }
    bundle_plan = ensure_run_bundle(
        tasks=[task.task for task in planned_tasks],
        run_id=run_id,
        output_root=bundle_root,
        mode="slurm",
        runtime="pyxis",
        eval_config_overrides=eval_config_overrides,
        allocation_defaults=allocation_defaults,
    )
    bundle_entries = bundle_plan.manifest.entry_map()
    entries: list[SlurmTaskEntry] = []

    for planned_task in planned_tasks:
        existing_entry = existing_entries.get(planned_task.task.task_id)
        bundle = bundle_plan.tasks[planned_task.task.task_id]
        bundle_entry = bundle_entries[planned_task.task.task_id]
        rendered = render_task_artifacts(
            planned_task=planned_task,
            task_bundle=bundle,
            source_dir=source_dir,
            activate_script=activate_script,
            env_file=env_file,
            readiness_timeout_s=readiness_timeout_s,
            prune_logs_on_success=prune_logs_on_success,
            node_gpus=node_gpus,
        )
        generated_dependency = placeholder_dependency(planned_task, task_order=task_order)
        state = existing_entry.state if existing_entry else "pending"
        slurm_job_id = existing_entry.slurm_job_id if existing_entry else None
        if state == "submitted" and not slurm_job_id:
            state = "pending"
        entries.append(
            SlurmTaskEntry(
                run_id=run_id,
                task_id=planned_task.task.task_id,
                task_slug=planned_task.task_slug,
                original_job_config_path=str(planned_task.task.job_config_path),
                original_job_config_checksum=bundle.spec.original_job_config_checksum,
                effective_job_config_path=bundle.spec.bundled_eval_config_path,
                bundled_eval_config_checksum=bundle.spec.bundled_eval_config_checksum,
                task_spec_path=bundle.spec.output_paths.task_spec_path,
                task_spec_checksum=bundle_entry.task_spec_checksum,
                allocation_path=bundle.spec.output_paths.allocation_path,
                state_path=bundle.spec.output_paths.state_path,
                tp_size=planned_task.tp_size,
                dp_size=planned_task.dp_size,
                effective_gpus=planned_task.effective_gpus,
                inner_run_id=planned_task.inner_run_id,
                restart_source=bundle.state.restart_source,
                restart_strategy=bundle.state.restart_source_strategy,
                script_path=str(rendered["script_path"]),
                generated_dependency=generated_dependency,
                base_dependency=planned_task.base_dependency,
                predecessor_task_id=planned_task.predecessor_task_id,
                chain_index=planned_task.chain_index,
                submission_order=planned_task.submission_order,
                job_name=planned_task.options.job_name,
                account=planned_task.options.account,
                slurm_job_id=slurm_job_id,
                state=state,
            )
        )

    manifest = SlurmBundleManifest(
        run_id=run_id,
        bundle_root=str(bundle_root),
        node_gpus=node_gpus,
        entries=entries,
    )
    if existing_manifest is not None:
        manifest.created_at = existing_manifest.created_at
    return manifest


def render_task_artifacts(
    *,
    planned_task: PlannedSlurmTask,
    task_bundle: PlannedTaskBundle,
    source_dir: Path,
    activate_script: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    node_gpus: int,
) -> dict[str, Path | str | None]:
    script_path = task_bundle.paths.submit_script_path
    write_script(
        script_path=script_path,
        planned_task=planned_task,
        task_bundle=task_bundle,
        source_dir=source_dir,
        activate_script=activate_script,
        env_file=env_file,
        readiness_timeout_s=readiness_timeout_s,
        prune_logs_on_success=prune_logs_on_success,
        node_gpus=node_gpus,
    )
    return {"script_path": script_path}


def write_script(
    *,
    script_path: Path,
    planned_task: PlannedSlurmTask,
    task_bundle: PlannedTaskBundle,
    source_dir: Path,
    activate_script: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    node_gpus: int,
) -> None:
    log_path = Path(task_bundle.spec.output_paths.root) / "slurm" / "job_%j.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={planned_task.options.job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-task={node_gpus}",
        f"#SBATCH --output={log_path}",
    ]
    option_lines = [
        _sbatch_line("--cpus-per-gpu", planned_task.options.cpus_per_gpu),
        _sbatch_line("--time", planned_task.options.time),
        _sbatch_line("--partition", planned_task.options.partition),
        _sbatch_line("--account", planned_task.options.account),
        _sbatch_line("--qos", planned_task.options.qos),
        _sbatch_line("--mail-type", planned_task.options.mail_type),
        _sbatch_line("--mail-user", planned_task.options.mail_user),
    ]
    lines.extend(line for line in option_lines if line is not None)
    if planned_task.options.slurm_resume:
        lines.append("#SBATCH --requeue")

    command = [
        "medarc-orchestrate",
        "--job-config",
        task_bundle.spec.bundled_eval_config_path,
        "--runtime",
        "pyxis",
        "--run-id",
        planned_task.inner_run_id,
        "--output-dir",
        str(Path(task_bundle.spec.output_paths.root) / "orchestrator"),
        "--no-uv-run",
    ]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    if readiness_timeout_s is not None:
        command.extend(["--readiness-timeout-s", str(readiness_timeout_s)])
    if prune_logs_on_success:
        command.append("--prune-logs-on-success")
    if planned_task.options.slurm_resume:
        command.append("--resume")

    body_lines = [
        "",
        "set -euo pipefail",
        "",
        f'SOURCE_DIR="${{SOURCE_DIR:-{source_dir}}}"',
        f'ACTIVATE_SCRIPT="${{ACTIVATE_SCRIPT:-{activate_script}}}"',
        "",
        'cd "$SOURCE_DIR"',
        "",
        'if [ -f "$SOURCE_DIR/.env" ]; then',
        '    set -a',
        '    source "$SOURCE_DIR/.env"',
        '    set +a',
        "fi",
        "",
        'if [ ! -f "$ACTIVATE_SCRIPT" ]; then',
        '    echo "Missing activation script: $ACTIVATE_SCRIPT" >&2',
        "    exit 1",
        "fi",
        "",
        'source "$ACTIVATE_SCRIPT"',
        f"export MEDARC_ALLOCATED_GPU_COUNT={node_gpus}",
        "",
        " ".join(shlex.quote(arg) for arg in command),
        "",
    ]
    script_path.write_text("\n".join(lines + body_lines), encoding="utf-8")


def _sbatch_line(flag: str, value: object) -> str | None:
    if value is None:
        return None
    return f"#SBATCH {flag}={value}"


__all__ = ["render_bundle", "render_task_artifacts", "write_script"]
