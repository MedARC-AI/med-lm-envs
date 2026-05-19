"""Artifact rendering for Slurm submission bundles."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shlex

from medarc_verifiers.orchestrate.bundle import (
    ExecutionAllocation,
    PlannedTaskBundle,
    SidecarSpec,
    ensure_run_bundle,
    write_execution_allocation,
)

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
    allocation_defaults = {
        task.task.task_id: ExecutionAllocation(
            task_id=task.task.task_id,
            allocated_gpus=task.allocated_gpus,
            server_port=_default_server_port(run_id, task.submission_order),
            require_contiguous_gpus=node_gpus > 1,
            slurm_job_id=(
                existing_entries.get(task.task.task_id).slurm_job_id
                if existing_entries.get(task.task.task_id)
                else None
            ),
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
        allocation_defaults=allocation_defaults,
    )
    bundle_entries = bundle_plan.manifest.entry_map()
    entries: list[SlurmTaskEntry] = []

    for planned_task in planned_tasks:
        existing_entry = existing_entries.get(planned_task.task.task_id)
        bundle = bundle_plan.tasks[planned_task.task.task_id]
        bundle_entry = bundle_entries[planned_task.task.task_id]
        write_execution_allocation(bundle.paths.allocation_path, allocation_defaults[planned_task.task.task_id])
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
                gpus=planned_task.gpus,
                allocated_gpus=planned_task.allocated_gpus,
                tensor_parallel_size=planned_task.tensor_parallel_size,
                data_parallel_size=planned_task.data_parallel_size,
                vllm_world_size=planned_task.vllm_world_size,
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
        "worker",
        "--task",
        task_bundle.spec.output_paths.task_spec_path,
        "--allocation",
        task_bundle.spec.output_paths.allocation_path,
        "--runtime",
        "pyxis",
        "--run-id",
        planned_task.inner_run_id,
        "--no-uv-run",
    ]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    if readiness_timeout_s is not None:
        command.extend(["--readiness-timeout-s", str(readiness_timeout_s)])
    if prune_logs_on_success:
        command.append("--prune-logs-on-success")

    body_lines = [
        "",
        "set -euo pipefail",
        "",
        f'SOURCE_DIR="${{SOURCE_DIR:-{source_dir}}}"',
        f'ACTIVATE_SCRIPT="${{ACTIVATE_SCRIPT:-{activate_script}}}"',
        "",
        'cd "$SOURCE_DIR"',
        "",
        'if [ ! -f "$ACTIVATE_SCRIPT" ]; then',
        '    echo "Missing activation script: $ACTIVATE_SCRIPT" >&2',
        "    exit 1",
        "fi",
        "",
        'source "$ACTIVATE_SCRIPT"',
        f"export MEDARC_ALLOCATED_GPU_COUNT={node_gpus}",
        "",
        *_render_sidecars(task_bundle.spec.sidecars, task_bundle=task_bundle),
        " ".join(shlex.quote(arg) for arg in command),
        "",
    ]
    script_path.write_text("\n".join(lines + body_lines), encoding="utf-8")


def _render_sidecars(sidecars: list[SidecarSpec], *, task_bundle: PlannedTaskBundle) -> list[str]:
    if not sidecars:
        return []
    lines: list[str] = [
        "SIDECAR_PIDS=()",
        "",
        "cleanup_sidecars() {",
        '    for pid in "${SIDECAR_PIDS[@]}"; do',
        '        kill "$pid" 2>/dev/null || true',
        "    done",
        "}",
        "trap cleanup_sidecars EXIT",
        "",
        "record_sidecar_failure() {",
        '    reason="$1"',
        '    message="$2"',
        "    medarc-orchestrate record-failure \\",
        f"        --task-spec {shlex.quote(task_bundle.spec.output_paths.task_spec_path)} \\",
        f"        --allocation {shlex.quote(task_bundle.spec.output_paths.allocation_path)} \\",
        '        --reason "$reason" \\',
        '        --message "$message"',
        "}",
        "",
    ]
    for sidecar in sidecars:
        lines.extend(_render_sidecar(sidecar, task_bundle=task_bundle))
        lines.append("")
    return lines


def _render_sidecar(sidecar: SidecarSpec, *, task_bundle: PlannedTaskBundle) -> list[str]:
    suffix = _sidecar_shell_suffix(sidecar.name)
    log_var = f"SIDECAR_LOG_{suffix}"
    pid_var = f"SIDECAR_PID_{suffix}"
    log_path = str(Path(task_bundle.spec.output_paths.sidecar_dir) / f"{sidecar.name}.log")
    lines: list[str] = [
        f"{log_var}={shlex.quote(log_path)}",
        f'mkdir -p "$(dirname "${log_var}")"',
        _render_sidecar_srun(sidecar, log_var=log_var),
        f"{pid_var}=$!",
        f'SIDECAR_PIDS+=("${{{pid_var}}}")',
    ]
    if sidecar.readiness.enabled:
        lines.extend(_render_sidecar_readiness(sidecar, log_var=log_var, pid_var=pid_var))
    return lines


def _render_sidecar_srun(sidecar: SidecarSpec, *, log_var: str) -> str:
    env_assignments = [f"{key}={shlex.quote(value)}" for key, value in sidecar.env.items()]
    command = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        f"--container-image={sidecar.image}",
        *sidecar.srun_args,
        *sidecar.command,
    ]
    tokens = [*env_assignments, *(shlex.quote(arg) for arg in command)]
    return f'{" ".join(tokens)} >"${log_var}" 2>&1 &'


def _render_sidecar_readiness(sidecar: SidecarSpec, *, log_var: str, pid_var: str) -> list[str]:
    name = sidecar.name
    message_exited = f"Sidecar {name} exited before readiness"
    message_timeout = f"Timed out waiting for sidecar {name}"
    probe = [
        "python3",
        "-c",
        "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read(1)",
        sidecar.readiness.url or "",
    ]
    return [
        f"deadline=$((SECONDS + {sidecar.readiness.timeout_s}))",
        f"until {' '.join(shlex.quote(arg) for arg in probe)} >/dev/null 2>&1; do",
        f'    if ! kill -0 "${pid_var}" 2>/dev/null; then',
        f"        echo {shlex.quote(message_exited)} >&2",
        f'        tail -100 "${log_var}" >&2 || true',
        f"        record_sidecar_failure sidecar_exited_before_readiness {shlex.quote(message_exited)}",
        "        exit 1",
        "    fi",
        '    if [ "$SECONDS" -ge "$deadline" ]; then',
        f"        echo {shlex.quote(message_timeout)} >&2",
        f'        tail -100 "${log_var}" >&2 || true',
        f"        record_sidecar_failure sidecar_readiness_timeout {shlex.quote(message_timeout)}",
        "        exit 1",
        "    fi",
        f"    sleep {sidecar.readiness.interval_s}",
        "done",
    ]


def _sidecar_shell_suffix(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "_", name).upper()


def _sbatch_line(flag: str, value: object) -> str | None:
    if value is None:
        return None
    return f"#SBATCH {flag}={value}"


def _default_server_port(run_id: str, submission_order: int, *, min_port: int = 8000, max_port: int = 65000) -> int:
    port_span = max_port - min_port + 1
    if submission_order >= port_span:
        raise ValueError(f"Submission order {submission_order} exceeds the available TCP port range.")
    run_seed = int(hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:8], 16)  # noqa: S324
    base_port = min_port + (run_seed % max(1, port_span - submission_order))
    return base_port + submission_order


__all__ = ["render_bundle", "render_task_artifacts", "write_script"]
