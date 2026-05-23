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
from medarc_verifiers.orchestrate.config import ConstructConfig, TeardownConfig
from medarc_verifiers.orchestrate.lifecycle import is_absolute_sqsh_image, materialized_image_path, resolve_construct_cache

from .manifest import SlurmBundleManifest, SlurmLifecycleEntry, SlurmTaskEntry
from .plan import PlannedSlurmTask


def render_bundle(
    *,
    planned_tasks: list[PlannedSlurmTask],
    bundle_root: Path,
    run_id: str = "bundle",
    source_dir: Path,
    activate_script: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    construct: ConstructConfig | None = None,
    teardown: TeardownConfig | None = None,
    existing_manifest: SlurmBundleManifest | None = None,
) -> SlurmBundleManifest:
    bundle_root.mkdir(parents=True, exist_ok=True)
    construct = construct or ConstructConfig()
    teardown = teardown or TeardownConfig()
    existing_entries = existing_manifest.eval_entry_map() if existing_manifest else {}
    existing_lifecycle = existing_manifest.lifecycle_entry_map() if existing_manifest else {}
    allocation_defaults = {
        task.task.task_id: ExecutionAllocation(
            task_id=task.task.task_id,
            allocated_gpus=task.allocated_gpus,
            server_port=_default_server_port(run_id, task.submission_order),
            require_contiguous_gpus=task.allocated_gpus > 1,
            slurm_job_id=(
                existing_entries.get(task.task.task_id).slurm_job_id
                if existing_entries.get(task.task.task_id)
                else None
            ),
            constraints={"scheduler": "slurm", "allocated_gpus": task.allocated_gpus},
        )
        for task in planned_tasks
    }
    construct_caches: dict[str, dict[str, object]] = {}
    container_image_overrides: dict[str, str] = {}
    teardowns: dict[str, dict[str, object]] = {}
    if construct.enabled:
        for task in planned_tasks:
            source = str((task.task.orchestrate.get("container") or {}).get("image") or "")
            should_materialize_image = construct.image_materialization_enabled and not is_absolute_sqsh_image(source)
            cache = resolve_construct_cache(
                config=construct,
                volume_mounts=list((task.task.orchestrate.get("container") or {}).get("volumes") or []),
                require_image_dir=should_materialize_image,
            )
            construct_caches[task.task.task_id] = cache.to_dict()
            if should_materialize_image:
                container_image_overrides[task.task.task_id] = str(materialized_image_path(source, str(cache.image_dir)))
    if teardown.enabled:
        for task in planned_tasks:
            teardowns[task.task.task_id] = teardown.model_dump()

    bundle_plan = ensure_run_bundle(
        tasks=[task.task for task in planned_tasks],
        run_id=run_id,
        output_root=bundle_root,
        mode="slurm",
        runtime="pyxis",
        allocation_defaults=allocation_defaults,
        construct_cache_by_task=construct_caches,
        teardown_by_task=teardowns,
        container_image_by_task=container_image_overrides,
    )
    bundle_entries = bundle_plan.manifest.entry_map()
    entries: list[SlurmTaskEntry] = []
    lifecycle_entries: list[SlurmLifecycleEntry] = []

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
        )
        if construct.enabled:
            write_lifecycle_script(
                phase="construct",
                script_path=bundle.paths.construct_script_path,
                planned_task=planned_task,
                task_bundle=bundle,
                source_dir=source_dir,
                activate_script=activate_script,
                env_file=env_file,
                construct=construct,
                teardown=teardown,
            )
            existing_life = existing_lifecycle.get((planned_task.task.task_id, "construct"))
            life_state = existing_life.state if existing_life else "pending"
            life_job = existing_life.slurm_job_id if existing_life else None
            if life_state == "submitted" and not life_job:
                life_state = "pending"
            lifecycle_entries.append(
                SlurmLifecycleEntry(
                    run_id=run_id,
                    task_id=planned_task.task.task_id,
                    task_slug=planned_task.task_slug,
                    phase="construct",
                    script_path=str(bundle.paths.construct_script_path),
                    generated_dependency=None,
                    base_dependency=planned_task.base_dependency,
                    submission_order=planned_task.submission_order,
                    job_name=f"{planned_task.options.job_name}-construct",
                    cpus=construct.cpus,
                    account=construct.account or planned_task.options.account,
                    slurm_job_id=life_job,
                    state=life_state,
                )
            )
        state = existing_entry.state if existing_entry else "pending"
        slurm_job_id = existing_entry.slurm_job_id if existing_entry else None
        if state == "submitted" and not slurm_job_id:
            state = "pending"
        entries.append(
            SlurmTaskEntry(
                run_id=run_id,
                task_id=planned_task.task.task_id,
                task_slug=planned_task.task_slug,
                suite_path=str(planned_task.task.suite_path),
                suite_checksum=bundle.spec.suite_checksum,
                target_endpoint_id=planned_task.task.target_endpoint_id,
                generated_eval_config_path=bundle.spec.bundled_eval_config_path,
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
                script_path=str(rendered["script_path"]),
                generated_dependency=None,
                base_dependency=None if construct.enabled else planned_task.base_dependency,
                submission_order=planned_task.submission_order,
                job_name=planned_task.options.job_name,
                account=planned_task.options.account,
                slurm_job_id=slurm_job_id,
                state=state,
            )
        )
        if teardown.enabled:
            write_lifecycle_script(
                phase="teardown",
                script_path=bundle.paths.teardown_script_path,
                planned_task=planned_task,
                task_bundle=bundle,
                source_dir=source_dir,
                activate_script=activate_script,
                env_file=env_file,
                construct=construct,
                teardown=teardown,
            )
            existing_life = existing_lifecycle.get((planned_task.task.task_id, "teardown"))
            life_state = existing_life.state if existing_life else "pending"
            life_job = existing_life.slurm_job_id if existing_life else None
            if life_state == "submitted" and not life_job:
                life_state = "pending"
            lifecycle_entries.append(
                SlurmLifecycleEntry(
                    run_id=run_id,
                    task_id=planned_task.task.task_id,
                    task_slug=planned_task.task_slug,
                    phase="teardown",
                    script_path=str(bundle.paths.teardown_script_path),
                    generated_dependency=f"afterany:${{{planned_task.task.task_id}:eval}}",
                    base_dependency=None,
                    submission_order=planned_task.submission_order,
                    job_name=f"{planned_task.options.job_name}-teardown",
                    cpus=teardown.cpus,
                    account=teardown.account or planned_task.options.account,
                    slurm_job_id=life_job,
                    state=life_state,
                )
            )
    if construct.enabled:
        for entry in entries:
            entry.generated_dependency = f"afterok:${{{entry.task_id}:construct}}"

    manifest = SlurmBundleManifest(
        run_id=run_id,
        bundle_root=str(bundle_root),
        entries=entries,
        lifecycle_entries=lifecycle_entries,
    )
    if existing_manifest is not None:
        manifest.created_at = existing_manifest.created_at
    return manifest


def write_lifecycle_script(
    *,
    phase: str,
    script_path: Path,
    planned_task: PlannedSlurmTask,
    task_bundle: PlannedTaskBundle,
    source_dir: Path,
    activate_script: Path,
    env_file: Path | None,
    construct: ConstructConfig,
    teardown: TeardownConfig,
) -> None:
    if phase == "construct":
        cpus = construct.cpus
        time = construct.time
        partition = construct.partition
        account = construct.account or planned_task.options.account
        qos = construct.qos or planned_task.options.qos
        nice = construct.nice if construct.nice is not None else planned_task.options.nice
        mail_type = construct.mail_type or planned_task.options.mail_type
        mail_user = construct.mail_user or planned_task.options.mail_user
        output_path = Path(task_bundle.spec.output_paths.construct_dir) / "slurm_%j.log"
        command = [
            "medarc-orchestrate",
            "construct",
            "--task",
            task_bundle.spec.output_paths.task_spec_path,
            "--allocation",
            task_bundle.spec.output_paths.construct_allocation_path,
        ]
        if construct.prefetch_enabled:
            command.append("--prefetch-model")
        if construct.image_materialization_enabled and task_bundle.spec.container_image_source is not None:
            command.append("--materialize-image")
    elif phase == "teardown":
        cpus = teardown.cpus
        time = teardown.time
        partition = teardown.partition
        account = teardown.account or planned_task.options.account
        qos = teardown.qos or planned_task.options.qos
        nice = teardown.nice if teardown.nice is not None else planned_task.options.nice
        mail_type = teardown.mail_type or planned_task.options.mail_type
        mail_user = teardown.mail_user or planned_task.options.mail_user
        output_path = Path(task_bundle.spec.output_paths.teardown_dir) / "slurm_%j.log"
        command = [
            "medarc-orchestrate",
            "teardown",
            "--task",
            task_bundle.spec.output_paths.task_spec_path,
            "--allocation",
            task_bundle.spec.output_paths.teardown_allocation_path,
        ]
    else:
        raise ValueError(f"Unknown lifecycle phase: {phase}")
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_path = (
        Path(task_bundle.spec.output_paths.construct_allocation_path)
        if phase == "construct"
        else Path(task_bundle.spec.output_paths.teardown_allocation_path)
    )
    write_execution_allocation(allocation_path, ExecutionAllocation(task_id=task_bundle.spec.task_id))
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={planned_task.options.job_name}-{phase}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --output={output_path}",
    ]
    option_lines = [
        _sbatch_line("--time", time),
        _sbatch_line("--partition", partition),
        _sbatch_line("--account", account),
        _sbatch_line("--qos", qos),
        _sbatch_line("--nice", nice),
        _sbatch_line("--mail-type", mail_type),
        _sbatch_line("--mail-user", mail_user),
    ]
    lines.extend(line for line in option_lines if line is not None)
    body = [
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
        " ".join(shlex.quote(arg) for arg in command),
        "",
    ]
    script_path.write_text("\n".join(lines + body), encoding="utf-8")


def render_task_artifacts(
    *,
    planned_task: PlannedSlurmTask,
    task_bundle: PlannedTaskBundle,
    source_dir: Path,
    activate_script: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
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
) -> None:
    log_path = Path(task_bundle.spec.output_paths.root) / "slurm" / "job_%j.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={planned_task.options.job_name}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-task={planned_task.allocated_gpus}",
        f"#SBATCH --output={log_path}",
    ]
    option_lines = [
        _sbatch_line("--cpus-per-gpu", planned_task.options.cpus_per_gpu),
        _sbatch_line("--time", planned_task.options.time),
        _sbatch_line("--partition", planned_task.options.partition),
        _sbatch_line("--account", planned_task.options.account),
        _sbatch_line("--qos", planned_task.options.qos),
        _sbatch_line("--nice", planned_task.options.nice),
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
        f"export MEDARC_ALLOCATED_GPU_COUNT={planned_task.allocated_gpus}",
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
    return f'{" ".join(tokens)} >>"${log_var}" 2>&1 &'


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
