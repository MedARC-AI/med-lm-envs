"""Artifact rendering for Slurm submission bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex

from medarc_verifiers.orchestrate.bundle import (
    ExecutionAllocation,
    PlannedTaskBundle,
    AuxiliaryImageSpec,
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
        *_render_auxiliary_images(task_bundle.spec.auxiliary_images, task_bundle=task_bundle),
        " ".join(shlex.quote(arg) for arg in command),
        "",
    ]
    script_path.write_text("\n".join(lines + body_lines), encoding="utf-8")


def _render_auxiliary_images(auxiliary_images: list[AuxiliaryImageSpec], *, task_bundle: PlannedTaskBundle) -> list[str]:
    if not auxiliary_images:
        return []
    dynamic_port_auxiliary_images = [auxiliary_image for auxiliary_image in auxiliary_images if _auxiliary_image_uses_dynamic_port(auxiliary_image)]
    lines: list[str] = [
        "AUX_IMAGE_PIDS=()",
        "AUX_IMAGE_PORTS=()",
        "",
        "cleanup_auxiliary_images() {",
        '    for pid in "${AUX_IMAGE_PIDS[@]}"; do',
        '        kill "$pid" 2>/dev/null || true',
        "    done",
        '    if [ "${#AUX_IMAGE_PORTS[@]}" -gt 0 ]; then',
        "        release_auxiliary_image_ports || true",
        "    fi",
        "}",
        "trap cleanup_auxiliary_images EXIT",
        "",
        "record_auxiliary_image_failure() {",
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
    if dynamic_port_auxiliary_images:
        lines.extend(_render_dynamic_port_helpers())
        lines.append("")
    for auxiliary_image in dynamic_port_auxiliary_images:
        lines.extend(_render_dynamic_port_setup(auxiliary_image))
        lines.append("")
    injection_lines = _render_eval_image_injections(auxiliary_images, task_bundle=task_bundle)
    if injection_lines:
        lines.extend(injection_lines)
        lines.append("")
    for auxiliary_image in auxiliary_images:
        lines.extend(_render_auxiliary_image(auxiliary_image, task_bundle=task_bundle))
        lines.append("")
    return lines


def _render_auxiliary_image(auxiliary_image: AuxiliaryImageSpec, *, task_bundle: PlannedTaskBundle) -> list[str]:
    suffix = _auxiliary_image_shell_suffix(auxiliary_image.name)
    port_var = _auxiliary_image_port_var(auxiliary_image) if _auxiliary_image_uses_dynamic_port(auxiliary_image) else None
    log_var = f"AUX_IMAGE_LOG_{suffix}"
    pid_var = f"AUX_IMAGE_PID_{suffix}"
    log_path = str(Path(task_bundle.spec.output_paths.auxiliary_image_dir) / f"{auxiliary_image.name}.log")
    lines: list[str] = [
        f"{log_var}={shlex.quote(log_path)}",
        f'mkdir -p "$(dirname "${log_var}")"',
        _render_auxiliary_image_srun(auxiliary_image, log_var=log_var, port_var=port_var),
        f"{pid_var}=$!",
        f'AUX_IMAGE_PIDS+=("${{{pid_var}}}")',
    ]
    if auxiliary_image.readiness.enabled:
        lines.extend(_render_auxiliary_image_readiness(auxiliary_image, log_var=log_var, pid_var=pid_var, port_var=port_var))
    return lines


def _render_auxiliary_image_srun(auxiliary_image: AuxiliaryImageSpec, *, log_var: str, port_var: str | None) -> str:
    env_assignments = [
        f"{key}={_shell_quote_with_port(value, port_var=port_var)}" for key, value in auxiliary_image.env.items()
    ]
    command = [
        "srun",
        "--overlap",
        "--nodes=1",
        "--ntasks=1",
        "--gpus=0",
        f"--container-image={auxiliary_image.image}",
        *auxiliary_image.srun_args,
        *auxiliary_image.command,
    ]
    tokens = [*env_assignments, *(_shell_quote_with_port(arg, port_var=port_var) for arg in command)]
    return f'{" ".join(tokens)} >>"${log_var}" 2>&1 &'


def _render_auxiliary_image_readiness(
    auxiliary_image: AuxiliaryImageSpec, *, log_var: str, pid_var: str, port_var: str | None
) -> list[str]:
    name = auxiliary_image.name
    message_exited = f"Auxiliary image {name} exited before readiness"
    message_timeout = f"Timed out waiting for auxiliary image {name}"
    probe = [
        "python3",
        "-c",
        "import sys, urllib.request; urllib.request.urlopen(sys.argv[1], timeout=2).read(1)",
        auxiliary_image.readiness.url or "",
    ]
    return [
        f"deadline=$((SECONDS + {auxiliary_image.readiness.timeout_s}))",
        f"until {' '.join(_shell_quote_with_port(arg, port_var=port_var) for arg in probe)} >/dev/null 2>&1; do",
        f'    if ! kill -0 "${pid_var}" 2>/dev/null; then',
        f"        echo {shlex.quote(message_exited)} >&2",
        f'        tail -100 "${log_var}" >&2 || true',
        f"        record_auxiliary_image_failure auxiliary_image_exited_before_readiness {shlex.quote(message_exited)}",
        "        exit 1",
        "    fi",
        '    if [ "$SECONDS" -ge "$deadline" ]; then',
        f"        echo {shlex.quote(message_timeout)} >&2",
        f'        tail -100 "${log_var}" >&2 || true',
        f"        record_auxiliary_image_failure auxiliary_image_readiness_timeout {shlex.quote(message_timeout)}",
        "        exit 1",
        "    fi",
        f"    sleep {auxiliary_image.readiness.interval_s}",
        "done",
    ]


def _render_dynamic_port_setup(auxiliary_image: AuxiliaryImageSpec) -> list[str]:
    port_var = _auxiliary_image_port_var(auxiliary_image)
    return [
        f'export {port_var}="$(allocate_auxiliary_image_port)"',
        f'AUX_IMAGE_PORTS+=("${{{port_var}}}")',
        f'echo "Selected auxiliary image port {auxiliary_image.name}: ${{{port_var}}}"',
    ]


def _render_dynamic_port_helpers() -> list[str]:
    return [
        'AUX_IMAGE_PORT_OWNER="${SLURM_JOB_ID:-manual}-$$"',
        'AUX_IMAGE_PORT_STATE="${TMPDIR:-/tmp}/medarc_aux_image_ports_$(hostname).json"',
        'AUX_IMAGE_PORT_LOCK="${AUX_IMAGE_PORT_STATE}.lock"',
        "",
        "allocate_auxiliary_image_port() {",
        "    python3 - \"$AUX_IMAGE_PORT_STATE\" \"$AUX_IMAGE_PORT_LOCK\" \"$AUX_IMAGE_PORT_OWNER\" <<'PY'",
        "import json",
        "import os",
        "import random",
        "import socket",
        "import sys",
        "import time",
        "import fcntl",
        "from pathlib import Path",
        "",
        "state_path = Path(sys.argv[1])",
        "lock_path = Path(sys.argv[2])",
        "owner = sys.argv[3]",
        "owner_pid = int(owner.rsplit('-', 1)[-1])",
        "lock_path.parent.mkdir(parents=True, exist_ok=True)",
        "with lock_path.open('w', encoding='utf-8') as lock:",
        "    fcntl.flock(lock, fcntl.LOCK_EX)",
        "    try:",
        "        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}",
        "    except Exception:",
        "        state = {}",
        "    live = {}",
        "    for port, payload in dict(state).items():",
        "        pid = int((payload or {}).get('pid') or -1)",
        "        try:",
        "            os.kill(pid, 0)",
        "        except OSError:",
        "            continue",
        "        live[str(port)] = payload",
        "    ports = list(range(20000, 61000))",
        "    random.shuffle(ports)",
        "    deadline = time.monotonic() + 10",
        "    while time.monotonic() < deadline:",
        "        for port in ports:",
        "            if str(port) in live:",
        "                continue",
        "            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:",
        "                try:",
        "                    sock.bind(('127.0.0.1', port))",
        "                except OSError:",
        "                    continue",
        "            live[str(port)] = {'owner': owner, 'pid': owner_pid}",
        "            state_path.write_text(json.dumps(live, sort_keys=True), encoding='utf-8')",
        "            print(port)",
        "            raise SystemExit(0)",
        "    raise RuntimeError('Could not allocate an auxiliary image port')",
        "PY",
        "}",
        "",
        "release_auxiliary_image_ports() {",
        "    python3 - \"$AUX_IMAGE_PORT_STATE\" \"$AUX_IMAGE_PORT_LOCK\" \"$AUX_IMAGE_PORT_OWNER\" <<'PY'",
        "import json",
        "import sys",
        "import fcntl",
        "from pathlib import Path",
        "",
        "state_path = Path(sys.argv[1])",
        "lock_path = Path(sys.argv[2])",
        "owner = sys.argv[3]",
        "if not state_path.exists():",
        "    raise SystemExit(0)",
        "with lock_path.open('w', encoding='utf-8') as lock:",
        "    fcntl.flock(lock, fcntl.LOCK_EX)",
        "    try:",
        "        state = json.loads(state_path.read_text(encoding='utf-8'))",
        "    except Exception:",
        "        state = {}",
        "    state = {port: payload for port, payload in state.items() if (payload or {}).get('owner') != owner}",
        "    state_path.write_text(json.dumps(state, sort_keys=True), encoding='utf-8')",
        "PY",
        "}",
    ]


def _render_eval_image_injections(auxiliary_images: list[AuxiliaryImageSpec], *, task_bundle: PlannedTaskBundle) -> list[str]:
    injections = [
        {
            "name": auxiliary_image.name,
            "evals": list(auxiliary_image.evals),
            "envs": list(auxiliary_image.envs),
            "port_env": _auxiliary_image_port_var(auxiliary_image) if _auxiliary_image_uses_dynamic_port(auxiliary_image) else None,
            "env_args": dict(auxiliary_image.inject_env_args),
        }
        for auxiliary_image in auxiliary_images
        if auxiliary_image.inject_env_args
    ]
    if not injections:
        return []
    config_path = task_bundle.spec.bundled_eval_config_path
    task_spec_path = task_bundle.spec.output_paths.task_spec_path
    injection_json = json.dumps(injections, sort_keys=True)
    return [
        f"python3 - {shlex.quote(config_path)} {shlex.quote(task_spec_path)} <<'PY'",
        "import hashlib",
        "import json",
        "import os",
        "import sys",
        "import tomllib",
        "from pathlib import Path",
        "from omegaconf import OmegaConf",
        "from medarc_verifiers.orchestrate.config import render_toml_mapping",
        "",
        f"injections = json.loads({json.dumps(injection_json)})",
        "path = Path(sys.argv[1])",
        "task_spec_path = Path(sys.argv[2])",
        "payload = tomllib.loads(path.read_text(encoding='utf-8'))",
        "for section_name in ('eval', 'ablation'):",
        "    entries = payload.get(section_name) or []",
        "    if isinstance(entries, dict):",
        "        entries = [entries]",
        "    for entry in entries:",
        "        if not isinstance(entry, dict):",
        "            continue",
        "        env_id = str(entry.get('env_id') or '')",
        "        env_args = dict(entry.get('env_args') or {})",
        "        for injection in injections:",
        "            evals = set(injection.get('evals') or [])",
        "            envs = set(injection.get('envs') or [])",
        "            env_match = bool(envs and env_id in envs)",
        "            eval_match = bool(evals and (env_id in evals or any(value.split(':', 1)[0] == env_id for value in evals)))",
        "            if not (env_match or eval_match):",
        "                continue",
        "            port_env = injection.get('port_env')",
        "            port = os.environ[port_env] if port_env else ''",
        "            for key, template in injection['env_args'].items():",
        "                env_args[key] = str(template).replace('{port}', port)",
        "        if env_args:",
        "            entry['env_args'] = env_args",
        "rendered = render_toml_mapping(payload)",
        "path.write_text(rendered, encoding='utf-8')",
        "checksum = hashlib.sha256(rendered.encode('utf-8')).hexdigest()",
        "task_spec = OmegaConf.to_container(OmegaConf.load(task_spec_path), resolve=True)",
        "task_spec['bundled_eval_config_checksum'] = checksum",
        "tmp_path = task_spec_path.with_suffix(task_spec_path.suffix + '.tmp')",
        "OmegaConf.save(config=OmegaConf.create(task_spec), f=str(tmp_path))",
        "tmp_path.replace(task_spec_path)",
        "PY",
    ]


def _auxiliary_image_uses_dynamic_port(auxiliary_image: AuxiliaryImageSpec) -> bool:
    values = [
        auxiliary_image.readiness.url or "",
        *auxiliary_image.env.values(),
        *auxiliary_image.srun_args,
        *auxiliary_image.command,
        *auxiliary_image.inject_env_args.values(),
    ]
    return any("{port}" in value for value in values)


def _auxiliary_image_port_var(auxiliary_image: AuxiliaryImageSpec) -> str:
    return f"AUX_IMAGE_PORT_{_auxiliary_image_shell_suffix(auxiliary_image.name)}"


def _shell_quote_with_port(value: str, *, port_var: str | None) -> str:
    if port_var is None or "{port}" not in value:
        return shlex.quote(value)
    parts = value.split("{port}")
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if part:
            rendered.append(shlex.quote(part))
        if index < len(parts) - 1:
            rendered.append(f'"${{{port_var}}}"')
    return "".join(rendered) or "''"


def _auxiliary_image_shell_suffix(name: str) -> str:
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
