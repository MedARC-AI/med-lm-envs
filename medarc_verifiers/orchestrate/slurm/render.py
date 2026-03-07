"""Artifact rendering for Slurm submission bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import shlex

from omegaconf import OmegaConf

from medarc_verifiers.orchestrate.config import load_job_config

from .manifest import SlurmBundleManifest, SlurmTaskEntry
from .plan import PlannedSlurmTask, placeholder_dependency


def render_bundle(
    *,
    planned_tasks: list[PlannedSlurmTask],
    bundle_root: Path,
    run_id: str,
    node_gpus: int,
    source_dir: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    existing_manifest: SlurmBundleManifest | None = None,
) -> SlurmBundleManifest:
    bundle_root.mkdir(parents=True, exist_ok=True)
    existing_entries = existing_manifest.entry_map() if existing_manifest else {}
    task_order = {task.task.task_id: task.submission_order for task in planned_tasks}
    entries: list[SlurmTaskEntry] = []

    for planned_task in planned_tasks:
        existing_entry = existing_entries.get(planned_task.task.task_id)
        rendered = render_task_artifacts(
            planned_task=planned_task,
            bundle_root=bundle_root,
            source_dir=source_dir,
            env_file=env_file,
            readiness_timeout_s=readiness_timeout_s,
            prune_logs_on_success=prune_logs_on_success,
            node_gpus=node_gpus,
            existing_entry=existing_entry,
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
                effective_job_config_path=str(rendered["effective_job_config_path"]),
                patched_job_config_path=_string_or_none(rendered["patched_job_config_path"]),
                tp_size=planned_task.tp_size,
                dp_size=planned_task.dp_size,
                effective_gpus=planned_task.effective_gpus,
                inner_run_id=planned_task.inner_run_id,
                restart_source=_string_or_none(rendered["restart_source"]),
                restart_strategy=_string_or_none(rendered["restart_strategy"]),
                script_path=str(rendered["script_path"]),
                generated_dependency=generated_dependency,
                base_dependency=planned_task.base_dependency,
                predecessor_task_id=planned_task.predecessor_task_id,
                chain_index=planned_task.chain_index,
                submission_order=planned_task.submission_order,
                job_name=planned_task.options.job_name,
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
    bundle_root: Path,
    source_dir: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    node_gpus: int,
    existing_entry: SlurmTaskEntry | None,
) -> dict[str, Path | str | None]:
    task_root = bundle_root / planned_task.task_slug
    slurm_dir = task_root / "slurm"
    slurm_dir.mkdir(parents=True, exist_ok=True)
    source_payload = dict(load_job_config(planned_task.task.job_config_path))
    source_orchestrate = dict(source_payload.get("orchestrate") or {})
    source_model_cfg = dict(source_orchestrate.get(planned_task.task.model_key) or {})
    restart_source, restart_strategy = resolve_restart_source(
        source_payload=source_payload,
        inner_run_id=planned_task.inner_run_id,
        existing_entry=existing_entry,
    )

    source_gpus = int(source_model_cfg.get("gpus", 1))
    source_dp_size = int(source_model_cfg.get("data_parallel_size", 1) or 1)
    patch_needed = (
        planned_task.effective_gpus != source_gpus
        or planned_task.dp_size != source_dp_size
        or restart_strategy != "source_config"
    )
    patched_job_config_path: Path | None = None
    effective_job_config_path = planned_task.task.job_config_path
    if patch_needed:
        patched_job_config_path = slurm_dir / "job-config.yaml"
        payload = _build_task_config_payload(
            source_payload=source_payload,
            planned_task=planned_task,
            restart_source=restart_source,
        )
        _write_yaml(patched_job_config_path, payload)
        effective_job_config_path = patched_job_config_path

    script_path = slurm_dir / "orchestrate.sh"
    write_script(
        script_path=script_path,
        planned_task=planned_task,
        source_dir=source_dir,
        env_file=env_file,
        readiness_timeout_s=readiness_timeout_s,
        prune_logs_on_success=prune_logs_on_success,
        node_gpus=node_gpus,
        effective_job_config_path=effective_job_config_path,
        bundle_root=bundle_root,
    )
    return {
        "effective_job_config_path": effective_job_config_path,
        "patched_job_config_path": patched_job_config_path,
        "restart_source": restart_source,
        "restart_strategy": restart_strategy,
        "script_path": script_path,
    }


def resolve_restart_source(
    *,
    source_payload: dict[str, Any],
    inner_run_id: str,
    existing_entry: SlurmTaskEntry | None,
) -> tuple[str | None, str]:
    if existing_entry and existing_entry.restart_source:
        return existing_entry.restart_source, "persisted"

    persisted_path = None
    if existing_entry and existing_entry.patched_job_config_path:
        persisted_path = Path(existing_entry.patched_job_config_path)
        if persisted_path.exists():
            persisted_payload = dict(load_job_config(persisted_path))
            persisted_restart = _extract_restart_source(persisted_payload)
            if persisted_restart:
                return persisted_restart, "persisted"

    source_restart = _extract_restart_source(source_payload)
    if source_restart:
        return source_restart, "source_config"
    return f"runs/raw/{inner_run_id}", "auto_injected"


def write_script(
    *,
    script_path: Path,
    planned_task: PlannedSlurmTask,
    source_dir: Path,
    env_file: Path | None,
    readiness_timeout_s: int | None,
    prune_logs_on_success: bool,
    node_gpus: int,
    effective_job_config_path: Path,
    bundle_root: Path,
) -> None:
    log_path = bundle_root / planned_task.task_slug / "slurm" / "job_%j.log"
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
        str(effective_job_config_path),
        "--runtime",
        "pyxis",
        "--run-id",
        planned_task.inner_run_id,
        "--output-dir",
        str(bundle_root / planned_task.task_slug / "orchestrator"),
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
        "",
        'cd "$SOURCE_DIR"',
        "",
        'if [ -f "$SOURCE_DIR/.env" ]; then',
        '    set -a',
        '    source "$SOURCE_DIR/.env"',
        '    set +a',
        "fi",
        "",
        'source "$SOURCE_DIR/.venv/bin/activate"',
        f"export MEDARC_ALLOCATED_GPU_COUNT={node_gpus}",
        "",
        " ".join(shlex.quote(arg) for arg in command),
        "",
    ]
    script_path.write_text("\n".join(lines + body_lines), encoding="utf-8")


def _build_task_config_payload(
    *,
    source_payload: dict[str, Any],
    planned_task: PlannedSlurmTask,
    restart_source: str | None,
) -> dict[str, Any]:
    payload = dict(source_payload)
    orchestrate = dict(payload.get("orchestrate") or {})
    model_cfg = dict(orchestrate.get(planned_task.task.model_key) or {})
    model_cfg["gpus"] = planned_task.effective_gpus
    if planned_task.dp_size > 1:
        model_cfg["data_parallel_size"] = planned_task.dp_size
    else:
        model_cfg.pop("data_parallel_size", None)
    orchestrate[planned_task.task.model_key] = model_cfg
    if restart_source:
        orchestrate["restart"] = restart_source
    payload["orchestrate"] = orchestrate
    return payload


def _extract_restart_source(payload: dict[str, Any]) -> str | None:
    orchestrate = payload.get("orchestrate")
    if not isinstance(orchestrate, dict):
        return None
    restart = orchestrate.get("restart")
    if restart is None:
        return None
    text = str(restart).strip()
    return text or None


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create(payload), f=str(path))


def _sbatch_line(flag: str, value: object) -> str | None:
    if value is None:
        return None
    return f"#SBATCH {flag}={value}"


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = ["render_bundle", "render_task_artifacts", "resolve_restart_source", "write_script"]
