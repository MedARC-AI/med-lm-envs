"""Planning helpers for Slurm-native orchestration submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

from medarc_verifiers.orchestrate.config import TaskSpec

_TASK_ALLOWED = re.compile(r"[^a-zA-Z0-9_.-]+")
_ALLOWED_SLURM_KEYS = {
    "job_name",
    "cpus_per_gpu",
    "time",
    "partition",
    "account",
    "qos",
    "mail_type",
    "mail_user",
    "slurm_resume",
}


def slug_task_id(task_id: str, *, fallback: str = "task") -> str:
    cleaned = _TASK_ALLOWED.sub("-", task_id).strip("-.")
    return cleaned or fallback


@dataclass(frozen=True)
class SlurmCliOverrides:
    cpus_per_gpu: int | None = None
    time: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    slurm_resume: bool | None = None


@dataclass(frozen=True)
class SlurmTaskOptions:
    job_name: str
    cpus_per_gpu: int | None = None
    time: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    slurm_resume: bool = False


@dataclass(frozen=True)
class PlannedSlurmTask:
    task: TaskSpec
    task_slug: str
    submission_order: int
    chain_index: int
    tp_size: int
    dp_size: int
    effective_gpus: int
    inner_run_id: str
    predecessor_task_id: str | None
    base_dependency: str | None
    options: SlurmTaskOptions


def build_submission_plan(
    tasks: list[TaskSpec],
    *,
    run_id: str,
    node_gpus: int,
    max_simultaneous_nodes: int,
    run_simultaneously: bool,
    base_dependency: str | None,
    cli_overrides: SlurmCliOverrides,
) -> list[PlannedSlurmTask]:
    if node_gpus < 1:
        raise ValueError("--node-gpus must be >= 1.")
    if max_simultaneous_nodes < 1:
        raise ValueError("--max-simultaneous-nodes must be >= 1.")

    prepared: list[tuple[int, TaskSpec, int, int, int, SlurmTaskOptions]] = []
    for original_index, task in enumerate(tasks):
        tp_size = resolve_tensor_parallel_size(task)
        if tp_size > node_gpus:
            raise ValueError(
                f"Task {task.task_id} ({task.job_config_path}) requires tensor_parallel_size={tp_size}, "
                f"which exceeds node_gpus={node_gpus}."
            )
        dp_size = max(1, node_gpus // tp_size)
        effective_gpus = tp_size * dp_size
        prepared.append(
            (
                original_index,
                task,
                tp_size,
                dp_size,
                effective_gpus,
                merge_slurm_options(task, cli_overrides=cli_overrides),
            )
        )
    prepared.sort(key=lambda item: (-item[2], item[0]))

    last_task_in_chain: dict[int, str] = {}
    planned: list[PlannedSlurmTask] = []
    for submission_order, (original_index, task, tp_size, dp_size, effective_gpus, options) in enumerate(prepared):
        if run_simultaneously:
            chain_index = submission_order
            predecessor_task_id = None
            task_base_dependency = base_dependency
        else:
            chain_index = submission_order % max_simultaneous_nodes
            predecessor_task_id = last_task_in_chain.get(chain_index)
            task_base_dependency = base_dependency if predecessor_task_id is None else None
        task_slug = slug_task_id(task.task_id)
        inner_run_id = f"{run_id}-{task_slug}"
        planned.append(
            PlannedSlurmTask(
                task=task,
                task_slug=task_slug,
                submission_order=submission_order,
                chain_index=chain_index,
                tp_size=tp_size,
                dp_size=dp_size,
                effective_gpus=effective_gpus,
                inner_run_id=inner_run_id,
                predecessor_task_id=predecessor_task_id,
                base_dependency=task_base_dependency,
                options=options,
            )
        )
        last_task_in_chain[chain_index] = task.task_id
    return planned


def resolve_tensor_parallel_size(task: TaskSpec) -> int:
    model_cfg = task.orchestrate.get(task.model_key, {}) or {}
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Task {task.task_id} orchestrate.{task.model_key} must be a mapping.")
    gpus = int(model_cfg.get("gpus", 1))
    data_parallel = int(model_cfg.get("data_parallel_size", 1) or 1)
    tensor_parallel = model_cfg.get("tensor_parallel_size")
    if tensor_parallel is not None:
        resolved = int(tensor_parallel)
    else:
        if gpus < 1:
            raise ValueError(f"Task {task.task_id} orchestrate.{task.model_key}.gpus must be >= 1.")
        if data_parallel < 1:
            raise ValueError(f"Task {task.task_id} orchestrate.{task.model_key}.data_parallel_size must be >= 1.")
        if gpus % data_parallel != 0:
            raise ValueError(
                f"Task {task.task_id} orchestrate.{task.model_key}.gpus={gpus} must be divisible by "
                f"data_parallel_size={data_parallel}."
            )
        resolved = gpus // data_parallel
    if resolved < 1:
        raise ValueError(f"Task {task.task_id} tensor_parallel_size must be >= 1.")
    return resolved


def merge_slurm_options(task: TaskSpec, *, cli_overrides: SlurmCliOverrides) -> SlurmTaskOptions:
    job_cfg = _validate_slurm_mapping(task.slurm, task_id=task.task_id)
    task_slug = slug_task_id(task.task_id)
    job_name = _slug_job_name(str(job_cfg.get("job_name") or task_slug))
    return SlurmTaskOptions(
        job_name=job_name,
        cpus_per_gpu=cli_overrides.cpus_per_gpu if cli_overrides.cpus_per_gpu is not None else _optional_int(job_cfg.get("cpus_per_gpu")),
        time=cli_overrides.time if cli_overrides.time is not None else _optional_str(job_cfg.get("time")),
        partition=cli_overrides.partition if cli_overrides.partition is not None else _optional_str(job_cfg.get("partition")),
        account=cli_overrides.account if cli_overrides.account is not None else _optional_str(job_cfg.get("account")),
        qos=cli_overrides.qos if cli_overrides.qos is not None else _optional_str(job_cfg.get("qos")),
        mail_type=cli_overrides.mail_type if cli_overrides.mail_type is not None else _optional_str(job_cfg.get("mail_type")),
        mail_user=cli_overrides.mail_user if cli_overrides.mail_user is not None else _optional_str(job_cfg.get("mail_user")),
        slurm_resume=cli_overrides.slurm_resume
        if cli_overrides.slurm_resume is not None
        else bool(job_cfg.get("slurm_resume", False)),
    )


def placeholder_dependency(task: PlannedSlurmTask, *, task_order: dict[str, int]) -> str | None:
    if task.predecessor_task_id is None:
        return None
    predecessor_order = task_order[task.predecessor_task_id] + 1
    return f"afterany:$JOBID_{predecessor_order}"


def _validate_slurm_mapping(mapping: Any, *, task_id: str) -> dict[str, Any]:
    if mapping is None:
        return {}
    if not isinstance(mapping, dict):
        raise ValueError(f"Task {task_id} slurm config must be a mapping.")
    unknown = sorted(set(mapping.keys()) - _ALLOWED_SLURM_KEYS)
    if unknown:
        raise ValueError(f"Task {task_id} has unknown slurm keys: {unknown}")
    return dict(mapping)


def _slug_job_name(value: str) -> str:
    return slug_task_id(value, fallback="job")


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "PlannedSlurmTask",
    "SlurmCliOverrides",
    "SlurmTaskOptions",
    "build_submission_plan",
    "merge_slurm_options",
    "placeholder_dependency",
    "resolve_tensor_parallel_size",
    "slug_task_id",
]
