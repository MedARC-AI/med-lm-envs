"""Planning helpers for Slurm-native orchestration submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from typing import Any
import re

from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.topology import ResolvedTopology, resolve_topology, task_sort_key

_TASK_ALLOWED = re.compile(r"[^a-zA-Z0-9_.-]+")
_ALLOWED_SLURM_KEYS = {
    "job_name",
    "cpus_per_gpu",
    "time",
    "partition",
    "account",
    "qos",
    "nice",
    "mail_type",
    "mail_user",
    "slurm_resume",
}


def slug_task_id(task_id: str, *, fallback: str = "task") -> str:
    cleaned = _TASK_ALLOWED.sub("-", task_id).strip("-.")
    return cleaned or fallback


class SlurmSubmissionOverrides(Protocol):
    cpus_per_gpu: int | None
    time: str | None
    partition: str | None
    account: str | None
    qos: str | None
    nice: int | None
    mail_type: str | None
    mail_user: str | None
    slurm_resume: bool | None


@dataclass(frozen=True)
class SlurmTaskOptions:
    job_name: str
    cpus_per_gpu: int | None = None
    time: str | None = None
    partition: str | None = None
    account: str | None = None
    qos: str | None = None
    nice: int | None = None
    mail_type: str | None = None
    mail_user: str | None = None
    slurm_resume: bool = False


@dataclass(frozen=True)
class PlannedSlurmTask:
    task: TaskSpec
    task_slug: str
    submission_order: int
    chain_index: int
    gpus: int
    allocated_gpus: int
    tensor_parallel_size: int
    data_parallel_size: int
    vllm_world_size: int
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
    submission_options: SlurmSubmissionOverrides,
) -> list[PlannedSlurmTask]:
    if max_simultaneous_nodes < 1:
        raise ValueError("--max-simultaneous-nodes must be >= 1.")

    prepared: list[tuple[tuple[int, int, str], int, TaskSpec, ResolvedTopology, SlurmTaskOptions]] = []
    for original_index, task in enumerate(tasks):
        topology = resolve_topology(task, allocated_gpus=node_gpus)
        prepared.append(
            (
                task_sort_key(task),
                original_index,
                task,
                topology,
                merge_slurm_options(task, submission_options=submission_options),
            )
        )
    prepared.sort(key=lambda item: (item[0], item[1]))

    last_task_in_chain: dict[int, str] = {}
    planned: list[PlannedSlurmTask] = []
    for submission_order, (_, _, task, topology, options) in enumerate(prepared):
        if run_simultaneously:
            chain_index = submission_order
            predecessor_task_id = None
            task_base_dependency = base_dependency
        else:
            chain_index = submission_order % max_simultaneous_nodes
            predecessor_task_id = last_task_in_chain.get(chain_index)
            task_base_dependency = base_dependency if predecessor_task_id is None else None
        task_slug = slug_task_id(task.task_id)
        planned.append(
            PlannedSlurmTask(
                task=task,
                task_slug=task_slug,
                submission_order=submission_order,
                chain_index=chain_index,
                gpus=topology.gpus,
                allocated_gpus=topology.allocated_gpus,
                tensor_parallel_size=topology.tensor_parallel_size,
                data_parallel_size=topology.data_parallel_size,
                vllm_world_size=topology.vllm_world_size,
                predecessor_task_id=predecessor_task_id,
                base_dependency=task_base_dependency,
                options=options,
            )
        )
        last_task_in_chain[chain_index] = task.task_id
    return planned


def merge_slurm_options(task: TaskSpec, *, submission_options: SlurmSubmissionOverrides) -> SlurmTaskOptions:
    job_cfg = _validate_slurm_mapping(task.slurm, task_id=task.task_id)
    task_slug = slug_task_id(task.task_id)
    job_name = _slug_job_name(str(job_cfg.get("job_name") or task_slug))
    return SlurmTaskOptions(
        job_name=job_name,
        cpus_per_gpu=submission_options.cpus_per_gpu
        if submission_options.cpus_per_gpu is not None
        else _optional_int(job_cfg.get("cpus_per_gpu")),
        time=submission_options.time if submission_options.time is not None else _optional_str(job_cfg.get("time")),
        partition=submission_options.partition
        if submission_options.partition is not None
        else _optional_str(job_cfg.get("partition")),
        account=submission_options.account
        if submission_options.account is not None
        else _optional_str(job_cfg.get("account")),
        qos=submission_options.qos if submission_options.qos is not None else _optional_str(job_cfg.get("qos")),
        nice=submission_options.nice if submission_options.nice is not None else _optional_int(job_cfg.get("nice")),
        mail_type=submission_options.mail_type
        if submission_options.mail_type is not None
        else _optional_str(job_cfg.get("mail_type")),
        mail_user=submission_options.mail_user
        if submission_options.mail_user is not None
        else _optional_str(job_cfg.get("mail_user")),
        slurm_resume=submission_options.slurm_resume
        if submission_options.slurm_resume is not None
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
    "SlurmSubmissionOverrides",
    "SlurmTaskOptions",
    "build_submission_plan",
    "merge_slurm_options",
    "placeholder_dependency",
    "slug_task_id",
]
