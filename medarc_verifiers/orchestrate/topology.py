"""Shared GPU topology derivation for orchestrator task planning and execution."""

from __future__ import annotations

from dataclasses import dataclass

from medarc_verifiers.orchestrate.bundle import ResolvedTaskSpec
from medarc_verifiers.orchestrate.config import TaskSpec

ALLOWED_ALLOCATED_GPU_SHAPES = frozenset({1, 2, 4, 8})


@dataclass(frozen=True)
class ResolvedTopology:
    gpus: int
    allocated_gpus: int
    tensor_parallel_size: int
    data_parallel_size: int
    vllm_world_size: int


def task_sort_key(task: TaskSpec) -> tuple[int, int, str]:
    minimum_gpus = minimum_required_gpus(task)
    tensor_parallel_size = configured_tensor_parallel_size(task)
    return (-minimum_gpus, -tensor_parallel_size, task.task_id)


def minimum_required_gpus(task: TaskSpec) -> int:
    model_cfg = _task_model_cfg(task)
    gpus = int(model_cfg.get("gpus", 1) or 1)
    if gpus < 1:
        raise ValueError(f"Task {task.task_id} orchestrate.{task.model_key}.gpus must be >= 1.")
    return gpus


def configured_tensor_parallel_size(task: TaskSpec) -> int:
    model_cfg = _task_model_cfg(task)
    tensor_parallel = int(model_cfg.get("tensor_parallel_size", 1) or 1)
    if tensor_parallel < 1:
        raise ValueError(f"Task {task.task_id} orchestrate.{task.model_key}.tensor_parallel_size must be >= 1.")
    return tensor_parallel


def resolve_topology(task: TaskSpec, *, allocated_gpus: int, allow_explicit_data_parallel: bool = True) -> ResolvedTopology:
    model_cfg = _task_model_cfg(task)
    return _resolve_topology(
        task_id=task.task_id,
        gpus=minimum_required_gpus(task),
        allocated_gpus=allocated_gpus,
        tensor_parallel_size=configured_tensor_parallel_size(task),
        explicit_data_parallel=model_cfg.get("data_parallel_size"),
        allow_explicit_data_parallel=allow_explicit_data_parallel,
    )


def resolve_task_spec_topology(
    task_spec: ResolvedTaskSpec,
    *,
    allocated_gpus: int,
    allow_explicit_data_parallel: bool = True,
) -> ResolvedTopology:
    return _resolve_topology(
        task_id=task_spec.task_id,
        gpus=task_spec.gpus,
        allocated_gpus=allocated_gpus,
        tensor_parallel_size=task_spec.tensor_parallel_size,
        explicit_data_parallel=task_spec.data_parallel_size,
        allow_explicit_data_parallel=allow_explicit_data_parallel,
    )


def _resolve_topology(
    *,
    task_id: str,
    gpus: int,
    allocated_gpus: int,
    tensor_parallel_size: int,
    explicit_data_parallel: object,
    allow_explicit_data_parallel: bool,
) -> ResolvedTopology:
    if gpus < 1:
        raise ValueError(f"Task {task_id} gpus must be >= 1.")
    if tensor_parallel_size < 1:
        raise ValueError(f"Task {task_id} tensor_parallel_size must be >= 1.")

    if allocated_gpus not in ALLOWED_ALLOCATED_GPU_SHAPES:
        raise ValueError(
            f"Task {task_id} allocated_gpus={allocated_gpus} is invalid; allowed shapes are "
            f"{sorted(ALLOWED_ALLOCATED_GPU_SHAPES)}."
        )
    if allocated_gpus < gpus:
        raise ValueError(
            f"Task {task_id} requires gpus={gpus} minimum outer allocation, but allocated_gpus={allocated_gpus}."
        )
    if allocated_gpus < tensor_parallel_size:
        raise ValueError(
            f"Task {task_id} allocated_gpus={allocated_gpus} is smaller than "
            f"tensor_parallel_size={tensor_parallel_size}."
        )
    if allocated_gpus % tensor_parallel_size != 0:
        raise ValueError(
            f"Task {task_id} allocated_gpus={allocated_gpus} must be divisible by "
            f"tensor_parallel_size={tensor_parallel_size}."
        )

    data_parallel_size = allocated_gpus // tensor_parallel_size
    if explicit_data_parallel is not None:
        explicit_value = int(explicit_data_parallel)
        if explicit_value != data_parallel_size:
            raise ValueError(
                f"Task {task_id} explicit data_parallel_size={explicit_value} does not match derived "
                f"value {data_parallel_size} for allocated_gpus={allocated_gpus} and "
                f"tensor_parallel_size={tensor_parallel_size}."
            )
        if not allow_explicit_data_parallel:
            raise ValueError(
                f"Task {task_id} must not set explicit data_parallel_size; it is derived from the allocation."
            )

    return ResolvedTopology(
        gpus=gpus,
        allocated_gpus=allocated_gpus,
        tensor_parallel_size=tensor_parallel_size,
        data_parallel_size=data_parallel_size,
        vllm_world_size=tensor_parallel_size * data_parallel_size,
    )


def _task_model_cfg(task: TaskSpec) -> dict[str, object]:
    model_cfg = task.orchestrate.get(task.model_key, {}) or {}
    if not isinstance(model_cfg, dict):
        raise ValueError(f"Task {task.task_id} orchestrate.{task.model_key} must be a mapping.")
    return model_cfg


__all__ = [
    "ALLOWED_ALLOCATED_GPU_SHAPES",
    "ResolvedTopology",
    "configured_tensor_parallel_size",
    "minimum_required_gpus",
    "resolve_topology",
    "resolve_task_spec_topology",
    "task_sort_key",
]
