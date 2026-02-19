from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.cli import _validate_schedule
from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import GpuInfo, ResourceError


def _gpu(index: int) -> GpuInfo:
    return GpuInfo(index=index, total_gb=80.0, free_gb=80.0)


def _task(tmp_path: Path, *, gpus: int) -> TaskSpec:
    return TaskSpec(
        task_id=f"task-{gpus}",
        job_config_path=tmp_path / f"job-{gpus}.yaml",
        model_key="foo",
        model_id="Foo/Bar",
        orchestrate={"foo": {"gpus": gpus}},
    )


def test_cli_validation_gpu_discovery_failure(monkeypatch, tmp_path: Path) -> None:
    def boom():
        raise ResourceError("boom")

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", boom)
    tasks = [_task(tmp_path, gpus=1)]

    with pytest.raises(ValueError, match="GPU discovery failed"):
        _validate_schedule(tasks, gpu_indices=None, port_range=(8000, 8001), max_parallel=1)


def test_cli_validation_gpu_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1)],
    )
    tasks = [_task(tmp_path, gpus=3)]

    with pytest.raises(ValueError, match="requests 3 GPUs"):
        _validate_schedule(tasks, gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


def test_cli_validation_contiguous_gpu_range(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1), _gpu(2), _gpu(3)],
    )
    tasks = [_task(tmp_path, gpus=2)]

    with pytest.raises(ValueError, match="contiguous"):
        _validate_schedule(tasks, gpu_indices=[0, 2, 4], port_range=(8000, 8003), max_parallel=1)
