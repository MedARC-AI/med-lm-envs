from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.cli import _validate_schedule, build_parser, main
from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import GpuInfo, ResourceError
from medarc_verifiers.orchestrate.run import OrchestratorRunner


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
        _validate_schedule(tasks, runtime="docker", gpu_indices=None, port_range=(8000, 8001), max_parallel=1)


def test_cli_validation_gpu_count(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1)],
    )
    tasks = [_task(tmp_path, gpus=3)]

    with pytest.raises(ValueError, match="requests 3 GPUs"):
        _validate_schedule(tasks, runtime="docker", gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


def test_cli_validation_contiguous_gpu_range(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1), _gpu(2), _gpu(3)],
    )
    tasks = [_task(tmp_path, gpus=2)]

    with pytest.raises(ValueError, match="contiguous"):
        _validate_schedule(tasks, runtime="docker", gpu_indices=[0, 2, 4], port_range=(8000, 8003), max_parallel=1)


def test_cli_validation_pyxis_skips_gpu_discovery(monkeypatch, tmp_path: Path) -> None:
    def boom():
        raise AssertionError("discover_gpus should not be called for pyxis validation")

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", boom)
    tasks = [_task(tmp_path, gpus=8)]

    _validate_schedule(tasks, runtime="pyxis", gpu_indices=None, port_range=(8000, 8003), max_parallel=2)


def test_cli_runtime_flag_parses() -> None:
    parser = build_parser()
    args = parser.parse_args(["--plan", "plan.yaml", "--runtime", "pyxis"])

    assert args.runtime == "pyxis"


def test_cli_runtime_precedence_cli_over_plan(monkeypatch, tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    job_cfg.write_text(
        """
models:
  foo:
    model: Foo/Bar
orchestrate:
  vllm-container:
    image: fake
  foo:
    gpus: 1
    serve: {}
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"""
job_configs:
  - {job_cfg.name}
runtime: docker
""".lstrip(),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["runtime"] = self._runtime

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(["--plan", str(plan_path), "--runtime", "pyxis"])

    assert rc == 0
    assert captured["runtime"] == "pyxis"
