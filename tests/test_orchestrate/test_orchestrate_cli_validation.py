from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.cli import _validate_schedule, build_parser, main
from medarc_verifiers.orchestrate.config import TaskSpec
from medarc_verifiers.orchestrate.resources import GpuInfo, PortOnlyResourceManager, ResourceError
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


def test_cli_job_config_flag_parses_multiple_values() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--job-config", "configs/job-a.yaml", "--job-config", "configs/job-b.yaml", "--runtime", "pyxis"]
    )

    assert args.job_configs == [Path("configs/job-a.yaml"), Path("configs/job-b.yaml")]
    assert args.plan is None
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


def test_cli_direct_job_configs_launch_without_plan(monkeypatch, tmp_path: Path) -> None:
    for name in ("job-a.yaml", "job-b.yaml"):
        (tmp_path / name).write_text(
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

    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["runtime"] = self._runtime
        captured["job_configs"] = list(self._plan.job_configs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", lambda: [_gpu(0), _gpu(1)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(
        [
            "--job-config",
            str(tmp_path / "job-a.yaml"),
            "--job-config",
            str(tmp_path / "job-b.yaml"),
            "--runtime",
            "pyxis",
            "--name",
            "bundle",
        ]
    )

    assert rc == 0
    assert captured["runtime"] == "pyxis"
    assert captured["job_configs"] == [(tmp_path / "job-a.yaml").resolve(), (tmp_path / "job-b.yaml").resolve()]


def test_cli_default_run_id_uses_shared_generator(monkeypatch, tmp_path: Path) -> None:
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

    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["run_id"] = self._options.run_id
        captured["output_root"] = self._options.output_root

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.generate_run_id", lambda name: "shared-run-id")
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(["--job-config", str(job_cfg), "--runtime", "pyxis"])

    assert rc == 0
    assert captured["run_id"] == "shared-run-id"
    assert captured["output_root"] == Path("outputs") / "orchestrate" / "shared-run-id"


def test_port_only_resource_manager_skips_gpus() -> None:
    rm = PortOnlyResourceManager(port_range=(9000, 9010))

    assert rm.available_gpus() == []
    assert rm.reserve_gpus("task-1", count=4) == []
    rm.release_gpus([0, 1, 2, 3])
