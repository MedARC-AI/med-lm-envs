from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.cli import _validate_schedule, build_local_parser, build_parser, main
from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.resources import GpuInfo, PortOnlyResourceManager, ResourceError
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner


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


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_gpu_discovery_failure(monkeypatch, tmp_path: Path, runtime: str) -> None:
    def boom():
        raise ResourceError("boom")

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", boom)
    tasks = [_task(tmp_path, gpus=1)]

    with pytest.raises(ValueError, match="GPU discovery failed"):
        _validate_schedule(tasks, runtime=runtime, gpu_indices=None, port_range=(8000, 8001), max_parallel=1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_gpu_count(monkeypatch, tmp_path: Path, runtime: str) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1)],
    )
    tasks = [_task(tmp_path, gpus=4)]

    with pytest.raises(ValueError, match=r"requires gpus=4"):
        _validate_schedule(tasks, runtime=runtime, gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_rejects_invalid_local_launch_shape(monkeypatch, tmp_path: Path, runtime: str) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1), _gpu(2), _gpu(3)],
    )
    tasks = [_task(tmp_path, gpus=3)]

    with pytest.raises(ValueError, match=r"allocated_gpus=3 is invalid; allowed shapes are \[1, 2, 4, 8\]"):
        _validate_schedule(tasks, runtime=runtime, gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_contiguous_gpu_range(monkeypatch, tmp_path: Path, runtime: str) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.discover_gpus",
        lambda: [_gpu(0), _gpu(1), _gpu(2), _gpu(3)],
    )
    tasks = [_task(tmp_path, gpus=2)]

    with pytest.raises(ValueError, match="contiguous"):
        _validate_schedule(tasks, runtime=runtime, gpu_indices=[0, 2, 4], port_range=(8000, 8003), max_parallel=1)


def test_cli_validation_pyxis_skips_gpu_discovery(monkeypatch, tmp_path: Path) -> None:
    def boom():
        raise AssertionError("discover_gpus should not be called for pyxis validation")

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", boom)
    tasks = [_task(tmp_path, gpus=8)]

    _validate_schedule(tasks, runtime="pyxis", gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


def test_cli_validation_pyxis_requires_single_task_allocation_use(tmp_path: Path) -> None:
    tasks = [_task(tmp_path, gpus=8)]

    with pytest.raises(ValueError, match="max_parallel must be 1"):
        _validate_schedule(tasks, runtime="pyxis", gpu_indices=None, port_range=(8000, 8003), max_parallel=2)


def test_root_parser_accepts_local_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["local", "--plan", "plan.yaml", "--runtime", "podman"])

    assert args.command == "local"
    assert args.runtime == "podman"


def test_local_parser_accepts_runtime_choices() -> None:
    parser = build_local_parser()
    args = parser.parse_args(["--plan", "plan.yaml", "--runtime", "pyxis"])

    assert args.runtime == "pyxis"


def test_root_parser_rejects_removed_top_level_forms(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--plan", "plan.yaml"])

    stderr = capsys.readouterr().err
    assert "usage: medarc-orchestrate" in stderr
    assert "{local,slurm}" in stderr


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
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    rc = main(["local", "--plan", str(plan_path), "--runtime", "pyxis"])

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
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    rc = main(
        [
            "local",
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
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    rc = main(["local", "--job-config", str(job_cfg), "--runtime", "pyxis"])

    assert rc == 0
    assert captured["run_id"] == "shared-run-id"
    assert captured["output_root"] == Path("outputs") / "orchestrate" / "shared-run-id"


def test_cli_local_pyxis_derives_allocated_gpu_count_from_visible_devices(monkeypatch, tmp_path: Path) -> None:
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
        captured["allocated_gpu_count"] = self._options.allocated_gpu_count

    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,4,6")

    rc = main(["local", "--job-config", str(job_cfg), "--runtime", "pyxis"])

    assert rc == 0
    assert captured["allocated_gpu_count"] == 4


def test_cli_local_pyxis_dry_run_does_not_require_allocation_env(monkeypatch, tmp_path: Path, capsys) -> None:
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

    monkeypatch.delenv("MEDARC_ALLOCATED_GPU_COUNT", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("NVIDIA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)

    rc = main(["local", "--job-config", str(job_cfg), "--runtime", "pyxis", "--dry-run"])

    assert rc == 0
    assert capsys.readouterr().out.strip() == f"job:foo\tFoo/Bar\t{job_cfg.resolve()}"


def test_cli_local_podman_uses_gpu_resource_manager(monkeypatch, tmp_path: Path) -> None:
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

    class FakeResourceManager:
        def __init__(self, *, gpu_indices, port_range) -> None:
            captured["resource_manager"] = "gpu"
            captured["gpu_indices"] = gpu_indices
            captured["port_range"] = port_range

    class FakePortOnlyResourceManager:
        def __init__(self, *, port_range) -> None:
            captured["resource_manager"] = "port-only"
            captured["port_range"] = port_range

    def fake_run(self) -> None:
        captured["runtime"] = self._runtime

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.ResourceManager", FakeResourceManager)
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.PortOnlyResourceManager", FakePortOnlyResourceManager)
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(["local", "--job-config", str(job_cfg), "--runtime", "podman"])

    assert rc == 0
    assert captured["runtime"] == "podman"
    assert captured["resource_manager"] == "gpu"
    assert captured["gpu_indices"] is None
    assert captured["port_range"] == (8000, 8999)


def test_cli_defaults_to_podman_when_docker_package_is_unavailable(monkeypatch, tmp_path: Path) -> None:
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
        captured["runtime"] = self._runtime

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.shutil.which",
        lambda name: None if name == "docker" else "/usr/bin/podman",
    )
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(["local", "--job-config", str(job_cfg)])

    assert rc == 0
    assert captured["runtime"] == "podman"


def test_cli_defaults_to_podman_when_docker_binary_exists_but_sdk_is_missing(monkeypatch, tmp_path: Path) -> None:
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
        captured["runtime"] = self._runtime

    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.shutil.which",
        lambda name: "/usr/bin/docker" if name == "docker" else "/usr/bin/podman",
    )
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(["local", "--job-config", str(job_cfg)])

    assert rc == 0
    assert captured["runtime"] == "podman"


def test_runner_builds_podman_adapter_by_runtime(tmp_path: Path) -> None:
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
    task = TaskSpec(
        task_id="task-1",
        job_config_path=job_cfg,
        model_key="foo",
        model_id="Foo/Bar",
        orchestrate={"vllm-container": {"image": "fake"}, "foo": {"gpus": 1, "serve": {}}},
    )
    plan = PlanConfig(job_configs=[job_cfg])
    runner = OrchestratorRunner(
        plan,
        [task],
        PortOnlyResourceManager(port_range=(9000, 9010)),
        options=OrchestratorOptions(
            run_id="run-1",
            output_root=tmp_path / "outputs",
            readiness_timeout_s=1,
            max_parallel=1,
        ),
        runtime="podman",
        use_dashboard=False,
    )

    assert runner._runtime == "podman"
    assert runner._runtime_adapter.__class__.__name__ == "PodmanRuntimeAdapter"


def test_cli_kill_orphans_uses_podman_cleanup_for_explicit_podman_runtime(monkeypatch, tmp_path: Path, capsys) -> None:
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

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_podman_orphans",
        lambda run_id=None: calls.append(("podman", run_id)) or ["podman-task"],
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_docker_orphans",
        lambda run_id=None: calls.append(("docker", run_id)) or ["docker-task"],
    )

    rc = main(["local", "--job-config", str(job_cfg), "--runtime", "podman", "--kill-orphans", "--run-id", "run-1"])

    assert rc == 0
    assert calls == [("podman", "run-1")]
    assert capsys.readouterr().out.strip() == "podman-task"


def test_cli_kill_orphans_uses_podman_cleanup_for_autodetected_runtime(monkeypatch, tmp_path: Path, capsys) -> None:
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

    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.shutil.which",
        lambda name: None if name == "docker" else "/usr/bin/podman",
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_podman_orphans",
        lambda run_id=None: calls.append(("podman", run_id)) or ["podman-task"],
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_docker_orphans",
        lambda run_id=None: calls.append(("docker", run_id)) or ["docker-task"],
    )

    rc = main(["local", "--job-config", str(job_cfg), "--kill-orphans"])

    assert rc == 0
    assert calls == [("podman", None)]
    assert capsys.readouterr().out.strip() == "podman-task"


def test_port_only_resource_manager_skips_gpus() -> None:
    rm = PortOnlyResourceManager(port_range=(9000, 9010))

    assert rm.available_gpus() == []
    assert rm.reserve_gpus("task-1", count=4) == []
    rm.release_gpus([0, 1, 2, 3])
