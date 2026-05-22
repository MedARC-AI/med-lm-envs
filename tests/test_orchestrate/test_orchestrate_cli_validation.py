from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.cli import build_local_parser, build_parser, main
from medarc_verifiers.orchestrate.config import PlanConfig, TaskSpec
from medarc_verifiers.orchestrate.launch import validate_local_schedule
from medarc_verifiers.orchestrate.resources import GpuInfo, PortOnlyResourceManager, ResourceError
from medarc_verifiers.orchestrate.run import OrchestratorOptions, OrchestratorRunner


def _gpu(index: int) -> GpuInfo:
    return GpuInfo(index=index, total_gb=80.0, free_gb=80.0)


def _write_eval_config(tmp_path: Path, name: str = "job.toml") -> Path:
    path = tmp_path / name
    path.write_text(
        """
endpoint_id = "foo"
endpoints_path = "endpoints.toml"

[[eval]]
env_id = "medqa"
num_examples = 1
rollouts_per_example = 1
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_endpoint_registry(tmp_path: Path, *, gpus: int = 1) -> Path:
    path = tmp_path / "endpoints.toml"
    path.write_text(
        f"""
[[endpoint]]
endpoint_id = "foo"
model = "Foo/Bar"

[endpoint.orchestrate.vllm]
gpus = {gpus}
tensor_parallel_size = {gpus}

[endpoint.orchestrate.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_plan(tmp_path: Path, job_cfg: Path, *, runtime: str | None = None) -> Path:
    plan = tmp_path / "plan.yaml"
    runtime_line = f"runtime: {runtime}\n" if runtime else ""
    plan.write_text(f"job_configs:\n  - {job_cfg.name}\n{runtime_line}", encoding="utf-8")
    return plan


def _task(tmp_path: Path, *, gpus: int) -> TaskSpec:
    return TaskSpec(
        task_id=f"task-{gpus}",
        job_config_path=tmp_path / f"job-{gpus}.toml",
        model_key="Foo-Bar",
        model_id="Foo/Bar",
        orchestrate={"vllm": {"gpus": gpus, "tensor_parallel_size": gpus}, "container": {"image": "fake"}},
    )


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_gpu_discovery_failure(monkeypatch, tmp_path: Path, runtime: str) -> None:
    def boom():
        raise ResourceError("boom")

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", boom)
    tasks = [_task(tmp_path, gpus=1)]

    with pytest.raises(ValueError, match="GPU discovery failed"):
        validate_local_schedule(tasks, runtime=runtime, gpu_indices=None, port_range=(8000, 8001), max_parallel=1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_gpu_count(monkeypatch, tmp_path: Path, runtime: str) -> None:
    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0), _gpu(1)])
    tasks = [_task(tmp_path, gpus=4)]

    with pytest.raises(ValueError, match=r"requires gpus=4"):
        validate_local_schedule(tasks, runtime=runtime, gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_rejects_invalid_local_launch_shape(monkeypatch, tmp_path: Path, runtime: str) -> None:
    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0), _gpu(1), _gpu(2), _gpu(3)])
    tasks = [_task(tmp_path, gpus=3)]

    with pytest.raises(ValueError, match=r"allocated_gpus=3 is invalid; allowed shapes are \[1, 2, 4, 8\]"):
        validate_local_schedule(tasks, runtime=runtime, gpu_indices=None, port_range=(8000, 8003), max_parallel=1)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_cli_validation_contiguous_gpu_range(monkeypatch, tmp_path: Path, runtime: str) -> None:
    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0), _gpu(1), _gpu(2), _gpu(3)])
    tasks = [_task(tmp_path, gpus=2)]

    with pytest.raises(ValueError, match="contiguous"):
        validate_local_schedule(tasks, runtime=runtime, gpu_indices=[0, 2, 4], port_range=(8000, 8003), max_parallel=1)


def test_cli_validation_pyxis_skips_gpu_discovery(monkeypatch, tmp_path: Path) -> None:
    def boom():
        raise AssertionError("discover_gpus should not be called for pyxis validation")

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", boom)
    validate_local_schedule(
        [_task(tmp_path, gpus=8)], runtime="pyxis", gpu_indices=None, port_range=(8000, 8003), max_parallel=1
    )


def test_cli_validation_pyxis_requires_single_task_allocation_use(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_parallel must be 1"):
        validate_local_schedule(
            [_task(tmp_path, gpus=8)], runtime="pyxis", gpu_indices=None, port_range=(8000, 8003), max_parallel=2
        )


def test_root_parser_accepts_run_subcommand() -> None:
    args = build_parser().parse_args(["run", "--plan", "plan.yaml", "--runtime", "podman"])
    assert args.command == "run"
    assert args.runtime == "podman"


def test_local_parser_accepts_runtime_choices() -> None:
    args = build_local_parser().parse_args(["--plan", "plan.yaml", "--runtime", "pyxis"])
    assert args.runtime == "pyxis"


def test_root_parser_rejects_removed_top_level_forms(capsys) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--plan", "plan.yaml"])
    stderr = capsys.readouterr().err
    assert "usage: medarc-orchestrate" in stderr
    assert "{run,local,slurm}" in stderr


def test_cli_runtime_precedence_cli_over_plan(monkeypatch, tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
    plan_path = _write_plan(tmp_path, job_cfg, runtime="docker")
    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["runtime"] = self._runtime

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    assert main(["run", "--plan", str(plan_path), "--runtime", "pyxis"]) == 0
    assert captured["runtime"] == "pyxis"


def test_cli_direct_job_configs_launch_without_plan(monkeypatch, tmp_path: Path) -> None:
    _write_eval_config(tmp_path, "job-a.toml")
    _write_eval_config(tmp_path, "job-b.toml")
    (tmp_path / "job-b.toml").write_text(
        (tmp_path / "job-b.toml").read_text(encoding="utf-8").replace('env_id = "medqa"', 'env_id = "pubmedqa"'),
        encoding="utf-8",
    )
    _write_endpoint_registry(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["runtime"] = self._runtime
        captured["job_configs"] = list(self._plan.job_configs)

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0), _gpu(1)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    rc = main(
        [
            "run",
            "--job-config",
            str(tmp_path / "job-a.toml"),
            "--job-config",
            str(tmp_path / "job-b.toml"),
            "--runtime",
            "pyxis",
            "--name",
            "bundle",
        ]
    )

    assert rc == 0
    assert captured["runtime"] == "pyxis"
    assert captured["job_configs"] == [(tmp_path / "job-a.toml").resolve(), (tmp_path / "job-b.toml").resolve()]


def test_cli_default_run_id_uses_shared_generator(monkeypatch, tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["run_id"] = self._options.run_id
        captured["output_root"] = self._options.output_root

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.generate_run_id", lambda name: "shared-run-id")
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")

    rc = main(
        ["run", "--job-config", str(job_cfg), "--runtime", "pyxis"]
    )

    assert rc == 0
    assert captured["run_id"] == "shared-run-id"
    assert captured["output_root"] == Path("outputs") / "orchestrate" / "shared-run-id"


def test_cli_local_pyxis_derives_allocated_gpu_count_from_visible_devices(monkeypatch, tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["allocated_gpu_count"] = self._options.allocated_gpu_count

    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,4,6")

    rc = main(
        ["run", "--job-config", str(job_cfg), "--runtime", "pyxis"]
    )

    assert rc == 0
    assert captured["allocated_gpu_count"] == 4


def test_cli_prune_logs_override_reaches_runner(monkeypatch, tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["prune_logs_on_success"] = self._options.prune_logs_on_success

    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)
    monkeypatch.setenv("MEDARC_ALLOCATED_GPU_COUNT", "1")

    rc = main(
        [
            "run",
            "--job-config",
            str(job_cfg),
            "--runtime",
            "pyxis",
            "--prune-logs-on-success",
        ]
    )

    assert rc == 0
    assert captured["prune_logs_on_success"] is True


def test_cli_local_pyxis_dry_run_does_not_require_allocation_env(monkeypatch, tmp_path: Path, capsys) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
    for key in (
        "MEDARC_ALLOCATED_GPU_COUNT",
        "SLURM_STEP_GPUS",
        "SLURM_JOB_GPUS",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "SLURM_GPUS_ON_NODE",
    ):
        monkeypatch.delenv(key, raising=False)

    rc = main(
        [
            "run",
            "--job-config",
            str(job_cfg),
            "--runtime",
            "pyxis",
            "--dry-run",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out.strip() == f"job:Foo-Bar\tFoo/Bar\t{job_cfg.resolve()}"


def test_cli_local_podman_uses_gpu_resource_manager(monkeypatch, tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
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

    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.ResourceManager", FakeResourceManager)
    monkeypatch.setattr("medarc_verifiers.orchestrate.cli.PortOnlyResourceManager", FakePortOnlyResourceManager)
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    rc = main(
        ["run", "--job-config", str(job_cfg), "--runtime", "podman"]
    )

    assert rc == 0
    assert captured["runtime"] == "podman"
    assert captured["resource_manager"] == "gpu"
    assert captured["gpu_indices"] is None
    assert captured["port_range"] == (8000, 8999)


def test_cli_defaults_to_podman_when_docker_probe_fails(monkeypatch, tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    _write_endpoint_registry(tmp_path)
    captured: dict[str, object] = {}

    def fake_run(self) -> None:
        captured["runtime"] = self._runtime

    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.docker_available",
        lambda: (False, "docker unavailable"),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.podman_available",
        lambda: (True, "podman ok"),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.pyxis_available_inside_slurm",
        lambda: (_ for _ in ()).throw(AssertionError("pyxis probe should not run after podman")),
    )
    monkeypatch.setattr("medarc_verifiers.orchestrate.launch.discover_gpus", lambda: [_gpu(0)])
    monkeypatch.setattr(OrchestratorRunner, "run", fake_run)

    assert main(["run", "--job-config", str(job_cfg)]) == 0
    assert captured["runtime"] == "podman"


def test_runner_builds_podman_adapter_by_runtime(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path)
    task = TaskSpec(
        task_id="task-1",
        job_config_path=job_cfg,
        model_key="Foo-Bar",
        model_id="Foo/Bar",
        orchestrate={"vllm": {"gpus": 1, "tensor_parallel_size": 1, "serve": {}}, "container": {"image": "fake"}},
    )
    runner = OrchestratorRunner(
        PlanConfig(job_configs=[job_cfg]),
        [task],
        PortOnlyResourceManager(port_range=(9000, 9010)),
        options=OrchestratorOptions(
            run_id="run-1", output_root=tmp_path / "outputs", readiness_timeout_s=1, max_parallel=1
        ),
        runtime="podman",
        use_dashboard=False,
    )

    assert runner._runtime == "podman"
    assert runner._runtime_adapter.__class__.__name__ == "PodmanRuntimeAdapter"


def test_cli_kill_orphans_uses_podman_cleanup_for_explicit_podman_runtime(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_podman_orphans",
        lambda run_id=None: calls.append(("podman", run_id)) or ["podman-task"],
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_docker_orphans",
        lambda run_id=None: calls.append(("docker", run_id)) or ["docker-task"],
    )

    rc = main(
        [
            "run",
            "--runtime",
            "podman",
            "--kill-orphans",
            "--run-id",
            "run-1",
        ]
    )

    assert rc == 0
    assert calls == [("podman", "run-1")]
    assert capsys.readouterr().out.strip() == "podman-task"


def test_cli_kill_orphans_defaults_to_docker_without_runtime_probes(monkeypatch, capsys) -> None:
    calls: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.docker_available",
        lambda: (_ for _ in ()).throw(AssertionError("docker probe should not run for cleanup")),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.podman_available",
        lambda: (_ for _ in ()).throw(AssertionError("podman probe should not run for cleanup")),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_podman_orphans",
        lambda run_id=None: calls.append(("podman", run_id)) or ["podman-task"],
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.cli.cleanup_docker_orphans",
        lambda run_id=None: calls.append(("docker", run_id)) or ["docker-task"],
    )

    rc = main(["run", "--kill-orphans"])

    assert rc == 0
    assert calls == [("docker", None)]
    assert capsys.readouterr().out.strip() == "docker-task"


def test_port_only_resource_manager_skips_gpus() -> None:
    rm = PortOnlyResourceManager(port_range=(9000, 9010))

    assert rm.available_gpus() == []
    assert rm.reserve_gpus("task-1", count=4) == []
    rm.release_gpus([0, 1, 2, 3])
