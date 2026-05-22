from argparse import Namespace
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import PlanConfig, expand_tasks, resolve_default_endpoints_path
from medarc_verifiers.orchestrate.launch import (
    derive_local_max_parallel,
    resolve_launch_plan,
    resolve_runtime,
    resolve_status_target,
    validate_slurm_plan_fields,
)


def _args(**overrides):
    defaults = {
        "plan": None,
        "job_configs": None,
        "name": None,
        "env_file": None,
        "runtime": "pyxis",
        "gpu_range": None,
        "port_range": None,
        "run_id": None,
        "output_dir": None,
        "max_parallel": None,
        "readiness_timeout_s": None,
        "no_uv_run": False,
        "eval_images_config": None,
        "endpoints_path": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def _write_endpoint_registry(path: Path, *, endpoint_id: str = "foo", model: str = "Foo/Bar", gpus: int = 1) -> Path:
    path.write_text(
        f"""
[[endpoint]]
endpoint_id = "{endpoint_id}"
model = "{model}"

[endpoint.orchestrate.vllm]
gpus = {gpus}

[endpoint.orchestrate.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )
    return path


def _write_eval_config(path: Path, *, endpoint_id: str = "foo", endpoints_path: str | None = None) -> Path:
    endpoint_line = f'endpoints_path = "{endpoints_path}"\n' if endpoints_path is not None else ""
    path.write_text(
        f"""
endpoint_id = "{endpoint_id}"
{endpoint_line}
[[eval]]
env_id = "medqa"
num_examples = 1
rollouts_per_example = 1
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_default_endpoint_path_prefers_medmarks_then_endpoints(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    endpoints = _write_endpoint_registry(configs / "endpoints.toml")

    assert resolve_default_endpoints_path(tmp_path) == endpoints.resolve()

    medmarks = _write_endpoint_registry(configs / "medmarks-endpoints.toml")
    assert resolve_default_endpoints_path(tmp_path) == medmarks.resolve()


def test_expand_tasks_honors_job_endpoint_before_default(tmp_path: Path) -> None:
    job_endpoints = _write_endpoint_registry(tmp_path / "job-endpoints.toml", model="Foo/Job")
    default_endpoints = _write_endpoint_registry(tmp_path / "default-endpoints.toml", model="Foo/Default")
    job = _write_eval_config(tmp_path / "job.toml", endpoints_path=job_endpoints.name)

    task = expand_tasks(PlanConfig(job_configs=[job]), default_endpoints_path=default_endpoints)[0]

    assert task.model_id == "Foo/Job"
    assert task.endpoints_path == job_endpoints.resolve()


def test_launch_resolver_uses_default_endpoint_when_job_has_none(tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    endpoints = _write_endpoint_registry(configs / "endpoints.toml")
    job = _write_eval_config(tmp_path / "job.toml")

    plan = resolve_launch_plan(_args(job_configs=[job]), backend="local", cwd=tmp_path)

    assert plan.endpoint_registry_paths == (endpoints.resolve(),)
    assert plan.tasks[0].model_id == "Foo/Bar"


def test_launch_resolver_explicit_endpoint_overrides_job_endpoint(tmp_path: Path) -> None:
    job_endpoints = _write_endpoint_registry(tmp_path / "job-endpoints.toml", model="Foo/Job")
    explicit = _write_endpoint_registry(tmp_path / "explicit.toml", model="Foo/Explicit")
    job = _write_eval_config(tmp_path / "job.toml", endpoints_path=job_endpoints.name)

    plan = resolve_launch_plan(_args(job_configs=[job], endpoints_path=explicit), backend="local", cwd=tmp_path)

    assert plan.endpoint_registry_paths == (explicit.resolve(),)
    assert plan.tasks[0].model_id == "Foo/Explicit"


def test_resolve_runtime_local_auto_uses_mocked_probe_order(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.docker_available",
        lambda: calls.append("docker") or (False, "no docker"),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.podman_available",
        lambda: calls.append("podman") or (True, "podman ok"),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.pyxis_available_inside_slurm",
        lambda: calls.append("pyxis") or (True, "pyxis ok"),
    )

    assert resolve_runtime(None, backend="local") == "podman"
    assert calls == ["docker", "podman"]


def test_resolve_runtime_slurm_skips_local_probes(monkeypatch) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.docker_available",
        lambda: (_ for _ in ()).throw(AssertionError("docker probe should not run")),
    )
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.runtime_probe.podman_available",
        lambda: (_ for _ in ()).throw(AssertionError("podman probe should not run")),
    )

    assert resolve_runtime(None, backend="slurm") == "pyxis"


def test_slurm_plan_rejects_local_only_fields(tmp_path: Path) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"], max_parallel=2)

    with pytest.raises(ValueError, match="local-only fields: max_parallel"):
        validate_slurm_plan_fields(plan)


def test_slurm_plan_rejects_disabled_uv_run(tmp_path: Path) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"], uv_run=False)

    with pytest.raises(ValueError, match="local-only fields: uv_run"):
        validate_slurm_plan_fields(plan)


def test_slurm_plan_rejects_explicit_default_uv_run(tmp_path: Path) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"], uv_run=True)

    with pytest.raises(ValueError, match="local-only fields: uv_run"):
        validate_slurm_plan_fields(plan)


def test_slurm_plan_rejects_explicit_default_resume_flags(tmp_path: Path) -> None:
    plan = PlanConfig(job_configs=[tmp_path / "job.toml"], resume=False, rerun_failed=False)

    with pytest.raises(ValueError, match="local-only fields: resume, rerun_failed"):
        validate_slurm_plan_fields(plan)


def test_status_target_does_not_expand_tasks_or_probe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.orchestrate.launch.generate_run_id",
        lambda name: (_ for _ in ()).throw(AssertionError("status should not generate without needing it")),
    )

    target = resolve_status_target(_args(run_id="run-1"), cwd=tmp_path)

    assert target.run_id == "run-1"
    assert target.output_root == Path("outputs") / "orchestrate" / "run-1"


def test_derive_local_max_parallel_uses_task_gpu_requirements_and_ports(tmp_path: Path) -> None:
    task_specs = [
        _task(tmp_path, "large", gpus=4),
        _task(tmp_path, "medium", gpus=2),
        _task(tmp_path, "small", gpus=1),
    ]

    assert derive_local_max_parallel(task_specs, gpu_count=6, port_capacity=10) == 2
    assert derive_local_max_parallel(task_specs, gpu_count=8, port_capacity=1) == 1


def _task(tmp_path: Path, name: str, *, gpus: int):
    from medarc_verifiers.orchestrate.config import TaskSpec

    return TaskSpec(
        task_id=name,
        job_config_path=tmp_path / f"{name}.toml",
        model_key=name,
        model_id=name,
        orchestrate={"vllm": {"gpus": gpus, "tensor_parallel_size": gpus}, "container": {"image": "fake"}},
    )
