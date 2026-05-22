from pathlib import Path

from medarc_verifiers.orchestrate.config import PlanConfig, expand_tasks, resolve_default_endpoints_path
from medarc_verifiers.orchestrate.launch import LaunchRequest, resolve_launch_plan, resolve_status_target


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

    plan = resolve_launch_plan(LaunchRequest(job_configs=(job,)), cwd=tmp_path)

    assert plan.tasks[0].endpoints_path == endpoints.resolve()
    assert plan.tasks[0].model_id == "Foo/Bar"


def test_launch_resolver_explicit_endpoint_overrides_job_endpoint(tmp_path: Path) -> None:
    job_endpoints = _write_endpoint_registry(tmp_path / "job-endpoints.toml", model="Foo/Job")
    explicit = _write_endpoint_registry(tmp_path / "explicit.toml", model="Foo/Explicit")
    job = _write_eval_config(tmp_path / "job.toml", endpoints_path=job_endpoints.name)

    plan = resolve_launch_plan(LaunchRequest(job_configs=(job,), endpoints_path=explicit), cwd=tmp_path)

    assert plan.tasks[0].endpoints_path == explicit.resolve()
    assert plan.tasks[0].model_id == "Foo/Explicit"


def test_status_target_resolves_absolute_run_path(tmp_path: Path) -> None:
    target = resolve_status_target(run_id="run-1", output_dir=None, cwd=tmp_path)

    assert target.run_id == "run-1"
    assert target.output_root == (tmp_path / "outputs" / "orchestrate" / "run-1").resolve()
