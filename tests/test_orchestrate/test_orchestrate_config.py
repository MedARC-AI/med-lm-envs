from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import expand_tasks, load_job_config, load_plan, make_plan


def _write_eval_config(path: Path, *, model: str = "Foo/Bar", env_id: str = "medqa") -> Path:
    path.write_text(
        f'''
model = "{model}"

[[eval]]
env_id = "{env_id}"
num_examples = 1
rollouts_per_example = 1
'''.lstrip(),
        encoding="utf-8",
    )
    return path


def _write_orchestrate_config(path: Path, *, model: str = "Foo/Bar", gpus: int = 1) -> Path:
    path.write_text(
        f'''
schema_version = 1

[[model]]
id = "{model}"
aliases = ["foo"]

[model.vllm]
gpus = {gpus}
tensor_parallel_size = {gpus}

[model.vllm.serve]
dtype = "bfloat16"

[model.container]
image = "vllm/vllm-openai:latest"
container_port = 8000

[model.slurm]
partition = "gpu"
time = "04:00:00"
slurm_resume = true
'''.lstrip(),
        encoding="utf-8",
    )
    return path


def test_plan_job_configs_and_registries_resolve_relative_to_plan_file(tmp_path: Path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    job_cfg = _write_eval_config(configs_dir / "job-foo.toml")
    orchestrate_cfg = _write_orchestrate_config(configs_dir / "orchestrate.toml")
    eval_images_cfg = configs_dir / "eval_images.toml"
    eval_images_cfg.write_text(
        """
schema_version = 1

[[eval_image]]
id = "fhir"
envs = ["medqa"]
runtime = "pyxis"
image = "/tmp/fhir.sqsh"
command = ["bash", "-lc", "serve-fhir"]

[eval_image.readiness]
url = "http://127.0.0.1:8080/health"
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
name: test
job_configs:
  - configs/job-foo.toml
orchestrate_config: configs/orchestrate.toml
eval_images_config: configs/eval_images.toml
gpu_range: "0-3"
port_range: "8100-8199"
run_id: "hello"
output_dir: "outputs/orchestrator/test-run"
max_parallel: 2
readiness_timeout_s: 123
resume: true
rerun_failed: true
kill_orphans: false
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.job_configs == [job_cfg.resolve()]
    assert plan.orchestrate_config == orchestrate_cfg.resolve()
    assert plan.eval_images_config == eval_images_cfg.resolve()
    assert plan.output_dir == (tmp_path / "outputs" / "orchestrator" / "test-run").resolve()
    assert plan.resume is True

    task = expand_tasks(plan)[0]
    assert task.job_config_path == job_cfg.resolve()
    assert task.model_id == "Foo/Bar"
    assert task.model_key == "Foo-Bar"
    assert task.orchestrate["container"]["image"] == "vllm/vllm-openai:latest"
    assert task.orchestrate["vllm"]["serve"] == {"dtype": "bfloat16"}
    assert task.slurm["partition"] == "gpu"
    assert task.eval_images[0]["id"] == "fhir"
    assert task.orchestrate_registry.path == str(orchestrate_cfg.resolve())
    assert task.eval_images_registry.path == str(eval_images_cfg.resolve())


def test_load_job_config_rejects_yaml_public_configs(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text("model: Foo/Bar\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected .toml"):
        load_job_config(path)


def test_make_plan_resolves_paths_relative_to_base_dir(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    job_cfg = _write_eval_config(configs_dir / "job-foo.toml")
    orchestrate_cfg = _write_orchestrate_config(configs_dir / "orchestrate.toml")

    plan = make_plan(
        job_configs=[Path("configs/job-foo.toml")],
        base_dir=tmp_path,
        name="bundle",
        orchestrate_config=Path("configs/orchestrate.toml"),
    )

    assert plan.name == "bundle"
    assert plan.job_configs == [job_cfg.resolve()]
    assert plan.orchestrate_config == orchestrate_cfg.resolve()


def test_expand_tasks_matches_model_alias_from_endpoint_registry(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text(
        """
endpoint_id = "foo-endpoint"

[[eval]]
env_id = "medqa"
""".lstrip(),
        encoding="utf-8",
    )
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        """
[[endpoint]]
endpoint_id = "foo-endpoint"
model = "Foo/Bar"
url = "http://localhost:8000/v1"
key = "OPENAI_API_KEY"
""".lstrip(),
        encoding="utf-8",
    )
    orchestrate_cfg = _write_orchestrate_config(tmp_path / "orchestrate.toml", model="Foo/Bar")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\nendpoints_path: {endpoints.name}\n",
        encoding="utf-8",
    )

    task = expand_tasks(load_plan(plan_path))[0]

    assert task.model_id == "Foo/Bar"
    assert task.endpoints_path == endpoints.resolve()


def test_expand_tasks_rejects_model_ablations(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text(
        """
model = "Foo/Bar"

[[eval]]
env_id = "medqa"

[[ablation]]
env_id = "medqa"

[ablation.sweep]
model = ["Foo/Bar", "Other/Model"]
""".lstrip(),
        encoding="utf-8",
    )
    orchestrate_cfg = _write_orchestrate_config(tmp_path / "orchestrate.toml")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="ablates model"):
        expand_tasks(load_plan(plan_path))


def test_orchestrate_registry_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml")
    orchestrate_cfg = tmp_path / "orchestrate.toml"
    orchestrate_cfg.write_text(
        """
schema_version = 1

[[model]]
id = "Foo/Bar"

[model.vllm]
gpus = 1
tensor_parallel_size = 1

[model.vllm.serve]
unknown = "bad"

[model.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unknown fields"):
        expand_tasks(load_plan(plan_path))


def test_job_local_relative_endpoints_path_matches_bundled_resolution(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text(
        """
endpoint_id = "foo-endpoint"
endpoints_path = "endpoints.toml"

[[eval]]
env_id = "medqa"
""".lstrip(),
        encoding="utf-8",
    )
    (tmp_path / "endpoints.toml").write_text(
        """
[[endpoint]]
endpoint_id = "foo-endpoint"
model = "Foo/Bar"
url = "http://localhost:8000/v1"
key = "OPENAI_API_KEY"
""".lstrip(),
        encoding="utf-8",
    )
    orchestrate_cfg = _write_orchestrate_config(tmp_path / "orchestrate.toml")
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\n", encoding="utf-8"
    )

    task = expand_tasks(load_plan(plan_path))[0]

    assert task.model_id == "Foo/Bar"


def test_orchestrate_registry_requires_gpu_sizing_fields(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml")
    orchestrate_cfg = tmp_path / "orchestrate.toml"
    orchestrate_cfg.write_text(
        """
schema_version = 1

[[model]]
id = "Foo/Bar"

[model.vllm]
gpus = 1

[model.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="tensor_parallel_size"):
        expand_tasks(load_plan(plan_path))


def test_eval_image_registry_requires_runtime_fields_and_selectors(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml")
    orchestrate_cfg = _write_orchestrate_config(tmp_path / "orchestrate.toml")
    eval_images_cfg = tmp_path / "eval_images.toml"
    eval_images_cfg.write_text(
        """
schema_version = 1

[[eval_image]]
id = "bad"
runtime = "pyxis"
image = "/tmp/image.sqsh"
command = ["bash"]
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        f"job_configs:\n  - {job_cfg.name}\norchestrate_config: {orchestrate_cfg.name}\neval_images_config: {eval_images_cfg.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one selector"):
        expand_tasks(load_plan(plan_path))
