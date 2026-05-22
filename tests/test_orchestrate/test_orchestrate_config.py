from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import (
    expand_tasks,
    load_endpoint_orchestration_registry,
    load_job_config,
    load_plan,
    make_plan,
)


def _write_endpoint_registry(
    path: Path,
    *,
    endpoint_id: str = "foo",
    model: str = "Foo/Bar",
    gpus: int = 1,
    tensor_parallel_size: int | None = None,
    extra_serve: str = 'dtype = "bfloat16"',
) -> Path:
    tensor_parallel_line = (
        f"tensor_parallel_size = {tensor_parallel_size}\n" if tensor_parallel_size is not None else ""
    )
    path.write_text(
        f'''
[[endpoint]]
endpoint_id = "{endpoint_id}"
model = "{model}"
api_client_type = "openai_chat_completions"

[endpoint.sampling_args]
temperature = 0.5

[endpoint.orchestrate.vllm]
gpus = {gpus}
{tensor_parallel_line}
[endpoint.orchestrate.vllm.serve]
{extra_serve}

[endpoint.orchestrate.container]
image = "vllm/vllm-openai:latest"
container_port = 8000

[endpoint.orchestrate.slurm]
partition = "gpu"
time = "04:00:00"
slurm_resume = true
'''.lstrip(),
        encoding="utf-8",
    )
    return path


def _write_eval_config(path: Path, *, endpoint_id: str = "foo", env_id: str = "medqa") -> Path:
    path.write_text(
        f'''
endpoint_id = "{endpoint_id}"
endpoints_path = "endpoints.toml"

[[eval]]
env_id = "{env_id}"
num_examples = 1
rollouts_per_example = 1
'''.lstrip(),
        encoding="utf-8",
    )
    return path


def test_plan_job_configs_and_registries_resolve_relative_to_plan_file(tmp_path: Path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    job_cfg = _write_eval_config(configs_dir / "job-foo.toml")
    endpoints = _write_endpoint_registry(configs_dir / "endpoints.toml")
    eval_images_cfg = configs_dir / "eval_images.toml"
    eval_images_cfg.write_text(
        """
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
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        """
name = "test"
job_configs = ["configs/job-foo.toml"]
eval_images_config = "configs/eval_images.toml"
run_id = "hello"
output_dir = "outputs/orchestrate/test-run"
readiness_timeout_s = 123
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.job_configs == [job_cfg.resolve()]
    assert plan.eval_images_config == eval_images_cfg.resolve()
    assert plan.output_dir == (tmp_path / "outputs" / "orchestrate" / "test-run").resolve()

    task = expand_tasks(plan)[0]
    assert task.job_config_path == job_cfg.resolve()
    assert task.model_id == "Foo/Bar"
    assert task.model_key == "Foo-Bar"
    assert task.orchestrate["container"]["image"] == "vllm/vllm-openai:latest"
    assert task.orchestrate["vllm"]["tensor_parallel_size"] == 1
    assert task.orchestrate["vllm"]["serve"] == {
        "gpu_memory_utilization": 0.90,
        "max_model_len": 32768,
        "async_scheduling": True,
        "enable_prefix_caching": True,
        "enable_auto_tool_choice": True,
        "dtype": "bfloat16",
    }
    assert task.slurm["partition"] == "gpu"
    assert task.eval_images[0]["id"] == "fhir"
    assert task.orchestrate_registry.path == str(endpoints.resolve())
    assert task.eval_images_registry.path == str(eval_images_cfg.resolve())


def test_load_job_config_rejects_yaml_public_configs(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text("model: Foo/Bar\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected .toml"):
        load_job_config(path)


def test_load_plan_rejects_yaml_configs(tmp_path: Path) -> None:
    path = tmp_path / "plan.yaml"
    path.write_text("job_configs:\n  - job.toml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected .toml"):
        load_plan(path)


def test_make_plan_resolves_paths_relative_to_base_dir(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    job_cfg = _write_eval_config(configs_dir / "job-foo.toml")
    _write_endpoint_registry(configs_dir / "endpoints.toml")

    plan = make_plan(
        job_configs=[Path("configs/job-foo.toml")],
        base_dir=tmp_path,
        name="bundle",
    )

    assert plan.name == "bundle"
    assert plan.job_configs == [job_cfg.resolve()]


def test_expand_tasks_requires_exact_endpoint_id_with_orchestrate_block(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml", endpoint_id="gpt-oss-120b-high")
    endpoints = _write_endpoint_registry(
        tmp_path / "endpoints.toml", endpoint_id="gpt-oss-120b", model="openai/gpt-oss-120b"
    )
    with endpoints.open("a", encoding="utf-8") as handle:
        handle.write(
            """

[[endpoint]]
endpoint_id = "gpt-oss-120b-high"
model = "openai/gpt-oss-120b"
"""
        )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(f'job_configs = ["{job_cfg.name}"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Known IDs: \\['gpt-oss-120b'\\]"):
        expand_tasks(load_plan(plan_path))


@pytest.mark.parametrize(
    "field", ["runtime", "gpu_range", "port_range", "max_parallel", "resume", "rerun_failed", "uv_run"]
)
def test_plan_rejects_deleted_local_only_fields(tmp_path: Path, field: str) -> None:
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(f'job_configs = ["job.toml"]\n{field} = "legacy"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid plan file"):
        load_plan(plan_path)


def test_endpoint_orchestration_registry_applies_defaults_and_default_tensor_parallel(tmp_path: Path) -> None:
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        """
[[endpoint]]
endpoint_id = "foo"
model = "Foo/Bar"

[endpoint.orchestrate.vllm]
gpus = 2

[endpoint.orchestrate.vllm.serve]
max_model_len = 40960
reasoning_parser = "qwen3"
""".lstrip(),
        encoding="utf-8",
    )

    registry = load_endpoint_orchestration_registry(endpoints)
    model = registry["model"][0]

    assert model["vllm"]["tensor_parallel_size"] == 2
    assert model["container"] == {
        "image": "vllm/vllm-openai:latest",
        "container_port": 8000,
        "ipc_mode": "host",
    }
    assert model["pyxis"] == {"srun_extra_args": ["--overlap"]}
    assert model["slurm"]["qos"] == "low"
    assert model["slurm"]["nice"] == 500
    assert model["vllm"]["serve"]["max_model_len"] == 40960
    assert model["vllm"]["serve"]["reasoning_parser"] == "qwen3"
    assert model["vllm"]["serve"]["async_scheduling"] is True


def test_endpoint_orchestration_rejects_unknown_nested_fields(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml")
    _write_endpoint_registry(tmp_path / "endpoints.toml", extra_serve='unknown = "bad"')
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(f'job_configs = ["{job_cfg.name}"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown fields"):
        expand_tasks(load_plan(plan_path))


def test_endpoint_orchestration_requires_gpus(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml")
    (tmp_path / "endpoints.toml").write_text(
        """
[[endpoint]]
endpoint_id = "foo"
model = "Foo/Bar"

[endpoint.orchestrate.vllm]

[endpoint.orchestrate.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(f'job_configs = ["{job_cfg.name}"]\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must set \\[endpoint.orchestrate.vllm\\].gpus"):
        expand_tasks(load_plan(plan_path))


def test_expand_tasks_uses_expanded_variant_templates_for_identities_and_eval_images(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    _write_endpoint_registry(tmp_path / "endpoints.toml")
    job_cfg.write_text(
        """
endpoint_id = "foo"
endpoints_path = "endpoints.toml"

[[eval]]
env_id = "medqa"
variant_id = "seed-{env_args.shuffle_seed}"

[eval.env_args]
shuffle_seed = 1

[[eval]]
env_id = "medqa"
variant_id = "seed-{env_args.shuffle_seed}"

[eval.env_args]
shuffle_seed = 2
""".lstrip(),
        encoding="utf-8",
    )
    eval_images_cfg = tmp_path / "eval_images.toml"
    eval_images_cfg.write_text(
        """
[[eval_image]]
id = "seed-two"
evals = ["medqa:seed-2"]
runtime = "pyxis"
image = "/tmp/seed-two.sqsh"
command = ["bash", "-lc", "serve"]
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        f'job_configs = ["{job_cfg.name}"]\neval_images_config = "{eval_images_cfg.name}"\n',
        encoding="utf-8",
    )

    task = expand_tasks(load_plan(plan_path))[0]

    assert task.eval_ids == ["medqa:seed-1", "medqa:seed-2"]
    assert [image["id"] for image in task.eval_images] == ["seed-two"]


def test_eval_image_registry_requires_runtime_fields_and_selectors(tmp_path: Path) -> None:
    job_cfg = _write_eval_config(tmp_path / "job.toml")
    _write_endpoint_registry(tmp_path / "endpoints.toml")
    eval_images_cfg = tmp_path / "eval_images.toml"
    eval_images_cfg.write_text(
        """
[[eval_image]]
id = "bad"
runtime = "pyxis"
image = "/tmp/image.sqsh"
command = ["bash"]
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        f'job_configs = ["{job_cfg.name}"]\neval_images_config = "{eval_images_cfg.name}"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one selector"):
        expand_tasks(load_plan(plan_path))
