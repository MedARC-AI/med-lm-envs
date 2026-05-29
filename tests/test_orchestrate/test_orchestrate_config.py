from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import (
    BenchOverrideConfig,
    PlanConfig,
    expand_tasks,
    load_endpoint_orchestration_registry,
    load_plan,
    load_suite_config,
    make_plan,
    materialize_task_eval_config,
)
from medarc_verifiers.orchestrate.lifecycle import materialized_image_path, resolve_construct_cache


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


def _write_suite(path: Path, *, env_id: str = "medqa", extra: str = "") -> Path:
    path.write_text(
        f'''
save_results = true
output_dir = "runs/evals"
{extra}

[[eval]]
env_id = "{env_id}"
num_examples = 1
rollouts_per_example = 1
max_concurrent = 12

[[ablation]]
env_id = "{env_id}"
name = "seed-{{env_args.shuffle_seed}}"
num_examples = 1
rollouts_per_example = 1

[ablation.sweep.env_args]
shuffle_seed = [1, 2]
'''.lstrip(),
        encoding="utf-8",
    )
    return path


def test_plan_suites_targets_and_registries_resolve_relative_to_plan_file(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    suite = _write_suite(configs_dir / "suite.toml")
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
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.toml"
    container_env = tmp_path / "container.env"
    container_env.write_text("HF_HOME=/root/.cache/huggingface\n", encoding="utf-8")
    plan_path.write_text(
        """
name = "test"
suite = "configs/suite.toml"
endpoints_path = "configs/endpoints.toml"
eval_images_config = "configs/eval_images.toml"
run_id = "hello"
bundle_dir = "outputs/orchestrate/test-run"
output_dir = "outputs/results/test-run"
readiness_timeout_s = 123

[container]
volumes = ["/host/cache:/root/.cache/huggingface"]
env_file = "container.env"

[bench]
max_concurrent = 768

[[target]]
endpoint_id = "foo"
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.suite == suite.resolve()
    assert plan.eval_images_config == eval_images_cfg.resolve()
    assert plan.bundle_dir == (tmp_path / "outputs" / "orchestrate" / "test-run").resolve()
    assert plan.output_dir == (tmp_path / "outputs" / "results" / "test-run").resolve()

    task = expand_tasks(plan)[0]
    assert task.suite_path == suite.resolve()
    assert task.target_endpoint_id == "foo"
    assert task.task_id == "foo:suite"
    assert task.model_id == "Foo/Bar"
    assert task.generated_eval_config["output_dir"] == str((tmp_path / "outputs" / "results" / "test-run").resolve())
    assert task.generated_eval_config["max_concurrent"] == 768
    assert task.generated_eval_config["eval"][0]["max_concurrent"] == 12
    assert task.orchestrate["container"]["volumes"] == ["/host/cache:/root/.cache/huggingface:rw"]
    assert task.orchestrate["container"]["env_file"] == container_env.resolve()
    assert task.eval_images[0]["id"] == "fhir"
    assert task.orchestrate_registry.path == str(endpoints.resolve())


def test_lifecycle_config_parses_and_resolves_cache_paths(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    suite = _write_suite(configs_dir / "suite.toml")
    endpoints = _write_endpoint_registry(configs_dir / "endpoints.toml")
    plan_path = tmp_path / "plan.toml"
    plan_path.write_text(
        """
suite = "configs/suite.toml"
endpoints_path = "configs/endpoints.toml"

[prepare]
enabled = true
cpus = 4
time = "01:00:00"
partition = "cpu-short"

[prepare.cache]
hf_home = "cache/hf"
hub_cache = "cache/hf/hub"
image_dir = "cache/images"
latest_link = false

[teardown]
enabled = true
cpus = 1
remove_model_weights = false

[[target]]
endpoint_id = "foo"
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)

    assert plan.prepare.enabled is True
    assert plan.prepare.prefetch_enabled is True
    assert plan.prepare.image_materialization_enabled is True
    assert plan.prepare.cpus == 4
    assert plan.prepare.cache.hf_home == (tmp_path / "cache" / "hf").resolve()
    assert plan.prepare.cache.image_dir == (tmp_path / "cache" / "images").resolve()
    assert plan.prepare.cache.latest_link is False
    assert plan.teardown.enabled is True
    assert plan.teardown.cpus == 1
    assert endpoints.exists() and suite.exists()


def test_prepare_operation_flags_enable_prepare(tmp_path: Path) -> None:
    plan = PlanConfig(
        suite=tmp_path / "suite.toml",
        endpoints_path=tmp_path / "endpoints.toml",
        prepare={"materialize_images": True, "prefetch_model_weights": False},
        targets=[{"endpoint_id": "foo"}],
    )

    assert plan.prepare.enabled is True
    assert plan.prepare.prefetch_enabled is False
    assert plan.prepare.image_materialization_enabled is True


def test_legacy_construct_section_normalizes_to_prepare(tmp_path: Path) -> None:
    plan = PlanConfig(
        suite=tmp_path / "suite.toml",
        endpoints_path=tmp_path / "endpoints.toml",
        construct={"enabled": True, "cpus": 3},
        targets=[{"endpoint_id": "foo"}],
    )

    assert plan.prepare.enabled is True
    assert plan.prepare.cpus == 3
    assert plan.construct is plan.prepare


def test_prepare_and_legacy_construct_together_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"\[prepare\].*\[construct\]"):
        PlanConfig(
            suite=tmp_path / "suite.toml",
            endpoints_path=tmp_path / "endpoints.toml",
            prepare={"enabled": True},
            construct={"enabled": True},
            targets=[{"endpoint_id": "foo"}],
        )


def test_construct_cache_resolves_hf_home_from_container_volume(tmp_path: Path) -> None:
    plan = PlanConfig(
        suite=tmp_path / "suite.toml",
        endpoints_path=tmp_path / "endpoints.toml",
        prepare={
            "enabled": True,
            "cache": {"image_dir": str(tmp_path / "images")},
        },
        targets=[{"endpoint_id": "foo"}],
    )

    cache = resolve_construct_cache(
        config=plan.construct,
        volume_mounts=[f"{tmp_path / 'hf'}:/root/.cache/huggingface:rw"],
    )

    assert cache.hf_home == str(tmp_path / "hf")
    assert cache.hub_cache == str(tmp_path / "hf" / "hub")
    assert cache.container_hf_home == "/root/.cache/huggingface"
    assert cache.container_hub_cache == "/root/.cache/huggingface/hub"


def test_construct_cache_requires_roots_for_enabled_operations(tmp_path: Path) -> None:
    plan = PlanConfig(
        suite=tmp_path / "suite.toml",
        endpoints_path=tmp_path / "endpoints.toml",
        prepare={"enabled": True, "cache": {"hf_home": str(tmp_path / "hf")}},
        targets=[{"endpoint_id": "foo"}],
    )

    with pytest.raises(ValueError, match="image_dir"):
        resolve_construct_cache(config=plan.construct, volume_mounts=[])


def test_materialized_image_path_preserves_namespace_and_uses_latest_link_path(tmp_path: Path) -> None:
    path = materialized_image_path("vllm/vllm-openai:v0.12.0", tmp_path)

    assert path.name.startswith("docker.io__vllm__vllm-openai--v0.12.0--")
    assert path.name.endswith(".sqsh")
    assert materialized_image_path("vllm/vllm-openai:latest", tmp_path) == tmp_path / "latest.sqsh"


def test_materialize_task_eval_config_forces_orchestrator_owned_fields(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path / "suite.toml", extra='env_dir_path = "envs"')
    endpoints = tmp_path / "endpoints.toml"

    payload = materialize_task_eval_config(
        suite_path=suite,
        endpoint_id="foo",
        endpoints_path=endpoints,
        bench_overrides=BenchOverrideConfig(timeout=900),
        output_dir=tmp_path / "bundle" / "bench",
    )

    assert payload["endpoint_id"] == "foo"
    assert payload["endpoints_path"] == str(endpoints.resolve())
    assert payload["output_dir"] == str(tmp_path / "bundle" / "bench")
    assert payload["env_dir_path"] == str((tmp_path / "envs").resolve())
    assert payload["timeout"] == 900
    assert payload["eval"][0]["max_concurrent"] == 12
    assert payload["ablation"][0]["sweep"]["env_args"]["shuffle_seed"] == [1, 2]


def test_materialize_task_eval_config_preserves_suite_output_dir_by_default(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path / "suite.toml")

    payload = materialize_task_eval_config(
        suite_path=suite,
        endpoint_id="foo",
        endpoints_path=tmp_path / "endpoints.toml",
        bench_overrides=BenchOverrideConfig(),
    )

    assert payload["output_dir"] == "runs/evals"


def test_suite_rejects_model_selection(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path / "suite.toml", extra='endpoint_id = "foo"')

    with pytest.raises(ValueError, match="orchestrator-owned"):
        materialize_task_eval_config(
            suite_path=suite,
            endpoint_id="foo",
            endpoints_path=tmp_path / "endpoints.toml",
            bench_overrides=BenchOverrideConfig(),
            output_dir=tmp_path / "bench",
        )


def test_load_suite_config_rejects_yaml_public_configs(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text("model: Foo/Bar\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expected .toml"):
        load_suite_config(path)


def test_make_plan_resolves_paths_relative_to_base_dir(tmp_path: Path) -> None:
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    suite = _write_suite(configs_dir / "suite.toml")

    plan = make_plan(
        suite=Path("configs/suite.toml"),
        targets=["foo"],
        base_dir=tmp_path,
        name="bundle",
    )

    assert plan.name == "bundle"
    assert plan.suite == suite.resolve()
    assert plan.targets[0].endpoint_id == "foo"


def test_duplicate_task_ids_are_rejected(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path / "suite.toml")
    endpoints = _write_endpoint_registry(tmp_path / "endpoints.toml")
    plan = PlanConfig(
        suite=suite,
        endpoints_path=endpoints,
        targets=[{"endpoint_id": "foo", "name": "dup"}, {"endpoint_id": "foo", "name": "dup"}],
    )

    with pytest.raises(ValueError, match="Duplicate orchestrated task id"):
        expand_tasks(plan)


def test_same_model_distinct_endpoint_targets_expand(tmp_path: Path) -> None:
    suite = _write_suite(tmp_path / "suite.toml")
    endpoints = tmp_path / "endpoints.toml"
    endpoints.write_text(
        """
[[endpoint]]
endpoint_id = "foo-instruct"
model = "Foo/Bar"

[endpoint.orchestrate.vllm]
gpus = 1

[endpoint.orchestrate.container]
image = "fake"

[[endpoint]]
endpoint_id = "foo-thinking"
model = "Foo/Bar"

[endpoint.orchestrate.vllm]
gpus = 1

[endpoint.orchestrate.container]
image = "fake"
""".lstrip(),
        encoding="utf-8",
    )
    plan = PlanConfig(
        suite=suite,
        endpoints_path=endpoints,
        targets=[{"endpoint_id": "foo-instruct"}, {"endpoint_id": "foo-thinking"}],
    )

    tasks = expand_tasks(plan)

    assert [task.task_id for task in tasks] == ["foo-instruct:suite", "foo-thinking:suite"]
    assert [task.model_id for task in tasks] == ["Foo/Bar", "Foo/Bar"]


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
decode_context_parallel_size = 2
dcp_comm_backend = "a2a"
dcp_kv_cache_interleave_size = 256
""".lstrip(),
        encoding="utf-8",
    )

    registry = load_endpoint_orchestration_registry(endpoints)
    model = registry["model"][0]

    assert model["vllm"]["tensor_parallel_size"] == 2
    assert model["container"]["image"] == "vllm/vllm-openai:latest"
    assert model["pyxis"] == {"srun_extra_args": ["--overlap"]}
    assert model["slurm"]["qos"] == "bottom"
    assert "nice" not in model["slurm"]
    assert model["vllm"]["serve"]["max_model_len"] == 40960
    assert model["vllm"]["serve"]["reasoning_parser"] == "qwen3"
    assert model["vllm"]["serve"]["decode_context_parallel_size"] == 2
    assert model["vllm"]["serve"]["dcp_comm_backend"] == "a2a"
    assert model["vllm"]["serve"]["dcp_kv_cache_interleave_size"] == 256
    assert model["vllm"]["serve"]["performance_mode"] == "throughput"
    assert model["vllm"]["serve"]["async_scheduling"] is True
