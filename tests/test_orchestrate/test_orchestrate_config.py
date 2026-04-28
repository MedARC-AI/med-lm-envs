import warnings
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import expand_tasks, load_plan


def test_plan_job_configs_resolve_relative_to_plan_file(tmp_path: Path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    job_cfg = configs_dir / "job-foo.yaml"
    job_cfg.write_text(
        """
models:
  foo:
    model: Foo/Bar
orchestrate:
  restart: runs/raw/example-run
  vllm-container:
    image: vllm/vllm-openai:latest
  foo:
    gpus: 1
    serve:
      dtype: bfloat16
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
name: test
job_configs:
  - configs/job-foo.yaml
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
    assert plan.gpu_range == "0-3"
    assert plan.port_range == "8100-8199"
    assert plan.run_id == "hello"
    assert plan.output_dir == (tmp_path / "outputs" / "orchestrator" / "test-run").resolve()
    assert plan.max_parallel == 2
    assert plan.readiness_timeout_s == 123
    assert plan.resume is True
    assert plan.rerun_failed is True
    assert plan.kill_orphans is False

    tasks = expand_tasks(plan)
    assert tasks[0].job_config_path == job_cfg.resolve()
    assert tasks[0].orchestrate.get("restart") == "runs/raw/example-run"
    assert "vllm-container" in tasks[0].orchestrate


def test_expand_tasks_accepts_toml_eval_config(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text(
        """
model = "Foo/Bar"

[[eval]]
env_id = "medqa"

[orchestrate.vllm-container]
image = "vllm/vllm-openai:latest"

[orchestrate.foo]
gpus = 1

[orchestrate.foo.serve]
dtype = "bfloat16"
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(f"job_configs:\n  - {job_cfg.name}\n", encoding="utf-8")

    tasks = expand_tasks(load_plan(plan_path))

    assert tasks[0].job_config_path == job_cfg.resolve()
    assert tasks[0].model_key == "foo"
    assert tasks[0].model_id == "Foo/Bar"
    assert tasks[0].orchestrate["vllm-container"]["image"] == "vllm/vllm-openai:latest"
    assert tasks[0].orchestrate["foo"]["serve"]["dtype"] == "bfloat16"


def test_expand_tasks_accepts_deprecated_vllm_docker_with_warning(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    job_cfg.write_text(
        """
models:
  foo:
    model: Foo/Bar
orchestrate:
  vllm-docker:
    image: vllm/vllm-openai:latest
  foo:
    gpus: 1
    serve: {}
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(f"job_configs:\n  - {job_cfg.name}\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tasks = expand_tasks(load_plan(plan_path))

    assert tasks[0].orchestrate["vllm-container"]["image"] == "vllm/vllm-openai:latest"
    assert "vllm-docker" not in tasks[0].orchestrate
    assert any("deprecated orchestrate.vllm-docker" in str(item.message) for item in caught)


def test_expand_tasks_rejects_ambiguous_container_keys(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    job_cfg.write_text(
        """
models:
  foo:
    model: Foo/Bar
orchestrate:
  vllm-container:
    image: new
  vllm-docker:
    image: old
  foo:
    gpus: 1
    serve: {}
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(f"job_configs:\n  - {job_cfg.name}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="defines both orchestrate.vllm-container and orchestrate.vllm-docker"):
        expand_tasks(load_plan(plan_path))
