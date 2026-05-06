from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.config import expand_tasks, load_plan


def test_plan_job_configs_resolve_relative_to_plan_file(tmp_path: Path):
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    job_cfg = configs_dir / "job-foo.toml"
    job_cfg.write_text(
        """
model = "Foo/Bar"

[medarc.orchestrate]
restart = "runs/raw/example-run"

[medarc.orchestrate.vllm-container]
image = "vllm/vllm-openai:latest"

[medarc.orchestrate.foo]
gpus = 1

[medarc.orchestrate.foo.serve]
dtype = "bfloat16"
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(
        """
name: test
job_configs:
  - configs/job-foo.toml
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

[medarc.orchestrate.vllm-container]
image = "vllm/vllm-openai:latest"

[medarc.orchestrate.foo]
gpus = 1

[medarc.orchestrate.foo.serve]
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


def test_expand_tasks_rejects_non_toml_job_config(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.yaml"
    job_cfg.write_text(
        """
model: Foo/Bar
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(f"job_configs:\n  - {job_cfg.name}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported job config format"):
        expand_tasks(load_plan(plan_path))


def test_expand_tasks_rejects_missing_vllm_container(tmp_path: Path) -> None:
    job_cfg = tmp_path / "job.toml"
    job_cfg.write_text(
        """
model = "Foo/Bar"

[medarc.orchestrate.foo]
gpus = 1
""".lstrip(),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(f"job_configs:\n  - {job_cfg.name}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must define medarc.orchestrate.vllm-container settings"):
        expand_tasks(load_plan(plan_path))
