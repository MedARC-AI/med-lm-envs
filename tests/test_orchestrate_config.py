from pathlib import Path

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
vllm:
  docker:
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
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan(plan_path)
    assert plan.job_configs == [job_cfg.resolve()]

    tasks = expand_tasks(plan)
    assert tasks[0].job_config_path == job_cfg.resolve()

