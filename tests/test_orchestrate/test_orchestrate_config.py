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
orchestrate:
  restart: runs/raw/example-run
  vllm-docker:
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
