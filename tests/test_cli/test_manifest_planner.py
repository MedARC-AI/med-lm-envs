from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._manifest import RunManifest
from medarc_verifiers.cli._manifest_planner import ManifestPlanner, _find_auto_resume_candidate
from medarc_verifiers.cli._schemas import EnvironmentConfigSchema, ModelConfigSchema


def _make_job(job_id: str = "job-a", env_id: str = "env-a", model_id: str = "model-a") -> ResolvedJob:
    env = EnvironmentConfigSchema(id=env_id, module=env_id)
    model = ModelConfigSchema(id=model_id, model="gpt-4.1-mini")
    return ResolvedJob(
        job_id=job_id,
        name=job_id,
        model=model,
        env=env,
        env_args={},
        sampling_args={},
    )


def _planner(
    *,
    tmp_path: Path,
    jobs: list[ResolvedJob],
    config_checksum: str = "abc123",
    run_id: str | None = None,
    restart_source: str | None = None,
    auto_resume: bool = True,
    persist: bool = True,
) -> ManifestPlanner:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: test\n", encoding="utf-8")
    env_args_map = {job.job_id: {} for job in jobs}
    sampling_args_map = {job.job_id: {} for job in jobs}
    return ManifestPlanner(
        output_dir=tmp_path / "runs",
        run_id=run_id,
        run_name="demo-run",
        config_path=config_path,
        config_checksum=config_checksum,
        jobs=jobs,
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        restart_source=restart_source,
        auto_resume=auto_resume,
        persist=persist,
    )


def test_restart_in_place_reuses_completed_job(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: test\n", encoding="utf-8")
    job = _make_job()
    env_args_map = {job.job_id: {}}
    sampling_args_map = {job.job_id: {}}
    run_dir = tmp_path / "runs" / "base-run"
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="base-run",
        run_name="demo-run",
        config_source=config_path,
        config_checksum="abc123",
        jobs=[job],
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=True,
        restart_source=None,
    )
    manifest.record_job_completion(
        job.job_id,
        duration_seconds=1.0,
        results_dir=run_dir / job.job_id,
        artifacts=[],
        avg_reward=None,
        metrics={},
        num_examples=job.env.num_examples,
        rollouts_per_example=job.env.rollouts_per_example,
    )

    planner = _planner(tmp_path=tmp_path, jobs=[job], restart_source="base-run")
    plan = planner.plan(force_all=False, forced_envs=set())

    assert plan.manifest.path == manifest.path
    assert plan.runnable_job_ids == set()
    assert plan.reused_job_ids == {job.job_id}


def test_auto_resume_prefers_incomplete_run(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: test\n", encoding="utf-8")
    job = _make_job()
    env_args_map = {job.job_id: {}}
    sampling_args_map = {job.job_id: {}}
    output_dir = tmp_path / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    incomplete_dir = output_dir / "incomplete-run"
    RunManifest.create(
        run_dir=incomplete_dir,
        run_id="incomplete-run",
        run_name="demo-run",
        config_source=config_path,
        config_checksum="abc123",
        jobs=[job],
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=True,
        restart_source=None,
    )

    complete_dir = output_dir / "complete-run"
    complete_manifest = RunManifest.create(
        run_dir=complete_dir,
        run_id="complete-run",
        run_name="demo-run",
        config_source=config_path,
        config_checksum="abc123",
        jobs=[job],
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=True,
        restart_source=None,
    )
    complete_manifest.record_job_completion(
        job.job_id,
        duration_seconds=1.0,
        results_dir=complete_dir / job.job_id,
        artifacts=[],
        avg_reward=None,
        metrics={},
        num_examples=job.env.num_examples,
        rollouts_per_example=job.env.rollouts_per_example,
    )

    candidate = _find_auto_resume_candidate(output_dir, expected_checksum="abc123")
    assert candidate == incomplete_dir

    planner = _planner(tmp_path=tmp_path, jobs=[job], auto_resume=True)
    plan = planner.plan(force_all=False, forced_envs=set())

    assert plan.manifest.run_dir == incomplete_dir
    assert plan.runnable_job_ids == {job.job_id}
    assert plan.reused_job_ids == set()


def test_auto_resume_with_checksum_mismatch_raises(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: test\n", encoding="utf-8")
    job = _make_job()
    env_args_map = {job.job_id: {}}
    sampling_args_map = {job.job_id: {}}
    run_dir = tmp_path / "runs" / "existing"
    RunManifest.create(
        run_dir=run_dir,
        run_id="existing",
        run_name="demo-run",
        config_source=config_path,
        config_checksum="different",
        jobs=[job],
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=True,
        restart_source=None,
    )

    planner = _planner(tmp_path=tmp_path, jobs=[job], run_id="existing", auto_resume=True, config_checksum="abc123")
    with pytest.raises(
        ValueError,
        match=(
            r"Run 'existing' was created from a different configuration\."
            r".*--no-auto-resume.*--restart existing"
        ),
    ):
        planner.plan(force_all=False, forced_envs=set())


def test_auto_resume_allows_resume_tolerant_model_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("config: test\n", encoding="utf-8")
    env = EnvironmentConfigSchema(id="env-a", module="env-a")
    model = ModelConfigSchema(id="model-a", model="gpt-4.1-mini", max_concurrent=16, timeout=30.0)
    job = ResolvedJob(
        job_id="job-a",
        name="job-a",
        model=model,
        env=env,
        env_args={},
        sampling_args={},
    )

    env_args_map = {job.job_id: {}}
    sampling_args_map = {job.job_id: {}}
    run_dir = tmp_path / "runs" / "existing"
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="existing",
        run_name="demo-run",
        config_source=config_path,
        config_checksum="abc123",
        jobs=[job],
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        persist=True,
        restart_source=None,
    )
    manifest.record_job_completion(
        job.job_id,
        duration_seconds=1.0,
        results_dir=run_dir / job.job_id,
        artifacts=[],
        avg_reward=None,
        metrics={},
        num_examples=job.env.num_examples,
        rollouts_per_example=job.env.rollouts_per_example,
    )

    planner = _planner(tmp_path=tmp_path, jobs=[job], run_id="existing", auto_resume=True, config_checksum="abc123")
    plan = planner.plan(force_all=False, forced_envs=set())

    assert plan.manifest.run_dir == run_dir
    assert plan.runnable_job_ids == set()
