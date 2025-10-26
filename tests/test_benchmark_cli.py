from __future__ import annotations

import asyncio
import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping
import sys
import types

import pytest
import yaml
from verifiers.types import GenerateMetadata, GenerateOutputs

randomize_stub = types.ModuleType("medarc_verifiers.utils.randomize_mcq")
randomize_stub.randomize_multiple_choice = lambda *args, **kwargs: None
randomize_stub.randomize_multiple_choice_hf_map = lambda *args, **kwargs: None
sys.modules.setdefault("medarc_verifiers.utils.randomize_mcq", randomize_stub)

from medarc_verifiers.cli import benchmark
from medarc_verifiers.cli._benchmark_utils import expand_jobs, load_run_config
from medarc_verifiers.cli.benchmark import prepare_run_config


def write_config(tmp_path: Path, data: Mapping[str, Any]) -> Path:
    path = tmp_path / "benchmark.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(data), handle, sort_keys=False)
    return path


def build_minimal_config(tmp_path: Path) -> Path:
    config = {
        "name": "unit-benchmark",
        "output_dir": str(tmp_path / "runs"),
        "env_dir_path": "./environments",
        "models": [
            {
                "id": "model-a",
                "params": {
                    "model": "a-001",
                    "env_args": {"shared": True},
                    "sampling_args": {"temperature": 0.2},
                },
            }
        ],
        "envs": [
            {
                "id": "env-a",
                "module": "env_a",
                "num_examples": 2,
                "rollouts_per_example": 1,
                "env_args": {"base_arg": 1},
            }
        ],
        "jobs": [{"model": "model-a", "env": "env-a"}],
    }
    return write_config(tmp_path, config)


def test_load_run_config_cartesian_jobs(tmp_path: Path) -> None:
    config_path = build_minimal_config(tmp_path)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    assert len(jobs) == 1
    job = jobs[0]
    assert job.model.id == "model-a"
    assert job.env.id == "env-a"


def test_expand_jobs_with_explicit_entries(tmp_path: Path) -> None:
    config = {
        "name": "multi",
        "output_dir": str(tmp_path / "runs"),
        "models": [{"id": "model-a"}, {"id": "model-b"}],
        "envs": [{"id": "env-a"}, {"id": "env-b"}],
        "jobs": [
            {"model": "model-b", "env": "env-b", "name": "job-b"},
            {"model": "model-a", "env": "env-b", "name": "job-a"},
        ],
    }
    config_path = write_config(tmp_path, config)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    assert [job.job_id for job in jobs] == [
        "model-b-env-b-job-b",
        "model-a-env-b-job-a",
    ]


def test_job_entry_with_envs_list(tmp_path: Path) -> None:
    config = {
        "name": "fanout",
        "output_dir": str(tmp_path / "runs"),
        "models": [{"id": "model-a"}],
        "envs": [{"id": "env-a"}, {"id": "env-b"}],
        "jobs": [
            {
                "model": "model-a",
                "envs": ["env-a", "env-b"],
                "env_args": {"foo": 1},
                "sampling_args": {"temperature": 0.7},
                "name": "fanout",
            }
        ],
    }
    config_path = write_config(tmp_path, config)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    assert {job.env.id for job in jobs} == {"env-a", "env-b"}
    assert all(job.model.id == "model-a" for job in jobs)
    for job in jobs:
        assert job.env_overrides["foo"] == 1
        assert job.sampling_overrides["temperature"] == 0.7
    assert {job.job_id for job in jobs} == {"model-a-env-a-fanout", "model-a-env-b-fanout"}


def test_job_entry_defaults_to_all_envs(tmp_path: Path) -> None:
    config = {
        "name": "defaults",
        "output_dir": str(tmp_path / "runs"),
        "models": [{"id": "model-a"}],
        "envs": [{"id": "env-a"}, {"id": "env-b"}, {"id": "env-c"}],
        "jobs": [
            {
                "model": "model-a",
                "name": "all",
            }
        ],
    }
    config_path = write_config(tmp_path, config)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    assert {job.env.id for job in jobs} == {"env-a", "env-b", "env-c"}
    assert {job.job_id for job in jobs} == {"model-a-env-a-all", "model-a-env-b-all", "model-a-env-c-all"}


def test_job_without_name_uses_model_env_slug(tmp_path: Path) -> None:
    config = {
        "name": "random",
        "output_dir": str(tmp_path / "runs"),
        "models": [{"id": "model-a"}],
        "envs": [{"id": "env-a"}],
        "jobs": [
            {
                "model": "model-a",
                "env": "env-a",
            }
        ],
    }
    config_path = write_config(tmp_path, config)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    assert len(jobs) == 1
    assert jobs[0].job_id == "model-a-env-a"
    assert jobs[0].name == "model-a-env-a"


def test_build_eval_config_merges_env_args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = build_minimal_config(tmp_path)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)

    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])

    eval_config = benchmark.build_eval_config(
        run_config=run_config,
        job=jobs[0],
        endpoints_cache={},
        verbose=False,
    )

    assert eval_config.env_args["base_arg"] == 1
    assert eval_config.env_args["shared"] is True
    assert "seed" not in eval_config.env_args
    assert eval_config.model == "a-001"
    assert eval_config.sampling_args["temperature"] == 0.2


async def _stub_success_eval(eval_config):  # type: ignore[no-untyped-def]
    path = Path(tempfile.mkdtemp(prefix="benchmark-test-"))
    (path / "results.jsonl").write_text(json.dumps({}), encoding="utf-8")
    (path / "metadata.json").write_text(json.dumps({}), encoding="utf-8")
    metadata = GenerateMetadata(
        env_id=eval_config.env_id,
        env_args=eval_config.env_args,
        model=eval_config.model,
        base_url=eval_config.client_config.api_base_url,
        num_examples=eval_config.num_examples,
        rollouts_per_example=eval_config.rollouts_per_example,
        sampling_args=eval_config.sampling_args,
        date="2024-01-01",
        time_ms=1.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=[],
        path_to_save=path,
    )
    conversation = [
        {"content": "system prompt", "role": "system"},
        {"content": "question text", "role": "user"},
    ]
    completion_message = [{"content": "answer text", "role": "assistant"}]
    timing = {"timing": {"generation_ms": 1, "scoring_ms": 1, "total_ms": 2}}
    return GenerateOutputs(
        prompt=[list(conversation) for _ in range(3)],
        completion=[list(completion_message) for _ in range(3)],
        answer=["answer"] * 3,
        state=[dict(timing) for _ in range(3)],
        task=["task"] * 3,
        info=[{} for _ in range(3)],
        example_id=[0, 1, 2],
        reward=[1.0, 0.0, 0.0],
        metrics={"pass_rate": [1.0, 0.0, 0.0]},
        metadata=metadata,
    )


async def _stub_fail_eval(eval_config):  # type: ignore[no-untyped-def]
    raise RuntimeError("boom")


def test_execute_jobs_records_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = build_minimal_config(tmp_path)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    run_config.run_id = "test-run"
    run_config.output_dir = tmp_path / "runs"
    monkeypatch.setattr(benchmark, "run_evaluation", _stub_success_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])

    outcomes = asyncio.run(
        benchmark.execute_jobs(
            run_config=run_config,
            jobs=jobs,
            verbose=False,
        )
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "succeeded"
    job_dir = Path(run_config.output_dir) / run_config.run_id / jobs[0].job_id
    assert Path(outcomes[0].results_path) == job_dir
    assert (job_dir / "results.jsonl").exists()
    assert (job_dir / "summary.json").exists()
    summary_payload = json.loads((job_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["avg_reward"] == pytest.approx(1 / 3)
    metadata_payload = json.loads((job_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata_payload["avg_reward"] == pytest.approx(1 / 3)
    assert metadata_payload["avg_metrics"]["pass_rate"] == pytest.approx(1 / 3)


def test_execute_jobs_records_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = build_minimal_config(tmp_path)
    run_config = load_run_config(config_path)
    jobs = expand_jobs(run_config)
    monkeypatch.setattr(benchmark, "run_evaluation", _stub_fail_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])

    outcomes = asyncio.run(
        benchmark.execute_jobs(
            run_config=run_config,
            jobs=jobs,
            verbose=False,
        )
    )
    assert len(outcomes) == 1
    assert outcomes[0].status == "failed"
    assert outcomes[0].error == "boom"


def test_load_config_from_split_files(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    envs_dir = tmp_path / "envs"
    configs_dir = tmp_path / "configs"
    models_dir.mkdir()
    envs_dir.mkdir()
    configs_dir.mkdir()

    (models_dir / "gpt4.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "model-a",
                "params": {
                    "model": "gpt-4.1-mini",
                    "sampling_args": {"temperature": 0.0},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (envs_dir / "medqa.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "medqa",
                "num_examples": 3,
                "rollouts_per_example": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (configs_dir / "jobs.yaml").write_text(
        yaml.safe_dump(
            [
                {"model": "model-a", "env": "medqa"},
            ],
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    root_config = {
        "name": "split",
        "output_dir": str(tmp_path / "runs"),
        "models": str(models_dir),
        "envs": str(envs_dir),
        "jobs": str(configs_dir / "jobs.yaml"),
    }
    config_path = write_config(tmp_path, root_config)

    run_config = load_run_config(config_path)
    assert run_config.models["model-a"].params.model == "gpt-4.1-mini"
    assert run_config.envs["medqa"].num_examples == 3
    jobs = expand_jobs(run_config)
    assert len(jobs) == 1


def test_prepare_run_config_defaults(tmp_path: Path) -> None:
    suite_dir = tmp_path / "suite"
    models_dir = suite_dir / "models"
    envs_dir = suite_dir / "envs"
    suite_dir.mkdir()
    models_dir.mkdir()
    envs_dir.mkdir()

    (models_dir / "model-a.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "model-a",
                "params": {"model": "model-a"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (envs_dir / "env-a.yaml").write_text(
        yaml.safe_dump(
            {
                "id": "env-a",
                "num_examples": 1,
                "rollouts_per_example": 1,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    jobs_file = suite_dir / "jobs.yaml"
    jobs_file.write_text(yaml.safe_dump([{"model": "model-a", "env": "env-a"}], sort_keys=False), encoding="utf-8")

    args = argparse.Namespace(
        jobs=str(jobs_file),
        models=None,
        envs=None,
        output_dir=None,
        run_id=None,
        name=None,
        env_dir_path=None,
        endpoints_path=None,
    )

    run_config, resolved_jobs_path, snapshot = prepare_run_config(args)
    assert resolved_jobs_path == jobs_file.resolve()
    assert set(run_config.models.keys()) == {"model-a"}
    assert set(run_config.envs.keys()) == {"env-a"}
    assert run_config.output_dir == (tmp_path / "runs").resolve()
    assert snapshot["output_dir"] == str((tmp_path / "runs").resolve())


def test_resume_skips_completed_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = build_minimal_config(tmp_path)
    run_id = "resume-run"
    monkeypatch.setattr(benchmark, "run_evaluation", _stub_success_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])
    args = ["--jobs", str(config_path), "--run-id", run_id]
    assert benchmark.main(args) == 0

    run_dir = tmp_path / "runs" / run_id
    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest_initial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_initial["last_resume_at"] is None

    async def _should_not_run(eval_config):  # type: ignore[no-untyped-def]
        raise AssertionError("run_evaluation should not be invoked on resume skip")

    monkeypatch.setattr(benchmark, "run_evaluation", _should_not_run)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])
    assert benchmark.main(["--jobs", str(config_path), "--resume", run_id]) == 0

    manifest_after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_after["last_resume_at"] is not None
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    statuses = {job["status"] for job in summary["jobs"]}
    assert statuses == {"succeeded"}


def test_resume_executes_new_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "name": "resume-new",
        "output_dir": str(tmp_path / "runs"),
        "models": [
            {"id": "model-a", "params": {"model": "model-a"}},
        ],
        "envs": [
            {"id": "env-a", "num_examples": 1, "rollouts_per_example": 1},
            {"id": "env-b", "num_examples": 1, "rollouts_per_example": 1},
        ],
        "jobs": [{"model": "model-a", "env": "env-a"}],
    }
    config_path = write_config(tmp_path, config)
    run_id = "resume-new"
    monkeypatch.setattr(benchmark, "run_evaluation", _stub_success_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])
    assert benchmark.main(["--jobs", str(config_path), "--run-id", run_id]) == 0

    # Add a new job targeting env-b
    config["jobs"] = [
        {"model": "model-a", "env": "env-a"},
        {"model": "model-a", "env": "env-b"},
    ]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    executed_envs: list[str] = []

    async def _tracking_eval(eval_config):  # type: ignore[no-untyped-def]
        executed_envs.append(eval_config.env_id)
        return await _stub_success_eval(eval_config)

    monkeypatch.setattr(benchmark, "run_evaluation", _tracking_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])
    assert benchmark.main(["--jobs", str(config_path), "--resume", run_id]) == 0

    assert executed_envs == ["env-b"]
    summary = json.loads(((tmp_path / "runs" / run_id) / "run_summary.json").read_text(encoding="utf-8"))
    assert len(summary["jobs"]) == 2
    assert {job["status"] for job in summary["jobs"]} == {"succeeded"}


def test_resume_force_reruns_jobs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = build_minimal_config(tmp_path)
    run_id = "resume-force"
    monkeypatch.setattr(benchmark, "run_evaluation", _stub_success_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])
    assert benchmark.main(["--jobs", str(config_path), "--run-id", run_id]) == 0

    invocations: list[str] = []

    async def _tracking_eval(eval_config):  # type: ignore[no-untyped-def]
        invocations.append(eval_config.env_id)
        return await _stub_success_eval(eval_config)

    monkeypatch.setattr(benchmark, "run_evaluation", _tracking_eval)
    monkeypatch.setattr(benchmark, "load_endpoints", lambda path: {})
    monkeypatch.setattr(benchmark, "gather_env_cli_metadata", lambda env_id: [])
    assert benchmark.main(["--jobs", str(config_path), "--resume", run_id, "--force"]) == 0

    assert invocations == ["env_a"]
    summary = json.loads(((tmp_path / "runs" / run_id) / "run_summary.json").read_text(encoding="utf-8"))
    assert {job["status"] for job in summary["jobs"]} == {"succeeded"}
