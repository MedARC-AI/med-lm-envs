from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli._config_loader import load_run_config
from medarc_verifiers.cli._job_builder import ResolvedJob, build_jobs


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def _stub_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )


def test_build_jobs_basic(monkeypatch, tmp_path: Path) -> None:
    _stub_metadata(monkeypatch)
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          gpt-mini:
            model: openai/gpt-mini
        envs:
          medqa:
            num_examples: 5
        jobs:
          - model: gpt-mini
            env: medqa
        """,
    )

    run_config = load_run_config(config_path)
    jobs = build_jobs(run_config)

    assert len(jobs) == 1
    job = jobs[0]
    assert isinstance(job, ResolvedJob)
    assert job.job_id == "gpt-mini-medqa"
    assert job.env.id == "medqa"
    assert job.env_args == {}
    assert job.sampling_args == {}


def test_env_args_precedence(monkeypatch, tmp_path: Path) -> None:
    _stub_metadata(monkeypatch)
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          gpt-mini:
            env_args:
              shared: model
              model_only: 1
            env_overrides:
              medqa:
                shared: override
                override_only: true
        envs:
          medqa:
            env_args:
              shared: env
              env_only: 2
        jobs:
          - model: gpt-mini
            env: medqa
            env_args:
              shared: job
              job_only: 3
        """,
    )

    run_config = load_run_config(config_path)
    jobs = build_jobs(run_config)

    assert len(jobs) == 1
    env_args = jobs[0].env_args
    assert env_args["env_only"] == 2
    assert env_args["model_only"] == 1
    assert env_args["override_only"] is True
    assert env_args["job_only"] == 3
    assert env_args["shared"] == "job"


def test_matrix_base_expansion(monkeypatch, tmp_path: Path) -> None:
    _stub_metadata(monkeypatch)
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          gpt-mini: {}
        envs:
          medqa:
            matrix:
              shuffle_seed: [1618, 9331]
            matrix_id_format: "{base}-r{shuffle_seed}"
        jobs:
          - model: gpt-mini
            env: medqa
        """,
    )

    run_config = load_run_config(config_path)
    jobs = build_jobs(run_config)

    job_ids = {job.env.id for job in jobs}
    assert job_ids == {"medqa-r1618", "medqa-r9331"}


def test_duplicate_job_ids_get_fingerprinted(monkeypatch, tmp_path: Path) -> None:
    _stub_metadata(monkeypatch)
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          gpt-mini: {}
        envs:
          medqa: {}
        jobs:
          - model: gpt-mini
            env: medqa
            env_args:
              variant: 1
          - model: gpt-mini
            env: medqa
            env_args:
              variant: 2
        """,
    )

    run_config = load_run_config(config_path)
    jobs = build_jobs(run_config)

    assert len(jobs) == 2
    job_ids = {job.job_id for job in jobs}
    assert len(job_ids) == 2
    assert any(job_id.startswith("gpt-mini-medqa-") for job_id in job_ids)


def test_unknown_model_raises(monkeypatch, tmp_path: Path) -> None:
    _stub_metadata(monkeypatch)
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        envs:
          medqa: {}
        jobs:
          - model: missing
            env: medqa
        """,
    )

    run_config = load_run_config(config_path)
    with pytest.raises(ValueError, match="unknown model"):
        build_jobs(run_config)


def test_unknown_environment_raises(monkeypatch, tmp_path: Path) -> None:
    _stub_metadata(monkeypatch)
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          gpt-mini: {}
        envs:
          medqa: {}
        jobs:
          - model: gpt-mini
            env: missing
        """,
    )

    run_config = load_run_config(config_path)
    with pytest.raises(ValueError, match="unknown environment"):
        build_jobs(run_config)
