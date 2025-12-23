from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._job_executor import ExecutorSettings, JobExecutionResult, execute_jobs
from medarc_verifiers.cli._schemas import EnvironmentConfigSchema, ModelConfigSchema
from medarc_verifiers.cli.utils.env_args import EnvParam


def _stub_metadata(required: bool = False) -> list[EnvParam]:
    return [
        EnvParam(
            name="seed",
            cli_name="seed",
            kind="int",
            default=None,
            required=required,
            help="Seed value",
            annotation=int,
            argparse_type=int,
            choices=None,
            action=None,
            is_list=False,
            element_type=None,
            unsupported_reason=None,
        )
    ]


def _settings(tmp_path: Path, **overrides: object) -> ExecutorSettings:
    base_kwargs = dict(
        run_id="run-1",
        output_dir=tmp_path / "runs",
        env_dir=tmp_path / "environments",
        endpoints_path=tmp_path / "endpoints.py",
        default_api_key_var="DEFAULT_KEY",
        default_api_base_url="https://api.default",
        log_level="INFO",
        verbose=False,
        save_results=True,
        save_to_hf_hub=False,
        hf_hub_dataset_name=None,
        max_concurrent_generation=None,
        max_concurrent_scoring=None,
        # New concurrency precedence: CLI (--max-concurrent) > env_cfg.max_concurrent > DEFAULT_BATCH_MAX_CONCURRENT (128)
        # Provide a placeholder so tests can inject a CLI override via overrides (max_concurrent=VALUE).
        max_concurrent=None,
        timeout=None,
        sleep=0.0,
        dry_run=False,
    )
    base_kwargs.update(overrides)
    return ExecutorSettings(**base_kwargs)


def _stub_results(value: float = 0.5) -> SimpleNamespace:
    metadata = SimpleNamespace(
        path_to_save="",
        avg_reward=value,
        num_examples=1,
        rollouts_per_example=1,
        avg_metrics={"pass_rate": value},
    )
    return SimpleNamespace(metadata=metadata, reward=[value], metrics={"pass_rate": [value]})


def test_execute_jobs_invokes_run_evaluation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    async def fake_run(config):
        captured["config"] = config
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {
            "alias": {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://api.resolved"}
        },
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: _stub_metadata(required=True),
    )

    model_cfg = ModelConfigSchema(id="alias", headers={"X-Test": "1"}, sampling_args={"temperature": 0.1})
    env_cfg = EnvironmentConfigSchema(id="medqa", env_args={"seed": 1}, num_examples=3)
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={"seed": 1},
        sampling_args={"temperature": 0.1},
    )

    results = execute_jobs([job], _settings(tmp_path))

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, JobExecutionResult)
    assert result.status == "succeeded"
    assert result.output_path == (tmp_path / "runs" / "run-1" / job.job_id)
    assert "config" in captured
    config = captured["config"]
    assert config.model == "resolved-model"
    assert config.client_config.api_key_var == "MODEL_KEY"
    assert config.client_config.api_base_url == "https://api.resolved"
    assert config.client_config.extra_headers == {"X-Test": "1"}
    assert config.env_args == {"seed": 1}
    # With no CLI override and no env-level max_concurrent, falls back to DEFAULT_BATCH_MAX_CONCURRENT (128)
    assert config.max_concurrent == 128


def test_execute_jobs_records_failures(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def failing_run(config):
        raise RuntimeError("boom")

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", failing_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {},
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: _stub_metadata(required=False),
    )

    model_cfg = ModelConfigSchema(id="alias")
    env_cfg = EnvironmentConfigSchema(id="medqa", env_args={"seed": 1})
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={"seed": 1},
        sampling_args={},
    )

    results = execute_jobs([job], _settings(tmp_path))

    assert len(results) == 1
    result = results[0]
    assert result.status == "failed"
    assert result.error is not None
    assert "boom" in result.error
    assert "alias-medqa" in result.error
    assert "env=medqa" in result.error
    assert result.output_path == (tmp_path / "runs" / "run-1" / job.job_id)


def test_execute_jobs_respects_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def raise_if_called(*args, **kwargs):
        raise AssertionError("run_evaluation should not be invoked during dry runs.")

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", raise_if_called)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {},
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: _stub_metadata(required=False),
    )

    model_cfg = ModelConfigSchema(id="alias")
    env_cfg = EnvironmentConfigSchema(id="medqa")
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={},
        sampling_args={},
    )

    results = execute_jobs([job], _settings(tmp_path, dry_run=True))

    assert results[0].status == "skipped"
    assert results[0].output_path == (tmp_path / "runs" / "run-1" / job.job_id)


def test_executor_timeout_precedence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    async def fake_run(config):
        captured["config"] = config
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {},
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: _stub_metadata(required=False),
    )

    model_cfg = ModelConfigSchema(id="alias", timeout=5.0)
    env_cfg = EnvironmentConfigSchema(id="medqa")
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={},
        sampling_args={},
    )

    # CLI override should take precedence when provided.
    execute_jobs([job], _settings(tmp_path, timeout=10.0))
    config = captured["config"]
    assert config.client_config.timeout == 10.0

    # Model-level timeout applies when CLI flag is absent.
    captured.clear()
    execute_jobs([job], _settings(tmp_path))
    config = captured["config"]
    assert config.client_config.timeout == 5.0


def test_cli_env_arg_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    async def fake_run(config):
        captured["config"] = config
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {},
    )
    metadata = [
        EnvParam(
            name="flag",
            cli_name="flag",
            kind="bool",
            default=False,
            required=False,
            help="Boolean flag",
            annotation=bool,
            argparse_type=None,
            choices=None,
            action="BooleanOptionalAction",
            is_list=False,
            element_type=None,
            unsupported_reason=None,
        )
    ]
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: metadata,
    )

    model_cfg = ModelConfigSchema(id="alias", env_args={"flag": True})
    env_cfg = EnvironmentConfigSchema(id="medqa", env_args={"flag": False})
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={"flag": False},
        sampling_args={},
    )

    results = execute_jobs([job], _settings(tmp_path, cli_env_args={"flag": True}))

    assert results[0].status == "succeeded"
    assert captured["config"].env_args["flag"] is True


def test_cli_sampling_arg_overrides_yaml(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    async def fake_run(config):
        captured["config"] = config
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {},
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: [],
    )

    model_cfg = ModelConfigSchema(id="alias", sampling_args={"temperature": 0.7})
    env_cfg = EnvironmentConfigSchema(id="medqa")
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={},
        sampling_args={"temperature": 0.5},
    )

    results = execute_jobs(
        [job],
        _settings(tmp_path, cli_sampling_args={"temperature": 0.2}),
    )

    assert results[0].status == "succeeded"
    assert captured["config"].sampling_args["temperature"] == 0.2


def test_execute_jobs_handles_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def interrupting_run(config):  # noqa: ARG001
        raise KeyboardInterrupt

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", interrupting_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {},
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: [],
    )

    model_cfg = ModelConfigSchema(id="alias")
    env_cfg = EnvironmentConfigSchema(id="medqa")
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={},
        sampling_args={},
    )

    results = execute_jobs([job], _settings(tmp_path))

    assert len(results) == 1
    result = results[0]
    assert result.status == "failed"
    assert result.error is not None
    assert "interrupted" in result.error.lower()


def test_job_sleep_overrides_cli(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleep_calls: list[float] = []

    async def fake_run(config):  # noqa: ARG001
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        lambda env_id, cache=None: _stub_metadata(required=False),
    )
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.sleep", lambda seconds: sleep_calls.append(seconds))

    model_cfg = ModelConfigSchema(id="alias")
    env_cfg = EnvironmentConfigSchema(id="medqa")

    jobs = [
        ResolvedJob(
            job_id="alias-medqa-a",
            name="alias-medqa-a",
            model=model_cfg,
            env=env_cfg,
            env_args={},
            sampling_args={},
            sleep=1.5,
        ),
        ResolvedJob(
            job_id="alias-medqa-b",
            name="alias-medqa-b",
            model=model_cfg,
            env=env_cfg,
            env_args={},
            sampling_args={},
            sleep=None,
        ),
    ]

    results = execute_jobs(jobs, _settings(tmp_path, sleep=0.25))

    assert all(result.status == "succeeded" for result in results)
    assert sleep_calls == [pytest.approx(1.5)]
