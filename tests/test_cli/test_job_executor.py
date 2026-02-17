from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli._constants import DEFAULT_ENDPOINTS_PATH
from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._job_executor import (
    ExecutorSettings,
    JobExecutionResult,
    _load_endpoints_for_model,
    execute_jobs,
)
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
        endpoints_path_explicit=False,
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


def _stub_results_metadata_only(value: float = 0.5) -> SimpleNamespace:
    metadata = SimpleNamespace(
        path_to_save="",
        avg_reward=value,
        num_examples=2,
        rollouts_per_example=3,
        avg_metrics={"pass_rate": value, "accuracy": value / 2},
    )
    return SimpleNamespace(metadata=metadata)


def test_execute_jobs_invokes_run_evaluation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured = {}

    async def fake_run(config):
        captured["config"] = config
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda path, cache=None: {
            "alias": [{"model": "resolved-model", "key": "MODEL_KEY", "url": "https://api.resolved"}]
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
    assert Path(str(config.resume_path)) == (tmp_path / "runs" / "run-1" / job.job_id)
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


def test_materialize_results_noop_logs_debug_when_source_matches_job_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_run(config):
        metadata = SimpleNamespace(
            path_to_save=str(config.resume_path),
            avg_reward=0.5,
            num_examples=1,
            rollouts_per_example=1,
            avg_metrics={"pass_rate": 0.5},
        )
        return SimpleNamespace(metadata=metadata, reward=[0.5], metrics={"pass_rate": [0.5]})

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda env_id, cache=None: [])

    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=ModelConfigSchema(id="alias"),
        env=EnvironmentConfigSchema(id="medqa"),
        env_args={},
        sampling_args={},
    )

    with caplog.at_level(logging.DEBUG):
        results = execute_jobs([job], _settings(tmp_path, log_level="DEBUG"))

    assert results[0].status == "succeeded"
    assert "Results already in job_dir; _materialize_results no-op" in caplog.text


def test_forced_job_archives_and_resets_existing_job_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def fake_run(config):
        captured["resume_path"] = config.resume_path
        metadata = SimpleNamespace(
            path_to_save=str(config.resume_path),
            avg_reward=0.5,
            num_examples=1,
            rollouts_per_example=1,
            avg_metrics={"pass_rate": 0.5},
        )
        return SimpleNamespace(metadata=metadata, reward=[0.5], metrics={"pass_rate": [0.5]})

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda env_id, cache=None: [])

    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=ModelConfigSchema(id="alias"),
        env=EnvironmentConfigSchema(id="medqa"),
        env_args={},
        sampling_args={},
    )
    run_dir = tmp_path / "runs" / "run-1"
    job_dir = run_dir / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "stale.txt").write_text("stale", encoding="utf-8")

    results = execute_jobs([job], _settings(tmp_path, forced_job_ids={job.job_id}))

    assert results[0].status == "succeeded"
    assert Path(str(captured["resume_path"])) == job_dir
    archived = sorted(run_dir.glob(f"{job.job_id}__old_*"))
    assert len(archived) == 1
    assert (archived[0] / "stale.txt").exists()
    assert job_dir.exists()
    assert not (job_dir / "stale.txt").exists()


def test_non_forced_invalid_nonempty_job_dir_fails_prescriptively(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def fail_if_called(_config):
        raise AssertionError("run_evaluation should not run when preflight fails")

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fail_if_called)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda env_id, cache=None: [])

    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=ModelConfigSchema(id="alias"),
        env=EnvironmentConfigSchema(id="medqa"),
        env_args={},
        sampling_args={},
    )
    job_dir = tmp_path / "runs" / "run-1" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "orphan.log").write_text("invalid state", encoding="utf-8")

    results = execute_jobs([job], _settings(tmp_path))

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error is not None
    assert "not a valid evaluation results path" in results[0].error
    assert "--force" in results[0].error
    assert "new run_id" in results[0].error


def test_batch_resume_mismatch_logs_saved_and_current_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=ModelConfigSchema(id="alias"),
        env=EnvironmentConfigSchema(id="medqa", num_examples=5, rollouts_per_example=3),
        env_args={},
        sampling_args={},
    )
    job_dir = tmp_path / "runs" / "run-1" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "results.jsonl").write_text("", encoding="utf-8")
    (job_dir / "metadata.json").write_text(
        (
            '{"env_id":"saved-env","model":"saved-model",'
            '"rollouts_per_example":2,"num_examples":8}'
        ),
        encoding="utf-8",
    )

    async def fake_run(_config):
        raise ValueError(
            f"Cannot resume from {job_dir}: metadata mismatch "
            "(env_id: saved='saved-env', current='medqa')"
        )

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda env_id, cache=None: [])

    with caplog.at_level(logging.ERROR):
        results = execute_jobs([job], _settings(tmp_path))

    assert len(results) == 1
    assert results[0].status == "failed"
    assert results[0].error is not None
    assert "incompatible prior results" in results[0].error
    assert "Resume metadata mismatch for job 'alias-medqa'" in caplog.text
    assert "env_id: saved='saved-env', current='medqa'" in caplog.text
    assert "model: saved='saved-model', current='alias'" in caplog.text
    assert "rollouts_per_example: saved=2, current=3" in caplog.text
    assert "num_examples: saved=8, current=5 (current must be >= saved)" in caplog.text


def test_execute_jobs_uses_metadata_averages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _ManifestStub:
        def __init__(self) -> None:
            self.started: list[str] = []
            self.completed: list[dict[str, object]] = []

        def record_job_start(self, job_id: str) -> None:
            self.started.append(job_id)

        def record_job_completion(self, job_id: str, **kwargs: object) -> None:
            payload = {"job_id": job_id}
            payload.update(kwargs)
            self.completed.append(payload)

        def record_job_failure(self, job_id: str, **kwargs: object) -> None:
            raise AssertionError(f"Job should not fail: {job_id}, {kwargs}")

    async def fake_run(config):
        return _stub_results_metadata_only(0.8)

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda env_id, cache=None: [])

    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=ModelConfigSchema(id="alias"),
        env=EnvironmentConfigSchema(id="medqa"),
        env_args={},
        sampling_args={},
    )
    manifest = _ManifestStub()

    results = execute_jobs([job], _settings(tmp_path), manifest=manifest)

    assert results[0].status == "succeeded"
    assert manifest.started == ["alias-medqa"]
    assert len(manifest.completed) == 1
    completed = manifest.completed[0]
    assert completed["job_id"] == "alias-medqa"
    assert completed["avg_reward"] == pytest.approx(0.8)
    metrics = completed["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["pass_rate"] == pytest.approx(0.8)
    assert metrics["accuracy"] == pytest.approx(0.4)
    assert completed["num_examples"] == 2
    assert completed["rollouts_per_example"] == 3


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


def test_execute_jobs_warns_for_deprecated_eval_knobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_run(config):  # noqa: ARG001
        return _stub_results()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda path, cache=None: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda env_id, cache=None: [])

    model_cfg = ModelConfigSchema(id="alias")
    env_cfg = EnvironmentConfigSchema(
        id="medqa",
        save_every=5,
        print_results=True,
    )
    job = ResolvedJob(
        job_id="alias-medqa",
        name="alias-medqa",
        model=model_cfg,
        env=env_cfg,
        env_args={},
        sampling_args={},
    )

    with caplog.at_level(logging.WARNING):
        results = execute_jobs(
            [job],
            _settings(
                tmp_path,
                max_concurrent_generation=2,
                max_concurrent_scoring=3,
            ),
        )

    assert results[0].status == "succeeded"
    assert "Environment 'medqa' sets deprecated eval knob(s): print_results, save_every" in caplog.text
    assert "Job 'alias-medqa' sets deprecated eval knob(s): max_concurrent_generation, max_concurrent_scoring" in caplog.text


def test_load_endpoints_for_model_missing_default_path_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    settings = _settings(tmp_path, endpoints_path=Path(DEFAULT_ENDPOINTS_PATH), endpoints_path_explicit=False)
    model_cfg = ModelConfigSchema(id="alias")

    endpoints = _load_endpoints_for_model(model_cfg, settings, cache=None)

    assert endpoints == {}


def test_load_endpoints_for_model_missing_explicit_path_raises(tmp_path: Path) -> None:
    settings = _settings(tmp_path, endpoints_path=tmp_path / "missing.toml", endpoints_path_explicit=True)
    model_cfg = ModelConfigSchema(id="alias")

    with pytest.raises(FileNotFoundError):
        _load_endpoints_for_model(model_cfg, settings, cache=None)
