from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli_new import main
from medarc_verifiers.cli_new.utils.env_args import EnvParam


def _write_config(path: Path, content: str) -> None:
    path.write_text(content)


def _patch_single_run_env(monkeypatch: pytest.MonkeyPatch, metadata: list[EnvParam]) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli_new._single_run.gather_env_cli_metadata",
        lambda env_id: metadata,
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli_new._single_run.load_endpoint_registry",
        lambda *args, **kwargs: {},
    )


def _make_env_param(
    name: str,
    *,
    kind: str = "str",
    required: bool = False,
    default: Any = None,
    choices: tuple[Any, ...] | None = None,
    action: str | None = None,
    annotation: Any = str,
    argparse_type: Any = str,
) -> EnvParam:
    return EnvParam(
        name=name,
        cli_name=name.replace("_", "-"),
        kind=kind,
        default=default,
        required=required,
        help=f"{name} help",
        annotation=annotation,
        argparse_type=argparse_type,
        choices=choices,
        action=action,
        is_list=False,
        element_type=None,
        unsupported_reason=None,
    )


def _stub_cli_result(value: float = 0.5) -> SimpleNamespace:
    metadata = SimpleNamespace(
        path_to_save="",
        avg_reward=value,
        num_examples=1,
        rollouts_per_example=1,
        avg_metrics={"pass_rate": value},
    )
    return SimpleNamespace(metadata=metadata, reward=[value], metrics={"pass_rate": [value]})


def test_cli_runs_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        models:
          model-a:
            model: alias-model
            headers:
              X-Test: one
        envs:
          medqa:
            env_args: {}
        jobs:
          - model: model-a
            env: medqa
        """,
    )

    captured = []

    async def fake_run(config):
        captured.append(config)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli_new._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.run_evaluation", fake_run)

    output_dir = tmp_path / "runs_out"
    env_dir = tmp_path / "envs"
    env_dir.mkdir()

    exit_code = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--env-dir",
            str(env_dir),
            "--max-concurrent",
            "5",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.model == "alias-model"
    assert config.env_dir_path == str(env_dir)
    assert config.client_config.extra_headers == {"X-Test": "one"}
    assert config.max_concurrent == 5
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    assert run_dirs[0].is_dir()
    manifest_path = run_dirs[0] / "run_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["summary"]["completed"] == 1
    assert manifest["jobs"][0]["status"] == "completed"


##


def test_regen_reuses_completed_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        models:
          model-a: {}
          model-b: {}
        envs:
          medqa: {}
        jobs:
          - model: model-a
            env: medqa
          - model: model-b
            env: medqa
        """,
    )

    monkeypatch.setattr("medarc_verifiers.cli_new._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def first_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.run_evaluation", first_run)

    output_dir = tmp_path / "runs_out"
    base_run = "base-run"
    assert (
        main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--run-id", base_run]) == 0
    )

    base_manifest_path = output_dir / base_run / "run_manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    base_manifest["jobs"][1]["status"] = "failed"
    base_manifest["jobs"][1]["reason"] = "boom"
    base_manifest_path.write_text(json.dumps(base_manifest, indent=2))

    calls: list[int] = []

    async def regen_run(config):
        calls.append(1)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.run_evaluation", regen_run)

    regen_id = "regen-run"
    exit_code = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--run-id",
            regen_id,
            "--regen",
            base_run,
        ]
    )
    assert exit_code == 0
    assert len(calls) == 1

    regen_manifest = json.loads((output_dir / regen_id / "run_manifest.json").read_text())
    reasons = {entry["job_id"]: entry.get("reason") for entry in regen_manifest["jobs"]}
    assert reasons["model-a-medqa"] == "up_to_date"
    assert regen_manifest["summary"]["completed"] == 2
    assert regen_manifest["regen_source"] == base_run


def test_auto_resume_discovery_without_run_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Auto-resume should discover a prior matching run when --run-id is omitted."""
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
models:
  model-a: {}
  model-b: {}
envs:
  medqa: {}
jobs:
  - model: model-a
    env: medqa
  - model: model-b
    env: medqa
        """,
    )

    # Avoid external dependencies
    monkeypatch.setattr("medarc_verifiers.cli_new._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def first_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.run_evaluation", first_run)

    output_dir = tmp_path / "runs_out"
    run_id = "discover-me"
    # Create the prior run
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--run-id", run_id]) == 0

    # Mark one job as failed to make the run incomplete
    manifest_path = output_dir / run_id / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"][1]["status"] = "failed"
    manifest["jobs"][1]["reason"] = "boom"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Now resume without specifying --run-id; it should discover the 'discover-me' run
    calls: list[int] = []

    async def resume_run(config):
        calls.append(1)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.run_evaluation", resume_run)

    exit_code = main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--auto-resume"])
    assert exit_code == 0
    assert len(calls) == 1  # only the failed job should be re-run

    # Verify the discovered run was updated to completion
    manifest_after = json.loads(manifest_path.read_text())
    assert manifest_after["summary"]["completed"] == 2
    assert manifest_after["summary"]["failed"] == 0


def test_auto_resume_no_candidate_fails_cleanly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Auto-resume without a matching prior run should fail and not create a new run."""
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
models:
  model-a: {}
envs:
  medqa: {}
jobs:
  - model: model-a
    env: medqa
        """,
    )

    # Stubs to avoid external calls; ensure execution is not invoked
    monkeypatch.setattr("medarc_verifiers.cli_new._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def fail_if_called(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("run_evaluation should not be called when no candidate to auto-resume")

    monkeypatch.setattr("medarc_verifiers.cli_new._job_executor.run_evaluation", fail_if_called)

    output_dir = tmp_path / "runs_out"
    # No prior runs exist; this should fail with exit code 1 and not create a run dir
    exit_code = main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--auto-resume"])
    assert exit_code == 1
    assert not output_dir.exists() or not any(output_dir.iterdir())


def test_single_run_help_lists_env_section_and_header_file(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = [
        _make_env_param(
            "difficulty",
            kind="str",
            default="easy",
            choices=("easy", "hard"),
        )
    ]
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(["medqa", "--help"])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Environment options (ENV=medqa)" in captured
    assert "--header-file" in captured


def test_single_run_missing_required_param_errors(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata = [
        _make_env_param(
            "threshold",
            kind="int",
            required=True,
            annotation=int,
            argparse_type=int,
        )
    ]
    _patch_single_run_env(monkeypatch, metadata)
    monkeypatch.setattr(
        "medarc_verifiers.cli_new._single_run.run_evaluation",
        lambda *args, **kwargs: pytest.fail("Should not run when args invalid."),
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main(["medqa"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Missing required environment arguments: threshold" in err


def test_single_run_boolean_negation_and_sampling_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = [
        _make_env_param(
            "use_think",
            kind="bool",
            default=True,
            annotation=bool,
            argparse_type=None,
            action="BooleanOptionalAction",
        )
    ]
    _patch_single_run_env(monkeypatch, metadata)

    captured = []

    async def fake_run(config):
        captured.append(config)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli_new._single_run.run_evaluation", fake_run)

    exit_code = main.main(
        [
            "medqa",
            "--no-use-think",
            "--sampling-arg",
            "max_tokens=64",
            "--max-tokens",
            "128",
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    eval_config = captured[0]
    assert eval_config.env_args["use_think"] is False
    assert eval_config.sampling_args["max_tokens"] == 64


def test_single_run_header_file_overrides_cli_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    header_file = tmp_path / "headers.txt"
    header_file.write_text("X-Test: file\n", encoding="utf-8")

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--header",
            "X-Test: cli",
            "--header-file",
            str(header_file),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    config = json.loads(output)
    assert config["client_config"]["extra_headers"] == {"X-Test": "file"}


def test_single_run_dry_run_outputs_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    async def fail_if_called(*args, **kwargs):
        raise AssertionError("run_evaluation should not execute during dry-run.")

    monkeypatch.setattr("medarc_verifiers.cli_new._single_run.run_evaluation", fail_if_called)

    exit_code = main.main(["medqa", "--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert '"env_id": "medqa"' in output


def test_env_must_be_first(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    exit_code = main.main(["--temperature", "0.1", "medqa", "--dry-run"])
    assert exit_code == 2
    err = capsys.readouterr().err
    assert "First argument must be ENV" in err

    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)
    exit_code = main.main(["medqa", "--temperature", "0.1", "--dry-run"])
    assert exit_code == 0


def test_removed_print_env_schema(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(["medqa", "--print-env-schema"])

    assert exit_code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments" in err


def test_sampling_precedence_and_none_behavior(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--sampling-arg",
            "max_tokens=64",
            "--max-tokens",
            "128",
        ]
    )
    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["sampling_args"]["max_tokens"] == 64

    exit_code = main.main(["medqa", "--dry-run"])
    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert "max_tokens" not in config["sampling_args"]
