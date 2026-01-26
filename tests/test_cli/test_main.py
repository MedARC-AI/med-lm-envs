from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import main
from medarc_verifiers.cli.process import ProcessResult
from medarc_verifiers.cli.utils.env_args import EnvParam


def _write_config(path: Path, content: str) -> None:
    path.write_text(content)


def _patch_single_run_env(monkeypatch: pytest.MonkeyPatch, metadata: list[EnvParam]) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._single_run.gather_env_cli_metadata",
        lambda env_id: metadata,
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._single_run.load_endpoint_registry",
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

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)

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


def test_batch_api_base_url_override_forces_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        models:
          model-a:
            model: alias-model
            api_base_url: https://config.example/v1
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

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda *args, **kwargs: {
            "alias-model": {"model": "resolved-model", "url": "https://endpoint.example/v1", "key": "REGISTRY_KEY"}
        },
    )
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)

    output_dir = tmp_path / "runs_out"
    env_dir = tmp_path / "envs"
    env_dir.mkdir()

    override_url = "http://127.0.0.1:8000/v1"
    assert (
        main.main(
            [
                "bench",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--env-dir",
                str(env_dir),
                "--api-base-url",
                override_url,
            ]
        )
        == 0
    )

    assert len(captured) == 1
    assert captured[0].client_config.api_base_url == override_url


def test_model_level_max_concurrent_applies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        models:
          model-a:
            model: alias-model
            max_concurrent: 7
        envs:
          medqa: {}
        jobs:
          - model: model-a
            env: medqa
        """,
    )

    captured = []

    async def fake_run(config):
        captured.append(config)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)

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
        ]
    )

    assert exit_code == 0
    assert len(captured) == 1
    config = captured[0]
    assert config.max_concurrent == 7


def test_env_rerun_flag_forces_completed_jobs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_config(
        config_path,
        """
        name: rerun-check
        models:
          model-a:
            model: alias-model
        envs:
          env-a:
            rerun: true
        jobs:
          - model: model-a
            env: env-a
        """,
    )

    captured = []

    async def fake_run(config):
        captured.append(config)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)

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
        ]
    )
    assert exit_code == 0
    assert len(captured) == 1
    run_dirs = list(output_dir.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    exit_code_second = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--env-dir",
            str(env_dir),
        ]
    )
    assert exit_code_second == 0
    assert len(captured) == 2

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    job_entry = manifest["jobs"][0]
    assert job_entry["status"] == "completed"
    assert job_entry["attempt"] == 2
    assert manifest["summary"]["completed"] == 1


def test_cli_env_config_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "jobs.yaml"
    _write_config(
        config_path,
        """
        models:
          model-a: {}
        envs:
          - custom_env
        jobs:
          - model: model-a
            env: custom_env
        """,
    )

    shared_envs = tmp_path / "shared_envs"
    shared_envs.mkdir()
    (shared_envs / "custom_env.yaml").write_text(
        """
        - id: custom_env
          module: custom_env
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def fake_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)

    output_dir = tmp_path / "runs_out"
    exit_code = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--env-config-root",
            str(shared_envs),
        ]
    )

    assert exit_code == 0


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

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def first_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", first_run)

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

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", regen_run)

    # Restart now uses the --restart flag and performs in-place extension of the seed run.
    exit_code = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--restart",
            base_run,
        ]
    )
    assert exit_code == 0
    assert len(calls) == 1

    # Manifest is updated in-place under the original run id (base_run)
    updated_manifest = json.loads((output_dir / base_run / "run_manifest.json").read_text())
    reasons = {entry["job_id"]: entry.get("reason") for entry in updated_manifest["jobs"]}
    assert reasons["model-a-medqa"] == "up_to_date"
    assert updated_manifest["summary"]["completed"] == 2
    # restart_source may remain None for in-place restarts; no assertion on legacy regen_source field.


def test_regen_accepts_path_to_run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--restart can be a direct path to a run directory, not only a run-id under output_dir."""
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

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def fake_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fake_run)

    output_dir = tmp_path / "runs_out"
    base_run = output_dir / "base-run"
    # First run to create a seed manifest
    assert (
        main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--run-id", "base-run"]) == 0
    )

    # Now use --restart with an explicit path to the run directory
    # Use --restart with explicit path to existing run directory; should update in place.
    # Mock interactive prompt to avoid stdin capture when all jobs are already completed.
    monkeypatch.setattr("medarc_verifiers.cli.main._prompt_completed_jobs_action", lambda: "continue")
    exit_code = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--restart",
            str(base_run),
        ]
    )
    assert exit_code == 0
    # Ensure manifest exists after restart-in-place; legacy regen_source not asserted.
    assert (output_dir / "base-run" / "run_manifest.json").exists()


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
    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def first_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", first_run)

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

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", resume_run)

    exit_code = main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)])
    assert exit_code == 0
    assert len(calls) == 1  # only the failed job should be re-run

    # Verify the discovered run was updated to completion
    manifest_after = json.loads(manifest_path.read_text())
    assert manifest_after["summary"]["completed"] == 2
    assert manifest_after["summary"]["failed"] == 0


def test_no_auto_resume_forces_new_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Passing --no-auto-resume should ignore existing manifests and start a new run."""
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

    monkeypatch.setattr("medarc_verifiers.cli._config_loader.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_env_metadata", lambda *args, **kwargs: [])
    monkeypatch.setattr("medarc_verifiers.cli._job_executor.load_endpoint_registry", lambda *args, **kwargs: {})

    async def first_run(config):
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", first_run)

    output_dir = tmp_path / "runs_out"
    run_id = "baseline-run"
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--run-id", run_id]) == 0

    manifest_path = output_dir / run_id / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["jobs"][0]["status"] = "failed"
    manifest["jobs"][0]["reason"] = "boom"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    calls: list[int] = []

    async def fresh_run(config):
        calls.append(1)
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fresh_run)

    preexisting = {child.name for child in output_dir.iterdir()}
    exit_code = main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--no-auto-resume"])
    assert exit_code == 0
    assert len(calls) == 2  # both jobs rerun in the fresh run

    post = {child.name for child in output_dir.iterdir()}
    new_runs = post - preexisting
    assert run_id in post
    assert len(new_runs) == 1
    new_run_id = next(iter(new_runs))
    assert new_run_id != run_id
    assert (output_dir / new_run_id / "run_manifest.json").exists()


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


def test_general_help_uses_invoked_binary_name(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["medarc-eval"])

    exit_code = main.main([])

    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "medarc-eval bench --help" in captured


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
        "medarc_verifiers.cli._single_run.run_evaluation",
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

    monkeypatch.setattr("medarc_verifiers.cli._single_run.run_evaluation", fake_run)

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

    monkeypatch.setattr("medarc_verifiers.cli._single_run.run_evaluation", fail_if_called)

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


def test_process_cli_builds_options(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_root = tmp_path / "envs"
    env_root.mkdir()
    (env_root / "demo.yaml").write_text(
        """
        - id: demo-env
          export:
            extra_columns: [debug]
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run(options, env_export_map):
        captured["options"] = options
        captured["env_export_map"] = env_export_map
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr("medarc_verifiers.cli.main.run_process", fake_run)

    exit_code = main.main(
        [
            "process",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--env-config-root",
            str(env_root),
            "--status",
            "completed",
            "--hf-repo",
            "medarc/demo",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    options = captured["options"]
    assert options.status_filter == ("completed",)
    assert options.hf_config is not None
    env_map = captured["env_export_map"]
    assert "demo-env" in env_map


def test_process_cli_applies_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_root = tmp_path / "envs"
    env_root.mkdir()
    (env_root / "demo.yaml").write_text(
        """
        - id: demo-env
          export:
            extra_columns: [debug]
        """,
        encoding="utf-8",
    )
    cfg_path = tmp_path / "process.yaml"
    cfg_path.write_text(
        f"""
        runs_dir: runs-from-config
        output_dir: processed-from-config
        env_config_root: {env_root}
        max_workers: 2
        hf:
          repo: medarc/demo
          branch: main
          token: secret-token
          private: true
          pull_policy: pull
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run(options, env_export_map):
        captured["options"] = options
        captured["env_export_map"] = env_export_map
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr("medarc_verifiers.cli.main.run_process", fake_run)

    exit_code = main.main(["process", "--config", str(cfg_path), "--dry-run"])
    assert exit_code == 0

    options = captured["options"]
    assert options.runs_dir == Path("runs-from-config")
    assert options.output_dir == Path("processed-from-config")
    assert options.max_workers == 2
    assert options.hf_pull_policy == "pull"
    assert options.hf_config is not None
    assert options.hf_config.repo_id == "medarc/demo"
    assert options.hf_config.branch == "main"
    assert options.hf_config.token == "secret-token"
    assert options.hf_config.private is True

    exit_code = main.main(["process", "--config", str(cfg_path), "--hf-token", "override", "--dry-run"])
    assert exit_code == 0
    options = captured["options"]
    assert options.hf_config is not None
    assert options.hf_config.token == "override"


def test_winrate_cli_applies_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "winrate.yaml"
    cfg_path.write_text(
        """
        processed_dir: runs-from-config
        output_name: from-config
        missing_policy: zero
        epsilon: 0.123
        min_common: 7
        weight_policy: equal
        weight_cap: 99
        include_models: [alpha, beta]
        exclude_model: gamma
        hf:
          repo: medarc/demo
          branch: main
          token: secret-token
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run_winrate(
        *, processed_dir, output_dir, output_path, output_name, config, processed_at, hf_config, hf_processed_pull
    ):
        captured["run_kwargs"] = {
            "processed_dir": processed_dir,
            "output_dir": output_dir,
            "output_path": output_path,
            "output_name": output_name,
            "config": config,
            "processed_at": processed_at,
            "hf_config": hf_config,
            "hf_processed_pull": hf_processed_pull,
        }
        return SimpleNamespace(
            output_path=tmp_path / "out.json",
            result={"models": {}},
            datasets=[("demo-env", [Path("demo-env.parquet")])],
        )

    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

    exit_code = main.main(["winrate", "--config", str(cfg_path), "--processed-at", "2024-01-01T00:00:00Z"])
    assert exit_code == 0

    assert captured["run_kwargs"]["processed_dir"] == Path("runs-from-config")
    cfg = captured["run_kwargs"]["config"]
    assert cfg.missing_policy == "zero"
    assert cfg.epsilon == pytest.approx(0.123)
    assert cfg.min_common == 7
    assert cfg.weight_policy == "equal"
    assert cfg.weight_cap == 99
    assert cfg.include_models == ("alpha", "beta")
    assert cfg.exclude_models == ("gamma",)

    exit_code = main.main(
        [
            "winrate",
            "--config",
            str(cfg_path),
            "--epsilon",
            "0.5",
            "--processed-at",
            "2024-01-01T00:00:00Z",
        ]
    )
    assert exit_code == 0
    cfg = captured["run_kwargs"]["config"]
    assert cfg.epsilon == pytest.approx(0.5)


def test_process_cli_requires_winrate_config_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    with pytest.raises(SystemExit):
        main.main(
            [
                "process",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--output-dir",
                str(tmp_path / "processed"),
                "--winrate",
                str(missing_path),
            ]
        )


def test_process_cli_runs_winrate_post_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "winrate.yaml"
    cfg_path.write_text(
        """
        processed_dir: ignored
        output_dir: winrate-out
        output_name: from-config
        missing_policy: zero
        hf_processed_repo: ignored/also
        hf_winrate_repo: medarc/winrate
        hf_token: secret-token
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run_process(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    def fake_run_winrate(
        *, processed_dir, output_dir, output_path, output_name, config, processed_at, hf_config, hf_processed_pull
    ):
        captured["run_kwargs"] = {
            "processed_dir": processed_dir,
            "output_dir": output_dir,
            "output_path": output_path,
            "output_name": output_name,
            "config": config,
            "processed_at": processed_at,
            "hf_config": hf_config,
            "hf_processed_pull": hf_processed_pull,
        }
        return SimpleNamespace(
            output_path=Path(output_dir) / "winrate.json",
            output_paths=[Path(output_dir) / "winrate.json"],
            result={"models": {}},
            datasets=[],
        )

    def fake_sync_files_to_hub(*, repo_id, output_dir, files, token, private, message, branch=None, dry_run=False):
        captured["upload"] = {
            "repo_id": repo_id,
            "output_dir": output_dir,
            "files": list(files),
            "token": token,
            "private": private,
            "message": message,
            "branch": branch,
            "dry_run": dry_run,
        }

    monkeypatch.setattr(main, "run_process", fake_run_process)
    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "sync_files_to_hub", fake_sync_files_to_hub)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

    exit_code = main.main(
        [
            "process",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--winrate",
            str(cfg_path),
        ]
    )
    assert exit_code == 0
    assert captured["run_kwargs"]["processed_dir"] == Path(tmp_path / "processed")
    assert captured["run_kwargs"]["output_dir"] == Path("winrate-out")
    assert captured["run_kwargs"]["hf_config"] is None
    assert captured["run_kwargs"]["hf_processed_pull"] is False
    upload = captured.get("upload")
    assert upload is not None
    assert upload["repo_id"] == "medarc/winrate"
    assert upload["token"] == "secret-token"
    assert upload["files"] == ["winrate.json"]


def test_process_config_sets_winrate_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "process.yaml"
    winrate_cfg = tmp_path / "winrate.yaml"
    winrate_cfg.write_text("output_dir: runs/winrate\n", encoding="utf-8")
    cfg_path.write_text(
        f"""
        runs_dir: runs/raw
        output_dir: runs/processed
        winrate: {winrate_cfg}
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run_process(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    def fake_run_winrate(
        *, processed_dir, output_dir, output_path, output_name, config, processed_at, hf_config, hf_processed_pull
    ):
        captured["run_kwargs"] = {"processed_dir": processed_dir, "output_dir": output_dir}
        return SimpleNamespace(
            output_path=Path(output_dir) / "winrate.json",
            output_paths=[Path(output_dir) / "winrate.json"],
            result={"models": {}},
            datasets=[],
        )

    monkeypatch.setattr(main, "run_process", fake_run_process)
    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

    exit_code = main.main(
        [
            "process",
            "--config",
            str(cfg_path),
        ]
    )
    assert exit_code == 0
    assert captured["run_kwargs"]["processed_dir"] == Path("runs/processed")


def test_process_cli_rejects_include_prompt_completion(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main.main(
            [
                "process",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--output-dir",
                str(tmp_path / "processed"),
                "--include-prompt-completion",
            ]
        )
