from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import main
from medarc_verifiers.cli.process import ProcessResult
from medarc_verifiers.cli._single_run import (
    _build_base_parser_layout,
    extract_env_cli_args,
    register_env_options,
)
from medarc_verifiers.cli.utils.env_args import EnvParam
from medarc_verifiers.utils.prime_inference import PRIME_INFERENCE_URL


def _write_config(path: Path, content: str) -> None:
    path.write_text(content)


def _patch_single_run_env(monkeypatch: pytest.MonkeyPatch, metadata: list[EnvParam]) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._single_run.gather_env_cli_metadata",
        lambda env_id: metadata,
    )


def _patch_single_run_metadata_only(monkeypatch: pytest.MonkeyPatch, metadata: list[EnvParam]) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._single_run.gather_env_cli_metadata",
        lambda env_id: metadata,
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


def _write_resume_artifacts(
    path: Path,
    *,
    env_id: str = "medqa",
    model: str = "gpt-4.1-mini",
    num_examples: int = 5,
    rollouts_per_example: int = 3,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "results.jsonl").write_text("", encoding="utf-8")
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": env_id,
                "model": model,
                "num_examples": num_examples,
                "rollouts_per_example": rollouts_per_example,
            }
        ),
        encoding="utf-8",
    )


def test_toml_bench_dry_run_expands_evals_and_ablations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "bench.toml"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"
        save_results = true

        [[eval]]
        env_id = "medqa"
        num_examples = 1
        rollouts_per_example = 1

        [[ablation]]
        env_id = "medqa"
        name = "shuffle_seed-{env_args.shuffle_seed}"
        num_examples = 1
        rollouts_per_example = 1
        env_args = { shuffle_answers = true }

        [ablation.sweep.env_args]
        shuffle_seed = [1618, 9331]
        """,
    )

    exit_code = main.main(
        [
            "bench",
            "--config",
            str(config_path),
            "--dry-run",
            "--output-dir",
            str(tmp_path / "evals"),
            "--max-concurrent",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TOML Bench Dry Run" in output
    assert "3 eval(s) to dry-run" in output
    assert "base" in output
    assert "shuffle_seed-1618" in output
    assert "shuffle_seed-9331" in output
    assert str(tmp_path / "evals" / "gpt-5-mini" / "medqa" / "base") in output


def test_repository_smoke_toml_config_dry_runs(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main.main(["bench", "--config", "configs/eval/smoke.toml", "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TOML Bench Dry Run" in output
    assert "medqa" in output
    assert "runs/evals/openai-gpt-4.1-mini/medqa" in output


def test_toml_bench_dry_run_accepts_medarc_orchestrate_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "bench.toml"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        num_examples = 1
        rollouts_per_example = 1

        [medarc.orchestrate.foo]
        gpus = 1

        [medarc.orchestrate.vllm-container]
        image = "vllm/vllm-openai:latest"
        """,
    )

    exit_code = main.main(["bench", "--config", str(config_path), "--dry-run"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "TOML Bench Dry Run" in output
    assert "medqa" in output


def test_bench_rejects_non_toml_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = tmp_path / "bench.yaml"
    _write_config(config_path, "models: {}\n")

    with pytest.raises(SystemExit) as excinfo:
        main.main(["bench", "--config", str(config_path), "--dry-run"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "medarc-eval bench now accepts upstream TOML configs only." in err


def test_bench_rejects_removed_yaml_runner_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main.main(["bench", "--config", "configs/eval/smoke.toml", "--restart"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "unrecognized arguments: --restart" in err


def test_repository_verified_toml_config_dry_run_shows_ablation_variants(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main.main(["bench", "--config", "configs/eval/medmarks-verified.toml", "--dry-run", "--eval-index", "9"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "medqa" in output
    assert "shuffle_seed-1618" in output
    assert "runs/evals/openai-gpt-4.1-mini/medqa/shuffle_seed-1618" in output


def test_repository_open_ended_toml_config_loads_expected_judge_args() -> None:
    configs = main.load_toml_eval_configs("configs/eval/medmarks-open_ended.toml")
    healthbench = next(config for config in configs if config["env_id"] == "healthbench")
    medrbench = [config for config in configs if config["env_id"] == "medrbench"]

    assert healthbench["env_args"]["judge_model"] == "openai/gpt-5-mini"
    assert healthbench["env_args"]["judge_base_url"] == "https://api.pinference.ai/api/v1"
    assert {config["env_args"]["task"] for config in medrbench} == {"oracle", "1turn", "free_turn"}


def test_toml_bench_dry_run_model_override(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "bench.toml"
    _write_config(
        config_path,
        """
        model = "config-model"

        [[eval]]
        env_id = "medqa"
        num_examples = 1
        rollouts_per_example = 1
        """,
    )

    exit_code = main.main(["bench", "--config", str(config_path), "--dry-run", "--model", "cli-model"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "cli-model" in output
    assert "config-model" not in output


def test_toml_bench_dry_run_uses_toml_output_dir(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "toml-output"
    _write_config(
        config_path,
        f"""
        model = "gpt-5-mini"
        output_dir = "{output_dir}"

        [[eval]]
        env_id = "medqa"
        """,
    )

    assert main.main(["bench", "--config", str(config_path), "--dry-run"]) == 0

    assert str(output_dir / "gpt-5-mini" / "medqa" / "base") in capsys.readouterr().out


def test_toml_bench_executes_sequentially_to_deterministic_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        num_examples = 1
        rollouts_per_example = 1
        """,
    )
    calls: list[Path] = []

    async def fake_run(config, on_progress=None, **_kwargs):
        results_path = Path(config.resume_path)
        calls.append(results_path)
        metadata = {"env_id": config.env_id, "model": config.model}
        if on_progress is not None:
            on_progress([], [], metadata)
        (results_path / "results.jsonl").write_text(json.dumps({"example_id": "0", "reward": 1.0}) + "\n")
        (results_path / "metadata.json").write_text(json.dumps(metadata))
        return {"outputs": [], "metadata": metadata}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    exit_code = main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)])

    results_path = output_dir / "gpt-5-mini" / "medqa" / "base"
    assert exit_code == 0
    assert calls == [results_path]
    assert (results_path / "results.jsonl").exists()
    metadata = json.loads((results_path / "metadata.json").read_text())
    assert "medarc_config_fingerprint" not in metadata
    assert "variant_id" not in metadata
    assert "variant_payload" not in metadata
    assert not (output_dir / "gpt-5-mini" / ".medarc_eval_metadata.json").exists()


def test_toml_bench_defaults_max_concurrent_to_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    captured: list[int] = []

    async def fake_run(config, **_kwargs):
        captured.append(config.max_concurrent)
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(tmp_path / "evals")]) == 0
    assert captured == [1]

    captured.clear()
    assert (
        main.main(
            [
                "bench",
                "--config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "evals-override"),
                "--max-concurrent",
                "4",
            ]
        )
        == 0
    )
    assert captured == [4]


def test_toml_bench_defaults_to_runs_evals(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "bench.toml"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    calls: list[Path] = []

    async def fake_run(config, **_kwargs):
        results_path = Path(config.resume_path)
        calls.append(results_path)
        (results_path / "metadata.json").write_text(json.dumps({"env_id": "medqa", "model": "gpt-5-mini"}))
        (results_path / "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path)]) == 0
    assert calls == [Path("runs/evals/gpt-5-mini/medqa/base")]


def test_toml_bench_auto_resumes_existing_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        env_args = { shuffle_seed = 1618 }
        """,
    )
    calls = 0

    async def fake_run(config, **_kwargs):
        nonlocal calls
        calls += 1
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        Path(config.resume_path, "metadata.json").write_text(json.dumps({"env_id": "medqa", "model": "gpt-5-mini"}))
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    assert calls == 2


def test_toml_bench_resume_refuses_malformed_existing_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    results_path = output_dir / "gpt-5-mini" / "medqa" / "base"
    (results_path / "metadata.json").mkdir(parents=True)
    (results_path / "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
    calls = 0

    async def fake_run(config, **_kwargs):
        nonlocal calls
        calls += 1
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 1
    assert calls == 0


def test_toml_bench_force_archives_existing_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )

    async def fake_run(config, **_kwargs):
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    results_path = output_dir / "gpt-5-mini" / "medqa" / "base"
    (results_path / "sentinel.txt").write_text("old")

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--force"]) == 0

    archived = list((output_dir / "gpt-5-mini" / "medqa").glob("base__old_*"))
    assert len(archived) == 1
    assert (archived[0] / "sentinel.txt").read_text() == "old"
    assert not (results_path / "sentinel.txt").exists()


def test_toml_bench_resume_preserves_existing_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    calls = 0

    async def fake_run(config, **_kwargs):
        nonlocal calls
        calls += 1
        results_path = Path(config.resume_path)
        if calls == 1:
            (results_path / "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
            (results_path / "metadata.json").write_text(
                json.dumps(
                    {
                        "avg_reward": 0.75,
                        "avg_metrics": {"accuracy": 0.75},
                        "total_tokens": 123,
                    }
                )
            )
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    metadata = json.loads((output_dir / "gpt-5-mini" / "medqa" / "base" / "metadata.json").read_text())
    assert metadata["avg_reward"] == 0.75
    assert metadata["avg_metrics"] == {"accuracy": 0.75}
    assert metadata["total_tokens"] == 123
    assert "medarc_config_fingerprint" not in metadata


def test_toml_bench_does_not_patch_upstream_metadata_saves(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import verifiers.envs.environment as environment_module

    config_path = tmp_path / "bench.toml"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    saved_metadata: list[dict[str, Any]] = []

    def fake_save_metadata(metadata, result_path):
        saved_metadata.append(dict(metadata))
        Path(result_path).mkdir(parents=True, exist_ok=True)
        Path(result_path, "metadata.json").write_text(json.dumps(metadata))

    async def fake_run(config, on_progress=None, **_kwargs):
        metadata = {}
        if on_progress is not None:
            on_progress([], [], metadata)
        environment_module.save_metadata({}, Path(config.resume_path))
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": metadata}

    monkeypatch.setattr(environment_module, "save_metadata", fake_save_metadata)
    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(tmp_path / "evals")]) == 0

    assert saved_metadata == [{}]
    metadata = json.loads((tmp_path / "evals" / "gpt-5-mini" / "medqa" / "base" / "metadata.json").read_text())
    assert "medarc_config_fingerprint" not in metadata
    assert "variant_id" not in metadata
    assert "variant_payload" not in metadata


def test_single_run_help_lists_env_section_and_header_option(
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
    assert "medqa environment options:" in captured
    assert "--header" in captured
    assert "--header-file" not in captured


def test_single_run_help_orders_env_group_before_core_options(
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
    env_idx = captured.index("medqa environment options:")
    core_idx = captured.index("medarc-eval options:")
    assert env_idx < core_idx


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


def test_single_run_headers_pass_through_to_eval_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--header",
            "X-Test: cli",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    config = json.loads(output)
    assert config["client_config"]["extra_headers"] == {"X-Test": "cli"}


def test_single_run_auto_adds_prime_team_header(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--api-base-url",
            PRIME_INFERENCE_URL,
            "--header",
            "X-Test: cli",
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    config = json.loads(output)
    assert config["client_config"]["extra_headers"] == {
        "X-Prime-Team-ID": "team-123",
        "X-Test": "cli",
    }


def test_single_run_prime_url_forces_prime_api_key_when_key_var_not_explicit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--api-base-url",
            PRIME_INFERENCE_URL,
        ]
    )

    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["client_config"]["api_key_var"] == "PRIME_API_KEY"


def test_single_run_explicit_api_key_var_is_respected_for_prime_url(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--api-base-url",
            PRIME_INFERENCE_URL,
            "--api-key-var",
            "OPENAI_API_KEY",
        ]
    )

    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["client_config"]["api_key_var"] == "OPENAI_API_KEY"


def test_single_run_endpoint_alias_uses_registry_url_and_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)
    endpoints_path = tmp_path / "endpoints.toml"
    endpoints_path.write_text(
        """
        [[endpoint]]
        endpoint_id = "openai-alias"
        model = "openai/resolved"
        url = "https://registry.example/v1"
        key = "REGISTRY_KEY"
        """,
        encoding="utf-8",
    )

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--model",
            "openai-alias",
            "--endpoints-path",
            str(endpoints_path),
        ]
    )

    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["endpoint_id"] == "openai-alias"
    assert config["model"] == "openai/resolved"
    assert config["client_config"]["api_base_url"] == "https://registry.example/v1"
    assert config["client_config"]["api_key_var"] == "REGISTRY_KEY"


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


def test_single_run_retry_flags_apply_to_client_and_eval_config(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--http-max-retries",
            "7",
            "--rollout-max-retries",
            "3",
        ]
    )

    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["client_config"]["max_retries"] == 7
    assert config["max_retries"] == 3


def test_single_run_resume_path_sets_eval_resume_path_and_forces_save_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)
    resume_dir = tmp_path / "resume-explicit"
    _write_resume_artifacts(resume_dir)

    exit_code = main.main(
        [
            "medqa",
            "--dry-run",
            "--resume",
            str(resume_dir),
            "--no-save-results",
        ]
    )

    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["resume_path"] == str(resume_dir)
    assert config["save_results"] is True


def test_single_run_resume_auto_discovery_sets_eval_resume_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)
    discovered = tmp_path / "resume-auto"
    _write_resume_artifacts(discovered)
    captured: dict[str, Any] = {}

    def fake_find_latest_incomplete_eval_results_path(**kwargs: Any) -> Path:
        captured.update(kwargs)
        return discovered

    monkeypatch.setattr(
        "medarc_verifiers.cli.utils.resume.find_latest_incomplete_eval_results_path",
        fake_find_latest_incomplete_eval_results_path,
    )

    exit_code = main.main(["medqa", "--dry-run", "--resume"])

    assert exit_code == 0
    config = json.loads(capsys.readouterr().out)
    assert config["resume_path"] == str(discovered)
    assert captured["env_id"] == "medqa"
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["rollouts_per_example"] == 3


def test_single_run_resume_mismatch_logs_saved_and_current_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)
    resume_dir = tmp_path / "resume-mismatch"
    _write_resume_artifacts(
        resume_dir,
        env_id="saved-env",
        model="saved-model",
        num_examples=8,
        rollouts_per_example=2,
    )

    async def fake_run(_config: Any) -> Any:
        raise ValueError(
            f"Cannot resume from {resume_dir}: metadata mismatch (env_id: saved='saved-env', current='medqa'). "
            "Use matching evaluation settings or provide a new results path."
        )

    monkeypatch.setattr("medarc_verifiers.cli._single_run.run_evaluation", fake_run)

    with caplog.at_level(logging.ERROR):
        exit_code = main.main(
            [
                "medqa",
                "--model",
                "current-model",
                "--num-examples",
                "5",
                "--rollouts-per-example",
                "3",
                "--resume",
                str(resume_dir),
            ]
        )

    assert exit_code == 1
    assert "Resume metadata mismatch for" in caplog.text
    assert "env_id: saved='saved-env', current='medqa'" in caplog.text
    assert "model: saved='saved-model', current='current-model'" in caplog.text
    assert "rollouts_per_example: saved=2, current=3" in caplog.text
    assert "num_examples: saved=8, current=5 (current must be >= saved)" in caplog.text
    assert "Evaluation failed" not in caplog.text


def test_single_run_uses_empty_registry_when_default_endpoints_path_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_metadata_only(monkeypatch, metadata)
    monkeypatch.chdir(tmp_path)

    exit_code = main.main(["medqa", "--dry-run"])

    assert exit_code == 0


def test_single_run_explicit_missing_endpoints_path_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_metadata_only(monkeypatch, metadata)
    monkeypatch.chdir(tmp_path)

    exit_code = main.main(["medqa", "--dry-run", "--endpoints-path", "does_not_exist.toml"])

    assert exit_code == 2


def test_single_run_warns_when_save_every_is_set(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    metadata: list[EnvParam] = []
    _patch_single_run_env(monkeypatch, metadata)

    async def fake_run(config):  # noqa: ARG001
        return _stub_cli_result()

    monkeypatch.setattr("medarc_verifiers.cli._single_run.run_evaluation", fake_run)

    with caplog.at_level(logging.WARNING):
        exit_code = main.main(["medqa", "--save-every", "10"])

    assert exit_code == 0
    assert "Single-run option --save-every is deprecated and ignored." in caplog.text


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


def test_load_env_export_map_adds_module_variant_keys(tmp_path: Path) -> None:
    env_root = tmp_path / "envs"
    env_root.mkdir()
    (env_root / "medcalc_bench.yaml").write_text(
        """
        - id: medcalc_bench_tools
          module: medcalc_bench
          env_args:
            version: verified
            add_python_tool: true
            add_calculator_tool: true
          export:
            extra_columns: [lower_bound, upper_bound]
            answer_column: ground_truth
        """,
        encoding="utf-8",
    )

    env_map = main._load_env_export_map(env_root)

    variant_key = "medcalc_bench::env_args.add_calculator_tool-true__env_args.add_python_tool-true__env_args.version-verified"
    assert "medcalc_bench_tools" in env_map
    assert variant_key in env_map
    assert env_map[variant_key].answer_column == "ground_truth"


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
        runs_dir: runs/raw-from-config
        process:
          dir: processed
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
    assert options.runs_dir == Path("runs/raw-from-config")
    assert options.output_dir == Path("runs/processed")
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

    exit_code = main.main(["process", "--config", str(cfg_path), "--hf-pull-policy", "continue-upload", "--dry-run"])
    assert exit_code == 0
    options = captured["options"]
    assert options.hf_pull_policy == "continue-upload"


def test_process_cli_resolves_hf_token_env_reference(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "process.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw-from-config
        process:
          dir: processed
        hf:
          repo: medarc/demo
          token: $HF_TOKEN
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "env-secret")

    captured: dict[str, Any] = {}

    def fake_run(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr("medarc_verifiers.cli.main.run_process", fake_run)

    exit_code = main.main(["process", "--config", str(cfg_path), "--dry-run"])
    assert exit_code == 0
    assert captured["options"].hf_config is not None
    assert captured["options"].hf_config.token == "env-secret"


def test_winrate_cli_applies_config_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "winrate.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw-from-config
        process:
          dir: processed
        winrate:
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
          winrate_dir: scorecards/latest
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
            output_paths=[tmp_path / "out.json"],
            result={"models": {}},
            datasets=[("demo-env", [Path("demo-env.parquet")])],
        )

    def fake_sync_files_to_hub(**kwargs):
        captured["upload"] = kwargs

    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "sync_files_to_hub", fake_sync_files_to_hub)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

    exit_code = main.main(["winrate", "--config", str(cfg_path), "--processed-at", "2024-01-01T00:00:00Z"])
    assert exit_code == 0

    assert captured["run_kwargs"]["processed_dir"] == Path("runs/processed")
    assert captured["run_kwargs"]["output_dir"] == Path("runs/processed") / "winrate"
    cfg = captured["run_kwargs"]["config"]
    assert cfg.missing_policy == "zero"
    assert cfg.epsilon == pytest.approx(0.123)
    assert cfg.min_common == 7
    assert cfg.weight_policy == "equal"
    assert cfg.weight_cap == 99
    assert cfg.include_models == ("alpha", "beta")
    assert cfg.exclude_models == ("gamma",)
    assert captured["run_kwargs"]["hf_config"] is not None
    assert captured["run_kwargs"]["hf_config"].repo_id == "medarc/demo"
    upload = captured.get("upload")
    assert upload is not None
    assert upload["repo_id"] == "medarc/demo"
    assert upload["path_in_repo_prefix"] == "scorecards/latest"

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


def test_winrate_cli_resolves_hf_token_braced_env_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "winrate.yaml"
    cfg_path.write_text(
        """
        processed_dir: runs/processed
        hf:
          repo: medarc/demo
          token: ${HF_TOKEN}
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("HF_TOKEN", "env-secret")

    captured: dict[str, Any] = {}

    def fake_run_winrate(
        *, processed_dir, output_dir, output_path, output_name, config, processed_at, hf_config, hf_processed_pull
    ):
        captured["hf_config"] = hf_config
        return SimpleNamespace(
            output_path=tmp_path / "out.json",
            output_paths=[tmp_path / "out.json"],
            result={"models": {}},
            datasets=[],
        )

    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(main, "sync_files_to_hub", lambda **_kwargs: None)

    exit_code = main.main(["winrate", "--config", str(cfg_path), "--processed-at", "2024-01-01T00:00:00Z"])
    assert exit_code == 0
    assert captured["hf_config"] is not None
    assert captured["hf_config"].token == "env-secret"


def test_process_cli_rejects_unset_hf_token_env_reference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / "process.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw-from-config
        process:
          dir: processed
        hf:
          repo: medarc/demo
          token: $HF_TOKEN
        """,
        encoding="utf-8",
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        main.main(["process", "--config", str(cfg_path), "--dry-run"])

    assert excinfo.value.code == 2
    assert "references unset environment variable 'HF_TOKEN'" in capsys.readouterr().err


def test_expand_embedded_process_config_promotes_process_section() -> None:
    payload = {
        "runs_dir": "runs/raw",
        "process": {
            "dir": "processed",
            "max_workers": 8,
            "replace_models": ["model-a"],
        },
        "winrate": {"dir": "scorecards"},
    }

    expanded = main._expand_embedded_pipeline_config(payload, mode="process")

    assert expanded["runs_dir"] == "runs/raw"
    assert expanded["output_dir"] == Path("runs/processed")
    assert expanded["max_workers"] == 8
    assert expanded["replace_models"] == ["model-a"]
    assert "winrate" not in expanded
    assert payload["process"]["dir"] == "processed"


def test_expand_embedded_winrate_config_resolves_relative_dirs() -> None:
    payload = {
        "runs_dir": "artifacts/raw",
        "process": {"dir": "processed"},
        "winrate": {
            "dir": "scorecards",
            "missing_policy": "zero",
            "hf_winrate_dir": "uploads/winrate",
        },
    }

    expanded = main._expand_embedded_pipeline_config(payload, mode="winrate")

    assert expanded["processed_dir"] == Path("artifacts/processed")
    assert expanded["output_dir"] == Path("artifacts/processed/scorecards")
    assert expanded["missing_policy"] == "zero"
    assert expanded["hf_winrate_dir"] == "uploads/winrate"


def test_expand_embedded_winrate_config_keeps_explicit_dirs() -> None:
    payload = {
        "processed_dir": "custom/processed",
        "output_dir": "custom/winrate",
        "runs_dir": "artifacts/raw",
        "process": {"dir": "processed"},
        "winrate": {"dir": "scorecards"},
    }

    expanded = main._expand_embedded_pipeline_config(payload, mode="winrate")

    assert expanded["processed_dir"] == "custom/processed"
    assert expanded["output_dir"] == "custom/winrate"


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


def test_process_cli_defaults_status_filter_to_completed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_process(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr(main, "run_process", fake_run_process)

    exit_code = main.main(
        [
            "process",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    options = captured["options"]
    assert options.status_filter == ("completed",)
    assert options.processed_with_args["status"] == ["completed"]
    assert options.max_results_missing_pct == pytest.approx(2.5)
    assert options.processed_with_args["max_results_missing_pct"] == pytest.approx(2.5)


def test_process_cli_uses_explicit_status_filter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_run_process(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr(main, "run_process", fake_run_process)

    exit_code = main.main(
        [
            "process",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output-dir",
            str(tmp_path / "processed"),
            "--status",
            "failed",
            "--max-results-missing-pct",
            "100",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    options = captured["options"]
    assert options.status_filter == ("failed",)
    assert options.processed_with_args["status"] == ["failed"]
    assert options.max_results_missing_pct == pytest.approx(100.0)


def test_process_cli_rejects_negative_max_results_missing_pct(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main.main(
            [
                "process",
                "--runs-dir",
                str(tmp_path / "runs"),
                "--output-dir",
                str(tmp_path / "processed"),
                "--max-results-missing-pct",
                "-1",
            ]
        )

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "--max-results-missing-pct must be non-negative." in err


def test_process_config_empty_status_uses_default_filter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "process.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw
        process:
          dir: processed
          status: []
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run_process(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr(main, "run_process", fake_run_process)

    exit_code = main.main(["process", "--config", str(cfg_path), "--dry-run"])

    assert exit_code == 0
    options = captured["options"]
    assert options.status_filter == ("completed",)
    assert options.processed_with_args["status"] == ["completed"]


def test_process_cli_runs_embedded_winrate_post_step(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "process.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw
        process:
          dir: processed
        winrate:
          dir: scorecards
          output_name: from-config
          missing_policy: zero
          hf_winrate_dir: winrate-post
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

    def fake_sync_files_to_hub(
        *, repo_id, output_dir, files, token, private, message, branch=None, dry_run=False, **_kw
    ):
        captured["upload"] = {
            "repo_id": repo_id,
            "output_dir": output_dir,
            "files": list(files),
            "token": token,
            "private": private,
            "message": message,
            "branch": branch,
            "dry_run": dry_run,
            **_kw,
        }

    monkeypatch.setattr(main, "run_process", fake_run_process)
    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "sync_files_to_hub", fake_sync_files_to_hub)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

    exit_code = main.main(
        [
            "process",
            "--config",
            str(cfg_path),
            "--hf-repo",
            "medarc/shared",
            "--hf-token",
            "secret-token",
        ]
    )
    assert exit_code == 0
    assert captured["run_kwargs"]["processed_dir"] == Path("runs/processed")
    assert captured["run_kwargs"]["output_dir"] == Path("runs/processed/scorecards")
    assert captured["run_kwargs"]["hf_config"] is None
    assert captured["run_kwargs"]["hf_processed_pull"] is False
    upload = captured.get("upload")
    assert upload is not None
    assert upload["repo_id"] == "medarc/shared"
    assert upload["token"] == "secret-token"
    assert upload["files"] == ["winrate.json"]
    assert upload["path_in_repo_prefix"] == "winrate-post"


def test_process_cli_defaults_winrate_output_dir_under_processed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "process.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw
        process:
          dir: processed
        winrate:
          missing_policy: zero
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
        }
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
    assert captured["run_kwargs"]["output_dir"] == Path("runs/processed/winrate")


def test_process_config_sets_winrate_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg_path = tmp_path / "process.yaml"
    fake_runs_dir = tmp_path / "runs" / "raw"
    fake_runs_dir.mkdir(parents=True)
    cfg_path.write_text(
        f"""
        runs_dir: {fake_runs_dir}
        process:
          dir: processed
        winrate:
          enabled: true
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
    expected_processed_dir = fake_runs_dir.parent / "processed"
    assert captured["run_kwargs"]["processed_dir"] == expected_processed_dir
    assert captured["run_kwargs"]["output_dir"] == expected_processed_dir / "winrate"


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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_workers", "not-an-int"),
        ("max_results_missing_pct", "not-a-float"),
        ("hf_request_timeout", "not-a-float"),
        ("hf_retries", "not-an-int"),
        ("hf_max_files_per_commit", "not-an-int"),
    ],
)
def test_process_cli_rejects_invalid_typed_config_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: str,
) -> None:
    cfg_path = tmp_path / "process-invalid.yaml"
    cfg_path.write_text(
        f"""
        runs_dir: runs/raw
        output_dir: runs/processed
        {field}: {value}
        """,
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main(["process", "--config", str(cfg_path)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"Invalid process config value for '{field}'" in err
    assert value in err


def test_process_cli_rejects_removed_top_level_max_run_missing_pct_config_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / "process-removed-top-level.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw
        output_dir: runs/processed
        max_run_missing_pct: 2.5
        """,
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main(["process", "--config", str(cfg_path)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Process config field 'max_run_missing_pct' was removed" in err
    assert "max_results_missing_pct" in err


def test_process_cli_rejects_removed_embedded_max_run_missing_pct_config_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg_path = tmp_path / "process-removed-embedded.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw
        process:
          dir: processed
          max_run_missing_pct: 2.5
        """,
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main(["process", "--config", str(cfg_path)])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "Process config field 'process.max_run_missing_pct' was removed" in err
    assert "process.max_results_missing_pct" in err


def test_winrate_cli_ignores_removed_process_only_missing_pct_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "winrate-process-key.yaml"
    cfg_path.write_text(
        """
        processed_dir: runs/processed
        process:
          max_run_missing_pct: 2.5
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run_winrate(
        *, processed_dir, output_dir, output_path, output_name, config, processed_at, hf_config, hf_processed_pull
    ):
        captured["processed_dir"] = processed_dir
        return SimpleNamespace(
            output_path=tmp_path / "out.json",
            output_paths=[tmp_path / "out.json"],
            result={"models": {}},
            datasets=[],
        )

    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

    exit_code = main.main(
        [
            "winrate",
            "--config",
            str(cfg_path),
            "--processed-at",
            "2024-01-01T00:00:00Z",
        ]
    )

    assert exit_code == 0
    assert captured["processed_dir"] == Path("runs/processed")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epsilon", "not-a-float"),
        ("min_common", "not-an-int"),
        ("weight_cap", "not-an-int"),
    ],
)
def test_winrate_cli_rejects_invalid_typed_config_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    field: str,
    value: str,
) -> None:
    cfg_path = tmp_path / "winrate-invalid.yaml"
    cfg_path.write_text(
        f"""
        processed_dir: runs/processed
        {field}: {value}
        """,
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        main.main(["winrate", "--config", str(cfg_path), "--processed-at", "2024-01-01T00:00:00Z"])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert f"Invalid winrate config value for '{field}'" in err
    assert value in err


def test_process_cli_allows_cli_override_of_malformed_numeric_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "process-invalid-override.yaml"
    cfg_path.write_text(
        """
        runs_dir: runs/raw
        output_dir: runs/processed
        max_workers: not-an-int
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run(options, env_export_map):
        captured["options"] = options
        return ProcessResult(records_processed=0, rows_processed=0, env_groups=[], env_summaries=[], hf_summary=None)

    monkeypatch.setattr(main, "_load_env_export_map", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(main, "run_process", fake_run)

    exit_code = main.main(["process", "--config", str(cfg_path), "--max-workers", "2", "--dry-run"])
    assert exit_code == 0
    assert captured["options"].max_workers == 2


def test_winrate_cli_allows_cli_override_of_malformed_numeric_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "winrate-invalid-override.yaml"
    cfg_path.write_text(
        """
        processed_dir: runs/processed
        epsilon: not-a-float
        """,
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def fake_run_winrate(
        *, processed_dir, output_dir, output_path, output_name, config, processed_at, hf_config, hf_processed_pull
    ):
        captured["config"] = config
        return SimpleNamespace(
            output_path=tmp_path / "out.json",
            output_paths=[tmp_path / "out.json"],
            result={"models": {}},
            datasets=[],
        )

    monkeypatch.setattr(main, "run_winrate", fake_run_winrate)
    monkeypatch.setattr(main, "print_winrate_summary_markdown", lambda *_args, **_kwargs: None)

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
    assert captured["config"].epsilon == pytest.approx(0.5)


def test_single_run_env_option_collision_uses_env_prefix() -> None:
    metadata = [
        _make_env_param(
            "model",
            kind="str",
            default=None,
            annotation=str,
            argparse_type=str,
        )
    ]
    parser, env_group, reserved_dests = _build_base_parser_layout(require_env=True, add_help=True, env_id="medqa")
    bindings = register_env_options(env_group, reserved_dests, "medqa", metadata)
    args = parser.parse_args(["medqa", "--model", "core-model", "--env-model", "env-model", "--dry-run"])
    explicit = extract_env_cli_args(args, bindings)

    assert args.model == "core-model"
    assert explicit["model"] == "env-model"
