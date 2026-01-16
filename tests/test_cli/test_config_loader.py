from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from medarc_verifiers.cli._config_loader import ConfigFormatError, load_run_config
from medarc_verifiers.cli._job_builder import build_jobs
from medarc_verifiers.cli._job_executor import ExecutorSettings, execute_jobs


@dataclass
class _FakeParam:
    name: str
    required: bool = False
    choices: tuple | None = None
    argparse_type: type | None = None
    is_list: bool = False
    element_type: type | None = None
    action: str | None = None
    kind: str | None = None
    supports_cli: bool = True


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_load_run_config_parses_basic_yaml(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        name: demo-run
        models:
          - id: gpt-mini
        envs:
          - id: medqa
        jobs:
          - model: gpt-mini
            env: medqa
        """,
    )

    config = load_run_config(config_path)

    assert config.name == "demo-run"
    assert len(config.models) == 1
    assert "gpt-mini" in config.models
    assert config.models["gpt-mini"].id == "gpt-mini"
    assert len(config.envs) == 1
    assert "medqa" in config.envs
    assert config.envs["medqa"].id == "medqa"
    assert len(config.jobs) == 1
    assert config.jobs[0].model == "gpt-mini"


def test_load_run_config_supports_mapped_format(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    config_path = _write_yaml(
        tmp_path / "mapped.yaml",
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

    config = load_run_config(config_path)

    assert set(config.models) == {"gpt-mini"}
    assert config.models["gpt-mini"].model == "openai/gpt-mini"
    assert set(config.envs) == {"medqa"}
    assert config.envs["medqa"].num_examples == 5


def test_load_run_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "invalid.yaml",
        """
        - not: a-mapping
        """,
    )

    with pytest.raises(ConfigFormatError):
        load_run_config(config_path)


def test_model_headers_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    config_path = _write_yaml(
        tmp_path / "headers.yaml",
        """
        models:
          - id: bad
            headers:
              - 123
        envs:
          - id: medqa
        jobs:
          - model: bad
            env: medqa
        """,
    )

    with pytest.raises(ValidationError):
        load_run_config(config_path)


def test_environment_num_examples_validation(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    config_path = _write_yaml(
        tmp_path / "envs.yaml",
        """
        envs:
          - id: medqa
            num_examples: 0
        jobs:
          - model: gpt
            env: medqa
        models:
          - id: gpt
        """,
    )

    with pytest.raises(ValidationError):
        load_run_config(config_path)


def test_environment_env_args_unknown(monkeypatch, tmp_path: Path) -> None:
    def fake_metadata(_env_id: str, cache=None):
        return [_FakeParam("shuffle_seed"), _FakeParam("shuffle_answers")]

    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        fake_metadata,
    )

    config_path = _write_yaml(
        tmp_path / "unknown_env_arg.yaml",
        """
        envs:
          - id: medqa
            env_args:
              invalid_param: true
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    with pytest.raises(ValueError):
        load_run_config(config_path)


def test_environment_env_args_known(monkeypatch, tmp_path: Path) -> None:
    def fake_metadata(_env_id: str, cache=None):
        return [_FakeParam("shuffle_answers"), _FakeParam("shuffle_seed")]

    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        fake_metadata,
    )

    config_path = _write_yaml(
        tmp_path / "known_env_arg.yaml",
        """
        envs:
          - id: medqa
            env_args:
              shuffle_answers: true
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    config = load_run_config(config_path)

    assert config.envs["medqa"].env_args == {"shuffle_answers": True}


def test_env_paths_resolve_with_cli_default_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    configs_dir = tmp_path / "configs"
    envs_dir = configs_dir / "envs"
    envs_dir.mkdir(parents=True)
    _write_yaml(
        envs_dir / "medqa.yaml",
        """
        - id: medqa
          module: medqa
        """,
    )
    config_path = _write_yaml(
        configs_dir / "jobs.yaml",
        """
        models:
          - id: gpt
        envs:
          - medqa
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    config = load_run_config(config_path, env_default_root=envs_dir)

    assert "medqa" in config.envs


def test_env_paths_use_env_config_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    shared_envs = tmp_path / "shared_envs"
    shared_envs.mkdir()
    _write_yaml(
        shared_envs / "custom_env.yaml",
        """
        - id: custom_env
          module: custom_env
        """,
    )
    config_path = _write_yaml(
        tmp_path / "jobs.yaml",
        """
        models:
          - id: gpt
        envs:
          - custom_env
        jobs:
          - model: gpt
            env: custom_env
        """,
    )

    config = load_run_config(config_path, env_default_root=shared_envs)

    assert "custom_env" in config.envs


def test_envs_auto_discovered_from_env_config_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    env_root = tmp_path / "auto_envs"
    env_root.mkdir()
    _write_yaml(
        env_root / "auto.yaml",
        """
        - id: auto_env
          module: auto_env
        """,
    )
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          - id: auto-model
        jobs:
          - model: auto-model
            env: auto_env
        """,
    )

    config = load_run_config(config_path, env_default_root=env_root)

    assert sorted(config.envs) == ["auto_env"]


def test_environment_env_args_missing_required(monkeypatch, tmp_path: Path) -> None:
    def fake_metadata(_env_id: str, cache=None):
        return [_FakeParam("subset", required=True)]

    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_env_metadata",
        fake_metadata,
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli._job_executor.load_endpoint_registry",
        lambda _path, cache=None: {},
    )

    async def fail_if_called(*_args, **_kwargs):  # pragma: no cover - sanity guard
        raise AssertionError("run_evaluation should not execute when env args are invalid.")

    monkeypatch.setattr("medarc_verifiers.cli._job_executor.run_evaluation", fail_if_called)

    config_path = _write_yaml(
        tmp_path / "missing_required.yaml",
        """
        envs:
          - id: medqa
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    config = load_run_config(config_path)
    jobs = build_jobs(config)

    env_dir = tmp_path / "envs"
    env_dir.mkdir()

    settings = ExecutorSettings(
        run_id="run-1",
        output_dir=tmp_path / "runs",
        env_dir=env_dir,
        endpoints_path=tmp_path / "endpoints.yaml",
        default_api_key_var="API_KEY",
        default_api_base_url="https://api.example",
        dry_run=False,
    )

    results = execute_jobs(jobs, settings)

    assert results[0].status == "failed"
    assert results[0].error is not None
    assert "Missing required environment arguments" in results[0].error


def test_environment_env_args_type_validation(monkeypatch, tmp_path: Path) -> None:
    def fake_metadata(_env_id: str, cache=None):
        return [_FakeParam("shuffle_seed", argparse_type=int)]

    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        fake_metadata,
    )

    config_path = _write_yaml(
        tmp_path / "invalid_type.yaml",
        """
        envs:
          - id: medqa
            env_args:
              shuffle_seed: wrong
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    with pytest.raises(ValueError):
        load_run_config(config_path)


def test_matrix_expansion_generates_variants(monkeypatch, tmp_path: Path) -> None:
    def fake_metadata(env_id: str, cache=None):  # noqa: ARG001
        return [
            _FakeParam("shuffle_answers"),
            _FakeParam("shuffle_seed", argparse_type=int),
        ]

    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        fake_metadata,
    )

    config_path = _write_yaml(
        tmp_path / "matrix.yaml",
        """
        envs:
          - id: medqa
            module: medqa
            env_args:
              shuffle_answers: true
            matrix:
              shuffle_seed: [1618, 9331]
            matrix_id_format: "{base}-r{shuffle_seed}"
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    config = load_run_config(config_path)

    env_ids = list(config.envs.keys())
    assert env_ids == ["medqa-r1618", "medqa-r9331"]
    assert all(env.matrix is None for env in config.envs.values())
    assert all(env.matrix_base_id == "medqa" for env in config.envs.values())
    assert {env.env_args["shuffle_seed"] for env in config.envs.values()} == {1618, 9331}
    assert all(env.env_args["shuffle_answers"] is True for env in config.envs.values())
    assert all(env.module == "medqa" for env in config.envs.values())


def test_duplicate_env_ids_from_files_expand_variants(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )

    env_dir = tmp_path / "envs"
    env_dir.mkdir()
    _write_yaml(
        env_dir / "longhealth.yaml",
        """
        - id: longhealth
          module: longhealth
          matrix:
            task: ["task1", "task2"]
          matrix_id_format: "{base}-{task}"
        - id: longhealth
          module: longhealth
          matrix:
            task: ["task3"]
          matrix_id_format: "{base}-{task}-alt"
        """,
    )

    config_path = _write_yaml(
        tmp_path / "config.yaml",
        f"""
        envs:
          - "{env_dir}"
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: longhealth
        """,
    )

    config = load_run_config(config_path)

    assert sorted(config.envs.keys()) == ["longhealth-task1", "longhealth-task2", "longhealth-task3-alt"]
    assert all(env.matrix_base_id == "longhealth" for env in config.envs.values())


def test_matrix_exclude_and_scalar_fields(monkeypatch, tmp_path: Path) -> None:
    def fake_metadata(env_id: str, cache=None):  # noqa: ARG001
        return [_FakeParam("shuffle_seed", argparse_type=int)]

    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        fake_metadata,
    )

    config_path = _write_yaml(
        tmp_path / "matrix_scalar.yaml",
        """
        envs:
          - id: medqa
            module: medqa
            matrix:
              num_examples: [10, 20]
              shuffle_seed: [1618, 9331]
            matrix_exclude:
              - num_examples: 20
                shuffle_seed: 9331
            matrix_id_format: "{base}-n{num_examples}-r{shuffle_seed}"
        models:
          - id: gpt
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    config = load_run_config(config_path)

    env_ids = sorted(config.envs)
    assert env_ids == ["medqa-n10-r1618", "medqa-n10-r9331", "medqa-n20-r1618"]

    num_examples = {env_id: env.num_examples for env_id, env in config.envs.items()}
    assert num_examples["medqa-n10-r1618"] == 10
    assert num_examples["medqa-n10-r9331"] == 10
    assert num_examples["medqa-n20-r1618"] == 20

    assert all("shuffle_seed" in env.env_args for env in config.envs.values())
    assert "medqa-n20-r9331" not in env_ids


def test_legacy_model_params_adapter(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )

    config_path = _write_yaml(
        tmp_path / "legacy_model.yaml",
        """
        models:
          - id: gpt
            params:
              model: openai/gpt
              env_overrides:
                medqa:
                  shuffle_answers: true
        envs:
          - id: medqa
            module: medqa
        jobs:
          - model: gpt
            env: medqa
        """,
    )

    config = load_run_config(config_path)

    assert len(config.models) == 1
    model_cfg = config.models["gpt"]
    assert model_cfg.model == "openai/gpt"
    assert model_cfg.env_overrides == {"medqa": {"shuffle_answers": True}}
    assert model_cfg.env_args == {}


def test_envs_can_reference_yaml_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    env_file = tmp_path / "envs.yaml"
    env_file.write_text(
        """
        - id: included-env
          module: included_env
          num_examples: 3
        """,
    )
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        f"""
        models:
          gpt-mini:
            model: openai/gpt-mini
        envs: "{env_file.name}"
        jobs:
          - model: gpt-mini
            env: included-env
        """,
    )

    config = load_run_config(config_path)

    assert set(config.envs) == {"included-env"}
    assert config.envs["included-env"].num_examples == 3


def test_envs_can_reference_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    env_dir = tmp_path / "envs"
    env_dir.mkdir()
    (env_dir / "a.yaml").write_text(
        """
        - id: env-a
          num_examples: 2
        """,
    )
    (env_dir / "b.yaml").write_text(
        """
        - id: env-b
          num_examples: 4
        """,
    )

    config_path = _write_yaml(
        tmp_path / "config.yaml",
        """
        models:
          gpt-mini:
            model: openai/gpt-mini
        envs: "envs"
        jobs:
          - model: gpt-mini
            env: env-a
        """,
    )

    config = load_run_config(config_path)

    assert set(config.envs) == {"env-a", "env-b"}
    assert config.envs["env-b"].num_examples == 4


def test_included_file_strict_shapes(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )

    bad_mapping = tmp_path / "bad_mapping.yaml"
    bad_mapping.write_text(
        """
        env-basic:
          id: env-basic
        env-invalid: 1
        """,
    )

    config_path = _write_yaml(
        tmp_path / "config_bad_mapping.yaml",
        f"""
        models:
          gpt-mini:
            model: openai/gpt-mini
        envs: "{bad_mapping.name}"
        jobs:
          - model: gpt-mini
            env: env-basic
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        load_run_config(config_path)
    assert "mapping of id" in str(excinfo.value)

    bad_list = tmp_path / "bad_list.yaml"
    bad_list.write_text(
        """
        - id: env-basic
        - 42
        """,
    )

    config_path = _write_yaml(
        tmp_path / "config_bad_list.yaml",
        f"""
        models:
          gpt-mini:
            model: openai/gpt-mini
        envs: "{bad_list.name}"
        jobs:
          - model: gpt-mini
            env: env-basic
        """,
    )

    with pytest.raises(ValueError) as excinfo:
        load_run_config(config_path)
    assert "must be a mapping" in str(excinfo.value)


def test_models_can_reference_yaml_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    models_file = tmp_path / "models.yaml"
    models_file.write_text(
        """
        - id: model-a
          model: openai/gpt-a
          sampling_args:
            max_tokens: 256
        """,
    )
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        f"""
        models: "{models_file.name}"
        envs:
          env-a:
            num_examples: 5
        jobs:
          - model: model-a
            env: env-a
        """,
    )

    config = load_run_config(config_path)

    assert set(config.models) == {"model-a"}
    assert config.models["model-a"].sampling_args["max_tokens"] == 256


def test_jobs_can_reference_yaml_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "medarc_verifiers.cli._config_loader.load_env_metadata",
        lambda _env_id, cache=None: [],
    )
    jobs_file = tmp_path / "jobs.yaml"
    jobs_file.write_text(
        """
        - model: gpt-mini
          env: env-a
        """,
    )
    config_path = _write_yaml(
        tmp_path / "config.yaml",
        f"""
        models:
          gpt-mini:
            model: openai/gpt-mini
        envs:
          env-a:
            num_examples: 5
        jobs: "{jobs_file.name}"
        """,
    )

    config = load_run_config(config_path)

    assert len(config.jobs) == 1
    assert config.jobs[0].model == "gpt-mini"
