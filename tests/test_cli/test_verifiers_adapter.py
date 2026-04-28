from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from medarc_verifiers.cli.verifiers_adapter import EvalConfigOverrides, build_eval_config, load_toml_eval_configs
from medarc_verifiers.utils.prime_inference import PRIME_INFERENCE_URL


def _write_endpoints(path: Path) -> Path:
    path.write_text(
        """
[[endpoint]]
endpoint_id = "openai-alias"
model = "openai/resolved"
url = "https://openai.example/v1"
key = "OPENAI_ALIAS_KEY"
headers = { "X-Registry" = "1" }

[[endpoint]]
endpoint_id = "replica-alias"
model = "replica/resolved"
url = "https://replica-a.example/v1"
key = "REPLICA_KEY_A"
headers = { "X-Replica" = "a" }

[[endpoint]]
endpoint_id = "replica-alias"
model = "replica/resolved"
url = "https://replica-b.example/v1"
key = "REPLICA_KEY_B"
headers = { "X-Replica" = "b" }
""".strip()
    )
    return path


def test_load_toml_eval_configs_expands_ablation(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    endpoints_path = _write_endpoints(tmp_path / "endpoints.toml")
    config_path.write_text(
        f"""
model = "openai/gpt-4.1-mini"
endpoints_path = "{endpoints_path}"
debug = true
headers_from_state = {{ "X-Trace" = "trace_id" }}
timeout = 30.0

[[eval]]
env_id = "medqa"

[[ablation]]
env_id = "medqa"
env_args = {{ shuffle_answers = true }}

[ablation.sweep.env_args]
shuffle_seed = [1618, 9331]
""".strip()
    )

    configs = load_toml_eval_configs(config_path)

    assert [config["env_id"] for config in configs] == ["medqa", "medqa", "medqa"]
    assert configs[0]["debug"] is True
    assert configs[0]["headers_from_state"] == {"X-Trace": "trace_id"}
    assert configs[0]["timeout"] == 30.0
    assert configs[1]["env_args"] == {"shuffle_answers": True, "shuffle_seed": 1618}
    assert configs[2]["env_args"] == {"shuffle_answers": True, "shuffle_seed": 9331}


def test_load_toml_eval_configs_strips_medarc_metadata(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
model = "openai/gpt-4.1-mini"

[[eval]]
env_id = "medqa"

[medarc.orchestrate.foo]
gpus = 1

[medarc.orchestrate.vllm-container]
image = "vllm/vllm-openai:latest"
""".strip()
    )

    configs = load_toml_eval_configs(config_path)

    assert len(configs) == 1
    assert "medarc" not in configs[0]
    assert configs[0]["env_id"] == "medqa"


def test_build_eval_config_resolves_endpoint_alias_and_core_fields(tmp_path: Path) -> None:
    endpoints_path = _write_endpoints(tmp_path / "endpoints.toml")
    resume_path = tmp_path / "resume"
    resume_path.mkdir()
    (resume_path / "results.jsonl").write_text("")
    (resume_path / "metadata.json").write_text("{}")

    config = build_eval_config(
        {
            "env_id": "medqa",
            "endpoint_id": "openai-alias",
            "endpoints_path": str(endpoints_path),
            "env_args": {"subset": "dev"},
            "sampling_args": {"temperature": 0.2},
            "max_tokens": 123,
            "num_examples": 7,
            "rollouts_per_example": 2,
            "max_concurrent": 4,
            "max_retries": 3,
            "num_workers": 2,
            "debug": True,
            "timeout": 45.0,
            "state_columns": ["question_id", "split"],
            "save_results": True,
            "resume_path": str(resume_path),
            "independent_scoring": True,
            "save_to_hf_hub": True,
            "hf_hub_dataset_name": "org/dataset",
            "headers": {"X-Eval": "table"},
            "header": ["X-Eval: list", "X-Extra: 1"],
            "headers_from_state": {"X-Trace": "trace_id"},
            "header_from_state": ["X-User: user_id"],
        }
    )

    assert config.env_id == "medqa"
    assert config.endpoint_id == "openai-alias"
    assert config.model == "openai/resolved"
    assert config.env_args == {"subset": "dev"}
    assert config.sampling_args["temperature"] == 0.2
    assert config.sampling_args["max_tokens"] == 123
    assert config.num_examples == 7
    assert config.rollouts_per_example == 2
    assert config.max_concurrent == 4
    assert config.max_retries == 3
    assert config.num_workers == 2
    assert config.debug is True
    assert config.extra_env_kwargs == {"timeout_seconds": 45.0}
    assert config.state_columns == ["question_id", "split"]
    assert config.save_results is True
    assert config.resume_path == resume_path
    assert config.independent_scoring is True
    assert config.save_to_hf_hub is True
    assert config.hf_hub_dataset_name == "org/dataset"
    assert config.client_config.api_base_url == "https://openai.example/v1"
    assert config.client_config.api_key_var == "OPENAI_ALIAS_KEY"
    assert config.client_config.extra_headers == {"X-Registry": "1", "X-Eval": "list", "X-Extra": "1"}
    assert config.client_config.extra_headers_from_state == {
        "X-Session-ID": "example_id",
        "X-Trace": "trace_id",
        "X-User": "user_id",
    }


def test_build_eval_config_supports_endpoint_replicas(tmp_path: Path) -> None:
    endpoints_path = _write_endpoints(tmp_path / "endpoints.toml")

    config = build_eval_config(
        {
            "env_id": "medqa",
            "endpoint_id": "replica-alias",
            "endpoints_path": str(endpoints_path),
        }
    )

    assert config.model == "replica/resolved"
    assert [endpoint.api_base_url for endpoint in config.client_config.endpoint_configs] == [
        "https://replica-a.example/v1",
        "https://replica-b.example/v1",
    ]
    assert [endpoint.api_key_var for endpoint in config.client_config.endpoint_configs] == [
        "REPLICA_KEY_A",
        "REPLICA_KEY_B",
    ]
    assert [endpoint.extra_headers for endpoint in config.client_config.endpoint_configs] == [
        {"X-Replica": "a"},
        {"X-Replica": "b"},
    ]


def test_build_eval_config_provider_and_cli_overrides_precede_toml(tmp_path: Path) -> None:
    endpoints_path = _write_endpoints(tmp_path / "endpoints.toml")

    config = build_eval_config(
        {
            "env_id": "medqa",
            "model": "openai-alias",
            "endpoints_path": str(endpoints_path),
            "provider": "openai",
            "api_base_url": "https://toml.example/v1",
            "api_key_var": "TOML_KEY",
            "max_concurrent": 8,
        },
        overrides=EvalConfigOverrides(
            provider="local",
            api_base_url="http://127.0.0.1:9000/v1",
            api_key_var="CLI_KEY",
            max_concurrent=1,
        ),
    )

    assert config.model == "openai/resolved"
    assert config.client_config.api_base_url == "http://127.0.0.1:9000/v1"
    assert config.client_config.api_key_var == "CLI_KEY"
    assert config.max_concurrent == 1
    assert config.client_config.endpoint_configs == []


def test_build_eval_config_unknown_model_uses_prime_provider_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")

    config = build_eval_config({"env_id": "medqa", "model": "prime-model", "sampling_args": {"top_k": 20}})

    assert config.model == "prime-model"
    assert config.client_config.api_base_url == PRIME_INFERENCE_URL
    assert config.client_config.api_key_var == "PRIME_API_KEY"
    assert config.client_config.extra_headers == {"X-Prime-Team-ID": "team-123"}
    assert config.sampling_args["extra_body"]["usage"] == {"include": True}
    assert config.sampling_args["extra_body"]["top_k"] == 20


def test_build_eval_config_sanitizes_unknown_sampling_args() -> None:
    config = build_eval_config(
        {
            "env_id": "medqa",
            "provider": "openai",
            "model": "openai/gpt-4.1-mini",
            "sampling_args": {"temperature": 0.4, "top_k": 40, "extra_body": {"known": True}},
        }
    )

    assert config.sampling_args["temperature"] == 0.4
    assert config.sampling_args["extra_body"] == {"known": True, "top_k": 40}


def test_build_eval_config_uses_env_pyproject_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_dir = tmp_path / "adapter_default_env_project"
    package_dir = project_dir / "adapter_default_env"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("")
    (project_dir / "pyproject.toml").write_text(
        """
[tool.verifiers.eval]
num_examples = 11
rollouts_per_example = 4
""".strip()
    )
    monkeypatch.syspath_prepend(str(project_dir))
    importlib.invalidate_caches()

    config = build_eval_config(
        {
            "env_id": "adapter-default-env",
            "provider": "openai",
            "model": "openai/gpt-4.1-mini",
        }
    )

    assert config.num_examples == 11
    assert config.rollouts_per_example == 4


def test_build_eval_config_rejects_invalid_resume_path(tmp_path: Path) -> None:
    invalid_resume_path = tmp_path / "missing"

    with pytest.raises(ValueError, match="not a valid evaluation results path"):
        build_eval_config(
            {
                "env_id": "medqa",
                "provider": "openai",
                "model": "openai/gpt-4.1-mini",
                "resume": str(invalid_resume_path),
            }
        )


def test_build_eval_config_auto_resume_uses_upstream_path_lookup(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "evals" / "medqa--openai--gpt-4.1-mini" / "abc12345"
    run_dir.mkdir(parents=True)
    (run_dir / "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "env_id": "medqa",
                "model": "openai/gpt-4.1-mini",
                "num_examples": 2,
                "rollouts_per_example": 1,
            }
        )
    )

    config = build_eval_config(
        {
            "env_id": "medqa",
            "provider": "openai",
            "model": "openai/gpt-4.1-mini",
            "num_examples": 2,
            "rollouts_per_example": 1,
            "output_dir": str(output_dir),
            "resume": True,
        }
    )

    assert config.resume_path == run_dir
