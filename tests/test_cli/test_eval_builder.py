from __future__ import annotations

import pytest

from medarc_verifiers.cli._eval_builder import build_client_config
from medarc_verifiers.cli._schemas import ModelConfigSchema


def test_build_client_config_populates_endpoint_configs_for_replicas() -> None:
    model_cfg = ModelConfigSchema(id="alias-model", headers={"X-Test": "1"})
    endpoints = {
        "alias-model": [
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-a.example/v1"},
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-b.example/v1"},
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-c.example/v1"},
        ]
    }

    resolved_model, client_config, sampling_overrides = build_client_config(
        model_cfg,
        endpoints=endpoints,
        default_api_key_var="DEFAULT_KEY",
        default_api_base_url="https://default.example/v1",
        api_base_url_override=None,
        http_max_retries_override=None,
        timeout_override=None,
        headers=None,
    )

    assert resolved_model == "resolved-model"
    assert client_config.api_base_url == "https://endpoint-a.example/v1"
    assert client_config.api_key_var == "MODEL_KEY"
    assert sampling_overrides == {}
    assert [entry.api_base_url for entry in client_config.endpoint_configs] == [
        "https://endpoint-a.example/v1",
        "https://endpoint-b.example/v1",
        "https://endpoint-c.example/v1",
    ]
    assert all(entry.api_key_var == "MODEL_KEY" for entry in client_config.endpoint_configs)
    assert all(entry.extra_headers == {"X-Test": "1"} for entry in client_config.endpoint_configs)


def test_build_client_config_api_base_url_override_suppresses_endpoint_configs() -> None:
    model_cfg = ModelConfigSchema(id="alias-model")
    endpoints = {
        "alias-model": [
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-a.example/v1"},
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-b.example/v1"},
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-c.example/v1"},
        ]
    }

    _, client_config, _ = build_client_config(
        model_cfg,
        endpoints=endpoints,
        default_api_key_var="DEFAULT_KEY",
        default_api_base_url="https://default.example/v1",
        api_base_url_override="http://127.0.0.1:8000/v1",
        http_max_retries_override=None,
        timeout_override=None,
        headers=None,
    )

    assert client_config.api_base_url == "http://127.0.0.1:8000/v1"
    assert client_config.endpoint_configs == []


def test_build_client_config_replicas_must_share_model_and_key() -> None:
    model_cfg = ModelConfigSchema(id="alias-model")
    endpoints = {
        "alias-model": [
            {"model": "resolved-model", "key": "MODEL_KEY", "url": "https://endpoint-a.example/v1"},
            {"model": "resolved-model", "key": "MODEL_KEY_B", "url": "https://endpoint-b.example/v1"},
        ]
    }

    with pytest.raises(ValueError, match="must agree on 'model' and 'key'"):
        build_client_config(
            model_cfg,
            endpoints=endpoints,
            default_api_key_var="DEFAULT_KEY",
            default_api_base_url="https://default.example/v1",
            api_base_url_override=None,
            http_max_retries_override=None,
            timeout_override=None,
            headers=None,
        )
