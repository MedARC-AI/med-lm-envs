import json
import sys
import types
from collections.abc import Callable
from typing import Any, Literal

import pytest

from medarc_verifiers.cli.eval import main


@pytest.fixture(autouse=True)
def stub_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid touching the real endpoints registry during tests."""
    monkeypatch.setattr("medarc_verifiers.cli.eval.load_endpoints", lambda _path: {})


@pytest.fixture
def capture_eval(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def fake_run_evaluation(config: Any) -> None:
        captured["config"] = config

    monkeypatch.setattr("medarc_verifiers.cli.eval.run_evaluation", fake_run_evaluation)
    return captured


@pytest.fixture
def module_registry():
    registered: list[str] = []
    yield registered
    for name in registered:
        sys.modules.pop(name, None)


@pytest.fixture
def register_env(module_registry: list[str]):
    def _register(module_name: str, load_fn: Callable[..., Any]) -> None:
        module = types.ModuleType(module_name)
        module.load_environment = load_fn  # type: ignore[attr-defined]
        sys.modules[module_name] = module
        module_registry.append(module_name)

    return _register


def test_cli_overrides_json(
    register_env: Callable[[str, Callable[..., Any]], None], capture_eval: dict[str, Any]
) -> None:
    def load_environment(
        required: int,
        optional: int = 11,
        flag: bool = False,
        labels: list[str] | None = None,
        config: dict[str, str] | None = None,
    ) -> None:
        """Loader mixing supported and unsupported params.

        Args:
            required: Mandatory integer.
            optional: Tunable integer.
            flag: Toggle feature.
            labels: Tags for filtering.
            config: Fallback to --env-args only.
        """

    register_env("cli_two_phase_env", load_environment)

    exit_code = main(
        [
            "cli_two_phase_env",
            "--required",
            "3",
            "--optional",
            "7",
            "--flag",
            "--labels",
            "alpha",
            "--labels",
            "beta",
            "--env-args",
            '{"optional": 2, "flag": false, "config": {"mode": "fast"}}',
        ]
    )

    assert exit_code == 0
    config = capture_eval["config"]
    assert config.env_id == "cli_two_phase_env"
    assert config.env_args == {
        "required": 3,
        "optional": 7,
        "flag": True,
        "labels": ["alpha", "beta"],
        "config": {"mode": "fast"},
    }


def test_conflicting_parameter_prefixed(
    register_env: Callable[[str, Callable[..., Any]], None], capture_eval: dict[str, Any]
) -> None:
    def load_environment(model: str = "base", limit: int = 1) -> None:
        """Loader with parameter colliding with global flag.

        Args:
            model: Environment-specific model identifier.
            limit: Limit to test override.
        """

    register_env("conflict_env", load_environment)

    exit_code = main(["conflict_env", "--env-model", "custom", "--limit", "7"])
    assert exit_code == 0
    assert capture_eval["config"].env_args == {"model": "custom", "limit": 7}


def test_help_includes_env_options(
    capsys: pytest.CaptureFixture[str],
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment(
        mode: Literal["fast", "accurate"] = "fast",
        use_cache: bool = False,
    ) -> None:
        """Loader documenting parameters.

        Args:
            mode: Selection mode.
            use_cache: Whether to reuse cache.
        """

    register_env("help_env", load_environment)
    exit_code = main(["help_env", "--help"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "--mode" in out
    assert "--use-cache" in out
    assert "config" not in capture_eval


def test_missing_required_param_errors(register_env: Callable[[str, Callable[..., Any]], None]) -> None:
    def load_environment(threshold: float) -> None:
        """Loader with required float."""

    register_env("missing_env", load_environment)

    with pytest.raises(SystemExit) as exc:
        main(["missing_env"])
    assert exc.value.code == 2


def test_json_provides_required_param(
    register_env: Callable[[str, Callable[..., Any]], None], capture_eval: dict[str, Any]
) -> None:
    def load_environment(threshold: float, mode: str = "auto") -> None:
        """Loader needing JSON fallback."""

    register_env("json_env", load_environment)

    exit_code = main(["json_env", "--env-args", '{"threshold": 0.25}'])
    assert exit_code == 0
    assert capture_eval["config"].env_args == {"threshold": 0.25}


def test_print_env_schema(
    capsys: pytest.CaptureFixture[str],
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment(alpha: int = 1, beta: Literal["x", "y"] = "x") -> None:
        """Loader for schema output."""

    register_env("schema_env", load_environment)

    exit_code = main(["schema_env", "--print-env-schema"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["env"] == "schema_env"
    assert any(param["name"] == "alpha" for param in data["parameters"])
    assert "config" not in capture_eval


def test_save_dataset_alias_sets_save_results(
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment() -> None:
        """Environment without required params."""

    register_env("save_alias_env", load_environment)

    exit_code = main(["save_alias_env", "--save-dataset"])
    assert exit_code == 0
    assert capture_eval["config"].save_results is True


def test_state_columns_parsing(
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment() -> None:
        """Environment without required params."""

    register_env("state_columns_env", load_environment)

    exit_code = main(
        [
            "state_columns_env",
            "--state-columns",
            "alpha, beta",
            "--state-columns",
            "gamma",
        ]
    )
    assert exit_code == 0
    assert capture_eval["config"].state_columns == ["alpha", "beta", "gamma"]


def test_sampling_args_precedence(
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment() -> None:
        """Environment without required params."""

    register_env("sampling_env", load_environment)

    exit_code = main(
        [
            "sampling_env",
            "--sampling-args",
            '{"max_tokens": 128}',
            "--max-tokens",
            "256",
            "--temperature",
            "0.75",
        ]
    )
    assert exit_code == 0
    config = capture_eval["config"]
    assert config.sampling_args["max_tokens"] == 128
    assert config.sampling_args["temperature"] == 0.75


def test_endpoint_registry_substitution(
    monkeypatch: pytest.MonkeyPatch,
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment() -> None:
        """Environment without required params."""

    register_env("endpoint_env", load_environment)

    monkeypatch.setattr(
        "medarc_verifiers.cli.eval.load_endpoints",
        lambda _path: {
            "alias-model": {
                "model": "resolved-model",
                "key": "REGISTRY_KEY",
                "url": "https://registry.example/v1",
            }
        },
    )

    exit_code = main(
        [
            "endpoint_env",
            "--model",
            "alias-model",
            "--api-key-var",
            "CLI_KEY",
            "--api-base-url",
            "https://cli.example/v1",
        ]
    )
    assert exit_code == 0
    config = capture_eval["config"]
    assert config.model == "resolved-model"
    assert config.client_config.api_key_var == "REGISTRY_KEY"
    assert config.client_config.api_base_url == "https://registry.example/v1"


def test_endpoint_registry_fallback(
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment() -> None:
        """Environment without required params."""

    register_env("fallback_env", load_environment)

    exit_code = main(
        [
            "fallback_env",
            "--model",
            "custom-model",
            "--api-key-var",
            "CLI_KEY",
            "--api-base-url",
            "https://cli.example/v1",
        ]
    )
    assert exit_code == 0
    config = capture_eval["config"]
    assert config.model == "custom-model"
    assert config.client_config.api_key_var == "CLI_KEY"
    assert config.client_config.api_base_url == "https://cli.example/v1"


def test_concurrency_and_save_every_options(
    register_env: Callable[[str, Callable[..., Any]], None],
    capture_eval: dict[str, Any],
) -> None:
    def load_environment() -> None:
        """Environment without required params."""

    register_env("concurrency_env", load_environment)

    exit_code = main(
        [
            "concurrency_env",
            "--max-concurrent",
            "5",
            "--max-concurrent-generation",
            "3",
            "--max-concurrent-scoring",
            "2",
            "--no-interleave-scoring",
            "--save-every",
            "10",
        ]
    )
    assert exit_code == 0
    config = capture_eval["config"]
    assert config.max_concurrent == 5
    assert config.max_concurrent_generation == 3
    assert config.max_concurrent_scoring == 2
    assert config.interleave_scoring is False
    assert config.save_every == 10
