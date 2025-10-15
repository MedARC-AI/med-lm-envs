import sys
import types
import json
from collections.abc import Callable
from typing import Any, Literal

import pytest

from medarc_verifiers.cli.eval import main


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
    monkeypatch: pytest.MonkeyPatch, register_env: Callable[[str, Callable[..., Any]], None]
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

    captured: dict[str, Any] = {}

    def fake_run_eval(**kwargs: Any) -> None:
        captured["env"] = kwargs["env"]
        captured["env_args"] = kwargs["env_args"]

    monkeypatch.setattr("medarc_verifiers.cli.eval.run_eval", fake_run_eval)

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
    assert captured["env"] == "cli_two_phase_env"
    assert captured["env_args"] == {
        "required": 3,
        "optional": 7,
        "flag": True,
        "labels": ["alpha", "beta"],
        "config": {"mode": "fast"},
    }


def test_conflicting_parameter_prefixed(
    monkeypatch: pytest.MonkeyPatch, register_env: Callable[[str, Callable[..., Any]], None]
) -> None:
    def load_environment(model: str = "base", limit: int = 1) -> None:
        """Loader with parameter colliding with global flag.

        Args:
            model: Environment-specific model identifier.
            limit: Limit to test override.
        """

    register_env("conflict_env", load_environment)

    received: dict[str, Any] = {}

    def fake_run_eval(**kwargs: Any) -> None:
        received["env_args"] = kwargs["env_args"]

    monkeypatch.setattr("medarc_verifiers.cli.eval.run_eval", fake_run_eval)

    exit_code = main(["conflict_env", "--env-model", "custom", "--limit", "7"])
    assert exit_code == 0
    assert received["env_args"] == {"model": "custom", "limit": 7}


def test_help_includes_env_options(
    capsys: pytest.CaptureFixture[str],
    register_env: Callable[[str, Callable[..., Any]], None],
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


def test_missing_required_param_errors(register_env: Callable[[str, Callable[..., Any]], None]) -> None:
    def load_environment(threshold: float) -> None:
        """Loader with required float."""

    register_env("missing_env", load_environment)

    with pytest.raises(SystemExit) as exc:
        main(["missing_env"])
    assert exc.value.code == 2


def test_json_provides_required_param(
    monkeypatch: pytest.MonkeyPatch, register_env: Callable[[str, Callable[..., Any]], None]
) -> None:
    def load_environment(threshold: float, mode: str = "auto") -> None:
        """Loader needing JSON fallback."""

    register_env("json_env", load_environment)

    received: dict[str, Any] = {}

    def fake_run_eval(**kwargs: Any) -> None:
        received["env_args"] = kwargs["env_args"]

    monkeypatch.setattr("medarc_verifiers.cli.eval.run_eval", fake_run_eval)

    exit_code = main(["json_env", "--env-args", '{"threshold": 0.25}'])
    assert exit_code == 0
    assert received["env_args"] == {"threshold": 0.25}


def test_print_env_schema(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    register_env: Callable[[str, Callable[..., Any]], None],
) -> None:
    def load_environment(alpha: int = 1, beta: Literal["x", "y"] = "x") -> None:
        """Loader for schema output."""

    register_env("schema_env", load_environment)

    def fail_run_eval(**kwargs: Any) -> None:  # pragma: no cover - should not run
        raise AssertionError("run_eval should not be invoked when printing schema")

    monkeypatch.setattr("medarc_verifiers.cli.eval.run_eval", fail_run_eval)

    exit_code = main(["schema_env", "--print-env-schema"])
    assert exit_code == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["env"] == "schema_env"
    assert any(param["name"] == "alpha" for param in data["parameters"])
