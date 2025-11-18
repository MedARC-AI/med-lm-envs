import enum
import sys
import types
from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

import pytest

from medarc_verifiers.utils.cli_env_args import gather_env_cli_metadata


@pytest.fixture
def module_registry():
    registered: list[str] = []
    yield registered
    for name in registered:
        sys.modules.pop(name, None)


def _register_env(module_name: str, load_fn: Callable[..., Any], module_registry: list[str]) -> None:
    module = types.ModuleType(module_name)
    module.load_environment = load_fn  # type: ignore[attr-defined]
    sys.modules[module_name] = module
    module_registry.append(module_name)


def test_gather_basic_types(module_registry: list[str]) -> None:
    def load_environment(
        alpha: int,
        beta: float = 1.5,
        use_flag: bool = False,
        name: str = "env",
    ) -> None:
        """Example loader.

        Args:
            alpha: Alpha values for testing.
            beta: Floating beta value.
            use_flag: Whether to enable the feature.
        """

    _register_env("test_env_basic", load_environment, module_registry)
    params = gather_env_cli_metadata("test_env_basic")
    assert [param.name for param in params] == ["alpha", "beta", "use_flag", "name"]

    alpha = params[0]
    assert alpha.required is True
    assert alpha.argparse_type is int
    assert alpha.help == "Alpha values for testing."

    flag = params[2]
    assert flag.action == "BooleanOptionalAction"
    assert flag.default is False
    assert flag.help == "Whether to enable the feature."


def test_optional_literal_and_list(module_registry: list[str]) -> None:
    def load_environment(
        mode: str,
        strategy: Literal["fast", "accurate"] = "fast",
        tags: list[str] | None = None,
    ) -> None:
        """Loader with literal and list options.

        Parameters
        ----------
        mode : str
            Execution mode.
        strategy : {'fast', 'accurate'}
            Selection strategy.
        tags : list[str], optional
            Tags for filtering.
        """

    _register_env("test_env_literal", load_environment, module_registry)
    params = {param.name: param for param in gather_env_cli_metadata("test_env_literal")}

    strategy = params["strategy"]
    assert strategy.choices == ("fast", "accurate")
    assert strategy.help == "Selection strategy."

    tags = params["tags"]
    assert tags.is_list is True
    assert tags.action == "append"
    assert tags.argparse_type is str
    assert tags.help == "Tags for filtering."


def test_unsupported_dict_and_union(module_registry: list[str]) -> None:
    def load_environment(config: dict[str, str], choice: int | str = 0) -> None:
        """Docstring without params."""

    _register_env("test_env_unsupported", load_environment, module_registry)
    params = gather_env_cli_metadata("test_env_unsupported")

    config = params[0]
    assert config.unsupported_reason == "dict unsupported"
    assert "requires --env-args" in config.help

    choice = params[1]
    assert choice.unsupported_reason == "non-optional union unsupported"


def test_enum_str_union(module_registry: list[str]) -> None:
    class AnswerFormat(enum.Enum):
        XML = "xml"
        BOXED = "boxed"

    def load_environment(answer_format: AnswerFormat | str = AnswerFormat.XML) -> None:
        """Loader mixing enum and str."""

    _register_env("test_env_enum_union", load_environment, module_registry)
    params = gather_env_cli_metadata("test_env_enum_union")
    answer_format = params[0]
    assert answer_format.choices == ("xml", "boxed")
    assert answer_format.unsupported_reason is None
    assert answer_format.default == AnswerFormat.XML


def test_missing_annotation_uses_default_type(module_registry: list[str]) -> None:
    def load_environment(limit=5) -> None:
        """Loader without annotations."""

    _register_env("test_env_defaults", load_environment, module_registry)
    params = gather_env_cli_metadata("test_env_defaults")
    limit = params[0]
    assert limit.argparse_type is int
    assert limit.supports_cli is True


def test_decorated_loader_follow_wrapped(module_registry: list[str]) -> None:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        return wraps(fn)(wrapper)

    @decorator
    def load_environment(threshold: float = 0.5) -> None:
        """Loader with decorator."""

    _register_env("test_env_decorated", load_environment, module_registry)
    params = gather_env_cli_metadata("test_env_decorated")
    threshold = params[0]
    assert threshold.argparse_type is float
    assert threshold.default == 0.5
