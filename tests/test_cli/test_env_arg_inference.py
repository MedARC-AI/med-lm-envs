from __future__ import annotations

from medarc_verifiers.cli.utils import env_args as cli_env_args
from medarc_verifiers.utils import cli_env_args as legacy_env_args


def _assert_union_list_supported(module) -> None:
    spec = module._infer_argparse_spec(str | list[str], module._EMPTY)
    assert spec.is_list is True
    assert spec.action == "append"
    assert spec.element_type is str
    assert spec.unsupported_reason is None

    optional_spec = module._infer_argparse_spec(str | list[str] | None, module._EMPTY)
    assert optional_spec.is_list is True
    assert optional_spec.element_type is str


def test_infer_argparse_spec_supports_union_list_cli() -> None:
    _assert_union_list_supported(cli_env_args)


def test_infer_argparse_spec_supports_union_list_legacy() -> None:
    _assert_union_list_supported(legacy_env_args)
