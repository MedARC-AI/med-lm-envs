from __future__ import annotations

from medarc_verifiers.cli.utils import env_args


def test_infer_argparse_spec_supports_union_list() -> None:
    spec = env_args._infer_argparse_spec(str | list[str], env_args._EMPTY)
    assert spec.is_list is True
    assert spec.action == "append"
    assert spec.element_type is str
    assert spec.unsupported_reason is None

    optional_spec = env_args._infer_argparse_spec(str | list[str] | None, env_args._EMPTY)
    assert optional_spec.is_list is True
    assert optional_spec.element_type is str
