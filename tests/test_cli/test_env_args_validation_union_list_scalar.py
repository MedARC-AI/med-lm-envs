from __future__ import annotations

import pytest

from medarc_verifiers.cli.utils.env_args import EnvParam, validate_env_args_or_raise


def _param_list_str_or_str(name: str) -> EnvParam:
    return EnvParam(
        name=name,
        cli_name=name.replace("_", "-"),
        kind="list",
        default=None,
        required=False,
        help="",
        annotation=str | list[str] | None,
        argparse_type=str,
        choices=None,
        action="append",
        is_list=True,
        element_type=str,
        unsupported_reason=None,
    )


def _param_list_str_or_str_non_optional(name: str) -> EnvParam:
    return EnvParam(
        name=name,
        cli_name=name.replace("_", "-"),
        kind="list",
        default=None,
        required=False,
        help="",
        annotation=str | list[str],
        argparse_type=str,
        choices=None,
        action="append",
        is_list=True,
        element_type=str,
        unsupported_reason=None,
    )


def _param_list_only_str(name: str) -> EnvParam:
    return EnvParam(
        name=name,
        cli_name=name.replace("_", "-"),
        kind="list",
        default=None,
        required=False,
        help="",
        annotation=list[str] | None,
        argparse_type=str,
        choices=None,
        action="append",
        is_list=True,
        element_type=str,
        unsupported_reason=None,
    )


def test_validate_env_args_allows_scalar_for_union_list_scalar() -> None:
    metadata = [_param_list_str_or_str("judge_base_url")]
    validate_env_args_or_raise("agentclinic", {"judge_base_url": "https://api.openai.com/v1"}, metadata=metadata)


def test_validate_env_args_allows_scalar_for_union_list_scalar_non_optional() -> None:
    metadata = [_param_list_str_or_str_non_optional("judge_base_url")]
    validate_env_args_or_raise("agentclinic", {"judge_base_url": "https://api.openai.com/v1"}, metadata=metadata)


def test_validate_env_args_rejects_scalar_for_list_only() -> None:
    metadata = [_param_list_only_str("judge_base_url")]
    with pytest.raises(ValueError, match=r"must be a list"):
        validate_env_args_or_raise("agentclinic", {"judge_base_url": "https://api.openai.com/v1"}, metadata=metadata)
