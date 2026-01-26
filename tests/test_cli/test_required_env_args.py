from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import pytest

from medarc_verifiers.cli._single_run import EnvOptionBinding, extract_env_cli_args
from medarc_verifiers.cli.utils.env_args import MissingEnvParamError, ensure_required_params


@dataclass(frozen=True)
class _ParamStub:
    name: str
    required: bool
    annotation: object
    # Fields used by extract_env_cli_args:
    is_list: bool = False
    action: str | None = None
    kind: str = "str"


def test_ensure_required_params_missing_key_raises() -> None:
    params = [_ParamStub(name="threshold", required=True, annotation=float)]
    with pytest.raises(MissingEnvParamError, match="threshold"):
        ensure_required_params(params, explicit={}, json_args={})


def test_ensure_required_params_required_none_disallowed_raises() -> None:
    params = [_ParamStub(name="threshold", required=True, annotation=float)]
    with pytest.raises(MissingEnvParamError, match="threshold"):
        ensure_required_params(params, explicit={}, json_args={"threshold": None})


def test_ensure_required_params_required_none_allowed_by_optional_passes() -> None:
    params = [_ParamStub(name="threshold", required=True, annotation=Optional[float])]
    ensure_required_params(params, explicit={}, json_args={"threshold": None})


def test_extract_env_cli_args_skips_unset_boolean_none() -> None:
    param = _ParamStub(
        name="flag",
        required=True,
        annotation=bool,
        action="BooleanOptionalAction",
        kind="bool",
    )
    bindings = {"flag": EnvOptionBinding(param=param, dest="flag", default=None)}
    ns = argparse.Namespace(flag=None)

    explicit = extract_env_cli_args(ns, bindings)
    assert explicit == {}


def test_extract_env_cli_args_includes_set_boolean() -> None:
    param = _ParamStub(
        name="flag",
        required=False,
        annotation=bool,
        action="BooleanOptionalAction",
        kind="bool",
    )
    bindings = {"flag": EnvOptionBinding(param=param, dest="flag", default=None)}
    ns = argparse.Namespace(flag=True)

    explicit = extract_env_cli_args(ns, bindings)
    assert explicit == {"flag": True}


