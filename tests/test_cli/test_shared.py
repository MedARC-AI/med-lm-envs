from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from medarc_verifiers.cli.utils.shared import (
    MissingEnvParamError,
    asdict_sanitized,
    build_headers,
    coerce_json_mapping,
    ensure_required_params,
    flatten_state_columns,
    merge_cli_override_args,
    merge_sampling_args,
    normalize_headers,
    resolve_endpoint_selection,
)
from medarc_verifiers.utils.cli_env_args import EnvParam


def _make_env_param(name: str, *, required: bool) -> EnvParam:
    return EnvParam(
        name=name,
        cli_name=name.replace("_", "-"),
        kind="int",
        default=None,
        required=required,
        help="",
        annotation=int,
        argparse_type=int,
        choices=None,
        action=None,
        is_list=False,
        element_type=None,
        unsupported_reason=None,
    )


def test_merge_sampling_args_precedence() -> None:
    merged = merge_sampling_args({"temperature": 0.2}, max_tokens=128, temperature=0.9)
    assert merged["temperature"] == 0.2  # existing key preserved
    assert merged["max_tokens"] == 128


def test_build_headers_accepts_list() -> None:
    headers = build_headers(["X-Trace: 123", "Authorization: Bearer token"])
    assert headers == {"X-Trace": "123", "Authorization": "Bearer token"}


def test_normalize_headers_file_overrides_cli(tmp_path: Path) -> None:
    header_file = tmp_path / "headers.txt"
    header_file.write_text("X-Trace: from-file\nX-New: added\n", encoding="utf-8")

    headers = normalize_headers(["X-Trace: from-cli"], header_file=header_file)

    assert headers == {"X-Trace": "from-file", "X-New": "added"}


def test_coerce_json_mapping_requires_object() -> None:
    with pytest.raises(ValueError):
        coerce_json_mapping([], flag="--example")


def test_resolve_endpoint_selection_prefers_registry() -> None:
    endpoints = {
        "alias": {"model": "true-model", "key": "SPECIAL_KEY", "url": "https://example.test"},
    }
    resolved = resolve_endpoint_selection(
        "alias",
        endpoints,
        default_key_var="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
    )
    assert resolved == ("true-model", "SPECIAL_KEY", "https://example.test")


def test_resolve_endpoint_selection_falls_back_to_cli_defaults() -> None:
    resolved = resolve_endpoint_selection(
        "unknown",
        {},
        default_key_var="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
    )
    assert resolved == ("unknown", "OPENAI_API_KEY", "https://api.openai.com/v1")


def test_merge_cli_override_args_cli_overrides_json() -> None:
    merged = merge_cli_override_args({"seed": 42}, {"seed": 1, "shuffle": True})
    assert merged == {"seed": 42, "shuffle": True}


def test_flatten_state_columns_merges_groups() -> None:
    flattened = flatten_state_columns([["a", "b"], ["c"]])
    assert flattened == ["a", "b", "c"]


def test_ensure_required_params_detects_missing() -> None:
    params = [_make_env_param("required_value", required=True)]
    with pytest.raises(MissingEnvParamError):
        ensure_required_params(params, explicit={}, json_args={})


def test_ensure_required_params_allows_json_provided() -> None:
    params = [_make_env_param("threshold", required=True)]
    ensure_required_params(params, explicit={}, json_args={"threshold": 0.5})


@dataclass(slots=True)
class _Inner:
    path: Path


@dataclass(slots=True)
class _Outer:
    name: str
    payload: _Inner
    tags: set[str]


def test_asdict_sanitized_handles_dataclasses_and_paths(tmp_path: Path) -> None:
    obj = _Outer(name="demo", payload=_Inner(path=tmp_path / "artifact.json"), tags={"x", "y"})
    serialized = asdict_sanitized(obj)

    assert serialized["name"] == "demo"
    assert serialized["payload"]["path"] == str(tmp_path / "artifact.json")
    assert sorted(serialized["tags"]) == ["x", "y"]
