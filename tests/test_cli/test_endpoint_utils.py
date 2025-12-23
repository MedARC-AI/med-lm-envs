from __future__ import annotations

from pathlib import Path

import pytest

from medarc_verifiers.cli.utils import endpoint_utils as utils


def test_load_endpoint_registry_uses_cache(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_loader(path: str):
        calls.append(path)
        return {"alias": {"model": "resolved"}}

    monkeypatch.setattr(utils, "load_endpoints", fake_loader)

    cache: dict[str, dict[str, dict[str, str]]] = {}
    endpoints_path = tmp_path / "endpoints.py"
    endpoints_path.write_text("# dummy")

    first = utils.load_endpoint_registry(endpoints_path, cache=cache)
    second = utils.load_endpoint_registry(endpoints_path, cache=cache)

    assert first == {"alias": {"model": "resolved"}}
    assert first is second  # Cached object returned
    assert len(calls) == 1


def test_load_env_metadata_uses_cache(monkeypatch) -> None:
    calls: list[str] = []

    def fake_gather(env_id: str):
        calls.append(env_id)
        return (env_id,)

    monkeypatch.setattr(utils, "gather_env_cli_metadata", fake_gather)

    cache: dict[str, tuple[str, ...]] = {}

    first = utils.load_env_metadata("medqa", cache=cache)
    second = utils.load_env_metadata("medqa", cache=cache)

    assert first == ("medqa",)
    assert first is second
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("alias", ("true-model", "SPECIAL_KEY", "https://example.test")),
        ("unknown", ("unknown", "OPENAI_API_KEY", "https://api.openai.com/v1")),
    ],
)
def test_resolve_model_endpoint_handles_aliases(model: str, expected: tuple[str, str, str]) -> None:
    registry = {"alias": {"model": "true-model", "key": "SPECIAL_KEY", "url": "https://example.test"}}

    resolved = utils.resolve_model_endpoint(
        model,
        registry,
        default_key_var="OPENAI_API_KEY",
        default_base_url="https://api.openai.com/v1",
    )

    assert resolved == expected
