import pytest

from medarc_verifiers.utils.prime_inference import (
    PRIME_INFERENCE_URL,
    prime_inference_overrides,
)


def test_prime_inference_overrides_with_prime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prime URL should inject team header and usage override when enabled."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling = prime_inference_overrides(PRIME_INFERENCE_URL)

    assert headers == {"X-Prime-Team-ID": "team-123"}
    assert sampling == {"extra_body": {"usage": {"include": True}}}


def test_prime_inference_overrides_without_team_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing PRIME_TEAM_ID should keep headers empty while preserving usage override behavior."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling = prime_inference_overrides(PRIME_INFERENCE_URL)

    assert headers == {}
    assert sampling == {"extra_body": {"usage": {"include": True}}}


def test_prime_inference_overrides_returns_two_tuple_without_api_key_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PRIME_API_KEY presence should not affect override outputs or return shape."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")

    with_key = prime_inference_overrides(PRIME_INFERENCE_URL)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    without_key = prime_inference_overrides(PRIME_INFERENCE_URL)

    assert len(with_key) == 2
    assert with_key == without_key
    assert with_key == ({}, {"extra_body": {"usage": {"include": True}}})


def test_prime_inference_overrides_non_prime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When not using Prime Inference URL, no overrides should be returned."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling = prime_inference_overrides("https://api.openai.com/v1")

    assert headers == {}
    assert sampling == {}


def test_prime_inference_overrides_explicit_include_usage_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit include_usage=True should include usage for non-Prime URL."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling = prime_inference_overrides("https://api.openai.com/v1", include_usage=True)

    assert headers == {}
    assert sampling == {"extra_body": {"usage": {"include": True}}}


def test_prime_inference_overrides_explicit_include_usage_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit include_usage=False should exclude usage even for Prime URL."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling = prime_inference_overrides(PRIME_INFERENCE_URL, include_usage=False)

    assert headers == {"X-Prime-Team-ID": "team-123"}
    assert sampling == {}


def test_prime_inference_overrides_env_var_include_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEDARC_INCLUDE_USAGE env var should control usage inclusion."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)

    # Test with env var set to true on non-Prime URL
    monkeypatch.setenv("MEDARC_INCLUDE_USAGE", "true")
    headers, sampling = prime_inference_overrides("https://api.openai.com/v1")
    assert headers == {}
    assert sampling == {"extra_body": {"usage": {"include": True}}}

    # Test with env var set to false on Prime URL
    monkeypatch.setenv("MEDARC_INCLUDE_USAGE", "false")
    headers, sampling = prime_inference_overrides(PRIME_INFERENCE_URL)
    assert headers == {}
    assert sampling == {}


def test_prime_inference_overrides_none_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When base_url is None, no overrides should be returned."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling = prime_inference_overrides(None)

    assert headers == {}
    assert sampling == {}
