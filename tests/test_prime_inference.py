import pytest

from medarc_verifiers.utils.prime_inference import (
    PRIME_INFERENCE_URL,
    prime_inference_overrides,
)


def test_prime_inference_overrides_with_prime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When using Prime Inference URL with PRIME_TEAM_ID and PRIME_API_KEY, all overrides should be returned."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides(PRIME_INFERENCE_URL)

    assert headers == {"X-Prime-Team-ID": "team-123"}
    assert sampling == {"extra_body": {"usage": {"include": True}}}
    assert api_key_var == "PRIME_API_KEY"


def test_prime_inference_overrides_without_team_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PRIME_TEAM_ID is not set, headers should be empty but usage and api_key should still work."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides(PRIME_INFERENCE_URL)

    assert headers == {}
    assert sampling == {"extra_body": {"usage": {"include": True}}}
    assert api_key_var == "PRIME_API_KEY"


def test_prime_inference_overrides_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """When PRIME_API_KEY is not set, api_key_var should be None."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides(PRIME_INFERENCE_URL)

    assert headers == {"X-Prime-Team-ID": "team-123"}
    assert sampling == {"extra_body": {"usage": {"include": True}}}
    assert api_key_var is None


def test_prime_inference_overrides_non_prime_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When not using Prime Inference URL, no overrides should be returned."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides("https://api.openai.com/v1")

    assert headers == {}
    assert sampling == {}
    assert api_key_var is None


def test_prime_inference_overrides_explicit_include_usage_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit include_usage=True should include usage for non-Prime URL."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides("https://api.openai.com/v1", include_usage=True)

    assert headers == {}
    assert sampling == {"extra_body": {"usage": {"include": True}}}
    assert api_key_var is None


def test_prime_inference_overrides_explicit_include_usage_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit include_usage=False should exclude usage even for Prime URL."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides(PRIME_INFERENCE_URL, include_usage=False)

    assert headers == {"X-Prime-Team-ID": "team-123"}
    assert sampling == {}
    assert api_key_var == "PRIME_API_KEY"


def test_prime_inference_overrides_env_var_include_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEDARC_INCLUDE_USAGE env var should control usage inclusion."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("PRIME_API_KEY", raising=False)

    # Test with env var set to true on non-Prime URL
    monkeypatch.setenv("MEDARC_INCLUDE_USAGE", "true")
    headers, sampling, api_key_var = prime_inference_overrides("https://api.openai.com/v1")
    assert headers == {}
    assert sampling == {"extra_body": {"usage": {"include": True}}}
    assert api_key_var is None

    # Test with env var set to false on Prime URL
    monkeypatch.setenv("MEDARC_INCLUDE_USAGE", "false")
    headers, sampling, api_key_var = prime_inference_overrides(PRIME_INFERENCE_URL)
    assert headers == {}
    assert sampling == {}
    assert api_key_var is None


def test_prime_inference_overrides_none_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """When base_url is None, no overrides should be returned."""
    monkeypatch.setenv("PRIME_TEAM_ID", "team-123")
    monkeypatch.setenv("PRIME_API_KEY", "secret-key")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    headers, sampling, api_key_var = prime_inference_overrides(None)

    assert headers == {}
    assert sampling == {}
    assert api_key_var is None
