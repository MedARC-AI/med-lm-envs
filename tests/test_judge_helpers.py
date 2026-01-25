import pytest

from medarc_verifiers.utils.judge_helpers import (
    PRIME_INFERENCE_URL,
    default_judge_api_key,
    judge_sampling_args_and_headers,
)


def test_judge_sampling_defaults_supports_fuzzy_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults should resolve when the judge slug only matches as a subsequence.

    When not using Prime Inference URL, extra_body.usage should NOT be included by default.
    """
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    result, _ = judge_sampling_args_and_headers("Claude-Sonnet-4.5")
    assert result == {"temperature": 0.5, "timeout": 300}
    assert "extra_body" not in result

    result, _ = judge_sampling_args_and_headers("openai/gpt-oss-120b")
    assert result.get("reasoning_effort") in {"low", "medium", "high"}
    result_without_effort = dict(result)
    result_without_effort.pop("reasoning_effort", None)
    assert result_without_effort == {
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_effort": "low",
        "extra_body": {"top_k": 0},  # top_k goes to extra_body, but not usage
        "timeout": 300,
    }


def test_judge_sampling_includes_usage_for_prime_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """When using Prime Inference URL, extra_body.usage should be included."""
    monkeypatch.setenv("PRIME_TEAM_ID", "prime-team-123")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    result, headers = judge_sampling_args_and_headers("gpt-4.1", base_url=PRIME_INFERENCE_URL)

    assert result["temperature"] == 0.5
    assert result["top_p"] == 1.0
    assert result["extra_body"] == {"usage": {"include": True}}
    assert headers == {"X-Prime-Team-ID": "prime-team-123"}


def test_judge_sampling_explicit_include_usage_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit include_usage=True should include usage regardless of URL."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    result, _ = judge_sampling_args_and_headers("Claude-Sonnet-4.5", include_usage=True)
    assert result == {"temperature": 0.5, "extra_body": {"usage": {"include": True}}, "timeout": 300}


def test_judge_sampling_explicit_include_usage_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit include_usage=False should exclude usage even for Prime Inference."""
    monkeypatch.setenv("PRIME_TEAM_ID", "prime-team-123")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    result, headers = judge_sampling_args_and_headers(
        "gpt-4.1", base_url=PRIME_INFERENCE_URL, include_usage=False
    )

    assert result["temperature"] == 0.5
    assert result["top_p"] == 1.0
    assert "extra_body" not in result
    assert headers == {"X-Prime-Team-ID": "prime-team-123"}


def test_judge_sampling_env_var_include_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    """MEDARC_INCLUDE_USAGE env var should control usage inclusion."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    # Test with env var set to true
    monkeypatch.setenv("MEDARC_INCLUDE_USAGE", "true")
    result, _ = judge_sampling_args_and_headers("Claude-Sonnet-4.5")
    assert result == {"temperature": 0.5, "extra_body": {"usage": {"include": True}}, "timeout": 300}

    # Test with env var set to false (should override Prime Inference auto-detect)
    monkeypatch.setenv("MEDARC_INCLUDE_USAGE", "false")
    monkeypatch.setenv("PRIME_TEAM_ID", "prime-team-123")
    result, _ = judge_sampling_args_and_headers("gpt-4.1", base_url=PRIME_INFERENCE_URL)
    assert "extra_body" not in result or result.get("extra_body", {}).get("usage") is None


def test_judge_sampling_defaults_injects_prime_team_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRIME_TEAM_ID", "prime-team-123")
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    result, headers = judge_sampling_args_and_headers("gpt-4.1", base_url=PRIME_INFERENCE_URL)

    assert result["temperature"] == 0.5
    assert result["top_p"] == 1.0
    assert result["extra_body"] == {"usage": {"include": True}}
    assert headers == {"X-Prime-Team-ID": "prime-team-123"}


def test_judge_sampling_defaults_unknown_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    with pytest.raises(KeyError):
        judge_sampling_args_and_headers("unknown-judge")


def test_judge_sampling_defaults_supports_multiple_names_per_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When not using Prime Inference URL, extra_body.usage should NOT be included."""
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)
    monkeypatch.delenv("MEDARC_INCLUDE_USAGE", raising=False)

    result_45, _ = judge_sampling_args_and_headers("glm-4.5")
    assert result_45 == {"temperature": 0.6, "top_p": 0.95, "timeout": 300}

    result_46, _ = judge_sampling_args_and_headers("glm-4.6")
    assert result_46 == {
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"top_k": 40},  # top_k goes to extra_body, but not usage
        "timeout": 300,
    }

    result_47, _ = judge_sampling_args_and_headers("glm-4.7")
    assert result_47 == result_46


def test_default_judge_api_key_prefers_pinference_judge_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JUDGE_API_KEY", "secret-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert default_judge_api_key(base_url=PRIME_INFERENCE_URL) == "secret-key"
