import pytest

from medarc_verifiers.utils.judge_helpers import default_judge_api_key, judge_sampling_args_and_headers


def test_judge_sampling_defaults_supports_fuzzy_match(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults should resolve when the judge slug only matches as a subsequence."""

    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    result, _ = judge_sampling_args_and_headers("Claude-Sonnet-4.5")
    assert result == {"temperature": 0.5, "extra_body": {"usage": {"include": True}}, "timeout": 300}

    result, _ = judge_sampling_args_and_headers("openai/gpt-oss-120b")
    assert result == {
        "temperature": 1.0,
        "top_p": 1.0,
        "reasoning_effort": "medium",
        "extra_body": {"usage": {"include": True}, "top_k": 0},
        "timeout": 300,
    }


def test_judge_sampling_defaults_injects_prime_team_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    base_url = "https://api.pinference.ai/api/v1"
    monkeypatch.setenv("PRIME_TEAM_ID", "prime-team-123")

    result, headers = judge_sampling_args_and_headers("gpt-4.1", base_url=base_url)

    assert result["temperature"] == 0.5
    assert result["top_p"] == 1.0
    assert headers == {"X-Prime-Team-ID": "prime-team-123"}


def test_judge_sampling_defaults_unknown_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    with pytest.raises(KeyError):
        judge_sampling_args_and_headers("unknown-judge")


def test_judge_sampling_defaults_supports_multiple_names_per_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PRIME_TEAM_ID", raising=False)

    result_45, _ = judge_sampling_args_and_headers("glm-4.5")
    assert result_45 == {"temperature": 0.6, "top_p": 0.95, "extra_body": {"usage": {"include": True}}, "timeout": 300}

    result_46, _ = judge_sampling_args_and_headers("glm-4.6")
    assert result_46 == {
        "temperature": 1.0,
        "top_p": 0.95,
        "extra_body": {"usage": {"include": True}, "top_k": 40},
        "timeout": 300,
    }

    result_47, _ = judge_sampling_args_and_headers("glm-4.7")
    assert result_47 == result_46


def test_default_judge_api_key_prefers_pinference_judge_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_url = "https://api.pinference.ai/api/v1"
    monkeypatch.setenv("JUDGE_API_KEY", "secret-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert default_judge_api_key(base_url=base_url) == "secret-key"
