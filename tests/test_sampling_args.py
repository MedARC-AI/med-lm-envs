from __future__ import annotations

import pytest

from medarc_verifiers.utils.sampling_args import sanitize_sampling_args, sanitize_sampling_args_for_openai


def test_openai_chat_keeps_reasoning_effort_and_moves_extras() -> None:
    result = sanitize_sampling_args(
        {
            "reasoning_effort": "low",
            "top_k": 20,
            "min_p": 0.1,
            "repetition_penalty": 1.1,
            "extra_body": {"usage": {"include": True}},
        },
        client_type="openai_chat_completions",
    )

    assert result["reasoning_effort"] == "low"
    assert result["extra_body"] == {
        "usage": {"include": True},
        "top_k": 20,
        "min_p": 0.1,
        "repetition_penalty": 1.1,
    }


def test_openai_chat_token_uses_chat_shape() -> None:
    result = sanitize_sampling_args(
        {"reasoning_effort": "medium", "top_k": 8},
        client_type="openai_chat_completions_token",
    )

    assert result["reasoning_effort"] == "medium"
    assert result["extra_body"] == {"top_k": 8}


def test_compatibility_wrapper_uses_chat_shape() -> None:
    result = sanitize_sampling_args_for_openai({"reasoning_effort": "low", "top_k": 1})

    assert result["reasoning_effort"] == "low"
    assert result["extra_body"] == {"top_k": 1}


def test_openai_chat_drops_framework_owned_request_keys() -> None:
    result = sanitize_sampling_args(
        {"model": "override", "messages": [], "tools": [], "extra_headers": {"x": "y"}, "top_k": 1},
        client_type="openai_chat_completions",
    )

    assert "model" not in result
    assert "messages" not in result
    assert "tools" not in result
    assert "extra_headers" not in result
    assert result["extra_body"] == {"top_k": 1}


def test_openai_responses_converts_reasoning_effort() -> None:
    result = sanitize_sampling_args(
        {"reasoning_effort": "low", "top_k": 20, "max_tokens": 128},
        client_type="openai_responses",
    )

    assert "reasoning_effort" not in result
    assert result["reasoning"] == {"effort": "low"}
    assert result["max_output_tokens"] == 128
    assert result["extra_body"] == {"top_k": 20}


def test_openai_responses_rejects_stop_sequences() -> None:
    with pytest.raises(ValueError, match="does not support stop sequences"):
        sanitize_sampling_args({"stop": ["END"]}, client_type="openai_responses")


def test_openai_responses_preserves_explicit_reasoning_effort() -> None:
    result = sanitize_sampling_args(
        {"reasoning": {"effort": "high", "summary": "auto"}, "reasoning_effort": "low"},
        client_type="openai_responses",
    )

    assert result["reasoning"] == {"effort": "high", "summary": "auto"}


def test_openai_responses_drops_framework_owned_request_keys() -> None:
    result = sanitize_sampling_args(
        {"model": "override", "input": "x", "prompt": "y", "tools": [], "extra_headers": {"x": "y"}, "top_k": 1},
        client_type="openai_responses",
    )

    assert "model" not in result
    assert "input" not in result
    assert "prompt" not in result
    assert "tools" not in result
    assert "extra_headers" not in result
    assert result["extra_body"] == {"top_k": 1}


def test_openai_completions_removes_reasoning_and_moves_extras() -> None:
    result = sanitize_sampling_args(
        {"prompt": "x", "reasoning_effort": "low", "reasoning": {"effort": "low"}, "top_k": 20},
        client_type="openai_completions",
    )

    assert "reasoning_effort" not in result
    assert "reasoning" not in result
    assert "prompt" not in result
    assert result["extra_body"] == {"top_k": 20}


@pytest.mark.parametrize("client_type", ["renderer", "nemorl_chat_completions"])
def test_passthrough_clients_only_drop_none(client_type: str) -> None:
    result = sanitize_sampling_args(
        {"reasoning_effort": "low", "top_k": 20, "temperature": None},
        client_type=client_type,
    )

    assert result == {"reasoning_effort": "low", "top_k": 20}


def test_anthropic_preserves_adaptive_thinking() -> None:
    result = sanitize_sampling_args(
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}, "top_k": 10},
        client_type="anthropic_messages",
    )

    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"] == {"effort": "medium"}
    assert result["top_k"] == 10


def test_anthropic_maps_reasoning_effort_to_adaptive_output_config() -> None:
    result = sanitize_sampling_args({"reasoning_effort": "high"}, client_type="anthropic_messages")

    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"] == {"effort": "high"}
    assert "reasoning_effort" not in result
    assert "effort" not in result


def test_anthropic_drops_framework_owned_request_keys() -> None:
    result = sanitize_sampling_args(
        {
            "model": "override",
            "messages": [],
            "system": "override",
            "tools": [],
            "extra_headers": {"x": "y"},
            "reasoning_effort": "low",
        },
        client_type="anthropic_messages",
    )

    assert "model" not in result
    assert "messages" not in result
    assert "system" not in result
    assert "tools" not in result
    assert "extra_headers" not in result
    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"] == {"effort": "low"}


def test_anthropic_does_not_put_effort_inside_thinking() -> None:
    result = sanitize_sampling_args(
        {"thinking": {"type": "adaptive", "effort": "low"}, "reasoning_effort": "medium"},
        client_type="anthropic_messages",
    )

    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"] == {"effort": "medium"}


@pytest.mark.parametrize(
    "sampling_args",
    [
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
        {"thinking": {"type": "adaptive", "budget_tokens": 4096}},
    ],
)
def test_anthropic_rejects_manual_budget_thinking(sampling_args: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="thinking"):
        sanitize_sampling_args(sampling_args, client_type="anthropic_messages")


def test_anthropic_validates_effort_values() -> None:
    with pytest.raises(ValueError, match="reasoning effort"):
        sanitize_sampling_args({"reasoning_effort": "extreme"}, client_type="anthropic_messages")


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_anthropic_accepts_sdk_documented_effort_values(effort: str) -> None:
    result = sanitize_sampling_args({"reasoning_effort": effort}, client_type="anthropic_messages")

    assert result["thinking"] == {"type": "adaptive"}
    assert result["output_config"] == {"effort": effort}


@pytest.mark.asyncio
async def test_openai_responses_client_receives_nested_reasoning() -> None:
    from verifiers.clients.openai_responses_client import OpenAIResponsesClient

    class Responses:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] | None = None

        async def create(self, **kwargs):
            self.kwargs = kwargs
            return object()

    class Client:
        def __init__(self) -> None:
            self.responses = Responses()

        async def close(self) -> None:
            pass

    raw_client = Client()
    client = OpenAIResponsesClient(raw_client)
    sampling_args = sanitize_sampling_args({"reasoning_effort": "low"}, client_type="openai_responses")

    await client.get_native_response([], "model", sampling_args)

    assert raw_client.responses.kwargs is not None
    assert "reasoning_effort" not in raw_client.responses.kwargs
    assert raw_client.responses.kwargs["reasoning"] == {"effort": "low"}
