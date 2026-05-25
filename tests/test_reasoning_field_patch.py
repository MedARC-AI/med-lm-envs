from types import SimpleNamespace

import pytest

from medarc_verifiers.utils.reasoning_field_patch import install_reasoning_field_patch
from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.types import AssistantMessage


@pytest.mark.asyncio
async def test_openai_chat_reasoning_field_patch_renames_reasoning_content() -> None:
    install_reasoning_field_patch()
    client = OpenAIChatCompletionsClient(object())
    client._config = SimpleNamespace(reasoning_field="reasoning")  # type: ignore[assignment]

    prompt, _ = await client.to_native_prompt([AssistantMessage(content="final", reasoning_content="hidden")])

    assert prompt[0]["reasoning"] == "hidden"
    assert "reasoning_content" not in prompt[0]


@pytest.mark.asyncio
async def test_openai_chat_reasoning_field_patch_can_strip_reasoning() -> None:
    install_reasoning_field_patch()
    client = OpenAIChatCompletionsClient(object())
    client._config = SimpleNamespace(reasoning_field="none")  # type: ignore[assignment]

    prompt, _ = await client.to_native_prompt([AssistantMessage(content="final", reasoning_content="hidden")])

    assert "reasoning" not in prompt[0]
    assert "reasoning_content" not in prompt[0]
