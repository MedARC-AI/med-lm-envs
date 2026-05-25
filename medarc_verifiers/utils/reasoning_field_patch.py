from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

VALID_REASONING_FIELDS = {"reasoning", "reasoning_content", "none"}


def install_reasoning_field_patch() -> bool:
    """Patch Verifiers' OpenAI chat client to honor MedARC's endpoint reasoning_field."""
    try:
        from verifiers.clients.openai_chat_completions_client import OpenAIChatCompletionsClient

        if getattr(OpenAIChatCompletionsClient, "_medarc_reasoning_field_patched", False):
            return True

        original_to_native_prompt = OpenAIChatCompletionsClient.to_native_prompt

        async def patched_to_native_prompt(self: Any, messages: Any) -> tuple[Any, dict]:
            native_messages, kwargs = await original_to_native_prompt(self, messages)
            config = getattr(self, "_config", None)
            reasoning_field = getattr(config, "reasoning_field", None)
            if reasoning_field is None:
                return native_messages, kwargs
            if reasoning_field not in VALID_REASONING_FIELDS:
                raise ValueError(
                    "reasoning_field must be one of "
                    f"{sorted(VALID_REASONING_FIELDS)}, got {reasoning_field!r}"
                )

            for message in native_messages:
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue

                if reasoning_field == "none":
                    message.pop("reasoning", None)
                    message.pop("reasoning_content", None)
                    continue

                if reasoning_field == "reasoning":
                    reasoning = message.pop("reasoning_content", None)
                    if reasoning is not None:
                        message["reasoning"] = reasoning
                    continue

                message.pop("reasoning", None)

            return native_messages, kwargs

        OpenAIChatCompletionsClient._medarc_original_to_native_prompt = original_to_native_prompt
        OpenAIChatCompletionsClient.to_native_prompt = cast(Any, patched_to_native_prompt)
        OpenAIChatCompletionsClient._medarc_reasoning_field_patched = True
        logger.debug("OpenAI chat reasoning_field patch installed")
        return True
    except Exception as exc:
        logger.warning("Failed to install OpenAI chat reasoning_field patch: %s", exc)
        return False
