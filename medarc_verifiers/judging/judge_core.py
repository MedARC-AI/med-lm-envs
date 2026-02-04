from __future__ import annotations

from typing import Any

from openai import APIError, APITimeoutError, RateLimitError
from verifiers.utils.async_utils import maybe_await


async def call_judge_model(
    judge_client: Any,
    judge_model: str,
    judge_prompt: str,
    judge_sampling_args: dict[str, Any] | None,
    logger,
) -> tuple[str, Any]:
    judge_args = dict(judge_sampling_args or {})
    if "max_tokens" in judge_args:
        if judge_args["max_tokens"] is None:
            judge_args.pop("max_tokens")
        else:
            judge_args["max_completion_tokens"] = judge_args.pop("max_tokens")
    if "max_completion_tokens" in judge_args and judge_args["max_completion_tokens"] is None:
        judge_args.pop("max_completion_tokens")
    judge_args = {k: v for k, v in judge_args.items() if v is not None}

    try:
        response_obj = await maybe_await(
            judge_client.chat.completions.create,
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            **judge_args,
        )
        response_text = str(response_obj.choices[0].message.content)
        return response_text, response_obj
    except RateLimitError as e:
        logger.warning(
            f"Rate limit exceeded when calling judge model '{judge_model}'. "
            f"Try reducing concurrency or waiting before retrying. Error: {str(e)}"
        )
        raise RuntimeError(
            f"Judge model rate limit exceeded. Try reducing concurrency or waiting before retrying. "
            f"Model: {judge_model}, Error: {str(e)}"
        ) from e
    except APITimeoutError as e:
        logger.warning(
            f"Timeout when calling judge model '{judge_model}'. "
            f"Increase timeout in judge_sampling_args or check model responsiveness. Error: {str(e)}"
        )
        raise RuntimeError(
            f"Judge model timeout. Increase timeout in judge_sampling_args or check model responsiveness. "
            f"Model: {judge_model}, Error: {str(e)}"
        ) from e
    except APIError as e:
        logger.warning(
            f"API error when calling judge model '{judge_model}'. Check model availability and API key. Error: {str(e)}"
        )
        raise RuntimeError(
            f"Judge model API error. Check model availability and API key. Model: {judge_model}, Error: {str(e)}"
        ) from e
    except Exception as e:
        logger.warning(f"Unexpected error when calling judge model '{judge_model}'. Error: {str(e)}")
        raise RuntimeError(f"Unexpected error when calling judge model '{judge_model}'. Error: {str(e)}") from e
