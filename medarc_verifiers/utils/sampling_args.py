import inspect
from collections.abc import Mapping
from functools import lru_cache
from typing import Any


def sanitize_sampling_args_for_openai(sampling_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return sampling args split into OpenAI-recognized kwargs and extra_body.

    Any parameters not recognized by the OpenAI Chat Completions API are moved under
    the `extra_body` key so they can be forwarded to compatible servers (e.g., vLLM/Qwen).
    """
    if not sampling_args:
        return {}

    allowed_keys = _get_openai_allowed_param_names()

    filtered: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in sampling_args.items():
        if key in allowed_keys:
            filtered[key] = value
        else:
            extras[key] = value

    if extras:
        # OpenAI python client forwards unknown params via `extra_body`.
        # If the caller already supplied an `extra_body` (e.g., to request `usage.include`),
        # merge rather than overwrite it.
        existing = filtered.get("extra_body")
        if existing is None:
            filtered["extra_body"] = extras
        elif isinstance(existing, Mapping):
            merged = dict(existing)
            for key, value in extras.items():
                merged.setdefault(key, value)
            filtered["extra_body"] = merged
        else:
            filtered["extra_body"] = {"_passthrough_extra_body": existing, **extras}
    return filtered


@lru_cache(maxsize=1)
def _get_openai_allowed_param_names() -> set[str]:
    """Infer allowed kwargs for OpenAI create() by inspecting client signatures.

    We union parameter names from:
      - openai.resources.chat.completions.AsyncCompletions.create
      - openai.resources.completions.AsyncCompletions.create

    On failure, return a conservative fallback. Always include 'extra_body'.
    """
    try:
        from openai.resources.chat.completions import AsyncCompletions as ChatAsyncCompletions  # type: ignore
        from openai.resources.completions import AsyncCompletions as TextAsyncCompletions  # type: ignore
    except Exception:
        return {
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "seed",
            "response_format",
            "tool_choice",
            "tools",
            "stream",
            "extra_body",
        }

    def _param_names(callable_obj: Any) -> set[str]:
        try:
            sig = inspect.signature(callable_obj)
        except Exception:
            return set()
        names: set[str] = set()
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                continue
            names.add(name)
        return names

    allowed = _param_names(ChatAsyncCompletions.create) | _param_names(TextAsyncCompletions.create)
    allowed.add("extra_body")
    return allowed
