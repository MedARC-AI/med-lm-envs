from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

_OPENAI_REASONING_KEYS = {"reasoning", "reasoning_effort", "thinking", "output_config"}
_ANTHROPIC_EFFORT_VALUES = {"low", "medium", "high"}
_OPENAI_CHAT_FALLBACK_TOP_LEVEL_KEYS = {
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
    "reasoning_effort",
}
_OPENAI_RESPONSES_FALLBACK_TOP_LEVEL_KEYS = {
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "n",
    "stop",
    "stream",
    "extra_body",
    "reasoning",
    "tools",
    "tool_choice",
}
_OPENAI_CHAT_VERIFIERS_WRAPPER_KEYS = {"max_tokens"}
_OPENAI_RESPONSES_VERIFIERS_WRAPPER_KEYS = {"max_tokens", "max_completion_tokens", "n", "stop", "modalities"}
_OPENAI_COMPLETIONS_FALLBACK_TOP_LEVEL_KEYS = {
    "temperature",
    "top_p",
    "max_tokens",
    "n",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "seed",
    "stream",
    "extra_body",
}


def sanitize_sampling_args(
    sampling_args: Mapping[str, Any] | None,
    *,
    client_type: str,
) -> dict[str, Any]:
    """Return sampling args in the request shape expected by the resolved client."""
    if not sampling_args:
        return {}

    if client_type in {"openai_chat_completions", "openai_chat_completions_token"}:
        return _sanitize_openai_chat(sampling_args)
    if client_type == "openai_responses":
        return _sanitize_openai_responses(sampling_args)
    if client_type == "openai_completions":
        return _sanitize_openai_completions(sampling_args)
    if client_type == "anthropic_messages":
        return _sanitize_anthropic_messages(sampling_args)
    if client_type in {"renderer", "nemorl_chat_completions"}:
        return _drop_none(sampling_args)
    return _drop_none(sampling_args)


def sanitize_sampling_args_for_openai(sampling_args: Mapping[str, Any] | None) -> dict[str, Any]:
    """Compatibility wrapper for existing OpenAI Chat Completions call sites."""
    return sanitize_sampling_args(sampling_args, client_type="openai_chat_completions")


def _sanitize_openai_chat(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args, preserve_none_keys={"max_tokens"})
    return _move_compatible_extras_to_extra_body(cleaned, allowed_top_level_keys=_get_openai_chat_allowed_param_names())


def _sanitize_openai_responses(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args, preserve_none_keys={"max_tokens"})
    reasoning_effort = cleaned.pop("reasoning_effort", None)
    if reasoning_effort is not None:
        existing_reasoning = cleaned.get("reasoning")
        if existing_reasoning is None:
            cleaned["reasoning"] = {"effort": reasoning_effort}
        elif isinstance(existing_reasoning, Mapping):
            reasoning = dict(existing_reasoning)
            reasoning.setdefault("effort", reasoning_effort)
            cleaned["reasoning"] = reasoning
        else:
            raise ValueError("sampling_args.reasoning must be a dict when used with openai_responses")
    return _move_compatible_extras_to_extra_body(
        cleaned, allowed_top_level_keys=_get_openai_responses_allowed_param_names()
    )


def _sanitize_openai_completions(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args, preserve_none_keys={"max_tokens"})
    for key in _OPENAI_REASONING_KEYS:
        cleaned.pop(key, None)
    return _move_compatible_extras_to_extra_body(
        cleaned, allowed_top_level_keys=_get_openai_completions_allowed_param_names()
    )


def _sanitize_anthropic_messages(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args)
    reasoning_effort = cleaned.pop("reasoning_effort", None)
    cleaned.pop("reasoning", None)
    cleaned.pop("effort", None)
    cleaned.pop("extra_body", None)

    thinking = cleaned.get("thinking")
    if thinking is not None:
        if not isinstance(thinking, Mapping):
            raise ValueError("sampling_args.thinking must be a dict when used with anthropic_messages")
        thinking_dict = dict(thinking)
        if thinking_dict.get("type") != "adaptive":
            raise ValueError("anthropic_messages only supports adaptive thinking configs")
        if "budget_tokens" in thinking_dict:
            raise ValueError("anthropic_messages does not support manual budget_tokens thinking configs")
        thinking_dict.pop("effort", None)
        cleaned["thinking"] = thinking_dict

    if reasoning_effort is not None:
        effort = _validate_anthropic_effort(reasoning_effort)
        cleaned["thinking"] = {"type": "adaptive"}
        output_config = cleaned.get("output_config")
        if output_config is None:
            cleaned["output_config"] = {"effort": effort}
        elif isinstance(output_config, Mapping):
            cleaned["output_config"] = {**dict(output_config), "effort": effort}
        else:
            raise ValueError("sampling_args.output_config must be a dict when used with anthropic_messages")
    elif "output_config" in cleaned:
        output_config = cleaned["output_config"]
        if not isinstance(output_config, Mapping):
            raise ValueError("sampling_args.output_config must be a dict when used with anthropic_messages")
        output_config_dict = dict(output_config)
        if "effort" in output_config_dict:
            output_config_dict["effort"] = _validate_anthropic_effort(output_config_dict["effort"])
        cleaned["output_config"] = output_config_dict

    return cleaned


def _validate_anthropic_effort(value: Any) -> str:
    if not isinstance(value, str) or value not in _ANTHROPIC_EFFORT_VALUES:
        raise ValueError(
            "anthropic_messages reasoning effort must be one of: "
            f"{', '.join(sorted(_ANTHROPIC_EFFORT_VALUES))}"
        )
    return value


def _move_compatible_extras_to_extra_body(
    sampling_args: Mapping[str, Any],
    *,
    allowed_top_level_keys: set[str],
) -> dict[str, Any]:
    filtered: dict[str, Any] = {}
    extras: dict[str, Any] = {}
    for key, value in sampling_args.items():
        if key in allowed_top_level_keys:
            filtered[key] = value
        else:
            extras[key] = value

    if not extras:
        return filtered

    existing = filtered.get("extra_body")
    if existing is None:
        filtered["extra_body"] = extras
    elif isinstance(existing, Mapping):
        filtered["extra_body"] = _deep_merge(extras, existing)
    else:
        filtered["extra_body"] = {"_passthrough_extra_body": existing, **extras}
    return filtered


@lru_cache(maxsize=1)
def _get_openai_chat_allowed_param_names() -> set[str]:
    try:
        from openai.resources.chat.completions import AsyncCompletions as ChatAsyncCompletions  # type: ignore
    except Exception:
        return set(_OPENAI_CHAT_FALLBACK_TOP_LEVEL_KEYS)

    allowed = _param_names(ChatAsyncCompletions.create) or set(_OPENAI_CHAT_FALLBACK_TOP_LEVEL_KEYS)
    allowed.add("extra_body")
    allowed.add("reasoning_effort")
    allowed.update(_OPENAI_CHAT_VERIFIERS_WRAPPER_KEYS)
    return allowed


@lru_cache(maxsize=1)
def _get_openai_responses_allowed_param_names() -> set[str]:
    try:
        from openai.resources.responses import AsyncResponses  # type: ignore
    except Exception:
        return set(_OPENAI_RESPONSES_FALLBACK_TOP_LEVEL_KEYS)

    allowed = _param_names(AsyncResponses.create) or set(_OPENAI_RESPONSES_FALLBACK_TOP_LEVEL_KEYS)
    allowed.add("extra_body")
    allowed.add("reasoning")
    allowed.update(_OPENAI_RESPONSES_VERIFIERS_WRAPPER_KEYS)
    return allowed


@lru_cache(maxsize=1)
def _get_openai_completions_allowed_param_names() -> set[str]:
    try:
        from openai.resources.completions import AsyncCompletions as TextAsyncCompletions  # type: ignore
    except Exception:
        return set(_OPENAI_COMPLETIONS_FALLBACK_TOP_LEVEL_KEYS)

    allowed = _param_names(TextAsyncCompletions.create) or set(_OPENAI_COMPLETIONS_FALLBACK_TOP_LEVEL_KEYS)
    allowed.add("extra_body")
    return allowed


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


def _drop_none(
    sampling_args: Mapping[str, Any],
    *,
    preserve_none_keys: set[str] | None = None,
) -> dict[str, Any]:
    preserve_none_keys = preserve_none_keys or set()
    return {key: value for key, value in sampling_args.items() if value is not None or key in preserve_none_keys}


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
