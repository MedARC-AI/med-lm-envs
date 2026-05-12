from __future__ import annotations

import inspect
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Literal, get_args, get_origin

_OPENAI_REASONING_KEYS = {"reasoning", "reasoning_effort", "thinking", "output_config"}
_FRAMEWORK_REQUEST_KEYS = {"model", "messages", "input", "prompt", "tools", "system", "extra_headers"}


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
    _drop_framework_request_keys(cleaned)
    return _move_compatible_extras_to_extra_body(cleaned, allowed_top_level_keys=_get_openai_chat_allowed_param_names())


def _sanitize_openai_responses(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args, preserve_none_keys={"max_tokens"})
    _normalize_openai_responses_sampling_args(cleaned)
    _drop_framework_request_keys(cleaned)
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


def _normalize_openai_responses_sampling_args(sampling_args: dict[str, Any]) -> None:
    n = sampling_args.pop("n", None)
    if n not in (None, 1):
        raise ValueError("Responses API client only supports n=1")

    max_tokens = sampling_args.pop("max_tokens", None)
    max_completion_tokens = sampling_args.pop("max_completion_tokens", None)
    if "max_output_tokens" not in sampling_args:
        output_tokens = max_tokens if max_tokens is not None else max_completion_tokens
        if output_tokens is not None:
            sampling_args["max_output_tokens"] = output_tokens

    if sampling_args.get("stop") is not None:
        raise ValueError("Responses API client does not support stop sequences")
    sampling_args.pop("stop", None)
    sampling_args.pop("modalities", None)


def _sanitize_openai_completions(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args, preserve_none_keys={"max_tokens"})
    _drop_framework_request_keys(cleaned)
    for key in _OPENAI_REASONING_KEYS:
        cleaned.pop(key, None)
    return _move_compatible_extras_to_extra_body(
        cleaned, allowed_top_level_keys=_get_openai_completions_allowed_param_names()
    )


def _sanitize_anthropic_messages(sampling_args: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = _drop_none(sampling_args)
    _drop_framework_request_keys(cleaned)
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

    allowed_keys = _get_anthropic_allowed_param_names()
    return {key: value for key, value in cleaned.items() if key in allowed_keys}


def _validate_anthropic_effort(value: Any) -> str:
    effort_values = _get_anthropic_effort_values()
    if not isinstance(value, str) or value not in effort_values:
        raise ValueError(f"anthropic_messages reasoning effort must be one of: {', '.join(sorted(effort_values))}")
    return value


def _drop_framework_request_keys(sampling_args: dict[str, Any]) -> None:
    for key in _FRAMEWORK_REQUEST_KEYS:
        sampling_args.pop(key, None)


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
def _get_anthropic_effort_values() -> set[str]:
    from anthropic.types import OutputConfigParam
    from typing import get_type_hints

    effort_type = get_type_hints(OutputConfigParam)["effort"]
    return _literal_string_values(effort_type)


def _literal_string_values(type_hint: Any) -> set[str]:
    values: set[str] = set()
    origin = get_origin(type_hint)
    if origin is None:
        return values
    if origin is Literal:
        return {value for value in get_args(type_hint) if isinstance(value, str)}
    for arg in get_args(type_hint):
        values.update(_literal_string_values(arg))
    return values


@lru_cache(maxsize=1)
def _get_openai_chat_allowed_param_names() -> set[str]:
    from openai.resources.chat.completions import AsyncCompletions as ChatAsyncCompletions  # type: ignore

    return _param_names(ChatAsyncCompletions.create)


@lru_cache(maxsize=1)
def _get_openai_responses_allowed_param_names() -> set[str]:
    from openai.resources.responses import AsyncResponses  # type: ignore

    return _param_names(AsyncResponses.create)


@lru_cache(maxsize=1)
def _get_openai_completions_allowed_param_names() -> set[str]:
    from openai.resources.completions import AsyncCompletions as TextAsyncCompletions  # type: ignore

    return _param_names(TextAsyncCompletions.create)


@lru_cache(maxsize=1)
def _get_anthropic_allowed_param_names() -> set[str]:
    from anthropic.resources.messages import AsyncMessages

    return _param_names(AsyncMessages.create)


def _param_names(callable_obj: Any) -> set[str]:
    sig = inspect.signature(callable_obj)
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
