from __future__ import annotations

import json
import re
from typing import Any

_CHAT_TEMPLATE_TOKEN_PATTERN = re.compile(r"<\|[^|>]+\|>")


def extract_tool_call_fields(tool_call: Any) -> tuple[str, str, Any]:
    """Extract (id, name, arguments) from nested or flat tool-call shapes."""
    if isinstance(tool_call, str):
        try:
            tool_call = json.loads(tool_call)
        except Exception:
            return "", "", ""

    call_id: Any = ""
    name: Any = ""
    raw_args: Any = ""

    if isinstance(tool_call, dict):
        call_id = tool_call.get("id", "") or ""
        function_payload = tool_call.get("function")
        if isinstance(function_payload, dict):
            name = function_payload.get("name", "")
            raw_args = function_payload.get("arguments", "")
        if not isinstance(name, str):
            name = tool_call.get("name", "")
        if raw_args is None:
            raw_args = tool_call.get("arguments", "")
    else:
        call_id = getattr(tool_call, "id", "") or ""
        function_payload = getattr(tool_call, "function", None)
        nested_name = getattr(function_payload, "name", None) if function_payload is not None else None
        nested_args = getattr(function_payload, "arguments", None) if function_payload is not None else None
        flat_name = getattr(tool_call, "name", None)
        flat_args = getattr(tool_call, "arguments", None)
        name = nested_name if isinstance(nested_name, str) else flat_name
        raw_args = nested_args if nested_args is not None else flat_args

    if not isinstance(name, str):
        name = ""
    if raw_args is None:
        raw_args = ""

    return str(call_id or ""), name, raw_args


def set_tool_call_name_and_arguments(tool_call: Any, *, name: str, arguments: str) -> None:
    """Mutate tool calls in-place across nested+flat dict/object representations."""
    try:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function")
            if isinstance(function_payload, dict):
                function_payload["name"] = name
                function_payload["arguments"] = arguments
                if "name" in tool_call:
                    tool_call["name"] = name
                if "arguments" in tool_call:
                    tool_call["arguments"] = arguments
            else:
                tool_call["name"] = name
                tool_call["arguments"] = arguments
            return

        function_payload = getattr(tool_call, "function", None)
        if function_payload is not None:
            try:
                setattr(function_payload, "name", name)
            except Exception:
                pass
            try:
                setattr(function_payload, "arguments", arguments)
            except Exception:
                pass

        try:
            setattr(tool_call, "name", name)
        except Exception:
            pass
        try:
            setattr(tool_call, "arguments", arguments)
        except Exception:
            pass
    except Exception:
        pass


def set_message_tool_calls(messages: Any, index: int, tool_calls: list[Any]) -> None:
    """Set message.tool_calls for dict and typed-message lists."""
    if not isinstance(messages, list):
        return
    if not messages:
        return
    resolved_index = index if index >= 0 else len(messages) + index
    if resolved_index < 0 or resolved_index >= len(messages):
        return

    message = messages[resolved_index]
    if isinstance(message, dict):
        message["tool_calls"] = tool_calls
        return

    try:
        setattr(message, "tool_calls", tool_calls)
        return
    except Exception:
        pass

    model_copy = getattr(message, "model_copy", None)
    if callable(model_copy):
        try:
            messages[resolved_index] = model_copy(update={"tool_calls": tool_calls})
        except Exception:
            pass


def sanitize_messages_content(messages: Any) -> Any:
    """Strip `<|...|>` chat-template tokens from string message content."""
    if not isinstance(messages, list):
        return messages

    sanitized_messages: list[Any] = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
        if not isinstance(content, str):
            sanitized_messages.append(message)
            continue

        sanitized_content = _CHAT_TEMPLATE_TOKEN_PATTERN.sub("", content)
        if sanitized_content == content:
            sanitized_messages.append(message)
            continue

        if isinstance(message, dict):
            updated = dict(message)
            updated["content"] = sanitized_content
            sanitized_messages.append(updated)
            continue

        model_copy = getattr(message, "model_copy", None)
        if callable(model_copy):
            try:
                sanitized_messages.append(model_copy(update={"content": sanitized_content}))
                continue
            except Exception:
                pass

        copy_method = getattr(message, "copy", None)
        if callable(copy_method):
            try:
                sanitized_messages.append(copy_method(update={"content": sanitized_content}))
                continue
            except Exception:
                pass

        try:
            setattr(message, "content", sanitized_content)
        except Exception:
            pass
        sanitized_messages.append(message)

    return sanitized_messages
