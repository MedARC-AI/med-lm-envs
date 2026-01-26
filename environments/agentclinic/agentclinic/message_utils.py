from typing import Any

import verifiers as vf


def message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item.get("content", "")))
        return "\n".join([p for p in parts if p])
    if content is None:
        return ""
    return str(content)


def extract_last_assistant_text(messages: vf.Messages) -> str:
    if isinstance(messages, str):
        return messages
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                return message_content_to_text(msg.get("content", ""))
        else:
            role = getattr(msg, "role", None)
            if role == "assistant":
                content = getattr(msg, "content", "")
                return message_content_to_text(content)
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        return message_content_to_text(last_msg.get("content", ""))
    return message_content_to_text(getattr(last_msg, "content", ""))
