"""AWS Bedrock Converse client for verifiers.

Implements the verifiers Client interface using boto3's Bedrock Converse API.
Registered as api_client_type = "bedrock_converse" via monkeypatch at CLI startup.
"""

from __future__ import annotations

import json
import time
from typing import Any

import boto3

from verifiers.clients.client import Client
from verifiers.types import (
    ClientConfig,
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    Tool,
    ToolCall,
    Usage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    TextMessage,
)


class BedrockConverseClient(Client[Any, dict, dict, dict]):
    """Verifiers client that calls AWS Bedrock Converse API directly."""

    def setup_client(self, config: ClientConfig) -> Any:
        # boto3 natively respects AWS_PROFILE, AWS_REGION, AWS_ACCESS_KEY_ID,
        # ~/.aws/config, instance roles, etc. No config repurposing needed.
        session = boto3.Session()
        return session.client("bedrock-runtime")

    async def close(self) -> None:
        pass

    async def to_native_prompt(self, messages: Messages) -> tuple[dict, dict]:
        system: list[dict] = []
        conversation: list[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                text = msg.content if isinstance(msg.content, str) else _flatten(msg.content)
                system.append({"text": text})
            elif isinstance(msg, UserMessage):
                conversation.append({"role": "user", "content": _to_content(msg.content)})
            elif isinstance(msg, AssistantMessage):
                content = msg.content if isinstance(msg.content, str) else _flatten(msg.content)
                conversation.append({"role": "assistant", "content": [{"text": content}]})
            elif isinstance(msg, TextMessage):
                conversation.append({"role": "user", "content": [{"text": msg.content}]})
            elif isinstance(msg, ToolMessage):
                # Bedrock tool results go as user messages
                conversation.append({
                    "role": "user",
                    "content": [{"toolResult": {
                        "toolUseId": msg.tool_call_id,
                        "content": [{"text": msg.content if isinstance(msg.content, str) else str(msg.content)}],
                    }}],
                })

        return {"messages": conversation, "system": system}, {}

    async def to_native_tool(self, tool: Tool) -> dict:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.parameters},
            }
        }

    async def get_native_response(
        self,
        prompt: dict,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        kwargs.pop("state", None)

        inference_config: dict[str, Any] = {}
        sa = dict(sampling_args) if sampling_args else {}

        if "max_tokens" in sa and sa["max_tokens"] is not None:
            inference_config["maxTokens"] = sa["max_tokens"]
        else:
            inference_config["maxTokens"] = 4096
        if "temperature" in sa and sa["temperature"] is not None:
            inference_config["temperature"] = sa["temperature"]
        if "top_p" in sa and sa["top_p"] is not None:
            inference_config["topP"] = sa["top_p"]
        if "stop" in sa and sa["stop"]:
            stops = sa["stop"]
            inference_config["stopSequences"] = [stops] if isinstance(stops, str) else stops

        call_kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": prompt["messages"],
        }
        if prompt["system"]:
            call_kwargs["system"] = prompt["system"]
        if inference_config:
            call_kwargs["inferenceConfig"] = inference_config
        if tools:
            call_kwargs["toolConfig"] = {"tools": tools}

        # boto3 is sync — run in executor to not block the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self.client.converse(**call_kwargs))
        return response

    async def raise_from_native_response(self, response: dict) -> None:
        pass

    async def from_native_response(self, response: dict) -> Response:
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts = []
        tool_calls = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu["toolUseId"],
                    name=tu["name"],
                    arguments=json.dumps(tu.get("input", {})),
                ))

        stop_reason = response.get("stopReason", "end_turn")
        finish_map: dict[str, FinishReason] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }

        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        return Response(
            id=response.get("ResponseMetadata", {}).get("RequestId", ""),
            model="",
            created=int(time.time()),
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                reasoning_tokens=0,
                total_tokens=input_tokens + output_tokens,
            ),
            message=ResponseMessage(
                content="".join(text_parts),
                reasoning_content=None,
                thinking_blocks=None,
                tool_calls=tool_calls or None,
                finish_reason=finish_map.get(stop_reason, "stop"),
                is_truncated=(stop_reason == "max_tokens"),
                tokens=None,
            ),
        )


def _to_content(content) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append({"text": part["text"]})
            elif isinstance(part, str):
                blocks.append({"text": part})
        return blocks or [{"text": ""}]
    return [{"text": str(content)}]


def _flatten(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return str(content)
