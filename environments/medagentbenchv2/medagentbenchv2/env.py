from __future__ import annotations

import asyncio
import inspect
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import verifiers as vf
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.types import Messages
from medarc_verifiers.utils.tool_call_compat import (
    extract_tool_call_fields,
    sanitize_messages_content,
    set_message_tool_calls,
    set_tool_call_name_and_arguments,
)
from verifiers.utils.async_utils import maybe_await

from . import graders
from .tools import build_tool_list

disable_progress_bar()


class TasksVariant(str, Enum):
    """MedAgentBench V2 task variant selector."""

    NEW_PATIENT_TASKS = "new_patient_tasks"
    NEW_TASKS = "new_tasks"
    TEST_DATA_V2 = "test_data_v2"


def _set_tool_call_function_field(tool_call: Any, field: str, value: Any) -> None:
    """Best-effort setter for tool_call.function fields across nested/flat shapes."""
    if field not in {"name", "arguments"}:
        return
    _, current_name, current_args = extract_tool_call_fields(tool_call)
    resolved_name = current_name
    resolved_args = current_args
    if field == "name":
        resolved_name = value
    if field == "arguments":
        resolved_args = value
    if not isinstance(resolved_name, str):
        resolved_name = str(resolved_name or "")
    if not isinstance(resolved_args, str):
        if isinstance(resolved_args, (dict, list)):
            try:
                resolved_args = json.dumps(resolved_args)
            except Exception:
                resolved_args = "{}"
        else:
            resolved_args = str(resolved_args or "")
    set_tool_call_name_and_arguments(tool_call, name=resolved_name, arguments=resolved_args)


class MedAgentBenchV2Env(vf.StatefulToolEnv):
    _supports_tool_defs_cache: bool | None = None

    @classmethod
    def _supports_tool_defs(cls) -> bool:
        cached = cls._supports_tool_defs_cache
        if cached is not None:
            return cached
        try:
            params = inspect.signature(vf.Environment.get_model_response).parameters
            cached = "tool_defs" in params
        except Exception:
            cached = False
        cls._supports_tool_defs_cache = cached
        return cached

    @staticmethod
    def _root_cause(exc: BaseException) -> BaseException:
        cur = exc
        seen: set[int] = set()
        while True:
            nxt = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
            if nxt is None:
                return cur
            nxt_id = id(nxt)
            if nxt_id in seen:
                return cur
            seen.add(nxt_id)
            cur = nxt

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status
        resp = getattr(exc, "response", None)
        if resp is not None:
            code = getattr(resp, "status_code", None)
            if isinstance(code, int):
                return code
        return None

    @classmethod
    def _sanitize_prompt(cls, prompt: Messages) -> Messages:
        return sanitize_messages_content(prompt)

    def __init__(
        self,
        *,
        fhir_api_base: str,
        max_turns: int = 8,
        tools: list[Callable[..., Any]] | None = None,
        **kwargs: Any,
    ):
        tool_list = tools or build_tool_list(fhir_api_base)
        super().__init__(
            tools=tool_list,
            max_turns=max_turns,
            error_formatter=lambda e: f"Error: {e}",
            **kwargs,
        )
        self.fhir_api_base = fhir_api_base
        self.max_turns = max_turns

    async def get_model_response(
        self,
        state: vf.State,
        prompt: Messages,
        client: Any | None = None,
        model: str | None = None,
        oai_tools: Any | None = None,
        tool_defs: Any | None = None,
        sampling_args: dict | None = None,
        message_type: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Like the base implementation, but tolerate certain provider 500s.

        Some OpenAI-compatible servers (especially OSS model endpoints) can 500 on prompts that
        contain literal chat-template tokens. We retry once with a sanitized prompt to avoid
        provider-side parser crashes.
        """
        supports_tool_defs = self._supports_tool_defs()
        resolved_message_type = message_type or getattr(self, "message_type", "chat")
        resolved_tool_defs = tool_defs if tool_defs is not None else getattr(self, "tool_defs", None)
        base_get_model_response = super(MedAgentBenchV2Env, self).get_model_response

        async def _call_base_model(current_prompt: Messages) -> Any:
            if supports_tool_defs:
                return await base_get_model_response(
                    state,
                    current_prompt,
                    client=client,
                    model=model,
                    tool_defs=resolved_tool_defs,
                    sampling_args=sampling_args,
                    **kwargs,
                )
            return await base_get_model_response(
                state,
                current_prompt,
                client=client,
                model=model,
                oai_tools=oai_tools,
                sampling_args=sampling_args,
                message_type=message_type,
                **kwargs,
            )

        try:
            return await _call_base_model(prompt)
        except vf.Error as exc:
            root = self._root_cause(exc)
            status = self._status_code(root)
            msg = str(root).lower()
            # Common deterministic failure from chat-template parsing on some endpoints.
            is_template_parse = "unexpected tokens remaining in message header" in msg
            if status in (500, 502, 503, 504) or is_template_parse:
                # Best-effort retry once with a further-sanitized prompt (covers cases where
                # the original prompt is already sanitized but the endpoint is flaky).
                try:
                    await asyncio.sleep(0.5)
                    sanitized_prompt = self._sanitize_prompt(prompt) if resolved_message_type == "chat" else prompt
                    return await _call_base_model(sanitized_prompt)
                except vf.Error as retry_exc:
                    retry_root = self._root_cause(retry_exc)
                    self.logger.error(
                        "Model call failed after retry (status=%s): %s",
                        self._status_code(retry_root),
                        retry_root,
                    )
                    raise retry_exc
            raise exc

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        if not isinstance(tool_args, dict):
            return {}
        return tool_args

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, state: vf.State, **kwargs: Any
    ) -> vf.Message:
        try:
            tool_func = self.tool_map[tool_name]
            result = await maybe_await(tool_func, **tool_args)
            if tool_name == "finish":
                state["final_answer"] = json.dumps(result)
            return {
                "role": "tool",
                "content": str(result),
                "tool_call_id": tool_call_id,
            }
        except Exception as exc:
            return {
                "role": "tool",
                "content": self.error_formatter(exc),
                "tool_call_id": tool_call_id,
            }

    async def env_response(self, messages: Messages, state: vf.State, **kwargs: Any) -> Messages:
        assert isinstance(messages, list)
        last_message = messages[-1]
        if isinstance(last_message, dict):
            tool_calls = last_message.get("tool_calls") or []
        else:
            message_get = getattr(last_message, "get", None)
            if callable(message_get):
                tool_calls = message_get("tool_calls") or []
            else:
                tool_calls = getattr(last_message, "tool_calls", None) or []

        tool_messages: list[vf.Message] = []
        sanitized_tool_calls: list[Any] = []
        finish_called = False
        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                try:
                    tool_call = json.loads(tool_call)
                except json.JSONDecodeError:
                    continue

            tool_call_id, tool_name, raw_args = extract_tool_call_fields(tool_call)

            if (
                not isinstance(tool_name, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_name)
                or tool_name not in self.tool_map
            ):
                # Rewrite malformed tool names to a valid configured tool so the next provider request
                # does not 400 while re-validating prior tool_calls in message history.
                safe_name = "calculator" if "calculator" in self.tool_map else next(iter(self.tool_map))
                _set_tool_call_function_field(tool_call, "name", safe_name)
                _set_tool_call_function_field(tool_call, "arguments", "{}")
                sanitized_tool_calls.append(tool_call)
                tool_messages.append(
                    {
                        "role": "tool",
                        "content": (f"Error: Invalid tool formatting for '{tool_name}'."),
                        "tool_call_id": tool_call_id or "",
                    }
                )
                continue

            try:
                if raw_args is None or str(raw_args).strip() == "":
                    tool_args = {}
                    _set_tool_call_function_field(tool_call, "arguments", "{}")
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
                else:
                    tool_args = json.loads(raw_args)
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}
                _set_tool_call_function_field(tool_call, "arguments", "{}")

            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            tool_message = await self.call_tool(tool_name, tool_args, tool_call_id, state=state)
            tool_messages.append(tool_message)
            sanitized_tool_calls.append(tool_call)
            if tool_name == "finish":
                finish_called = True

        if tool_calls:
            set_message_tool_calls(messages, -1, sanitized_tool_calls)

        if finish_called:
            # Ensure we don't make an extra model call after the finish tool executes.
            # MultiTurnEnv.rollout() skips generation when final_env_response is set.
            state["final_env_response"] = tool_messages

        return tool_messages


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent / "_vendor" / "prompts" / "new_system.txt"
    return prompt_path.read_text()


def _build_user_message(instruction: str, context: str | None) -> str:
    content = f"""<instruction>
{instruction}
</instruction>
"""
    if context:
        content += f"""<context>
{context}
</context>
Make sure to format the tool calls correctly and use the finish tool when you have the final answer.
"""
    return content


def _resolve_tasks_path(tasks_path: str | None, tasks_variant_value: str) -> Path:
    if tasks_path:
        path = Path(tasks_path)
        if not path.is_absolute() and path.parent == Path("."):
            return Path(__file__).resolve().parent / path
        return path
    base = Path(__file__).resolve().parent / "_vendor" / "MedAgentBench" / "data" / "medagentbench"
    mapping = {
        "new_patient_tasks": base / "new_patient_tasks.json",
        "new_tasks": base / "new_tasks.json",
        "test_data_v2": base / "test_data_v2.json",
    }
    return mapping[tasks_variant_value]


def load_environment(
    fhir_api_base: str,
    tasks_variant: TasksVariant | str = TasksVariant.NEW_PATIENT_TASKS,
    tasks_path: str | None = None,
    task_types: list[str] | None = None,
    max_turns: int = 8,
) -> vf.Environment:
    if not fhir_api_base.endswith("/"):
        fhir_api_base = f"{fhir_api_base}/"

    # Normalize tasks_variant to enum
    tasks_variant = TasksVariant(tasks_variant) if isinstance(tasks_variant, str) else tasks_variant

    system_prompt = _load_system_prompt()
    tasks_file = _resolve_tasks_path(tasks_path, tasks_variant.value)
    tasks = json.loads(tasks_file.read_text())

    if task_types:
        tasks = [task for task in tasks if any(task["id"].startswith(t) for t in task_types)]

    def _map(task: dict[str, Any]) -> dict[str, Any]:
        prompt = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": _build_user_message(task["instruction"], task.get("context")),
            },
        ]
        return {
            "id": task["id"],
            "prompt": prompt,
            "info": dict(task),
            "answer": "",
            "task": "medagentbenchv2",
        }

    eval_dataset = Dataset.from_list([_map(task) for task in tasks])

    def medagentbench_reward(
        completion: list[Any],
        state: vf.State,
        info: dict[str, Any] | None = None,
        **_: Any,
    ) -> float:
        task = info or {}
        final_answer = state.get("final_answer")
        passed = graders.evaluate_task(task, completion, final_answer, fhir_api_base)
        return 1.0 if passed else 0.0

    rubric = vf.Rubric(funcs=[medagentbench_reward], weights=[1.0], parser=vf.Parser())

    return MedAgentBenchV2Env(
        eval_dataset=eval_dataset,
        fhir_api_base=fhir_api_base,
        max_turns=max_turns,
        rubric=rubric,
    )
