from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable, Literal

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
import verifiers as vf
from verifiers.utils.async_utils import maybe_await
from verifiers.envs.environment import get_overlong_prompt_dummy_response

from . import graders
from .tools import build_tool_list

disable_progress_bar()


class MedAgentBenchV2Env(vf.StatefulToolEnv):
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

    @staticmethod
    def _sanitize_message_content(value: Any) -> Any:
        """Strip common chat-template tokens that can crash some OpenAI-compatible servers.

        We keep this conservative: only remove `<|...|>` style tokens that some OSS chat templates
        emit literally (e.g. `<|end|><|start|>assistant<|channel|>commentary`).
        """
        if not isinstance(value, str):
            return value
        # Remove `<|...|>` tokens (llama/chatml-ish) while leaving surrounding text intact.
        return re.sub(r"<\|[^|>]+\|>", "", value)

    @classmethod
    def _sanitize_prompt(cls, prompt: vf.Messages) -> vf.Messages:
        if not isinstance(prompt, list):
            return prompt
        sanitized: list[dict[str, Any]] = []
        for msg in prompt:
            if not isinstance(msg, dict):
                sanitized.append(msg)  # type: ignore[arg-type]
                continue
            if "content" not in msg:
                sanitized.append(msg)
                continue
            content = msg.get("content")
            # Only sanitize string content; multimodal content is left untouched.
            if isinstance(content, str):
                new_msg = dict(msg)
                new_msg["content"] = cls._sanitize_message_content(content)
                sanitized.append(new_msg)
            else:
                sanitized.append(msg)
        return sanitized

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
        client: Any,
        model: str,
        prompt: vf.Messages,
        oai_tools: Any | None = None,
        sampling_args: dict | None = None,
        message_type: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Like the base implementation, but tolerate certain provider 500s.

        Some OpenAI-compatible servers (especially OSS model endpoints) can 500 on prompts that
        contain literal chat-template tokens. Instead of crashing the entire evaluation, we
        sanitize the prompt and, if the call still fails, return Verifiers' "overlong-prompt"
        dummy response so the rollout terminates cleanly.
        """
        resolved_message_type = message_type or getattr(self, "message_type", "chat")
        try:
            return await super().get_model_response(
                client,
                model,
                prompt,
                oai_tools=oai_tools,
                sampling_args=sampling_args,
                message_type=message_type,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            status = self._status_code(exc)
            msg = str(exc).lower()
            # Common deterministic failure from chat-template parsing on some endpoints.
            is_template_parse = "unexpected tokens remaining in message header" in msg
            if status in (500, 502, 503, 504) or is_template_parse:
                # Best-effort retry once with a further-sanitized prompt (covers cases where
                # the original prompt is already sanitized but the endpoint is flaky).
                try:
                    await asyncio.sleep(0.5)
                    sanitized_prompt = self._sanitize_prompt(prompt) if resolved_message_type == "chat" else prompt
                    return await super().get_model_response(
                        client,
                        model,
                        sanitized_prompt,
                        oai_tools=oai_tools,
                        sampling_args=sampling_args,
                        message_type=message_type,
                        **kwargs,
                    )
                except Exception:  # noqa: BLE001
                    self.logger.error(
                        "Model call failed with status=%s (%s); terminating rollout with dummy response.",
                        status,
                        exc,
                    )
                    return get_overlong_prompt_dummy_response(resolved_message_type)
            raise

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        if not isinstance(tool_args, dict):
            return {}
        return tool_args

    async def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> bool:
        if state.get("status") == "completed":
            return True

        completed = await super().is_completed(messages, state, **kwargs)
        if not completed:
            return False

        last_msg = messages[-1] if messages else {}
        is_assistant = isinstance(last_msg, dict) and last_msg.get("role") == "assistant"
        no_tool_calls = not last_msg.get("tool_calls") if isinstance(last_msg, dict) else False

        if is_assistant and no_tool_calls:
            state["final_answer"] = "[]"
            state["status"] = "completed"
        elif await self.max_turns_reached(state):
            state["final_answer"] = "[]"
            state["status"] = "completed"
        elif state.get("prompt_too_long"):
            state["final_answer"] = "[]"
            state["status"] = "completed"

        return state.get("status") == "completed"

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, state: vf.State, **kwargs: Any
    ) -> vf.Message:
        try:
            tool_func = self.tool_map[tool_name]
            result = await maybe_await(tool_func, **tool_args)
            if tool_name == "finish":
                state["final_answer"] = json.dumps(result)
                state["status"] = "completed"
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

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> tuple[vf.Messages, vf.State]:
        assert isinstance(messages, list)
        tool_calls = messages[-1].get("tool_calls") or []

        def _set_tool_call_arguments(tc: Any, args: str) -> None:
            try:
                if isinstance(tc, dict):
                    func = tc.get("function")
                    if isinstance(func, dict):
                        func["arguments"] = args
                    return
                func = getattr(tc, "function", None)
                if func is not None:
                    try:
                        func.arguments = args
                    except Exception:
                        pass
            except Exception:
                pass

        def _set_tool_call_name(tc: Any, name: str) -> None:
            try:
                if isinstance(tc, dict):
                    func = tc.get("function")
                    if isinstance(func, dict):
                        func["name"] = name
                    return
                func = getattr(tc, "function", None)
                if func is not None:
                    try:
                        func.name = name
                    except Exception:
                        pass
            except Exception:
                pass

        tool_messages: list[vf.Message] = []
        sanitized_tool_calls: list[Any] = []
        for tool_call in tool_calls:
            if isinstance(tool_call, str):
                try:
                    tool_call = json.loads(tool_call)
                except json.JSONDecodeError:
                    continue

            tool_call_id = getattr(tool_call, "id", None) or (
                tool_call.get("id", "") if isinstance(tool_call, dict) else ""
            )
            func = getattr(tool_call, "function", None) or (
                tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
            )
            tool_name = getattr(func, "name", None) or (func.get("name", "") if isinstance(func, dict) else "")
            raw_args = getattr(func, "arguments", None) or (func.get("arguments", "") if isinstance(func, dict) else "")

            if (
                not isinstance(tool_name, str)
                or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_name)
                or tool_name not in self.tool_map
            ):
                safe_name = "calculator" if "calculator" in self.tool_map else next(iter(self.tool_map))
                _set_tool_call_name(tool_call, safe_name)
                _set_tool_call_arguments(tool_call, "{}")
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
                    _set_tool_call_arguments(tool_call, "{}")
                elif isinstance(raw_args, dict):
                    tool_args = raw_args
                else:
                    tool_args = json.loads(raw_args)
                    if isinstance(tool_args, str):
                        tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {}
                _set_tool_call_arguments(tool_call, "{}")

            tool_args = self.update_tool_args(tool_name, tool_args, messages, state, **kwargs)
            tool_message = await self.call_tool(tool_name, tool_args, tool_call_id, state=state)
            tool_messages.append(tool_message)
            sanitized_tool_calls.append(tool_call)

        if tool_calls:
            messages[-1]["tool_calls"] = sanitized_tool_calls
        return tool_messages, state


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


def _resolve_tasks_path(tasks_path: str | None, tasks_variant: str) -> Path:
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
    return mapping[tasks_variant]


def load_environment(
    fhir_api_base: str,
    tasks_variant: Literal["new_patient_tasks", "new_tasks", "test_data_v2"] = "new_patient_tasks",
    tasks_path: str | None = None,
    task_types: list[str] | None = None,
    max_turns: int = 8,
) -> vf.Environment:
    if not fhir_api_base.endswith("/"):
        fhir_api_base = f"{fhir_api_base}/"

    system_prompt = _load_system_prompt()
    tasks_file = _resolve_tasks_path(tasks_path, tasks_variant)
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
        completion: list[dict[str, Any]],
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
