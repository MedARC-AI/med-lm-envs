from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

from ._vendor.medagentbenchevals import new_refsol


@dataclass
class ChatHistoryItem:
    role: str
    content: str


@dataclass
class TaskResult:
    result: str | int | float
    history: list[ChatHistoryItem] = field(default_factory=list)


def _api_mapping(fhir_api_base: str) -> dict[str, tuple[str, str]]:
    return {
        "patient_search": ("GET", urljoin(fhir_api_base, "Patient")),
        "fhir_vitals_create": ("POST", urljoin(fhir_api_base, "Observation")),
        "fhir_medication_request_create": (
            "POST",
            urljoin(fhir_api_base, "MedicationRequest"),
        ),
        "fhir_service_request_create": (
            "POST",
            urljoin(fhir_api_base, "ServiceRequest"),
        ),
        "fhir_observation_search": ("GET", urljoin(fhir_api_base, "Observation")),
        "fhir_medication_request_search": (
            "GET",
            urljoin(fhir_api_base, "MedicationRequest"),
        ),
        "fhir_vitals_search": ("GET", urljoin(fhir_api_base, "Observation")),
        "fhir_procedure_search": ("GET", urljoin(fhir_api_base, "Procedure")),
        "fhir_condition_search": ("GET", urljoin(fhir_api_base, "Condition")),
    }


def _coerce_tool_call(tool_call: Any) -> tuple[str, str, Any]:
    if isinstance(tool_call, dict):
        func = tool_call.get("function") or {}
        name = func.get("name", "") if isinstance(func, dict) else ""
        args = func.get("arguments", "") if isinstance(func, dict) else ""
        call_id = tool_call.get("id", "") or ""
        return name, call_id, args

    func = getattr(tool_call, "function", None)
    name = getattr(func, "name", "") if func is not None else ""
    args = getattr(func, "arguments", "") if func is not None else ""
    call_id = getattr(tool_call, "id", "") or ""
    return name, call_id, args


def _parse_tool_args(raw_args: Any) -> Any:
    if isinstance(raw_args, dict):
        return raw_args
    if raw_args is None:
        return {}
    try:
        parsed = json.loads(raw_args)
        if isinstance(parsed, str):
            return json.loads(parsed)
        return parsed
    except Exception:
        return {}


def build_task_result(
    completion: list[dict[str, Any]],
    final_answer: str | list[Any] | None,
    fhir_api_base: str,
) -> TaskResult:
    history: list[ChatHistoryItem] = []
    tool_calls: dict[str, str] = {}
    api_mapping = _api_mapping(fhir_api_base)

    if final_answer is None:
        result_value: str | int | float = "[]"
    elif isinstance(final_answer, str):
        result_value = final_answer
    else:
        result_value = json.dumps(final_answer)

    for msg in completion:
        role = msg.get("role")
        if role == "assistant":
            content = msg.get("content") or ""
            if content:
                history.append(ChatHistoryItem(role="agent", content=content))

            for tool_call in msg.get("tool_calls") or []:
                tool_name, call_id, raw_args = _coerce_tool_call(tool_call)
                tool_args = _parse_tool_args(raw_args)
                if tool_name in api_mapping:
                    method, url = api_mapping[tool_name]
                    history.append(
                        ChatHistoryItem(
                            role="agent",
                            content=f"{method} {url}\n{json.dumps(tool_args)}",
                        )
                    )
                else:
                    history.append(
                        ChatHistoryItem(
                            role="agent", content=f"{tool_name}({tool_args})"
                        )
                    )
                if call_id:
                    tool_calls[call_id] = tool_name

        if role == "tool":
            call_id = msg.get("tool_call_id", "") or ""
            tool_name = tool_calls.get(call_id)
            if tool_name in api_mapping:
                method, _ = api_mapping[tool_name]
                history.append(
                    ChatHistoryItem(role="agent", content=f"{method} request accepted")
                )

    return TaskResult(result=result_value, history=history)


def get_task_category(task_id: str) -> int:
    return int(task_id.split("_")[0][4:])


category_to_eval_fn = {
    1: new_refsol.task1,
    2: new_refsol.task2,
    3: new_refsol.task3,
    4: new_refsol.task4,
    5: new_refsol.task5,
    6: new_refsol.task6,
    7: new_refsol.task7,
    8: new_refsol.task8,
    9: new_refsol.task9,
    10: new_refsol.task10,
}


def evaluate_task(
    task: dict[str, Any],
    completion: list[dict[str, Any]],
    final_answer: str | list[Any] | None,
    fhir_api_base: str,
) -> bool:
    try:
        task_id = task.get("id")
        if not isinstance(task_id, str):
            return False
        category = get_task_category(task_id)
        eval_fn = category_to_eval_fn.get(category)
        if eval_fn is None:
            return False
        result = build_task_result(completion, final_answer, fhir_api_base)
        return bool(eval_fn(task, result, fhir_api_base))
    except Exception:
        return False
