import json

import pytest
from datasets import Dataset

from environments.medagentbenchv2.medagentbenchv2.env import MedAgentBenchV2Env
from environments.medagentbenchv2.medagentbenchv2.tools import (
    ProcedureSearchParams,
    build_tool_list,
    calculator,
    create_finish,
)
from environments.medagentbenchv2.medagentbenchv2 import graders


def test_tool_name_list() -> None:
    tools = build_tool_list("http://localhost:8080/fhir/")
    names = {tool.__name__ for tool in tools}
    assert names == {
        "patient_search",
        "fhir_observation_search",
        "fhir_vitals_search",
        "fhir_procedure_search",
        "fhir_condition_search",
        "fhir_medication_request_search",
        "fhir_vitals_create",
        "fhir_medication_request_create",
        "fhir_service_request_create",
        "calculator",
        "finish",
    }


def test_prompt_tool_name_consistency() -> None:
    prompt_path = "environments/medagentbenchv2/medagentbenchv2/_vendor/prompts/new_system.txt"
    content = open(prompt_path).read()
    assert "patient_search" in content
    assert "fhir_patient_search" not in content


def test_grader_returns_false_on_invalid_task() -> None:
    assert (
        graders.evaluate_task(
            task={"id": 123},
            completion=[],
            final_answer=None,
            fhir_api_base="http://localhost:8080/fhir/",
        )
        is False
    )


def test_calculator_accepts_datetime_and_math() -> None:
    result = calculator(expression="math.sqrt(4) + datetime.timedelta(days=7).days")
    assert result == "9.0"


def test_calculator_rejects_bad_chars() -> None:
    assert calculator(expression="__import__('os').system('ls')").startswith("Error:")


def test_tool_schemas_have_descriptions() -> None:
    env = MedAgentBenchV2Env(
        fhir_api_base="http://localhost:8080/fhir/",
        eval_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": "hi"}], "answer": ""}]),
    )
    tool_by_name = {t["function"]["name"]: t for t in env.oai_tools}
    assert tool_by_name["patient_search"]["function"]["description"]
    assert tool_by_name["fhir_vitals_create"]["function"]["description"]
    # Ensure at least one parameter has a human-readable description (not just a title).
    assert (
        tool_by_name["patient_search"]["function"]["parameters"]["properties"]["identifier"].get("description")
        is not None
    )


def test_procedure_search_date_is_optional() -> None:
    params = ProcedureSearchParams(code="IMGCT0491", patient="Patient/S6315806")
    assert params.date is None


@pytest.mark.asyncio
async def test_env_response_sanitizes_tool_args() -> None:
    env = MedAgentBenchV2Env(
        fhir_api_base="http://localhost:8080/fhir/",
        tools=[calculator, create_finish()],
        eval_dataset=Dataset.from_list([{"prompt": [{"role": "user", "content": "hi"}], "answer": ""}]),
    )
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "function": {"name": "calculator", "arguments": "{bad"}},
            ],
        }
    ]
    state = {}
    tool_messages, _ = await env.env_response(messages, state)
    assert messages[-1]["tool_calls"][0]["function"]["arguments"] == "{}"
    assert tool_messages[0]["tool_call_id"] == "call_1"
    assert "Error:" in tool_messages[0]["content"]
