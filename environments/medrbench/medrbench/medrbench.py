import json
import os
import re
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, concatenate_datasets
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.utils import default_judge_api_key, download_file, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

from medrbench.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    DIAGNOSIS_JUDGE_PROMPT,
    DIAGNOSIS_TASK_PROMPT,
    MULTI_TURN_FIRST_TURN_PROMPT,
    MULTI_TURN_FOLLOWING_TURN_PROMPT,
    PATIENT_AGENT_PROMPT,
    SINGLE_TURN_FINAL_TURN_PROMPT,
    SINGLE_TURN_FIRST_TURN_PROMPT,
    TREATMENT_JUDGE_PROMPT,
    TREATMENT_TASK_PROMPT,
)

disable_progress_bar()


class Split(str, Enum):
    """MedRBench dataset splits."""

    DIAGNOSIS = "diagnosis"
    TREATMENT = "treatment"
    ALL = "all"


class Task(str, Enum):
    ORACLE = "oracle"
    ONE_TURN = "1turn"
    FREE_TURN = "free_turn"


# Data source
BASE_URL = "https://raw.githubusercontent.com/MAGIC-AI4Med/MedRBench/refs/heads/main/data/MedRBench"
DIAGNOSIS_FILENAME = "diagnosis_957_cases_with_rare_disease_491.json"
TREATMENT_FILENAME = "treatment_496_cases_with_rare_disease_165.json"


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    """Resolve the cache directory path."""
    if cache_dir is None:
        env_override = os.getenv("MEDRBENCH_CACHE_DIR")
        if env_override:
            return Path(env_override)
        return Path.home() / ".cache" / "medrbench"
    return Path(cache_dir)


def _fetch_data(filename: str, cache_path: Path) -> dict[str, Any]:
    """Fetch JSON data from URL with caching."""
    # Prefer local `external/MedRBench` data (when available) to match the canonical
    # implementation in this repo and avoid upstream dataset drift.
    try:
        repo_root = Path(__file__).resolve().parents[3]
        local_path = repo_root / "external" / "MedRBench" / "data" / "MedRBench" / filename
    except Exception:
        local_path = None

    if local_path is not None and local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)

    json_path = cache_path / filename
    download_file(url=f"{BASE_URL}/{filename}", dest=json_path, verify=False)

    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _to_vf_format_diagnosis(data: dict[str, Any], rare_disease_only: bool = False, task: Task = Task.ORACLE) -> Dataset:
    """Convert diagnosis data to verifiers format."""
    records = []
    task = Task(task)
    for pmc_id, case in data.items():
        # Filter for rare disease cases if requested
        if rare_disease_only and not case.get("checked_rare_disease"):
            continue

        generate_case = case.get("generate_case", {})
        case_summary = generate_case.get("case_summary", "")
        differential_diagnosis = generate_case.get("differential_diagnosis", "")
        diagnosis_results = generate_case.get("diagnosis_results", "")

        case_without_tests, ancillary_tests = _split_case_at_ancillary_tests(case_summary)

        if task == Task.ORACLE:
            question = DIAGNOSIS_TASK_PROMPT.format(case=case_summary)
        elif task == Task.ONE_TURN:
            question = SINGLE_TURN_FIRST_TURN_PROMPT.format(case=case_without_tests)
        else:
            question = MULTI_TURN_FIRST_TURN_PROMPT.format(case=case_without_tests)

        records.append(
            {
                "question": question,
                "answer": diagnosis_results,
                "task": f"medrbench-diagnosis-{task.value}",
                "info": {
                    "pmc_id": pmc_id,
                    "case_summary": case_summary,
                    "case_without_tests": case_without_tests,
                    "ancillary_tests": ancillary_tests,
                    "differential_diagnosis": differential_diagnosis,
                    "reference_response": diagnosis_results,
                    "task_type": "medrbench-diagnosis",
                    "body_category": case.get("body_category", []),
                    "disorder_category": case.get("disorder_category", []),
                    "checked_rare_disease": case.get("checked_rare_disease", []),
                },
            }
        )

    return Dataset.from_list(records)


def _to_vf_format_treatment(data: dict[str, Any], rare_disease_only: bool = False) -> Dataset:
    """Convert treatment data to verifiers format."""
    records = []
    for pmc_id, case in data.items():
        # Filter for rare disease cases
        if rare_disease_only and not case.get("checked_rare_disease"):
            continue

        generate_case = case.get("generate_case", {})
        case_summary = generate_case.get("case_summary", "")
        treatment_planning_analysis = generate_case.get("treatment_planning_analysis", "")
        treatment_plan_results = generate_case.get("treatment_plan_results", "")

        # Only use case_summary as input - treatment_planning_analysis likely contains hints
        # The model must derive the treatment plan from the case summary alone
        question = TREATMENT_TASK_PROMPT.format(case=case_summary)

        records.append(
            {
                "question": question,
                "answer": treatment_plan_results,
                "task": "medrbench-treatment",
                "info": {
                    "pmc_id": pmc_id,
                    "case_summary": case_summary,
                    "treatment_planning_analysis": treatment_planning_analysis,
                    "reference_response": treatment_plan_results,
                    "task_type": "medrbench-treatment",
                    "body_category": case.get("body_category", []),
                    "disorder_category": case.get("disorder_category", []),
                    "checked_rare_disease": case.get("checked_rare_disease", []),
                },
            }
        )

    return Dataset.from_list(records)


def _split_case_at_ancillary_tests(case_summary: str) -> tuple[str, str]:
    if "ancillary tests" not in case_summary.lower():
        return case_summary, ""
    lines = case_summary.strip().split("\n")
    for idx, line in enumerate(lines):
        if "ancillary tests" in line.lower():
            return "\n".join(lines[:idx]), "\n".join(lines[idx:])
    return case_summary, ""


def _extract_section(text: str, header: str) -> str:
    pattern = rf"###\s*{re.escape(header)}:\s*(.*?)(?=\n###|\Z)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _extract_additional_info_required(text: str) -> str:
    return _extract_section(text, "Additional Information Required")


def _extract_conclusion(text: str) -> str:
    return _extract_section(text, "Conclusion")


def _extract_answer(text: str) -> str:
    answer_patterns = [
        r"###\s*Answer:\s*\n?(.*)",
        r"\*\*Answer:\*\*\s*\n?(.*)",
        r"Answer:\s*\n?(.*)",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"```\s*$", "", answer).strip()
            return answer
    return text.strip()


def _message_content_to_text(content: Any) -> str:
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


def _extract_last_assistant_text(messages: Messages) -> str:
    if isinstance(messages, str):
        return messages
    if not messages:
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict):
            if msg.get("role") == "assistant":
                return _message_content_to_text(msg.get("content", ""))
        else:
            role = getattr(msg, "role", None)
            if role == "assistant":
                content = getattr(msg, "content", "")
                return _message_content_to_text(content)
    last_msg = messages[-1]
    if isinstance(last_msg, dict):
        return _message_content_to_text(last_msg.get("content", ""))
    return _message_content_to_text(getattr(last_msg, "content", ""))


def _parse_judge_result(judge_response: str) -> bool:
    """Parse judge response to determine if prediction is correct.

    Following the original MedRBench logic: is_correct = 'correct' in evaluation_result.lower()
    """
    return "correct" in judge_response.lower()


def _normalize_openai_chat_args(args: dict[str, Any]) -> dict[str, Any]:
    """Normalize AsyncOpenAI chat.completions.create args across client versions."""
    normalized = dict(args)
    if "max_tokens" in normalized:
        if normalized["max_tokens"] is None:
            normalized.pop("max_tokens")
        else:
            normalized["max_completion_tokens"] = normalized.pop("max_tokens")
    if "max_completion_tokens" in normalized and normalized["max_completion_tokens"] is None:
        normalized.pop("max_completion_tokens")
    return {k: v for k, v in normalized.items() if v is not None}


async def _get_patient_agent_response(
    client: AsyncOpenAI,
    model: str,
    case_without_tests: str,
    ancillary_tests: str,
    additional_info_required: str,
    cache: dict[str, str] | None = None,
) -> str:
    cache_key = additional_info_required.strip()
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    system_prompt = PATIENT_AGENT_PROMPT.format(case=case_without_tests, ancillary_test_results=ancillary_tests)
    user_prompt = f"The junior physician wants the following information:\n{additional_info_required}"

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
    except Exception:
        content = "There is no relevant ancillary test information available for this request."

    if cache is not None:
        cache[cache_key] = content
    return content


class MedRBenchFreeTurnEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        patient_agent_client: AsyncOpenAI,
        patient_agent_model: str,
        max_turns: int = 5,
        **kwargs: Any,
    ):
        super().__init__(max_turns=max_turns, **kwargs)
        self.patient_agent_client = patient_agent_client
        self.patient_agent_model = patient_agent_model

    async def setup_state(self, state: State) -> State:
        info = state.get("info", {})
        case_summary = str(info.get("case_summary") or "")
        case_without_tests = str(info.get("case_without_tests") or "")
        ancillary_tests = str(info.get("ancillary_tests") or "")
        if not case_without_tests and case_summary:
            case_without_tests, ancillary_tests = _split_case_at_ancillary_tests(case_summary)
        state["case_without_tests"] = case_without_tests
        state["ancillary_tests"] = ancillary_tests
        state["patient_agent_cache"] = {}
        return state

    async def is_completed(self, messages: Messages, state: State, **kwargs: Any) -> bool:
        # verifiers<stop-decorator> uses the legacy is_completed() override pattern.
        if await super().is_completed(messages, state, **kwargs):
            return True
        # Don't early-stop before the first model response; prompts may contain
        # the words "not required" in instructions.
        if int(state.get("turn", 0) or 0) == 0:
            return False
        last_text = _extract_last_assistant_text(messages)
        additional_info = _extract_additional_info_required(last_text)
        return "not required" in additional_info.lower()

    async def env_response(self, messages: Messages, state: State, **kwargs: Any) -> tuple[Messages, State]:
        last_text = _extract_last_assistant_text(messages)
        additional_info_required = _extract_additional_info_required(last_text)
        if "not required" in additional_info_required.lower():
            return [], state

        case_without_tests = str(state.get("case_without_tests") or "")
        ancillary_tests = str(state.get("ancillary_tests") or "")
        cache = state.get("patient_agent_cache")
        patient_agent_response = await _get_patient_agent_response(
            client=self.patient_agent_client,
            model=self.patient_agent_model,
            case_without_tests=case_without_tests,
            ancillary_tests=ancillary_tests,
            additional_info_required=additional_info_required,
            cache=cache if isinstance(cache, dict) else None,
        )

        response_content = MULTI_TURN_FOLLOWING_TURN_PROMPT.format(additional_information=patient_agent_response)
        if state.get("turn", 0) == self.max_turns - 1:
            response_content = (
                "In the next turn, you cannot ask any additional infomation and must make a final diagnoisis.\n"
                + response_content
            )
        return [{"role": "user", "content": response_content}], state


class MedRBenchOneTurnEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        patient_agent_client: AsyncOpenAI,
        patient_agent_model: str,
        **kwargs: Any,
    ):
        super().__init__(max_turns=2, **kwargs)
        self.patient_agent_client = patient_agent_client
        self.patient_agent_model = patient_agent_model

    async def setup_state(self, state: State) -> State:
        info = state.get("info", {})
        case_summary = str(info.get("case_summary") or "")
        case_without_tests = str(info.get("case_without_tests") or "")
        ancillary_tests = str(info.get("ancillary_tests") or "")
        if not case_without_tests and case_summary:
            case_without_tests, ancillary_tests = _split_case_at_ancillary_tests(case_summary)
        state["case_without_tests"] = case_without_tests
        state["ancillary_tests"] = ancillary_tests
        state["patient_agent_cache"] = {}
        return state

    async def env_response(self, messages: Messages, state: State, **kwargs: Any) -> tuple[Messages, State]:
        last_text = _extract_last_assistant_text(messages)
        additional_info_required = _extract_additional_info_required(last_text)

        case_without_tests = str(state.get("case_without_tests") or "")
        ancillary_tests = str(state.get("ancillary_tests") or "")
        cache = state.get("patient_agent_cache")
        patient_agent_response = await _get_patient_agent_response(
            client=self.patient_agent_client,
            model=self.patient_agent_model,
            case_without_tests=case_without_tests,
            ancillary_tests=ancillary_tests,
            additional_info_required=additional_info_required,
            cache=cache if isinstance(cache, dict) else None,
        )

        response_content = SINGLE_TURN_FINAL_TURN_PROMPT.format(additional_information=patient_agent_response)
        return [{"role": "user", "content": response_content}], state


def load_environment(
    split: str | Split = Split.ALL,
    rare_disease_only: bool = False,
    cache_dir: Path | str | None = None,
    task: str | Task = Task.ORACLE,
    max_turns: int = 5,
    judge_model: str = "gpt-5-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    patient_agent_model: str = "gpt-5-mini",
    patient_agent_base_url: str | None = None,
    patient_agent_api_key: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """
    Load the MedRBench evaluation environment.

    This implementation matches the original MedRBench evaluation setup:
    - Uses system prompt "You are a professional doctor"
    - Uses original task prompts that specify "### Answer:" format
    - Uses original judge prompts from MedRBench

    Args:
        split: Dataset split - "diagnosis", "treatment", or "all" (default: "all")
        rare_disease_only: If True, only include cases with rare diseases
        cache_dir: Directory for caching downloaded data (defaults to ~/.cache/medrbench
            or MEDRBENCH_CACHE_DIR env var)
        task: Diagnosis task mode - "oracle", "1turn", or "free_turn"
        max_turns: Max model calls for free-turn diagnosis
        judge_model: Model to use for LLM-as-judge evaluation (default: gpt-4o as in original)
        judge_base_url: Custom API base URL for judge model
        judge_api_key: API key for judge model
        patient_agent_model: Model for the patient agent in interactive modes
        patient_agent_base_url: Custom API base URL for patient agent
        patient_agent_api_key: API key for patient agent
        system_prompt: Custom system prompt (defaults to "You are a professional doctor")
        **kwargs: Additional arguments passed to SingleTurnEnv

    Returns:
        A verifiers Environment configured for MedRBench evaluation
    """
    # Setup cache directory
    cache_path = _resolve_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Normalize split and task mode
    split = Split(split) if isinstance(split, str) else split
    task = Task(task) if isinstance(task, str) else task

    if task != Task.ORACLE and split in {Split.TREATMENT, Split.ALL}:
        warnings.warn(
            "Non-oracle task requested but only supported for diagnosis split; switching split to 'diagnosis'.",
            UserWarning,
        )
        split = Split.DIAGNOSIS

    # Load and convert data based on split
    if split == Split.DIAGNOSIS:
        data = _fetch_data(DIAGNOSIS_FILENAME, cache_path)
        dataset = _to_vf_format_diagnosis(data, rare_disease_only=rare_disease_only, task=task)
    elif split == Split.TREATMENT:
        data = _fetch_data(TREATMENT_FILENAME, cache_path)
        dataset = _to_vf_format_treatment(data, rare_disease_only=rare_disease_only)
    elif split == Split.ALL:
        # Load both diagnosis and treatment datasets and combine
        diag_data = _fetch_data(DIAGNOSIS_FILENAME, cache_path)
        diag_dataset = _to_vf_format_diagnosis(diag_data, rare_disease_only=rare_disease_only, task=task)
        treat_data = _fetch_data(TREATMENT_FILENAME, cache_path)
        treat_dataset = _to_vf_format_treatment(treat_data, rare_disease_only=rare_disease_only)
        dataset = concatenate_datasets([diag_dataset, treat_dataset])
    else:
        raise ValueError(f"Invalid split: {split}. Must be 'diagnosis', 'treatment', or 'all'.")

    system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    if task == Task.ORACLE:
        parser = vf.Parser(extract_fn=_extract_answer)
    else:
        parser = vf.Parser(extract_fn=_extract_conclusion)

    # Setup judge
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)

    judge_rubric = vf.JudgeRubric(
        judge_client=AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers),
        judge_model=judge_model,
        judge_prompt="{question}",
        parser=parser,
        judge_sampling_args=sampling_args,
    )

    async def judge_rubric_reward(completion: Messages, info: Info, state: State, **kwargs: Any) -> float:
        """Evaluate model completion using LLM judge with original MedRBench prompts."""
        gold_response = str(info.get("reference_response") or "")
        extracted_answer = parser.parse_answer(completion) or ""

        task_name = state.get("task") or info.get("task_type") or "medrbench-diagnosis"

        if str(task_name).startswith("medrbench-treatment"):
            # Use original MedRBench treatment judge prompt
            # Note: original prompt expects additional_info for web search results,
            # we removed it as we don't use web search in this implementation which requires BING search API.
            judge_prompt = TREATMENT_JUDGE_PROMPT.format(
                pred_treatment=extracted_answer,
                gt_treatment=gold_response,
            )
        else:
            # Use original MedRBench diagnosis judge prompt
            judge_prompt = DIAGNOSIS_JUDGE_PROMPT.format(
                pred_diagnose=extracted_answer,
                gt_diagnose=gold_response,
            )

        try:
            judge_raw = await judge_rubric.judge(judge_prompt, completion, gold_response, state)
            is_correct = _parse_judge_result(str(judge_raw))
        except Exception:
            is_correct = False
            judge_raw = "Error during judge evaluation"

        # Store judge feedback in info
        info.setdefault("judge_feedback", []).append(
            {
                "is_correct": is_correct,
                "raw_judge": judge_raw,
            }
        )

        return 1.0 if is_correct else 0.0

    judge_rubric.add_reward_func(judge_rubric_reward, weight=1.0)

    patient_agent_base_url = judge_base_url if patient_agent_base_url is None else patient_agent_base_url
    patient_agent_api_key = api_key if patient_agent_api_key is None else patient_agent_api_key
    patient_agent_client = AsyncOpenAI(
        base_url=patient_agent_base_url,
        api_key=patient_agent_api_key,
        default_headers=default_headers,
    )

    env_kwargs = dict(kwargs)
    env_kwargs.pop("max_turns", None)

    if task == Task.ORACLE:
        return vf.SingleTurnEnv(
            eval_dataset=dataset,
            system_prompt=system_prompt,
            rubric=judge_rubric,
            parser=parser,
            **env_kwargs,
        )
    if task == Task.ONE_TURN:
        return MedRBenchOneTurnEnv(
            eval_dataset=dataset,
            system_prompt=system_prompt,
            rubric=judge_rubric,
            parser=parser,
            patient_agent_client=patient_agent_client,
            patient_agent_model=patient_agent_model,
            **env_kwargs,
        )
    return MedRBenchFreeTurnEnv(
        eval_dataset=dataset,
        system_prompt=system_prompt,
        rubric=judge_rubric,
        parser=parser,
        patient_agent_client=patient_agent_client,
        patient_agent_model=patient_agent_model,
        max_turns=max_turns,
        **env_kwargs,
    )
