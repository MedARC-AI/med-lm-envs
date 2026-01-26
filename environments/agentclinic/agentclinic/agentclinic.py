"""
AgentClinic Environment
"""

import json
import os
import re

from pathlib import Path
from typing import Any, Dict, List, Optional

import verifiers as vf
from datasets import Dataset
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI

from agentclinic.message_utils import extract_last_assistant_text
from agentclinic.prompts import (
    DatasetType,
    DoctorBias,
    JUDGE_PROMPT,
    NORMAL_READINGS,
    NEXT_TO_LAST_TURN_HINT,
    PatientBias,
    PROGRESS_TURN_HINT_TEMPLATE,
    FINAL_TURN_HINT,
    doctor_system_prompt,
    measurement_system_prompt,
    normalize_bias,
    patient_system_prompt,
)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _detect_dataset_type(cases: List[Dict[str, Any]]) -> DatasetType:
    if not cases:
        raise ValueError("Empty dataset")
    first_case = cases[0]
    if "OSCE_Examination" in first_case:
        return DatasetType.MEDQA
    if "image_url" in first_case and "answers" in first_case:
        return DatasetType.NEJM
    raise ValueError(f"Unknown dataset format. Keys: {list(first_case.keys())}")


def _resolve_dataset_path(dataset_path: Optional[str]) -> str:
    module_dir = Path(__file__).resolve().parent
    package_dir = module_dir.parent  # environments/agentclinic/
    filename = dataset_path or "agentclinic_medqa_extended.jsonl"

    for base in [Path.cwd(), module_dir, package_dir]:
        candidate = base / filename if not Path(filename).is_absolute() else Path(filename)
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Dataset not found: {filename}")


def _build_initial_question(objective: str) -> str:
    return (
        f"SYSTEM: Below is all of the information you have.\n{objective} "
        "\n\nRemember, you must discover their disease by asking them questions. You are also able to request tests or perform exams."
    )


def _turn_count(state: vf.State) -> int:
    # `verifiers==0.1.7.post0` uses `state["turn"]` (incremented after each model call).
    # Newer verifiers versions may use `state["trajectory"]`; support both to avoid subtle bugs.
    if isinstance(state.get("trajectory"), list):
        return len(state["trajectory"])
    try:
        return int(state.get("turn", 0) or 0)
    except Exception:
        return 0


_PATIENT_PREFIX_RE = re.compile(r"(?is)^\s*(\*\*)?\s*patient\s*:\s*")
_RESULTS_PREFIX_RE = re.compile(r"(?is)^\s*(\*\*)?\s*results\s*:\s*")


def _progress_checkpoints(max_turns: int) -> set[int]:
    """Return a small set of "used responses so far" checkpoints for progress banners."""
    if max_turns <= 0:
        return set()
    if max_turns < 6:
        return {max(1, max_turns // 2)}
    if max_turns < 12:
        return {max(1, max_turns // 3), max(1, (2 * max_turns) // 3)}
    return {max(1, max_turns // 4), max(1, max_turns // 2), max(1, (3 * max_turns) // 4)}


def _ensure_patient_prefix(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return "Patient:"
    if _PATIENT_PREFIX_RE.search(normalized):
        return normalized
    return f"Patient: {normalized}"


def _ensure_results_prefix(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized:
        return "RESULTS:"
    if _RESULTS_PREFIX_RE.search(normalized):
        return normalized
    return f"RESULTS: {normalized}"


class Scenario:
    def __init__(self, scenario_dict: Dict[str, Any]):
        osce = scenario_dict.get("OSCE_Examination", scenario_dict) or {}
        self.tests = osce.get("Test_Results", {}) or {}
        self.diagnosis = osce.get("Correct_Diagnosis", "") or ""
        self.patient_info = osce.get("Patient_Actor", {}) or {}
        self.examiner_info = osce.get("Objective_for_Doctor", "") or ""
        self.physical_exams = osce.get("Physical_Examination_Findings", {}) or {}

    def patient_information(self) -> Dict[str, Any]:
        return self.patient_info

    def examiner_information(self) -> str:
        return self.examiner_info

    def exam_information(self) -> Dict[str, Any]:
        exams = dict(self.physical_exams)
        exams["tests"] = self.tests
        return exams

    def diagnosis_information(self) -> str:
        return self.diagnosis


class NEJMScenario:
    def __init__(self, scenario_dict: Dict[str, Any]):
        self.question = scenario_dict.get("question", "")
        self.image_url = scenario_dict.get("image_url", "")
        answers = scenario_dict.get("answers", [])
        self.diagnosis = next((a["text"] for a in answers if a.get("correct")), "")
        self.patient_info = scenario_dict.get("patient_info", "")
        self.physical_exams = scenario_dict.get("physical_exams", "")

    def patient_information(self) -> Dict[str, Any]:
        return {"Description": self.patient_info, "Image_URL": self.image_url}

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> Dict[str, Any]:
        return {"Physical_Examination": self.physical_exams, "Image_URL": self.image_url}

    def diagnosis_information(self) -> str:
        return self.diagnosis


class PatientAgent:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict[str, Any],
        max_completion_tokens: int,
        bias: str | None,
    ):
        self.client = client
        self.model = model
        self.sampling_args = sampling_args
        self.max_completion_tokens = max_completion_tokens
        self.bias = bias
        self.agent_hist = ""
        self.symptoms: dict[str, Any] = {}

    def reset(self, patient_info: Dict[str, Any]):
        self.agent_hist = ""
        self.symptoms = patient_info

    def system_prompt(self) -> str:
        return patient_system_prompt(self.symptoms, self.bias)

    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"

    async def inference_patient(self, question: str) -> str:
        prompt = (
            f"\nHere is a history of your dialogue: {self.agent_hist}\n"
            f"Here was the doctor response: {question}\n"
            "Now please continue your dialogue\nPatient: "
        )
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                **self.sampling_args,
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[PatientAgent] Error: {e}")
            answer = ""
        if not answer:
            answer = "I'm not sure about that."
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer


class MeasurementAgent:
    def __init__(
        self,
        scenario_data: Dict[str, Any],
        client: AsyncOpenAI,
        model: str,
        sampling_args: dict[str, Any],
        max_completion_tokens: int,
    ):
        self.agent_hist = ""
        self.information = scenario_data
        self.client = client
        self.model = model
        self.sampling_args = sampling_args
        self.max_completion_tokens = max_completion_tokens

    def system_prompt(self) -> str:
        return measurement_system_prompt(self.information)

    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"

    async def inference_measurement(self, question: str) -> str:
        prompt = (
            f"\nHere is a history of the dialogue: {self.agent_hist}\n"
            f"Here was the doctor measurement request: {question}"
        )
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_completion_tokens,
                **self.sampling_args,
            )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[MeasurementAgent] Error: {e}")
            answer = ""
        if not answer:
            answer = NORMAL_READINGS
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer


class AgentClinicEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        scenarios: list[Scenario | NEJMScenario],
        max_turns: int,
        patient_client: AsyncOpenAI,
        patient_model: str,
        patient_sampling_args: dict[str, Any],
        measurement_client: AsyncOpenAI,
        measurement_model: str,
        measurement_sampling_args: dict[str, Any],
        aux_max_tokens: int,
        doctor_bias: str | None,
        patient_bias: str | None,
        dataset: Dataset,
        name: str,
        **kwargs: Any,
    ):
        system_prompt = doctor_system_prompt(
            max_turns=max_turns,
            doctor_bias=doctor_bias,
        )
        super().__init__(name=name, dataset=dataset, system_prompt=system_prompt, max_turns=max_turns, **kwargs)
        self._scenarios = scenarios
        self._patient_client = patient_client
        self._patient_model = patient_model
        self._patient_sampling_args = patient_sampling_args
        self._measurement_client = measurement_client
        self._measurement_model = measurement_model
        self._measurement_sampling_args = measurement_sampling_args
        self._aux_max_tokens = aux_max_tokens
        self._patient_bias = patient_bias

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        info = state.get("info", {})
        case_index = info.get("case_id", 0)
        scenario = self._scenarios[case_index]

        patient_agent = PatientAgent(
            client=self._patient_client,
            model=self._patient_model,
            sampling_args=self._patient_sampling_args,
            max_completion_tokens=self._aux_max_tokens,
            bias=self._patient_bias,
        )
        patient_agent.reset(scenario.patient_information())

        measurement_agent = MeasurementAgent(
            scenario_data=scenario.exam_information(),
            client=self._measurement_client,
            model=self._measurement_model,
            sampling_args=self._measurement_sampling_args,
            max_completion_tokens=self._aux_max_tokens,
        )

        state["case_index"] = case_index
        state["_patient_agent"] = patient_agent
        state["_measurement_agent"] = measurement_agent
        state["scenario"] = scenario
        state.setdefault("num_tests_requested", 0)
        state.setdefault("num_exams_requested", 0)
        return state

    async def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        if _turn_count(state) == 0:
            return False
        last_text = extract_last_assistant_text(messages)
        return "DIAGNOSIS READY" in (last_text or "").upper()

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any):
        patient_agent: PatientAgent = state["_patient_agent"]
        measurement_agent: MeasurementAgent = state["_measurement_agent"]

        doctor_dialogue = extract_last_assistant_text(messages)
        # Match original AgentClinic: trigger on substring and forward full doctor dialogue.
        upper_dialogue = (doctor_dialogue or "").upper()

        is_measurement_request = ("REQUEST TEST" in upper_dialogue) or ("PERFORM EXAM" in upper_dialogue)
        if "REQUEST TEST" in upper_dialogue:
            state["num_tests_requested"] = int(state.get("num_tests_requested", 0) or 0) + 1
        if "PERFORM EXAM" in upper_dialogue:
            state["num_exams_requested"] = int(state.get("num_exams_requested", 0) or 0) + 1

        if is_measurement_request:
            response_text = await measurement_agent.inference_measurement(doctor_dialogue)
            patient_agent.add_hist(response_text)
            response_text = _ensure_results_prefix(response_text)
        else:
            response_text = await patient_agent.inference_patient(doctor_dialogue)
            measurement_agent.add_hist(response_text)
            response_text = _ensure_patient_prefix(response_text)

        used = _turn_count(state)
        system_lines: list[str] = []

        # Countdown warnings (always when applicable). These take precedence over
        # generic progress banners to avoid redundant stacked system messages.
        if self.max_turns >= 3 and used == self.max_turns - 2:
            system_lines.append(NEXT_TO_LAST_TURN_HINT)
        elif self.max_turns >= 2 and used == self.max_turns - 1:
            system_lines.append(FINAL_TURN_HINT)
        # Periodic progress banner (subset only).
        elif used in _progress_checkpoints(self.max_turns):
            system_lines.append(PROGRESS_TURN_HINT_TEMPLATE.format(used=used, max_turns=self.max_turns))

        if system_lines:
            content = "\n".join(system_lines) + "\n\n" + response_text
        else:
            content = response_text

        return ([{"role": "user", "content": content}], state)


def load_environment(
    dataset_path: Optional[str] = None,
    dataset_type: DatasetType | str = DatasetType.MEDQA,
    max_turns: int = 20,
    aux_max_tokens: int = 500,
    doctor_bias: DoctorBias | str | None = None,
    patient_bias: PatientBias | str | None = None,
    patient_model: str = "gpt-5-mini",
    patient_base_url: Optional[str] = None,
    patient_api_key: Optional[str] = None,
    measurement_model: str = "gpt-5-mini",
    measurement_base_url: Optional[str] = None,
    measurement_api_key: Optional[str] = None,
    judge_model: str = "gpt-5-mini",
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    patient_reasoning_effort: str | None = None,
    measurement_reasoning_effort: str | None = None,
    patient_temperature: float | None = None,
    measurement_temperature: float | None = None,
    **kwargs: Any,
) -> vf.Environment:
    dataset_path = _resolve_dataset_path(dataset_path)
    cases = read_jsonl(dataset_path)
    if not cases:
        raise ValueError(f"No cases loaded from: {dataset_path}")

    if dataset_type is None:
        dataset_type = _detect_dataset_type(cases)

    dataset_type = DatasetType(dataset_type.lower()) if isinstance(dataset_type, str) else dataset_type
    doctor_bias = normalize_bias(doctor_bias, DoctorBias, "doctor")
    patient_bias = normalize_bias(patient_bias, PatientBias, "patient")

    scenarios: list[Scenario | NEJMScenario]
    if dataset_type == DatasetType.MEDQA:
        scenarios = [Scenario(c) for c in cases]
    else:
        scenarios = [NEJMScenario(c) for c in cases]

    records = []
    for i, scenario in enumerate(scenarios):
        objective = scenario.examiner_information()
        question = _build_initial_question(objective)
        info = {
            "gold": scenario.diagnosis_information(),
            "reference_response": scenario.diagnosis_information(),
            "case_id": i,
            "dataset_type": dataset_type,
        }
        records.append(
            {
                "question": question,
                "answer": scenario.diagnosis_information(),
                "task": f"agentclinic-{dataset_type}",
                "info": info,
            }
        )

    dataset = Dataset.from_list(records)

    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    judge_sampling_args, judge_default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)

    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=judge_default_headers)

    # Default helper agents to the judge credentials for convenience (MedRBench pattern),
    # but still fall back to OPENAI_API_KEY for backwards compatibility.
    patient_api_key = patient_api_key or api_key or os.environ.get("OPENAI_API_KEY")
    measurement_api_key = measurement_api_key or api_key or os.environ.get("OPENAI_API_KEY")

    patient_sampling_args, patient_headers = judge_sampling_args_and_headers(
        patient_model,
        patient_base_url,
        temperature=patient_temperature,
        reasoning_effort=patient_reasoning_effort,
    )

    measurement_sampling_args, measurement_headers = judge_sampling_args_and_headers(
        measurement_model,
        measurement_base_url,
        temperature=measurement_temperature,
        reasoning_effort=measurement_reasoning_effort,
    )

    patient_client = AsyncOpenAI(base_url=patient_base_url, api_key=patient_api_key, default_headers=patient_headers)
    measurement_client = AsyncOpenAI(
        base_url=measurement_base_url, api_key=measurement_api_key, default_headers=measurement_headers
    )

    # Match original AgentClinic moderator scoring: judge sees the doctor's final turn
    # (whatever the model emitted alongside "DIAGNOSIS READY"), not a separate
    # post-processed "diagnosis-only" string.
    parser = vf.Parser(extract_fn=lambda text: (text or "").strip())

    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        parser=parser,
        judge_sampling_args=judge_sampling_args,
    )

    async def diagnosis_reward_func(completion: vf.Messages, info: vf.Info, state: vf.State, **_kwargs: Any) -> float:
        info.setdefault("turns_used", _turn_count(state))
        info.setdefault("exam_requests", state.get("num_exams_requested", 0))
        info.setdefault("test_requests", state.get("num_tests_requested", 0))
        gold = str(info.get("reference_response") or info.get("gold") or "")
        response = parser.parse_answer(completion) or ""

        judge_text = await rubric.judge(JUDGE_PROMPT.format(answer=gold, response=response), completion, "", state)
        judge_text = judge_text.lower().strip()
        is_correct = "yes" in judge_text and "no" not in judge_text
        info.setdefault("judge_feedback", []).append({"raw_judge": judge_text, "is_correct": is_correct})
        return 1.0 if is_correct else 0.0

    rubric.add_reward_func(diagnosis_reward_func, weight=1.0)

    env_kwargs = dict(kwargs)
    env_kwargs.pop("max_turns", None)
    env_kwargs.pop("task_mode", None)

    env = AgentClinicEnv(
        scenarios=scenarios,
        max_turns=max_turns,
        patient_client=patient_client,
        patient_model=patient_model,
        patient_sampling_args=patient_sampling_args,
        measurement_client=measurement_client,
        measurement_model=measurement_model,
        measurement_sampling_args=measurement_sampling_args,
        aux_max_tokens=aux_max_tokens,
        doctor_bias=doctor_bias,
        patient_bias=patient_bias,
        dataset=dataset,
        name=f"AgentClinic-{dataset_type.upper()}",
        parser=parser,
        rubric=rubric,
        **env_kwargs,
    )
    return env
