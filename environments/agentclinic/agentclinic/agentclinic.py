"""
AgentClinic Environment
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import verifiers as vf
from datasets import Dataset
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.utils.data_utils import extract_boxed_answer

from agentclinic.message_utils import extract_last_assistant_text
from agentclinic.prompts import (
    DOCTOR_BIASES,
    FINAL_TURN_HINT,
    NORMAL_READINGS,
    PATIENT_BIASES,
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


def _detect_dataset_type(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        raise ValueError("Empty dataset")
    first_case = cases[0]
    if "OSCE_Examination" in first_case:
        return "medqa"
    if "image_url" in first_case and "answers" in first_case:
        return "nejm"
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


def _has_diagnosis_ready(text: str) -> bool:
    return re.search(r"diagnosis\s*ready\s*[:\-]?", text, re.IGNORECASE) is not None


_TEST_REQUEST_RE = re.compile(r"(?im)^\s*REQUEST\s+TEST\s*:\s*(.+?)\s*$")


def _extract_test_request(text: str) -> str | None:
    if not text:
        return None
    match = _TEST_REQUEST_RE.search(text)
    if not match:
        return None
    requested = match.group(1).strip()
    return requested or None


def _extract_diagnosis_text(text: str, use_think: bool) -> str:
    if not text:
        return ""
    match = re.search(r"diagnosis\s*ready\s*[:\-]?\s*", text, re.IGNORECASE)
    if match:
        text = text[match.end() :]
    if use_think:
        try:
            boxed = extract_boxed_answer(text)
            if boxed:
                text = boxed
        except Exception:
            pass
    return text.strip()


def _build_initial_question(objective: str, asked_so_far: int) -> str:
    return (
        f"Below is all of the information you have. {objective}. "
        "\n\nRemember, you must discover their disease by asking them questions. You are also able to provide exams."
        f"\n\nYou have asked {asked_so_far} questions so far."
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
        temperature: float,
        max_tokens: int,
        bias: str | None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
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
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
        temperature: float,
        max_tokens: int,
    ):
        self.agent_hist = ""
        self.information = scenario_data
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

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
                temperature=self.temperature,
                max_tokens=self.max_tokens,
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
        use_think: bool,
        patient_client: AsyncOpenAI,
        patient_model: str,
        patient_temperature: float,
        measurement_client: AsyncOpenAI,
        measurement_model: str,
        measurement_temperature: float,
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
            use_think=use_think,
        )
        super().__init__(name=name, dataset=dataset, system_prompt=system_prompt, max_turns=max_turns, **kwargs)
        self._scenarios = scenarios
        self._use_think = use_think
        self._patient_client = patient_client
        self._patient_model = patient_model
        self._patient_temperature = patient_temperature
        self._measurement_client = measurement_client
        self._measurement_model = measurement_model
        self._measurement_temperature = measurement_temperature
        self._aux_max_tokens = aux_max_tokens
        self._patient_bias = patient_bias

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        info = state.get("info", {})
        case_index = info.get("case_id", 0)
        scenario = self._scenarios[case_index]

        patient_agent = PatientAgent(
            client=self._patient_client,
            model=self._patient_model,
            temperature=self._patient_temperature,
            max_tokens=self._aux_max_tokens,
            bias=self._patient_bias,
        )
        patient_agent.reset(scenario.patient_information())

        measurement_agent = MeasurementAgent(
            scenario_data=scenario.exam_information(),
            client=self._measurement_client,
            model=self._measurement_model,
            temperature=self._measurement_temperature,
            max_tokens=self._aux_max_tokens,
        )

        state["case_index"] = case_index
        state["_patient_agent"] = patient_agent
        state["_measurement_agent"] = measurement_agent
        state["scenario"] = scenario
        return state

    async def is_completed(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> bool:
        if await super().is_completed(messages, state, **kwargs):
            return True
        if _turn_count(state) == 0:
            return False
        last_text = extract_last_assistant_text(messages)
        return _has_diagnosis_ready(last_text)

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any):
        patient_agent: PatientAgent = state["_patient_agent"]
        measurement_agent: MeasurementAgent = state["_measurement_agent"]

        doctor_dialogue = extract_last_assistant_text(messages)
        requested_test = _extract_test_request(doctor_dialogue)

        if requested_test is not None:
            response_text = await measurement_agent.inference_measurement(doctor_dialogue)
            patient_agent.add_hist(response_text)
        else:
            response_text = await patient_agent.inference_patient(doctor_dialogue)
            measurement_agent.add_hist(response_text)

        include_final_hint = _turn_count(state) == self.max_turns - 1
        response_with_hint = response_text
        if include_final_hint:
            response_with_hint = response_with_hint + "\n" + FINAL_TURN_HINT

        asked_so_far = _turn_count(state)
        response_with_hint = response_with_hint + f"\n\nYou have asked {asked_so_far} questions so far."

        return ([{"role": "user", "content": response_with_hint}], state)


JUDGE_PROMPT = (
    "You are responsible for determining if the corrent diagnosis and the doctor diagnosis are the same disease. "
    "Please respond only with Yes or No. Nothing else.\n\n"
    "Here is the correct diagnosis: {answer}\n Here was the doctor dialogue: {response}\nAre these the same?"
)


def load_environment(
    dataset_path: Optional[str] = None,
    dataset_type: Optional[str] = None,
    task_mode: str = "free_turn",
    use_think: bool = False,
    max_turns: int = 20,
    patient_model: str = "gpt-4o-mini",
    patient_base_url: Optional[str] = None,
    patient_api_key: Optional[str] = None,
    measurement_model: str = "gpt-4o-mini",
    measurement_base_url: Optional[str] = None,
    measurement_api_key: Optional[str] = None,
    patient_temperature: float = 0.05,
    measurement_temperature: float = 0.05,
    aux_max_tokens: int = 200,
    doctor_bias: Optional[str] = None,
    patient_bias: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    judge_timeout_s: Optional[float] = None,
    **kwargs: Any,
) -> vf.Environment:
    dataset_path = _resolve_dataset_path(dataset_path)
    cases = read_jsonl(dataset_path)
    if not cases:
        raise ValueError(f"No cases loaded from: {dataset_path}")

    if dataset_type is None:
        dataset_type = _detect_dataset_type(cases)
    dataset_type = dataset_type.lower()
    if dataset_type not in {"medqa", "nejm"}:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'medqa' or 'nejm'")

    doctor_bias = normalize_bias(doctor_bias, DOCTOR_BIASES, "doctor")
    patient_bias = normalize_bias(patient_bias, PATIENT_BIASES, "patient")

    task_mode = task_mode.lower()
    if task_mode not in {"free_turn", "oracle"}:
        raise ValueError("task_mode must be 'free_turn' or 'oracle'")

    scenarios: list[Scenario | NEJMScenario]
    if dataset_type == "medqa":
        scenarios = [Scenario(c) for c in cases]
    else:
        scenarios = [NEJMScenario(c) for c in cases]

    records = []
    for i, scenario in enumerate(scenarios):
        objective = scenario.examiner_information()
        question = _build_initial_question(objective, 0)
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
    judge_sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    if judge_timeout_s is not None:
        judge_sampling_args = dict(judge_sampling_args or {})
        judge_sampling_args["timeout"] = judge_timeout_s

    client_cache: dict[tuple[Optional[str], Optional[str], tuple[tuple[str, str], ...]], AsyncOpenAI] = {}

    def get_client(base_url: Optional[str], key: Optional[str], headers: Optional[dict[str, str]]) -> AsyncOpenAI:
        headers_tuple = tuple(sorted((headers or {}).items()))
        cache_key = (base_url, key, headers_tuple)
        if cache_key not in client_cache:
            client_cache[cache_key] = AsyncOpenAI(base_url=base_url, api_key=key, default_headers=headers)
        return client_cache[cache_key]

    judge_client = get_client(judge_base_url, api_key, default_headers)

    # Default helper agents to the judge credentials for convenience (MedRBench pattern),
    # but still fall back to OPENAI_API_KEY for backwards compatibility.
    patient_api_key = patient_api_key or api_key or os.environ.get("OPENAI_API_KEY")
    measurement_api_key = measurement_api_key or api_key or os.environ.get("OPENAI_API_KEY")

    patient_headers = default_headers if patient_base_url == judge_base_url else None
    measurement_headers = default_headers if measurement_base_url == judge_base_url else None
    patient_client = get_client(patient_base_url, patient_api_key, patient_headers)
    measurement_client = get_client(measurement_base_url, measurement_api_key, measurement_headers)

    parser = vf.Parser(extract_fn=lambda text: _extract_diagnosis_text(text, use_think))

    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
        parser=parser,
        judge_sampling_args=judge_sampling_args,
    )

    async def diagnosis_reward_func(completion: vf.Messages, info: vf.Info, state: vf.State, **_kwargs: Any) -> float:
        gold = str(info.get("reference_response") or info.get("gold") or "")
        try:
            judge_text = await rubric.judge("", completion, gold, state)
        except Exception:
            judge_text = "Error during judge evaluation"
        is_correct = judge_text.lower().strip().startswith("yes")
        info.setdefault("judge_feedback", []).append({"raw_judge": judge_text, "is_correct": is_correct})
        return 1.0 if is_correct else 0.0

    rubric.add_reward_func(diagnosis_reward_func, weight=1.0)

    env_kwargs = dict(kwargs)
    env_kwargs.pop("max_turns", None)

    if task_mode == "oracle":
        oracle_records = []
        for i, scenario in enumerate(scenarios):
            full_info = {
                "patient_info": scenario.patient_information(),
                "physical_exams": scenario.exam_information(),
            }
            question = (
                "Below is all of the information you have. "
                f"{scenario.examiner_information()}\n\n"
                f"{json.dumps(full_info, ensure_ascii=False)}"
            )
            info = {
                "gold": scenario.diagnosis_information(),
                "reference_response": scenario.diagnosis_information(),
                "case_id": i,
                "dataset_type": dataset_type,
            }
            oracle_records.append(
                {
                    "question": question,
                    "answer": scenario.diagnosis_information(),
                    "task": f"agentclinic-{dataset_type}-oracle",
                    "info": info,
                }
            )
        oracle_dataset = Dataset.from_list(oracle_records)
        system_prompt = doctor_system_prompt(max_turns=1, doctor_bias=doctor_bias, use_think=use_think)
        return vf.SingleTurnEnv(
            eval_dataset=oracle_dataset,
            system_prompt=system_prompt,
            rubric=rubric,
            parser=parser,
            **env_kwargs,
        )

    env = AgentClinicEnv(
        scenarios=scenarios,
        max_turns=max_turns,
        use_think=use_think,
        patient_client=patient_client,
        patient_model=patient_model,
        patient_temperature=patient_temperature,
        measurement_client=measurement_client,
        measurement_model=measurement_model,
        measurement_temperature=measurement_temperature,
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


def load_medqa_environment(**kwargs) -> vf.Environment:
    dataset_path = kwargs.pop("dataset_path", None)
    if dataset_path is None:
        dataset_path = str(Path(__file__).resolve().parent / "agentclinic_medqa_extended.jsonl")
    return load_environment(dataset_path=dataset_path, dataset_type="medqa", **kwargs)


def load_nejm_environment(**kwargs) -> vf.Environment:
    dataset_path = kwargs.pop("dataset_path", None)
    if dataset_path is None:
        dataset_path = str(Path(__file__).resolve().parent / "agentclinic_nejm_extended.jsonl")
    return load_environment(dataset_path=dataset_path, dataset_type="nejm", **kwargs)


def get_environment(*args: Any, **kwargs: Any) -> vf.Environment:
    return load_environment(*args, **kwargs)
