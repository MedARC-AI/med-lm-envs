"""
AgentClinic Environment for Prime Intellect Verifiers
Reimplemented to match the original paper exactly.
Supports multiple LLM backends: OpenAI, Anthropic, Gemini, Mistral, vLLM
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import os
import urllib.request
import urllib.error

import verifiers as vf
from datasets import Dataset
from verifiers.utils.data_utils import THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

from llm_client import LLMClient


# ============================================================
#                    Utility Functions
# ============================================================

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ============================================================
#                    Agent Classes
# ============================================================

class PatientAgent:
    """
    Simulates patient responses using configured LLM backend.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.agent_hist = ""
        self.symptoms = {}

    def reset(self, patient_info: Dict[str, Any]):
        """Reset for new scenario """
        self.agent_hist = ""
        self.symptoms = patient_info

    def system_prompt(self) -> str:
        base = (
            "You are a patient in a clinic who only responds in the form of dialogue. "
            "You are being inspected by a doctor who will ask you questions and will "
            "perform exams on you in order to understand your disease. "
            "Your answer will only be 1-3 sentences in length."
        )
        symptoms = (
            f"\n\nBelow is all of your information. {json.dumps(self.symptoms, ensure_ascii=False)}. "
            "\n\nRemember, you must not reveal your disease explicitly but may only convey "
            "the symptoms you have in the form of dialogue if you are asked."
        )
        return base + symptoms

    def inference_patient(self, question: str) -> str:
        prompt = (
            f"\nHere is a history of your dialogue: {self.agent_hist}\n"
            f"Here was the doctor response: {question}\n"
            "Now please continue your dialogue\nPatient: "
        )

        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt}
        ]

        answer = self.llm_client.generate(messages)
        if not answer:
            answer = "I'm not sure about that."

        # Update history like the paper
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer


class MeasurementAgent:
    """
    Returns test results from the scenario data using configured LLM backend
    """

    def __init__(self, scenario_data: Dict[str, Any], llm_client: LLMClient):
        self.agent_hist = ""
        self.information = scenario_data
        self.llm_client = llm_client

    def system_prompt(self) -> str:
        base = "You are a measurement reader who responds with medical test results. Please respond in the format \"RESULTS: [results here]\""
        presentation = f"\n\nBelow is all of the information you have. {json.dumps(self.information, ensure_ascii=False)}. \n\nIf the requested results are not in your data then you can respond with NORMAL READINGS."
        return base + presentation

    def inference_measurement(self, question: str) -> str:
        prompt = (
            f"\nHere is a history of the dialogue: {self.agent_hist}\n"
            f"Here was the doctor measurement request: {question}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": prompt}
        ]

        answer = self.llm_client.generate(messages)
        if not answer:
            answer = "RESULTS: NORMAL READINGS"

        # Update history 
        self.agent_hist += question + "\n\n" + answer + "\n\n"
        return answer


# ============================================================
#                    Scenario Classes
# ============================================================

class Scenario:
    """Scenario wrapper for MedQA Extended cases"""

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
    """Scenario wrapper for NEJM cases (image-based clinical cases)."""

    def __init__(self, scenario_dict: Dict[str, Any]):
        self.question = scenario_dict.get("question", "")
        self.image_url = scenario_dict.get("image_url", "")

        # Extract correct answer from answers array
        answers = scenario_dict.get("answers", [])
        self.diagnosis = next((a["text"] for a in answers if a.get("correct")), "")

        self.patient_info = scenario_dict.get("patient_info", "")
        self.physical_exams = scenario_dict.get("physical_exams", "")

    def patient_information(self) -> Dict[str, Any]:
        return {
            "Description": self.patient_info,
            "Image_URL": self.image_url
        }

    def examiner_information(self) -> str:
        return "What is the most likely diagnosis?"

    def exam_information(self) -> Dict[str, Any]:
        return {
            "Physical_Examination": self.physical_exams,
            "Image_URL": self.image_url
        }

    def diagnosis_information(self) -> str:
        return self.diagnosis


# ============================================================
#                    Doctor Prompts
# ============================================================

def _compose_doctor_system(use_think: bool, max_infs: int, current_infs: int) -> str:
    """Compose doctor system prompt with turn info."""
    base = (
        f"You are a doctor named Dr. Agent who only responds in the form of dialogue. "
        f"You are inspecting a patient who you will ask questions in order to understand their disease. "
        f"You are only allowed to ask {max_infs} questions total before you must make a decision. "
        f"You have asked {current_infs} questions so far. "
        "You can request test results using the format \"REQUEST TEST: [test]\". "
        "For example, \"REQUEST TEST: Chest_X-Ray\". "
        "Your dialogue will only be 1-3 sentences in length. "
        "Once you have decided to make a diagnosis please type \"DIAGNOSIS READY: [diagnosis here]\""
    )

    if use_think:
        return THINK_BOXED_SYSTEM_PROMPT + "\n\n" + base
    return base


# ============================================================
#                    Verifiers Environment
# ============================================================

class AgentClinicEnv(vf.MultiTurnEnv):
    """
    AgentClinic environment matching the paper's main() loop.
    Doctor is the evaluated model, Patient and Measurement are helper agents.
    """

    def __init__(
        self,
        cases: List[Dict[str, Any]],
        max_turns: int = 20,
        use_think: bool = False,
        name: str = "AgentClinic",
        # Patient agent config
        patient_model: str = "gpt-4o-mini",
        patient_api_base: Optional[str] = None,
        patient_api_key: Optional[str] = None,
        patient_backend: str = "auto",
        # Measurement agent config
        measurement_model: str = "gpt-4o-mini",
        measurement_api_base: Optional[str] = None,
        measurement_api_key: Optional[str] = None,
        measurement_backend: str = "auto",
        # Moderator/judge config
        moderator_model: str = "gpt-4o-mini",
        moderator_api_base: Optional[str] = None,
        moderator_api_key: Optional[str] = None,
        moderator_backend: str = "auto",
    ):
        """
        Initialize AgentClinic environment.

        Args:
            cases: List of case dicts
            max_turns: Maximum conversation turns
            use_think: Whether to use chain-of-thought prompting
            name: Environment name
            patient_model: Model name for Patient agent
            patient_api_base: API base URL for patient agent
            patient_api_key: API key for patient agent
            patient_backend: Backend type for patient agent
            measurement_model: Model name for Measurement agent
            measurement_api_base: API base URL for measurement agent
            measurement_api_key: API key for measurement agent
            measurement_backend: Backend type for measurement agent
            moderator_model: Model name for Moderator/Judge
            moderator_api_base: API base URL for moderator
            moderator_api_key: API key for moderator
            moderator_backend: Backend type for moderator
        """
        self._raw_cases = cases
        self._scenarios = [Scenario(c) for c in cases]
        self._max_turns = max_turns
        self._use_think = use_think

        # Store patient agent LLM configuration
        self._patient_model = patient_model
        self._patient_api_base = patient_api_base
        self._patient_api_key = patient_api_key
        self._patient_backend = patient_backend

        # Store measurement agent LLM configuration
        self._measurement_model = measurement_model
        self._measurement_api_base = measurement_api_base
        self._measurement_api_key = measurement_api_key
        self._measurement_backend = measurement_backend

        # Store moderator LLM configuration
        self._moderator_model = moderator_model
        self._moderator_api_base = moderator_api_base
        self._moderator_api_key = moderator_api_key
        self._moderator_backend = moderator_backend

        # Create moderator client for scoring
        self._moderator_client = LLMClient(
            model=moderator_model,
            api_base=moderator_api_base,
            api_key=moderator_api_key,
            backend=moderator_backend,
            temperature=0.0,
            max_tokens=10,
        )

        # Build dataset for verifiers
        prompts = []
        infos = []

        for i, scenario in enumerate(self._scenarios):
            # Initial doctor prompt with objective
            objective = scenario.examiner_information()
            initial_prompt = [
                {"role": "system", "content": _compose_doctor_system(use_think, max_turns, 0)},
                {"role": "user", "content": f"Below is all of the information you have. {objective}. \n\nRemember, you must discover their disease by asking them questions. You are also able to provide exams."}
            ]
            prompts.append(initial_prompt)
            infos.append({
                "gold": scenario.diagnosis_information(),
                "case_id": i
            })

        dataset = Dataset.from_dict({
            "id": list(range(len(cases))),
            "prompt": prompts,
            "info": infos
        })

        super().__init__(name=name, dataset=dataset)

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        """
        Override MultiTurnEnv.setup_state to initialize agents for each case.
        This is called by the rollout() method with the initial state.
        """
        # Get the case index from info (passed through from dataset)
        info = state.get("info", {})
        case_index = info.get("case_id", 0)

        scenario = self._scenarios[case_index]

        # Create separate LLM clients for each agent
        patient_client = LLMClient(
            model=self._patient_model,
            api_base=self._patient_api_base,
            api_key=self._patient_api_key,
            backend=self._patient_backend,
            temperature=0.05,
            max_tokens=200,
        )

        measurement_client = LLMClient(
            model=self._measurement_model,
            api_base=self._measurement_api_base,
            api_key=self._measurement_api_key,
            backend=self._measurement_backend,
            temperature=0.05,
            max_tokens=200,
        )

        # Create fresh agents for this case (like paper's initialization)
        patient_agent = PatientAgent(patient_client)
        patient_info = scenario.patient_information()
        patient_agent.reset(patient_info)

        measurement_agent = MeasurementAgent(scenario.exam_information(), measurement_client)

        # Add our agents to the state
        state["case_index"] = case_index
        state["_patient_agent"] = patient_agent
        state["_measurement_agent"] = measurement_agent
        state["scenario"] = scenario

        return state

    async def is_completed(self, messages: vf.Messages, state: vf.State, info: Dict[str, Any] | None = None) -> bool:
        """Check if conversation is complete."""
        turns = state.get("turn", 0)

        # Check last assistant message for DIAGNOSIS READY (like paper)
        last_assistant = None
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                last_assistant = m.get("content", "")
                break

        if last_assistant and "DIAGNOSIS READY" in last_assistant:
            return True

        # Max turns reached
        if turns >= self._max_turns:
            return True

        return False

    async def env_response(self, messages: vf.Messages, state: vf.State, info: Dict[str, Any] | None = None):
        """
        Generate environment response - either patient reply or test results.
        Matches the paper's main loop logic.
        """
        new_state = dict(state)
        new_state["turn"] = state.get("turn", 0) + 1

        # Get agents from state
        patient_agent = new_state["_patient_agent"]
        measurement_agent = new_state["_measurement_agent"]

        # Get last doctor message
        doctor_dialogue = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                doctor_dialogue = m.get("content", "")
                break

        # Final turn nudge
        if new_state["turn"] >= self._max_turns:
            return (
                [{"role": "user", "content": "This is the final question. Please provide a diagnosis.\nDIAGNOSIS READY: "}],
                new_state
            )

        # Check if doctor requested test (like paper's main loop)
        if "REQUEST TEST" in doctor_dialogue:
            # Measurement agent responds
            result = measurement_agent.inference_measurement(doctor_dialogue)
            patient_agent.agent_hist += result + "\n\n"  # Add to patient history too
            return ([{"role": "user", "content": result}], new_state)

        # Otherwise, patient responds
        pi_dialogue = patient_agent.inference_patient(doctor_dialogue)
        measurement_agent.agent_hist += pi_dialogue + "\n\n"  # Add to measurement history too

        return ([{"role": "user", "content": pi_dialogue}], new_state)


# ============================================================
#                    Scoring (LLM Judge)
# ============================================================

def compare_results_llm(
    prediction: str,
    gold: str,
    moderator_client: Optional[LLMClient] = None
) -> float:
    """
    LLM judge matching paper's compare_results().
    Uses configurable moderator LLM.
    """
    # Fallback to default OpenAI if no client provided
    if moderator_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return 0.0
        moderator_client = LLMClient(
            model="gpt-4o-mini",
            backend="openai",
            api_key=api_key,
            temperature=0.0,
            max_tokens=10
        )

    # Extract diagnosis from "DIAGNOSIS READY: [diagnosis]" format
    if "DIAGNOSIS READY:" in prediction:
        prediction = prediction.split("DIAGNOSIS READY:")[-1].strip()

    # Also handle \boxed{} format for verifiers compatibility
    try:
        boxed = extract_boxed_answer(prediction)
        if boxed:
            prediction = boxed
    except:
        pass

    system = "You are responsible for determining if the correct diagnosis and the doctor diagnosis are the same disease. Please respond only with Yes or No. Nothing else."
    user = f"\nHere is the correct diagnosis: {gold}\nHere was the doctor dialogue: {prediction}\nAre these the same?"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]

    try:
        answer = moderator_client.generate(messages).lower()
        return 1.0 if "yes" in answer else 0.0
    except:
        return 0.0


class AccuracyReward:
    """Reward function class that holds reference to moderator client."""

    # Add __name__ for verifiers framework compatibility
    __name__ = "accuracy_reward"

    def __init__(self, moderator_client: LLMClient):
        self.moderator_client = moderator_client

    def __call__(self, prompt: str, completion: str, answer: str, state: Dict[str, Any]) -> float:
        """Reward function for verifiers."""
        gold = (state.get("info") or {}).get("gold", "") or answer

        # Extract final diagnosis from completion
        if isinstance(completion, list):
            completion_text = ""
            for msg in reversed(completion):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    completion_text = msg.get("content", "")
                    break
        else:
            completion_text = str(completion)

        return compare_results_llm(completion_text, gold, self.moderator_client)


# ============================================================
#                    Public Loaders
# ============================================================

def _detect_dataset_type(cases: List[Dict[str, Any]]) -> str:
    """Auto-detect dataset type from structure."""
    if not cases:
        raise ValueError("Empty dataset")

    first_case = cases[0]
    if "OSCE_Examination" in first_case:
        return "medqa"
    elif "image_url" in first_case and "answers" in first_case:
        return "nejm"
    else:
        raise ValueError(f"Unknown dataset format. Keys: {list(first_case.keys())}")


def load_environment(
    dataset_path: Optional[str] = None,
    dataset_type: Optional[str] = None,
    use_think: bool = False,
    max_turns: int = 20,
    # Patient agent config
    patient_model: str = "gpt-4o-mini",
    patient_api_base: Optional[str] = None,
    patient_api_key: Optional[str] = None,
    patient_backend: str = "auto",
    # Measurement agent config
    measurement_model: str = "gpt-4o-mini",
    measurement_api_base: Optional[str] = None,
    measurement_api_key: Optional[str] = None,
    measurement_backend: str = "auto",
    # Moderator config
    moderator_model: str = "gpt-4o-mini",
    moderator_api_base: Optional[str] = None,
    moderator_api_key: Optional[str] = None,
    moderator_backend: str = "auto",
    **kwargs,
) -> vf.Environment:
    """
    Load the AgentClinic environment.

    Args:
        dataset_path: Path to JSONL dataset file (optional)
        dataset_type: Dataset type - 'medqa' or 'nejm' (auto-detected if None)
        use_think: Whether to use chain-of-thought prompting
        max_turns: Maximum conversation turns
        patient_model: Model name for Patient agent
        patient_api_base: API base URL for patient agent
        patient_api_key: API key for patient agent
        patient_backend: Backend type for patient agent
        measurement_model: Model name for Measurement agent
        measurement_api_base: API base URL for measurement agent
        measurement_api_key: API key for measurement agent
        measurement_backend: Backend type for measurement agent
        moderator_model: Model name for Moderator/Judge
        moderator_api_base: API base URL for moderator
        moderator_api_key: API key for moderator
        moderator_backend: Backend type for moderator
        **kwargs: Additional arguments passed to AgentClinicEnv

    Returns:
        AgentClinic environment instance
    """
    # Find dataset file
    if dataset_path:
        # User specified a path via --env-args
        # Check if it's an absolute path
        if os.path.isabs(dataset_path):
            found = dataset_path
        else:
            # Try relative to current working directory first
            cwd_path = os.path.join(os.getcwd(), dataset_path)
            if os.path.exists(cwd_path):
                found = cwd_path
            else:
                # Try relative to this module's directory
                module_path = os.path.join(os.path.dirname(__file__), dataset_path)
                if os.path.exists(module_path):
                    found = module_path
                else:
                    raise FileNotFoundError(
                        f"Dataset not found: {dataset_path}\n"
                        f"Tried: {cwd_path}, {module_path}"
                    )

        if not os.path.exists(found):
            raise FileNotFoundError(f"Dataset not found: {found}")
    else:
        # Default to MedQA Extended
        found = os.path.join(os.path.dirname(__file__), "agentclinic_medqa_extended.jsonl")
        if not os.path.exists(found):
            raise FileNotFoundError(
                f"Default MedQA dataset not found: {found}\n"
                "Pass dataset_path parameter via --env-args to specify a different dataset."
            )

    cases = read_jsonl(found)
    if not cases:
        raise ValueError(f"No cases loaded from: {found}")

    # Auto-detect dataset type if not specified
    if dataset_type is None:
        dataset_type = _detect_dataset_type(cases)

    dataset_type = dataset_type.lower()
    print(f"Loaded {len(cases)} cases from {found} (type: {dataset_type})")

    # Create scenarios based on dataset type
    if dataset_type == "medqa":
        scenarios = [Scenario(c) for c in cases]
    elif dataset_type == "nejm":
        scenarios = [NEJMScenario(c) for c in cases]
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Use 'medqa' or 'nejm'")

    # Create environment
    env = AgentClinicEnv(
        cases=cases,
        max_turns=max_turns,
        use_think=use_think,
        name=f"AgentClinic-{dataset_type.upper()}",
        patient_model=patient_model,
        patient_api_base=patient_api_base,
        patient_api_key=patient_api_key,
        patient_backend=patient_backend,
        measurement_model=measurement_model,
        measurement_api_base=measurement_api_base,
        measurement_api_key=measurement_api_key,
        measurement_backend=measurement_backend,
        moderator_model=moderator_model,
        moderator_api_base=moderator_api_base,
        moderator_api_key=moderator_api_key,
        moderator_backend=moderator_backend,
        **kwargs,
    )

    # Override scenarios with typed versions
    env._scenarios = scenarios

    # Set rubric with moderator client from environment
    accuracy_reward_func = AccuracyReward(env._moderator_client)
    env.rubric = vf.Rubric(
        funcs=[accuracy_reward_func],
        names=["accuracy_reward"]
    )

    return env


def load_medqa_environment(**kwargs) -> vf.Environment:
    """Load MedQA Extended benchmark."""
    dataset_path = kwargs.pop("dataset_path", None)
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "agentclinic_medqa_extended.jsonl"
        )
    return load_environment(dataset_path=dataset_path, dataset_type="medqa", **kwargs)


def load_nejm_environment(**kwargs) -> vf.Environment:
    """Load NEJM Extended benchmark (image-based cases)."""
    dataset_path = kwargs.pop("dataset_path", None)
    if dataset_path is None:
        dataset_path = os.path.join(
            os.path.dirname(__file__),
            "agentclinic_nejm_extended.jsonl"
        )
    return load_environment(dataset_path=dataset_path, dataset_type="nejm", **kwargs)


def get_environment(*args, **kwargs) -> vf.Environment:
    """Alias for load_environment."""
    return load_environment(*args, **kwargs)
