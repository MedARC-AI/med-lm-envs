import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import verifiers as vf
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer
from verifiers.parsers.think_parser import ThinkParser
from datasets import Dataset
import re




def _normalize_text(value: Optional[str]) -> str:
    """Normalize text for comparison."""
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"Warning: Skipping malformed JSONL line: {line[:200]} ({exc})")
    return rows


class AgentClinicMedqaMultiTurn(vf.MultiTurnEnv):
    """ AgentClinic environment with enhanced multiturn capabilities."""
    
    def __init__(
        self,
        cases: List[Dict[str, Any]],
        max_turns: int = 10,
        use_think: bool = False,
        name: str = "AgentClinicMedqa",
    ) -> None:
        self._cases_raw = cases
        self._cases = [_extract_case_fields(x) for x in cases]
        self._max_turns = max_turns
        self._use_think = use_think

        system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT

        num = len(self._cases)
        infos: List[Dict[str, Any]] = []
        prompts: List[List[Dict[str, str]]] = []
        for i in range(num):
            fields_i = self._cases[i]
            info = {
                "gold": fields_i.get("correct", ""),
                "kind": fields_i.get("kind"),
                "answers": fields_i.get("answers"),
            }
            infos.append(info)
            intro = _format_patient_intro(fields_i)
            prompts.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intro}
            ])
        
        minimal_dataset = Dataset.from_dict({
            "id": list(range(num)),
            "prompt": prompts,
            "info": infos,
        })
        super().__init__(name=name, dataset=minimal_dataset)

    async def is_completed(self, messages: vf.Messages, state: vf.State, info: Dict[str, Any] | None = None) -> bool:
        """Check if conversation is completed."""
        turns = state.get("turn", 0)
        print(f"DEBUG: is_completed called with {len(messages)} messages, turn {turns}")
        
        if turns >= self._max_turns:
            print("DEBUG: Max turns reached")
            return True
        if not messages:
            print("DEBUG: No messages")
            return False
        
        # Check if the last message contains a boxed answer
        last_message = messages[-1]
        if isinstance(last_message, dict):
            content = last_message.get("content", "")
        else:
            content = str(last_message)
            
        print(f"DEBUG: Last message content: '{content[:100]}...'")
        
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", content)
        if boxed_match:
            boxed_content = boxed_match.group(1).strip()
            # Only consider it a real answer if it's not just "..." or instruction text
            if (boxed_content and 
                boxed_content != "..." and 
                not content.strip().startswith("Instructions:") and
                len(boxed_content) > 2):
                print(f"DEBUG: Found boxed answer: '{boxed_content}', completing")
                return True
        return False

    async def env_response(self, messages: vf.Messages, state: vf.State, info: Dict[str, Any] | None = None) -> Tuple[vf.Messages, vf.State]:
        """Generate environment response based on agent message."""
        idx = state.get("case_index", 0)
        fields = self._cases[idx]
        turn = state.get("turn", 0)
        new_state = dict(state)
        new_state["turn"] = turn + 1

        if turn == 0:
            intro = _format_patient_intro(fields)
            return ([{"role": "user", "content": intro}], new_state)

        agent_msg = ""
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") in ("assistant", "tool"):
                agent_msg = str(m.get("content", ""))
                break
        agent_lower = agent_msg.lower()
        reveals: List[str] = []

        if any(k in agent_lower for k in ["physical exam", "examination", "exam", "vital signs", "neurological"]):
            reveals.append(_format_section("Physical examination", fields.get("physical")))
        if any(k in agent_lower for k in ["test", "labs", "imaging", "x-ray", "ct", "mri", "blood", "biopsy"]):
            reveals.append(_format_section("Tests and results", fields.get("tests")))
        if any(k in agent_lower for k in ["symptom", "complaint", "history"]):
            symptoms = fields.get("symptoms", {})
            if symptoms:
                reveals.append(_format_section("Detailed symptoms", symptoms))
        if any(k in agent_lower for k in ["demographic", "age", "gender"]):
            demo = fields.get("demographics", "")
            if demo:
                reveals.append(f"Demographics: {demo}")
        if any(k in agent_lower for k in ["past medical", "medical history", "previous"]):
            past_med = fields.get("past_medical", "")
            if past_med:
                reveals.append(f"Past Medical History: {past_med}")
        if any(k in agent_lower for k in ["social", "smoking", "drinking", "occupation"]):
            social = fields.get("social", "")
            if social:
                reveals.append(f"Social History: {social}")
        if any(k in agent_lower for k in ["review", "systems", "other symptoms"]):
            review = fields.get("review_systems", "")
            if review:
                reveals.append(f"Review of Systems: {review}")
        if fields.get("kind") == "NEJM" and ("option" in agent_lower or "choice" in agent_lower):
            options_lines: List[str] = []
            for i, a in enumerate(fields.get("answers", []), start=1):
                options_lines.append(f"{i}. {a.get('text', '')}")
            if options_lines:
                reveals.append("Answer options:\n" + "\n".join(options_lines))
        if not reveals:
            reveals.append("You may request 'physical exam', 'tests', 'symptoms', 'demographics', 'past medical history', 'social history', or 'review of systems'. Provide your final answer inside \\boxed{...} when ready.")

        return ([{"role": "user", "content": "\n\n".join(reveals)}], new_state)

    def get_num_examples(self) -> int:
        """Get number of examples in the dataset."""
        return len(self._cases)

    def get_example_info(self, index: int) -> Dict[str, Any]:
        """Get information about a specific example."""
        fields = self._cases[index]
        return {
            "gold": fields.get("correct", ""),
            "kind": fields.get("kind"),
            "answers": fields.get("answers"),
        }

    def build_initial_state(self, index: int) -> vf.State:
        """Build initial state for a case."""
        fields = self._cases[index]
        return {
            "case_index": index, 
            "turn": 0,
            "info": {
                "gold": fields.get("correct", ""),
                "kind": fields.get("kind"),
                "answers": fields.get("answers"),
            }
        }


def _extract_case_fields(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Extract case fields from raw data."""
    kind = _infer_dataset_kind(sample)
    if kind == "MEDQA_EXTENDED":
        osce = sample.get("OSCE_Examination", {})
        patient = osce.get("Patient_Actor", {})
        phys = osce.get("Physical_Examination_Findings", {})
        tests = osce.get("Test_Results", {})
        symptoms = patient.get("Symptoms", {})
        
        return {
            "kind": kind,
            "objective": osce.get("Objective_for_Doctor", ""),
            "patient_info": patient,
            "physical": phys,
            "tests": tests,
            "symptoms": symptoms,
            "correct": osce.get("Correct_Diagnosis", ""),
            "demographics": patient.get("Demographics", ""),
            "history": patient.get("History", ""),
            "past_medical": patient.get("Past_Medical_History", ""),
            "social": patient.get("Social_History", ""),
            "review_systems": patient.get("Review_of_Systems", ""),
        }
    else:
        answers = sample.get("answers", [])
        correct_answer = ""
        for a in answers:
            if a.get("correct"):
                correct_answer = a.get("text", "")
                break
        return {
            "kind": kind,
            "objective": sample.get("question", ""),
            "patient_info": sample.get("patient_info", ""),
            "physical": sample.get("physical_exams", ""),
            "tests": sample.get("physical_exams", ""),
            "answers": answers,
            "correct": correct_answer,
            "image_url": sample.get("image_url"),
        }


def _infer_dataset_kind(sample: Dict[str, Any]) -> str:
    """Infer the type of dataset from sample structure."""
    if "OSCE_Examination" in sample:
        return "MEDQA_EXTENDED"
    return "NEJM"


def _format_patient_intro(fields: Dict[str, Any]) -> str:
    """Format patient introduction for the case."""
    parts: List[str] = []
    if fields.get("kind") == "MEDQA_EXTENDED":
        patient = fields.get("patient_info", {})
        parts.append("You are interacting with a standardized patient. Gather history, exam, and tests as needed to reach a final diagnosis.")
        if fields.get("objective"):
            parts.append(f"Objective: {fields['objective']}")
        
        # Add demographics
        demo = fields.get("demographics", "")
        if demo:
            parts.append(f"Demographics: {demo}")
        
        # Add history
        hist = fields.get("history", "")
        if hist:
            parts.append(f"History: {hist}")
        
        # Add symptoms if available
        symptoms = fields.get("symptoms", {})
        if symptoms:
            primary = symptoms.get("Primary_Symptom", "")
            secondary = symptoms.get("Secondary_Symptoms", [])
            if primary:
                parts.append(f"Primary Symptom: {primary}")
            if secondary:
                parts.append(f"Secondary Symptoms: {', '.join(secondary)}")
        
        # Add past medical history
        past_med = fields.get("past_medical", "")
        if past_med:
            parts.append(f"Past Medical History: {past_med}")
        
        # Add social history
        social = fields.get("social", "")
        if social:
            parts.append(f"Social History: {social}")
        
        # Add review of systems
        review = fields.get("review_systems", "")
        if review:
            parts.append(f"Review of Systems: {review}")
            
    else:
        if fields.get("objective"):
            parts.append(fields["objective"])
        if fields.get("image_url"):
            parts.append(f"Image URL: {fields['image_url']}")
        if fields.get("patient_info"):
            parts.append(f"Patient info: {fields['patient_info']}")

    parts.append(
        "Instructions: Provide your answer inside a boxed format. "
        "Begin your final line with '\\boxed{...}' containing the concise disease name."
    )
    return "\n".join(parts)


def _format_section(title: str, content: Any) -> str:
    """Format a section with title and content."""
    if not content:
        return f"{title}: No additional information available."
    if isinstance(content, str):
        return f"{title}: {content}"
    try:
        return f"{title}:\n" + json.dumps(content, indent=2, ensure_ascii=False)
    except Exception:
        return f"{title}: {str(content)}"


def _match_correct(prediction: str, correct: str, answers: Optional[List[Dict[str, Any]]] = None) -> bool:
    """Check if prediction matches correct answer."""
    p = _normalize_text(prediction)
    c = _normalize_text(correct)
    if not p:
        return False
    if c and c in p:
        return True
    if answers:
        for idx, a in enumerate(answers):
            text = _normalize_text(a.get("text", ""))
            if text and (text in p):
                return a.get("correct", False) is True
    if p and c and (p in c or c in p):
        return True
    return False


def accuracy_reward(prompt: str, completion: str, answer: str, state: Dict[str, Any]) -> float:
    """Calculate accuracy reward for a completion."""

    info = state.get("info", {})
    gold = info.get("gold", "") or answer
    
    if isinstance(completion, list):
        completion_text = ""
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                completion_text = msg.get("content", "")
                break
    else:
        completion_text = str(completion)
    
    try:
        prediction = extract_boxed_answer(completion_text) or ""
    except Exception:
        prediction = completion_text or ""

    answers = info.get("answers")
    correct = _match_correct(prediction, gold, answers)
    
    print(f"DEBUG: prediction='{prediction}', gold='{gold}', correct={correct}")
    
    return 1.0 if correct else 0.0


def load_environment(
    dataset_path: str | None = None,
    use_think: bool = False,
    max_turns: int = 10,
    **kwargs
) -> vf.Environment:

    if dataset_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(current_dir, "..", "..", "agentclinic", "agentclinic_medqa_extended.jsonl")
        if not os.path.exists(dataset_path):
            dataset_path = "agentclinic/agentclinic_medqa_extended.jsonl"
    
    # Load dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    cases = _read_jsonl(dataset_path)
    if not cases:
        raise ValueError("No cases loaded from dataset")
    
    print(f"Loaded {len(cases)} cases from {dataset_path}")
    
    rubric = vf.Rubric(
        funcs=[accuracy_reward],
        names=["accuracy"]
    )
    
    env = AgentClinicMedqaMultiTurn(
        cases=cases,
        max_turns=max_turns,
        use_think=use_think,
        **kwargs
    )
    env.rubric = rubric
    
    return env
