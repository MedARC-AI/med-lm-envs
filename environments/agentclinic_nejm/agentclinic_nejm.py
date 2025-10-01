"""
AgentClinic NEJM multiturn evaluation environment

- Assistant asks; env replies with HISTORY / EXAM / TESTS / IMAGE
- Episode completes ONLY when the last **assistant** message contains a final answer brace-wrapped { ... } 
  AND the assistant has made at least one information request earlier in the episode
- Reward = 1 if normalized prediction matches gold.
"""

from __future__ import annotations
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import verifiers as vf
from datasets import Dataset

# ---------- Config ----------
DEFAULT_DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "../../agentclinic/agentclinic_nejm_extended.jsonl",
)
SYSTEM_PROMPT = (
    "Think step by step. When you are ready, output ONE diagnosis inside {Diagnosis} "
    "with no extra words and no punctuation inside the braces."
)
LOG_DIR = os.path.join(os.path.dirname(__file__), "log")
LOG_PATH = os.path.join(LOG_DIR, "debug.log")


# ---------- Helpers ----------
def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _is_correct_flag(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return False


def _extract_gold(answers: List[Dict[str, Any]]) -> str:
    for a in answers or []:
        if _is_correct_flag(a.get("correct")):
            return str(a.get("text", ""))
    return ""


def _prepare_case(raw: Dict[str, Any]) -> Dict[str, Any]:
    answers = raw.get("answers", []) or []
    return {
        "question": raw.get("question", ""),
        "patient_info": raw.get("patient_info", ""),
        "physical_exams": raw.get("physical_exams", ""),
        "answers": answers,
        "image_url": raw.get("image_url", "") or raw.get("image", ""),
        "gold": _extract_gold(answers),
        "type": raw.get("type", []),
    }


def _brace_content(text: str) -> str:
    matches = re.findall(r"\{([^}]+)\}", text)
    return matches[-1].strip() if matches else ""


def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.strip().lower()
    s = s.strip("{}(). ")
    return s


def accuracy_reward(prompt: str, completion: Any, info: Dict[str, Any]) -> float:
    gold = (info or {}).get("gold", "") or ""
    answers = (info or {}).get("answers", []) or []

    comp_text = _to_text(completion)
    pred = _brace_content(comp_text) or comp_text
    pred_clean = pred.strip().strip(".{} ")

    # If gold missing, fallback to correct answer from answers
    if not gold and answers:
        gold = _extract_gold(answers)

    gold_norm = _normalize(gold)
    pred_norm = _normalize(pred_clean)

    ok = gold_norm and (pred_norm == gold_norm)

    _log_debug(
        f"[score] raw_gold={gold!r} raw_pred={pred_clean!r} "
        f"norm_gold={gold_norm!r} norm_pred={pred_norm!r} ok={ok}"
    )

    # Extra fallback: check if pred_norm equals any other correct option
    if not ok and answers:
        for opt in answers:
            if _normalize(opt.get("text", "")) == pred_norm and _is_correct_flag(opt.get("correct")):
                ok = True
                break

    return 1.0 if ok else 0.0


def _to_text(completion: Any) -> str:
    if isinstance(completion, list):
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content", ""))
    return str(completion or "")


def _last_assistant_text(messages: vf.Messages) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""


def _log_debug(line: str) -> None:
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        pass


def _build_prompt(case: Dict[str, Any]) -> Dict[str, str]:
    opts = "\n".join(f"- {opt.get('text','')}" for opt in case.get("answers", []))
    px = case.get("physical_exams") or "No additional examination or test results were reported."
    user = (
        "You are interacting with a standardized patient. Ask for HISTORY, EXAM, TESTS, or IMAGE "
        "as needed to reach a final diagnosis.\n\n"
        "Objective: Assess and diagnose the patient based on the provided information and medical image.\n\n"
        f"Case Information:\n{case.get('question','')}\n\n"
        "Patient Information:\n"
        f"{case.get('patient_info','')}\n\n"
        "Physical Examination and Tests:\n"
        f"{px}\n\n"
        "Available Information:\n"
        "- Patient history and symptoms\n"
        "- Physical examination findings and test results\n"
        "- Medical image (if applicable)\n\n"
        "Answer Choices (choose EXACTLY one):\n"
        f"{opts}\n\n"
        "Instructions:\n"
        "1) Ask for HISTORY, EXAM, TESTS, or IMAGE as needed (you can request them by name).\n"
        "2) When ready, provide ONE diagnosis as EXACTLY one of the answer choices inside {Diagnosis} "
        "with no extra words and no punctuation inside the braces."
    )
    return {"system": SYSTEM_PROMPT, "user": user}


# ---------- Scoring (signature-agnostic) ----------
def _extract_completion_and_info(*args, **kwargs) -> Tuple[str, Dict[str, Any]]:

    completion = None
    info: Dict[str, Any] = {}

    # Look through args
    for a in args:
        if isinstance(a, (list, str)): 
            completion = a
        elif isinstance(a, dict) and ("gold" in a or "answers" in a or "case_index" in a):
            info = a
        elif isinstance(a, dict) and ("info" in a or "case_index" in a):
            if "info" in a and isinstance(a["info"], dict):
                info = a["info"]

    # Look through kwargs
    if not info:
        cand = kwargs.get("info") or kwargs.get("state")
        if isinstance(cand, dict):
            if "gold" in cand or "answers" in cand:
                info = cand
            elif "info" in cand and isinstance(cand["info"], dict):
                info = cand["info"]

    return _to_text(completion), (info or {})


# ---------- Environment ----------
class AgentClinicNEJMMultiTurn(vf.MultiTurnEnv):

    def __init__(
        self,
        cases: Optional[List[Dict[str, Any]]] = None,
        *,
        dataset_path: Optional[str] = None,
        max_turns: int = 10,
        name: str = "AgentClinicNEJM",
    ) -> None:
        if cases is None:
            data_path = dataset_path or DEFAULT_DATASET_PATH
            raw_cases = _read_jsonl(data_path)
        else:
            raw_cases = cases
        prepared = [_prepare_case(c) for c in raw_cases]

        prompts: List[List[Dict[str, str]]] = []
        infos: List[Dict[str, Any]] = []
        for idx, case in enumerate(prepared):
            p = _build_prompt(case)
            prompts.append(
                [
                    {"role": "system", "content": p["system"]},
                    {"role": "user", "content": p["user"]},
                ]
            )
            infos.append(
                {
                    "case_index": idx,
                    "gold": case.get("gold", ""),
                    "answers": case.get("answers", []),
                    "image_url": case.get("image_url", ""),
                    "question": case.get("question", ""),
                    "patient_info": case.get("patient_info", ""),
                    "physical_exams": case.get("physical_exams", ""),
                }
            )

        dataset = Dataset.from_dict(
            {
                "id": list(range(len(prepared))),
                "prompt": prompts,  
                "info": infos, 
            }
        )

        super().__init__(name=name, dataset=dataset)
        self._cases = prepared
        self._max_turns = max_turns

        try:
            self.rubric = vf.Rubric(funcs=[accuracy_reward], weights=[1.0], names=["accuracy"])
        except TypeError:
            self.rubric = vf.Rubric(reward_fns=[accuracy_reward], names=["accuracy"])

    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def build_initial_state(self, prompt: vf.Messages, info: Dict[str, Any]) -> vf.State:
        # Track whether the assistant has elicited at least one env reply 
        return {"turn": 0, "info": info, "asked": False}

    async def is_completed(
        self, messages: vf.Messages, state: vf.State, info: Optional[Dict[str, Any]] = None
    ) -> bool:
        # Stop if we hit max_turns 
        if state.get("turn", 0) >= self._max_turns:
            return True

        last_asst = _last_assistant_text(messages)
        if not last_asst:
            # model hasn't replied yet → not complete
            return False

        # Require that the assistant has asked for at least one section before allowing completion
        if not state.get("asked", False):
            return False

        # Done if assistant provided brace-wrapped answer
        if _brace_content(last_asst):
            return True

        current_info = info or state.get("info", {}) or {}
        answers = current_info.get("answers", []) or []
        norm = _normalize(last_asst)
        for opt in answers:
            if _normalize(str(opt.get("text", ""))) in norm:
                return True

        return False

    async def env_response(
        self, messages: vf.Messages, state: vf.State, info: Optional[Dict[str, Any]] = None
    ) -> Tuple[vf.Messages, vf.State]:
        turn = state.get("turn", 0)
        new_state = dict(state)
        new_state["turn"] = turn + 1

        last_asst = _last_assistant_text(messages)

        # If the assistant gave a brace answer but hasn't asked yet, DO NOT end the episode.
        # We still reply with a nudge or with the requested section; only end when is_completed() allows it.
        if last_asst and _brace_content(last_asst) and state.get("asked", False):
            return [], new_state

        # Otherwise reply with the requested section 
        text_lower = (last_asst or "").lower()
        case_info = state.get("info", {})

        asked = False  # will flip to True only when we deliver a requested section

        if "history" in text_lower or "symptom" in text_lower:
            reply = f"Patient History\n{case_info.get('patient_info', 'No additional history available.')}"
            asked = True
        elif "exam" in text_lower or "physical" in text_lower:
            reply = f"Physical Examination Findings\n{case_info.get('physical_exams', 'No additional exam findings available.')}"
            asked = True
        elif any(k in text_lower for k in ["test", "lab", "result", "imaging", "x-ray", "ct", "mri"]):
            reply = f"Test Results\n{case_info.get('physical_exams', 'No additional test results available.')}"
            asked = True
        elif "image" in text_lower or "photo" in text_lower or "picture" in text_lower:
            reply = case_info.get("image_url") or "No medical image is available for this case."
            asked = True
        else:
            reply = (
                "You can request HISTORY, EXAM, TESTS, or IMAGE. "
                "When ready, give ONE diagnosis inside {Diagnosis} with no extra words and no punctuation."
            )

        if asked:
            new_state["asked"] = True

        return [{"role": "user", "content": reply}], new_state


# ---------- Loader ----------
def load_environment(
    dataset_path: Optional[str] = None,
    max_turns: int = 10,
    **kwargs: Any,
) -> vf.Environment:

    env = AgentClinicNEJMMultiTurn(
        cases=None,
        dataset_path=dataset_path or DEFAULT_DATASET_PATH,
        max_turns=max_turns,
        **kwargs,
    )
    return env
