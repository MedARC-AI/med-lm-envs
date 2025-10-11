from __future__ import annotations
from typing import Any, Optional
from datasets import load_dataset
import verifiers as vf


# Helper Functions

def _get_text_from_completion(completion: Any) -> str:
    """Extract plain text from completion."""
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
        return str(last).strip()
    return str(completion).strip()


def _first_letter(text: str) -> Optional[str]:
    """Extract the first uppercase A–Z letter."""
    for ch in (text or "").upper():
        if "A" <= ch <= "Z":
            return ch
    return None

# Prompt Construction

def _build_prompt(question: str, options: dict[str, str]) -> str:
    """Create a polished clinical MCQ prompt."""
    formatted_opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    letters = ", ".join(options.keys())
    return (
        "You are a board-certified clinician taking a medical reasoning test.\n"
        "Read the following question carefully and choose the most appropriate answer.\n\n"
        f"Question:\n{question.strip()}\n\n"
        f"Options:\n{formatted_opts}\n\n"
        f"Respond with only the option letter ({letters}), nothing else."
    )

# Main Environment

def load_environment(split: str = "test") -> vf.Environment:
    """
    CareQA multiple-choice evaluation environment.
    Uses vf.SingleTurnEnv + MCQ accuracy rubric.
    """
    ds = load_dataset("HPAI-BSC/CareQA",'CareQA_en', split=split)

    def _map(ex):
        options = {"A": ex["op1"], "B": ex["op2"], "C": ex["op3"], "D": ex["op4"]}
        gold_letter = ["A", "B", "C", "D"][ex["cop"] - 1] 
        # The key change is here: format the single prompt string as a list of dicts (ChatML format)
        return {
            "prompt": [
                {
                    "role": "user", 
                    "content": _build_prompt(ex["question"], options)
                }
            ],
            "answer": gold_letter,
        }

    mapped = ds.map(_map, remove_columns=ds.column_names)

    def mcq_accuracy(completion, answer):
        pred = _first_letter(_get_text_from_completion(completion))
        return 1.0 if pred == str(answer).upper() else 0.0

    rubric = vf.Rubric(funcs=[mcq_accuracy], weights=[1.0])

    return vf.SingleTurnEnv(
        dataset=mapped,
        eval_dataset=mapped,
        rubric=rubric,
        system_prompt=None,
    )
