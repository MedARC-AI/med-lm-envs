from __future__ import annotations
from typing import Dict, Optional, Any
from datasets import load_dataset
import verifiers as vf

def _get_text_from_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)

def _first_letter(text: str) -> Optional[str]:
    t = (text or "").upper()
    for ch in t:
        if "A" <= ch <= "Z":
            return ch
    return None

def _build_prompt(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    letters = ", ".join(sorted(options.keys()))
    return (
        "You are a clinician. Choose exactly ONE option letter.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"Answer with ONLY the letter ({letters})."
    )

def load_environment() -> vf.Environment:
    """
    MedQA-USMLE-4-options multiple-choice evaluation
    - Train split = dataset
    - Test split = eval_dataset
    - Scoring: accuracy (predicted letter == gold letter)
    """
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")

    def _map(ex):
        q: str = ex["question"]
        options: Dict[str, str] = ex["options"]
        gold_letter: str = ex["answer_idx"].strip().upper()

        return {
            "question": _build_prompt(q, options),
            "answer": gold_letter,
        }

    train_mapped = ds["train"].map(_map, remove_columns=ds["train"].column_names)
    test_mapped = ds["test"].map(_map, remove_columns=ds["test"].column_names)

    def accuracy_reward(completion, answer):
        pred = _first_letter(_get_text_from_completion(completion))
        gold = str(answer).strip().upper()
        return 1.0 if (pred is not None and pred == gold) else 0.0

    rubric = vf.Rubric(funcs=[accuracy_reward], weights=[1.0])

    return vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=test_mapped,
        system_prompt=None,
        rubric=rubric,
    )
