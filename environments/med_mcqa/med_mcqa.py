# environments/med_mcqa/med_mcqa.py
from __future__ import annotations
from typing import Dict, Optional, Any
from datasets import load_dataset
import verifiers as vf


def _get_text_from_completion(completion: Any) -> str:
    """Best-effort extraction of a text string from a model completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
        return str(last)
    return str(completion)


def _first_letter_ad(text: str) -> Optional[str]:
    """Return the first A–D (case-insensitive) letter in text, or None."""
    t = (text or "").upper()
    for ch in t:
        if ch in ("A", "B", "C", "D"):
            return ch
    return None


def _build_prompt(question: str, options: Dict[str, str]) -> str:
    """Render a strict MCQ prompt requiring ONLY a letter reply."""
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    letters = ", ".join(sorted(options.keys()))
    return (
        "You are a clinician. Choose exactly ONE option letter.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{opts}\n\n"
        f"Answer with ONLY the letter ({letters})."
    )


def load_environment(split: str = "test") -> vf.Environment:
    """
    Med MCQA multiple-choice accuracy eval (single-turn, string prompt).

    - Loads HuggingFace 'lighteval/med_mcqa' split
    - Builds a prompt per example with options A–D
    - Maps 'cop' to the gold letter (handles 1-based 1..4; safe fallback for 0..3)
    - Scores accuracy by matching the model's first A–D letter
    """
    ds = load_dataset("lighteval/med_mcqa", split=split)

    def _map(ex):
        q = ex.get("question", "") or ""
        options = {"A": ex.get("opa","") or "",
                "B": ex.get("opb","") or "",
                "C": ex.get("opc","") or "",
                "D": ex.get("opd","") or ""}
        # default outputs so columns are always present
        out = {
            "question": _build_prompt(q, options),
            "answer": None,
        }
        cop_raw = ex.get("cop", None)
        try:
            cop_int = int(cop_raw)
        except Exception:
            return out  # keep columns, answer stays None

        out["answer"] = "ABCD"[cop_int - 1]
        return out

    # Strip all original columns; keep only ['question', 'answer']
    mapped = ds.map(_map, remove_columns=ds.column_names).filter(lambda r: r is not None)
    # Hard accuracy on first A–D letter from completion
    def accuracy_reward(completion, answer):
        pred = _first_letter_ad(_get_text_from_completion(completion))
        gold = str(answer).strip().upper()
        return 1.0 if (pred is not None and pred == gold) else 0.0

    rubric = vf.Rubric(funcs=[accuracy_reward], weights=[1.0])

    env = vf.SingleTurnEnv(
        dataset=mapped,
        eval_dataset=mapped,
        system_prompt=None,
        rubric=rubric,
    )

    return env

