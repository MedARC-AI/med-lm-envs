from __future__ import annotations
from typing import Any, Optional
from datasets import load_dataset
import verifiers as vf

# Prompt Construction

def _build_open_prompt(question: str) -> str:
    """Create an open-ended clinical QA prompt."""
    return (
        "You are an expert clinician answering medical questions.\n"
        "Read the following question carefully and provide a detailed, concise answer.\n\n"
        f"Question:\n{question.strip()}\n\n"
        "Answer:"
    )
    
# Load Open-Ended Environment

def load_environment(split: str = "test") -> vf.SingleTurnEnv:
    ds = load_dataset("HPAI-BSC/CareQA", 'CareQA_en_open', split=split)

    def _map(ex):
        system_content = "You are an expert clinician answering medical questions."

        user_content = (
            "Read the following question carefully and provide a detailed, concise answer.\n\n"
            f"Question:\n{ex['question'].strip()}\n\n"
            "Answer:"
        )

        return {
            "prompt": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "answer": ex.get("answer_explanation", ex.get("answer", "")),
        }

    mapped = ds.map(_map, remove_columns=ds.column_names)

    rubric = vf.JudgeRubric()

    return vf.SingleTurnEnv(
        dataset=mapped,
        eval_dataset=mapped,
        rubric=rubric,
        system_prompt=None,
    )
