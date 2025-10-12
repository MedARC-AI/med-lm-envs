from __future__ import annotations
from typing import Dict, Optional
from datasets import load_dataset
import verifiers as vf
from verifiers.utils.data_utils import (
    extract_boxed_answer,
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
)


def _build_prompt(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question:{question}\n{opts}\nAnswer:"


def exact_match(parser: vf.Parser, completion: str, answer: str, **kwargs) -> float:
    """Reward exact matches."""
    response = parser.parse_answer(completion).lower().strip()
    return 1.0 if response == answer.lower().strip() else 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
) -> vf.Environment:
    """
    MedQA-USMLE-4-options multiple-choice evaluation
    - Train split = dataset
    - Test split = eval_dataset
    - Supports reasoning (use_think=True) or non-reasoning models
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

    # Use boxed parser; ThinkParser if use_think is True
    parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)

    rubric = vf.Rubric(funcs=[exact_match], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=test_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
