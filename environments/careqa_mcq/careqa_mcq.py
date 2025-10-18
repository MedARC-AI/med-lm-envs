from __future__ import annotations
from typing import Any, Optional
from datasets import load_dataset
import verifiers as vf
from verifiers.utils.data_utils import (
    extract_boxed_answer,
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
)

# Prompt Construction

def _build_prompt(question: str, options: dict[str, str]) -> str:
    """Create an MCQ prompt."""
    formatted_opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question:{question}\nChoices:{formatted_opts}\nAnswer:"
    
def exact_match(parser: vf.Parser, completion: str, answer: str, **kwargs) -> float:
    """Reward exact matches."""
    response = parser.parse_answer(completion).strip().upper()
    return 1.0 if response == answer.strip().upper() else 0.0

# Main Environment

def load_environment(
    split: str = "test",
    use_think: bool = False,
    system_prompt: Optional[str] = None
    ) -> vf.Environment:
    
    """
    CareQA multiple-choice evaluation environment.
    Uses vf.SingleTurnEnv + MCQ accuracy rubric.
    """
    ds = load_dataset("HPAI-BSC/CareQA",'CareQA_en', split=split)

    def _map(ex):
        options = {"A": ex["op1"], "B": ex["op2"], "C": ex["op3"], "D": ex["op4"]}
        gold_letter = ["A", "B", "C", "D"][ex["cop"] - 1] 
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
    
    parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)

    rubric = vf.Rubric(funcs=[exact_match], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=mapped,
        eval_dataset=mapped,
        rubric=rubric,
        parser = parser,
        system_prompt=system_prompt,
    )
