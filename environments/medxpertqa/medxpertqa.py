import verifiers as vf
from verifiers.utils.data_utils import extract_boxed_answer
from datasets import load_dataset

import os
import re

def _get_system_prompt(use_think: bool) -> str:
    think_system_prompt = (
        "You are a helpful medical assistant. Think step-by-step inside <think>...</think> tags."
        "Put your final answer within \boxed{{}}.\nQ:"
        )
    no_think_system_prompt = (
        "You are a helpful medical assistant."
        "Put your final answer within \boxed{{}}.\nQ:"
        )
    system_prompt = think_system_prompt if use_think else no_think_system_prompt
    return system_prompt

def _build_prompt(question: str, use_think: bool) -> str:
    return _get_system_prompt(use_think) + " " + question

def load_environment(
    use_think: bool = False,
) -> vf.Environment:
    """
    MedXpertQA environment using equality with the "label" column as the eval criterion.
    This environment loads the MedXpertQA dataset and compares model responses (diagnosis) with the ground truth in "label" column.
    """
    full_dataset = load_dataset("TsinghuaC3I/MedXpertQA", "Text")
    test_dataset = full_dataset["test"].map(
        lambda x: {
            "question": x["question"],
            "answer": x["label"],
            "task": "medxpertqa",
        }
    )

    def _map(ex):
        q: str = ex["question"]
        a: str = ex["label"]

        return {
            "question": _build_prompt(q, use_think),
            "answer": a,
        }


    mapped = test_dataset.map(_map).filter(lambda r: r is not None)

    async def medxpertqa_reward_func(
        completion: str,
        answer: str,
    ) -> float:
        """
        Reward function for MedXpertQA environment.
        Compares the model response with the ground truth answer.
        Returns 1.0 if they match (case-insensitive), else 0.0.
        """
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
        final_answer = parser.parse(completion).strip()
        if final_answer.lower() == answer.lower():
            return 1.0
        else:
            return 0.0
        
    rubric = vf.Rubric(
        funcs=[medxpertqa_reward_func],
        weights=[1.0],
    )

    vf_env = vf.SingleTurnEnv(
        eval_dataset=mapped,
        system_prompt=None,
        rubric=rubric
    )

    return vf_env
