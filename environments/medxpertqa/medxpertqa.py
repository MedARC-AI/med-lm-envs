import verifiers as vf
from datasets import load_dataset

import os
import re

def _get_system_prompt() -> str:
    system_prompt = (
        "You are a medical question-answering model. You must think step-by-step and reason carefully about "
        "the following medical question before providing your answer. You should enclose your thoughts "
        "and reasoning inside <think> </think> tags, and then provide your final option letter enclosed inside <answer> </answer> tags. Choose exactly ONE option letter."
    )
    return system_prompt

def load_environment() -> vf.Environment:
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

    async def medxpertqa_reward_func(
        completion: str,
        answer: str,
    ) -> float:
        """
        Reward function for MedXpertQA environment.
        Compares the model response with the ground truth answer.
        Returns 1.0 if they match (case-insensitive), else 0.0.
        """

        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL | re.IGNORECASE)
        if match:
            final_answer = match.group(1).strip()
        else:
            final_answer = completion.strip()

        if final_answer.lower() == answer.lower():
            return 1.0
        else:
            return 0.0
        
    rubric = vf.Rubric(
        funcs=[medxpertqa_reward_func],
        weights=[1.0],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=test_dataset,
        eval_dataset=test_dataset,
        system_prompt=_get_system_prompt(),
        rubric=rubric
    )

    return vf_env
