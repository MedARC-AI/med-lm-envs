"""
SCT-Bench Public Environment

This script defines an evaluation environment for the public SCT-Bench dataset compatible with the Verifiers framework.

Prompts, templates, and scoring logic are adapted from:
https://github.com/SCT-Bench/sctpublic

MIT License, Copyright (c) 2025 liamgmccoy

Reference:
@article{mccoy2025assessment,
  title={Assessment of large language models in clinical reasoning: a novel benchmarking study},
  author={McCoy, Liam G and Swamy, Rajiv and Sagar, Nidhish and Wang, Minjia and Bacchi, Stephen and Fong, Jie Ming Nigel and Tan, Nigel CK and Tan, Kevin and Buckley, Thomas A and Brodeur, Peter and others},
  journal={NEJM AI},
  volume={2},
  number={10},
  pages={AIdbp2500120},
  year={2025},
  publisher={Massachusetts Medical Society}
}
"""

from typing import Any
import os
import re

import numpy as np
import pandas as pd
from datasets import Dataset

import verifiers as vf


# File paths and directories
BASE_PATH = os.path.dirname(__file__)
INPUT_PATH = os.path.join(BASE_PATH, "data", "sct_cleaned_full.csv")
TEMPLATE_PATH = os.path.join(BASE_PATH, "data")


def generate_prompt_template(guideline: str, testcase: str, reason: bool, few_shot: bool, examples) -> str:
    """
    Generate a prompt template based on the given guideline, testcase and examples.

    Args:
    guideline (str): The guideline for the SCT task.
    testcase (str): The case template for which the SCT task is being generated.
    reason (bool): Whether to include the explanation in the prompt.
    few_shot (bool): Whether to include few-shot examples in the prompt.
    examples (dict): A dictionary of examples with keys as the ratings (str) and values as the example text (str).

    Returns:
    str: The generated prompt template.
    """
    if not reason:
        guideline = guideline.replace(" and a brief explanation for your choice", "")
    prompt = guideline
    if few_shot:
        prompt += "\n## Examples with Response Labels\n"
        for rating in ["-2", "-1", "0", "+1", "+2"]:
            prompt += examples[rating]
    prompt += testcase
    return prompt


def generate_prompt(reason: bool, few_shot: bool) -> str:
    examples = {}
    for rating in ["-2", "-1", "0", "+1", "+2"]:
        with open(f"{TEMPLATE_PATH}/example_{rating}.md", "r") as f:
            examples[rating] = f.read()

    with open(f"{TEMPLATE_PATH}/guideline.md", "r") as f:
        guideline = f.read()

    with open(f"{TEMPLATE_PATH}/testcase.md", "r") as f:
        testcase = f.read()

    prompt_template = generate_prompt_template(guideline, testcase, reason, few_shot, examples)
    return prompt_template


def sct_prompts(df: pd.DataFrame, reason: bool, few_shot: bool) -> pd.DataFrame:
    prompt_template = generate_prompt(reason, few_shot)

    def parse_prompt(row):
        return (
            prompt_template.replace("{{ scenario }}", row["sct_stem"])
            .replace("{{ hypothesis }}", row["question"])
            .replace("{{ additional information }}", row["additional_info"])
        )

    df["info_prompt"] = df.apply(parse_prompt, axis=1)
    return df


class SCTParser(vf.Parser):
    """Parser for SCT numeric ratings (-2 to +2)."""

    def parse_answer(self, completion: Any) -> str | None:
        response = getattr(completion, "content", str(completion))
        try:
            rating = response.split("Rating: ")[1][:5]
            pattern = r"\+2|\+1|0|-1|-2"
            matches = re.findall(pattern, rating)
            label = int("".join(matches))
            assert label in [-2, -1, 0, 1, 2]
            return str(label)
        except Exception:
            return None


def sct_rubric(completion: Any, answer: str, parser: vf.Parser, info: dict | None = None) -> float:
    """
    Computes SCT score. Normalizes answer distribution so that the greatest score is always 1
    """
    parsed = parser.parse_answer(completion)
    if parsed is None or info is None:
        return 0.0
    ans_distribution = info["ans_distribution"]
    llm_response = int(parsed)
    normalized_dist = ans_distribution / np.max(ans_distribution)
    score = normalized_dist[0, llm_response + 2]
    return float(score)


def load_environment(
    reason: bool = False,
    few_shot: bool = False,
) -> vf.Environment:
    """Load the SCT-Bench Public Environment.

    Args:
        reason: supports reasoning if True else standard evaluation
        few_shot: if True, use few-shot prompting else zero-shot
    """
    df = pd.read_csv(INPUT_PATH, encoding_errors="ignore")
    df = df.dropna(subset=["sct_stem", "question", "additional_info"])

    df = sct_prompts(df, reason, few_shot)

    def _map_example(row: pd.Series) -> dict[str, Any]:
        prompt = row["info_prompt"]
        ans_dist = row[["-2", "-1", "0", "1", "2"]].to_numpy().reshape(1, 5)
        gold_label = str(np.argmax(ans_dist) - 2)

        example = {"question": prompt, "answer": gold_label, "info": {"ans_distribution": ans_dist}}
        return example

    val_mapped = [_map_example(row) for _, row in df.iterrows()]
    val_dataset = Dataset.from_list(val_mapped)

    parser = SCTParser()

    rubric = vf.Rubric(funcs=[sct_rubric], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=None,
        eval_dataset=val_dataset,
        system_prompt=None,
        parser=parser,
        rubric=rubric,
    )

    return env
