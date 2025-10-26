"""
MedMCQA Environment

This script defines a MedMCQA evaluation environment compatible with the Verifiers framework.

The `med_mcqa` function is adapted from LightEval's default prompts:
https://github.com/huggingface/lighteval/blob/ecef2c662b9418866b6447d33b5e7d5dedd74af8/src/lighteval/tasks/default_prompts.py

Originally licensed MIT, Copyright (c) 2024 Hugging Face

Reference:
@misc{lighteval,
  author = {Habib, Nathan and Fourrier, Clémentine and Kydlíček, Hynek and Wolf, Thomas and Tunstall, Lewis},
  title = {LightEval: A lightweight framework for LLM evaluation},
  year = {2023},
  version = {0.8.0},
  url = {https://github.com/huggingface/lighteval}
}
"""

from typing import Any, Dict

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

LETTER_INDICES = ["A", "B", "C", "D"]


def med_mcqa(line: Dict[str, Any]) -> Dict[str, Any]:
    """Build the standard MedMCQA multiple-choice question prompt."""
    query = f"Give a letter answer among A, B, C or D.\nQuestion: {line['question']}\n"
    query += "".join(
        [
            f"{key}. {choice}\n"
            for key, choice in zip(LETTER_INDICES, [line["opa"], line["opb"], line["opc"], line["opd"]])
        ]
    )
    query += "Answer:"

    result = {
        "question": query,
        "answer": LETTER_INDICES[line["cop"] - 1],
        "choices": LETTER_INDICES,
        "gold_index": line["cop"] - 1,
        "instruction": "Give a letter answer among A, B, C or D.\n",
    }
    return result


def _get_text_from_completion(completion: Any) -> str:
    """Extract a text string from a model completion."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last_item = completion[-1]
        if isinstance(last_item, dict):
            return str(last_item.get("content", ""))
        return str(last_item)
    return str(completion)


def _first_letter_ad(text: str) -> str | None:
    """Return the first A–D letter found in the text, preferring boxed answers if present."""
    t = text or ""
    boxed_ans = extract_boxed_answer(t)
    if boxed_ans:
        for ch in boxed_ans.upper():
            if ch in LETTER_INDICES:
                return ch
    for ch in t.upper():
        if ch in LETTER_INDICES:
            return ch
    return None


def exact_match_reward(completion: Any, answer: str | None = None, **kwargs) -> float:
    """Reward 1.0 if the predicted answer matches the correct answer; 0.0 otherwise."""
    text = _get_text_from_completion(completion)
    predicted_ans = _first_letter_ad(text)
    correct_ans = answer.strip().upper() if answer else None
    return float(predicted_ans == correct_ans)


def load_environment(
    use_think: bool = False,
    system_prompt: str | None = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    """
    Load the MedMCQA environment with train and validation splits.
    Supports reasoning (use_think=True) or standard evaluation.
    Returns a SingleTurnEnv ready for model evaluation.
    """
    train_ds = load_dataset("lighteval/med_mcqa", split="train")
    val_ds = load_dataset("lighteval/med_mcqa", split="validation")

    def _map_example(example: Dict[str, Any]) -> Dict[str, Any] | None:
        cop = example.get("cop", -1)
        if not isinstance(cop, int) or cop not in [1, 2, 3, 4]:
            return None

        question = (example.get("question") or "").strip()
        choices = [(example.get(k) or "").strip() for k in ["opa", "opb", "opc", "opd"]]

        if not question or not any(choices):
            return None

        line = {
            "question": question,
            "opa": choices[0],
            "opb": choices[1],
            "opc": choices[2],
            "opd": choices[3],
            "cop": cop,
        }
        mapped = med_mcqa(line)
        return mapped

    columns_to_remove = ["question", "opa", "opb", "opc", "opd", "cop"]
    train_mapped = train_ds.map(_map_example, remove_columns=columns_to_remove).filter(lambda x: x is not None)
    val_mapped = val_ds.map(_map_example, remove_columns=columns_to_remove).filter(lambda x: x is not None)

    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    rubric = vf.Rubric(funcs=[exact_match_reward], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
