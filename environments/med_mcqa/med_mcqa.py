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

from typing import Any

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.rewards.mcq_accuracy import multiple_choice_accuracy
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

LETTER_INDICES = ["A", "B", "C", "D"]


def med_mcqa(line: dict[str, Any]) -> dict[str, Any]:
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

    def _map_example(example: dict[str, Any]) -> dict[str, Any] | None:
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
        mapped["info"] = {"answer_text": choices[cop - 1]}
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

    def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
