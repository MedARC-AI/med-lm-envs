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
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets progress indicators

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
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
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

        options = [choices[0], choices[1], choices[2], choices[3]]
        answer_idx = cop - 1

        if shuffle_answers:
            shuffled_options, _, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=LETTER_INDICES,
                seed=shuffle_seed,
                row_id=question,
            )
            options = shuffled_options

        line = {
            "question": question,
            "opa": options[0],
            "opb": options[1],
            "opc": options[2],
            "opd": options[3],
            "cop": answer_idx + 1,
        }
        mapped = med_mcqa(line)
        mapped["info"] = {"answer_text": options[answer_idx]}
        return mapped

    columns_to_remove = ["question", "opa", "opb", "opc", "opd", "cop"]
    # Disable the Datasets cache when shuffling answers
    load_from_cache_file = False if shuffle_answers else True
    train_mapped = train_ds.map(
        _map_example,
        remove_columns=columns_to_remove,
        load_from_cache_file=load_from_cache_file,
    ).filter(lambda x: x is not None, load_from_cache_file=load_from_cache_file)
    val_mapped = val_ds.map(
        _map_example,
        remove_columns=columns_to_remove,
        load_from_cache_file=load_from_cache_file,
    ).filter(lambda x: x is not None, load_from_cache_file=load_from_cache_file)

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
