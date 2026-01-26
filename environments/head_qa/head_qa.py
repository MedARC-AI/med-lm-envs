"""
HEAD-QA environment

This script defines an evaluation environment for HEAD-QA compatible with the Verifiers framework.

The prompts were adapted from the HEAD-QA v2 paper.

MIT License

Citation:

@inproceedings{vilares-gomez-rodriguez-2019-head,
    title = "{HEAD}-{QA}: A Healthcare Dataset for Complex Reasoning",
    author = "Vilares, David  and
      Gomez-Rodriguez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1092",
    doi = "10.18653/v1/P19-1092",
    pages = "960--966",
    abstract = "We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans. We then consider monolingual (Spanish) and cross-lingual (to English) experiments with information retrieval and neural techniques. We show that: (i) HEAD-QA challenges current methods, and (ii) the results lag well behind human performance, demonstrating its usefulness as a benchmark for future work.",
}
"""

from typing import Any

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer


def zero_shot_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Build the Zero-shot prompt."""
    question_text = (example.get("qtext") or "").strip()
    answers = example.get("answers", [])
    options_text = "\n".join([f"{a['aid']}. {a['atext'].strip()}" for a in answers])

    prompt = (
        "You are an expert in specialized scientific and health disciplines. "
        "Respond to the following multiple-choice question:\n"
        f"{question_text}\n{options_text}\n"
    )

    correct_answer = example.get("ra", -1)

    result = {
        "question": prompt,
        "answer": str(correct_answer),
        "choices": [str(a["aid"]) for a in answers],
        "gold_index": correct_answer - 1,
        "info": {"answer_text": answers[correct_answer - 1]["atext"].strip()},
    }
    return result


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: str | None = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    """
    Load the HEAD-QA environment with train and validation splits.
    Returns a SingleTurnEnv ready for model evaluation.

    Args:
        use_think: Supports reasoning if True
        shuffle_answers: Shuffle answer options
        shuffle_seed: Seed for shuffling
    """
    train_ds = load_dataset("EleutherAI/headqa", "en", split="train")
    val_ds = load_dataset("EleutherAI/headqa", "en", split="validation")

    def _map_example(example: dict[str, Any]) -> dict[str, Any] | None:
        correct_answer = example.get("ra", -1)
        question_text = (example.get("qtext") or "").strip()
        answers = example.get("answers", [])

        if not question_text or not answers or not (1 <= correct_answer <= len(answers)):
            return None

        options = [a["atext"].strip() for a in answers]
        answer_idx = correct_answer - 1

        if shuffle_answers:
            indices = [str(i + 1) for i in range(len(options))]
            shuffled_options, _, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=indices,
                seed=shuffle_seed,
                row_id=question_text,
            )
            options = shuffled_options

        temp_example = {
            "qtext": question_text,
            "answers": [{"aid": i + 1, "atext": opt} for i, opt in enumerate(options)],
            "ra": answer_idx + 1,
        }
        mapped = zero_shot_prompt(temp_example)
        return mapped

    columns_to_remove = ["qtext", "answers", "ra"]
    # Disable the Datasets cache when shuffling answers
    load_from_cache_file = False if shuffle_answers else True
    train_mapped = train_ds.map(
        _map_example,
        remove_columns=columns_to_remove,
        load_from_cache_file=load_from_cache_file,
    )
    val_mapped = val_ds.map(
        _map_example,
        remove_columns=columns_to_remove,
        load_from_cache_file=load_from_cache_file,
    )

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

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
