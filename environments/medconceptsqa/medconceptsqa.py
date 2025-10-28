from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets progress indicators

_VOCAB_CHOICES = ["atc", "icd10cm", "icd10proc", "icd9cm", "icd9proc"]
_LEVEL_CHOICES = ["easy", "hard", "medium"]
_OPTION_LABELS = ("A", "B", "C", "D")
_OPTION_SEPARATORS = (".", ")", ":", "-")


def _extract_question_and_options(row: dict) -> tuple[str, dict[str, str]]:
    """Return the stem without embedded options and an ordered map of options."""
    question = row.get("question", "") or ""
    options: dict[str, str] = {}
    for idx, label in enumerate(_OPTION_LABELS, start=1):
        value = row.get(f"option{idx}", "")
        if value not in ("", None):
            options[label] = value

    def _looks_like_option(line: str) -> bool:
        candidate = line.strip()
        for label in _OPTION_LABELS:
            for sep in _OPTION_SEPARATORS:
                if candidate.startswith(f"{label}{sep}"):
                    return True
        return False

    question_lines = [line for line in question.splitlines() if not _looks_like_option(line)]
    question_stem = "\n".join(question_lines).strip()

    return question_stem, options


def _format_stem_with_options(question: str, options: dict[str, str]) -> str:
    option_block = "\n".join(f"{label}. {text}" for label, text in options.items())
    return f"{question}\n{option_block}".strip()


def _create_few_shot_data(few_shot_set: Dataset, num_few_shot: int, answer_format: AnswerFormat) -> dict[tuple, str]:
    """Create few-shot examples from the dev set, grouped by (vocab, level).
    Args:
        few_shot_set: the dev set to draw few-shot examples from
        num_few_shot: number of few-shot examples to include per (vocab, level) pair
    Returns:
        dict mapping (vocab, level) to concatenated few-shot examples
    """
    few_shot_examples = {}

    for row in few_shot_set:
        key = (row["vocab"], row["level"])
        if key not in few_shot_examples:
            few_shot_examples[key] = []
        question_stem, options = _extract_question_and_options(row)
        formatted_question = _format_stem_with_options(question_stem, options)
        if len(few_shot_examples[key]) < num_few_shot:
            if answer_format == AnswerFormat.XML:
                prompt = f"{formatted_question}\nAnswer: <answer>{row['answer_id']}</answer>\n\n".replace("  ", "")
            elif answer_format == AnswerFormat.BOXED:
                prompt = f"{formatted_question}\nAnswer: \\boxed{{{row['answer_id']}}}\n\n".replace("  ", "")
            else:
                raise ValueError(f"Unsupported answer format: {answer_format=}")
            few_shot_examples[key].append(prompt)

    for key in few_shot_examples:
        few_shot_examples[key] = "".join(few_shot_examples[key])

    return few_shot_examples


def load_environment(
    num_few_shot: int = 4,
    use_think: bool = False,
    vocab: str | None = None,
    level: str | None = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    """MedConceptsQA multiple-choice evaluation
    - Loads HF 'ofir408/MedConceptsQA' (contains only dev and test split)
    - Builds a prompt per item, with optional few-shot examples from the dev set
    - Scores accuracy by comparing the model's A/B/C/D answer to the gold answer
    - Supports reasoning (use_think=True) or non-reasoning models

    Args:
        num_few_shot: number of few-shot examples to include in the prompt (default: 4)
        use_think: whether to use a ThinkParser and reasoning system prompt (default: False)
        vocab: vocabulary to subset dataset, if `None` - choses `all` (default: None)
        level: difficulty level to subset dataset (used in conjunction with vocab).
            cannot be None if vocab is not None (default: None)
    Returns:
        vf.Environment: the single-turn evaluation environment
    """
    subset = "all"
    if vocab is not None:
        if vocab not in _VOCAB_CHOICES:
            raise ValueError(f"Invalid vocab choice {vocab}, must be one of {_VOCAB_CHOICES}")
        if level is None or level not in _LEVEL_CHOICES:
            raise ValueError(f"Invalid level choice {level}, must be one of {_LEVEL_CHOICES}")
        subset = f"{vocab}_{level}"

    # load the entire dataset, should contain dev and test
    ds = load_dataset("ofir408/MedConceptsQA", subset)
    test = ds["test"]

    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if num_few_shot > 0:
        # few-shot examples are chosen based on the `vocab` and `level`
        few_shot_data = _create_few_shot_data(ds["dev"], num_few_shot, answer_format=answer_format)

    def _map(row: dict, idx: int | None = None) -> dict:
        vocab = row["vocab"]
        level = row["level"]
        question_stem, options = _extract_question_and_options(row)
        row_id = row.get("id") or row.get("concept_id") or idx or question_stem
        answer = row["answer_id"]
        few_shot_prompt = few_shot_data.get((vocab, level), "") if num_few_shot > 0 else ""

        if shuffle_answers and answer in options:
            options, answer, _ = randomize_multiple_choice(
                options=options,
                answer_choice=answer,
                seed=shuffle_seed,
                row_id=row_id,
            )

        formatted_question = _format_stem_with_options(question_stem, options)

        full_question = (
            "Answer A, B, C, D according to the answer to this multiple choice question.\n"
            + few_shot_prompt
            + ("\n" if len(few_shot_prompt) > 0 else "")
            + formatted_question
            + "\nAnswer:"
        )
        answer_text = options.get(answer, row.get("answer"))
        info: dict[str, Any] = {"answer_text": answer_text}
        if shuffle_answers:
            info["options"] = options

        return {"question": full_question, "answer": answer, "info": info}

    load_from_cache_file = not shuffle_answers
    mapped = test.map(
        _map,
        with_indices=True,
        remove_columns=test.column_names,
        load_from_cache_file=load_from_cache_file,
    )

    if answer_format == AnswerFormat.XML:
        system_prompt = THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict | None = None) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        eval_dataset=mapped,
        rubric=rubric,
        system_prompt=system_prompt,
        parser=parser,
    )
