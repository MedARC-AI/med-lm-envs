from enum import Enum
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer


disable_progress_bar()  # suppress datasets mapping progress bar


class Vocab(str, Enum):
    ATC = "atc"
    ICD10CM = "icd10cm"
    ICD10PROC = "icd10proc"
    ICD9CM = "icd9cm"
    ICD9PROC = "icd9proc"
    ICD10CM_SAMPLE = "icd10cm_sample"
    ALL = "all"


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def _extract_question_and_options(row: dict) -> tuple[str, dict[str, str]]:
    """Return the stem without embedded options and an ordered map of options."""
    question = row.get("question", "") or ""
    options: dict[str, str] = {}
    for idx, label in enumerate(("A", "B", "C", "D"), start=1):
        value = row.get(f"option{idx}", "")
        if value not in ("", None):
            options[label] = value

    def _looks_like_option(line: str) -> bool:
        candidate = line.strip()
        for label in ("A", "B", "C", "D"):
            for sep in (".", ")", ":", "-"):
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
    vocab: Vocab | str = Vocab.ICD10CM_SAMPLE,
    difficulty: Difficulty | str = Difficulty.EASY,
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
        vocab: code vocabulary to subset dataset, if `icd10cm_sample` choses a subsample of
            `icd10cm`, if `all` choses the entire dataset (default: icd10cm_sample)
        difficulty: difficulty level to subset dataset (used in conjunction with vocab).
            Ignored if vocab is `all` (default: easy)
        shuffle_answers: whether to shuffle the answer choices (default: False)
        shuffle_seed: deterministic seed forwarded to the shuffler (default: 1618)
        answer_format: format of the answer in the model's output (default: XML)
    Returns:
        vf.Environment: the single-turn evaluation environment
    """
    vocab = Vocab(vocab) if isinstance(vocab, str) else vocab
    level = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty

    if vocab is Vocab.ALL:
        subset = "all"
    else:
        subset = f"{vocab.value}_{level.value}"

    if vocab == Vocab.ICD10CM_SAMPLE:
        # load only the sample subset, should contain dev and test
        ds = load_dataset("sameedkhan/medconceptsqa-sample_medarc_2k", subset.replace("_sample", ""))
        test = ds["test"]
    else:
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

    load_from_cache_file = False if shuffle_answers else True
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
