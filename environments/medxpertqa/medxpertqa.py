from enum import Enum
import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import extract_boxed_answer

disable_progress_bar()  # suppress datasets progress indicators


class QuestionType(str, Enum):
    REASONING = "reasoning"
    UNDERSTANDING = "understanding"
    ALL = "all"


def _get_system_prompt(use_think: bool, answer_format: AnswerFormat) -> str:
    if answer_format == AnswerFormat.BOXED:
        think_system_prompt = "You are a helpful medical assistant. Think step-by-step inside <think>...</think> tags. Put your final answer within \\boxed{}."
        no_think_system_prompt = (
            "You are a helpful medical assistant. Think step-by-step and put your final answer within \\boxed{}."
        )
    elif answer_format == AnswerFormat.XML:
        think_system_prompt = "You are a helpful medical assistant. Think step-by-step inside <think>...</think> tags. Put your final answer within <answer>...</answer> tags."
        no_think_system_prompt = "You are a helpful medical assistant. Think step-by-step and put your final answer within <answer>...</answer> tags."
    else:
        raise ValueError(f"Unsupported answer format: {answer_format}")
    return think_system_prompt if use_think else no_think_system_prompt


def _format_question_with_options(question: str, options: dict[str, str]) -> str:
    """Attach the answer choices to the stem in the original layout."""
    if not options:
        return question
    formatted_options = " ".join(f"({letter}) {text}" for letter, text in options.items())
    if not formatted_options:
        return question
    if "Answer Choices:" in question:
        stem, _, _ = question.partition("Answer Choices:")
        stem = stem.strip()
    else:
        stem = question.strip()
    return f"{stem}\nAnswer Choices: {formatted_options}"


def load_environment(
    question_type: str | QuestionType = QuestionType.ALL,
    use_think: bool = False,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    """
    MedXpertQA environment using equality with the "label" column as the eval criterion.
    This environment loads the MedXpertQA dataset and compares model responses (diagnosis) with the ground truth in "label" column.
    """
    full_dataset = load_dataset("TsinghuaC3I/MedXpertQA", "Text")
    test_dataset = full_dataset["test"]
    question_type = QuestionType(question_type) if isinstance(question_type, str) else question_type
    if question_type != QuestionType.ALL:
        test_dataset = test_dataset.filter(lambda x: str(x["question_type"]).strip().lower() == question_type.value)

    def _map(example: dict) -> dict:
        raw_options = example.get("options") or {}
        options = dict(raw_options) if isinstance(raw_options, dict) else {}

        answer_letter = str(example.get("label", "")).strip().upper()
        if shuffle_answers and answer_letter and answer_letter in options:
            randomized_options, answer_letter, _ = randomize_multiple_choice(
                options=options,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=example.get("question", None),
            )
            options = randomized_options

        answer_text = options.get(answer_letter)

        info = dict(example)
        if shuffle_answers:
            info["options"] = options
            info["label"] = answer_letter
        info["answer_text"] = answer_text

        return {
            "question": _format_question_with_options(example.get("question", ""), options),
            "answer": answer_letter if answer_letter else "",
            "info": info,
            "task": question_type.value,
        }

    # Disable the Datasets cache when shuffling answers
    load_from_cache_file = False if shuffle_answers else True
    mapped = test_dataset.map(
        _map,
        remove_columns=test_dataset.column_names,
        load_from_cache_file=load_from_cache_file,
    )

    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    system_prompt = _get_system_prompt(use_think, answer_format)

    if answer_format == AnswerFormat.XML:
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(eval_dataset=mapped, system_prompt=system_prompt, rubric=rubric, parser=parser)
