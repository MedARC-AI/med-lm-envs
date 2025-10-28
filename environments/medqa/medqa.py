from typing import Dict, Optional

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets progress indicators


def _build_prompt(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question:{question}\n{opts}\nAnswer:"


def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
    """Reward based on shared multiple-choice accuracy grading."""
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    """
    MedQA-USMLE-4-options multiple-choice evaluation
    - Train split = dataset
    - Test split = eval_dataset
    - Supports reasoning (use_think=True) or non-reasoning models
    """
    ds = load_dataset("GBaker/MedQA-USMLE-4-options")

    def _map(ex, idx=None):
        q: str = ex["question"]
        options: Dict[str, str] = ex["options"]
        gold_letter: str = ex["answer_idx"].strip().upper()

        if shuffle_answers and gold_letter in options:
            options, gold_letter, _ = randomize_multiple_choice(
                options=options,
                answer_choice=gold_letter,
                seed=shuffle_seed,
                row_id=ex.get("id", idx),
            )

        return {
            "question": _build_prompt(q, options),
            "answer": gold_letter,
            "info": {
                "answer_text": options.get(gold_letter, ""),
                **({"options": options} if shuffle_answers else {}),
            },
        }

    load_from_cache_file = not shuffle_answers
    train_mapped = ds["train"].map(
        _map,
        with_indices=True,
        remove_columns=ds["train"].column_names,
        load_from_cache_file=load_from_cache_file,
    )
    test_mapped = ds["test"].map(
        _map,
        with_indices=True,
        remove_columns=ds["test"].column_names,
        load_from_cache_file=load_from_cache_file,
    )

    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
        system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=test_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
