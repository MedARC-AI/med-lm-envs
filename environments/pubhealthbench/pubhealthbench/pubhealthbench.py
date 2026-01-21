import re
from enum import Enum
from typing import Optional

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers.json_parser import JSONParser
from medarc_verifiers.parsers.xml_parser import XMLParser
from medarc_verifiers.prompts import AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from openai import AsyncOpenAI
from verifiers.types import Info, State
from verifiers.utils.data_utils import extract_boxed_answer

from pubhealthbench.prompts import (
    ANSWER_INSTRUCTION_BOXED,
    ANSWER_INSTRUCTION_XML,
    JUDGE_TEMPLATE,
    QUESTION_TEMPLATE_FREEFORM,
    QUESTION_TEMPLATE_MCQ,
    QUESTION_SYSTEM_PROMPT,
)

disable_progress_bar()


class Split(str, Enum):
    """Dataset split selector for PubHealthBench."""

    FULL = "full"  # test split (7929) - MCQ
    VALIDATION = "validation"  # validation split (161) - MCQ
    REVIEWED = "reviewed"  # reviewed split (760) - MCQ
    FREEFORM = "freeform"  # reviewed split (760) - LLM-as-judge
    FREEFORM_VALID = "freeform_valid"  # validation split (161) - LLM-as-judge


class Category(str, Enum):
    """Subject category selector for PubHealthBench."""

    ALL = "all"
    BLOOD_SAFETY_HEPATITIS_STIS_AND_HIV = "blood_safety_hepatitis_stis_and_hiv"
    CHEMICALS_TOXICOLOGY = "chemicals_toxicology"
    CLIMATE_AND_HEALTH = "climate_and_health"
    GASTRO_FOOD_SAFETY = "gastro_food_safety"
    HCAI_FUNGAL_AMR_ANTIMICROBIAL_USE_SEPSIS = "hcai_fungal_amr_antimicrobial_use_sepsis"
    HEALTH_PROTECTION_IN_INCLUSION_HEALTH_SETTINGS = "health_protection_in_inclusion_health_settings"
    OTHER = "other"
    RADIATION = "radiation"
    TUBERCULOSIS_TRAVEL_ZOONOTIC_AND_EMERGING_INFECTIONS = "tuberculosis_travel_zoonotic_and_emerging_infections"
    VPDS_IMMUNISATION = "vpds_immunisation"


def extract_answer_section(completion_text: str) -> str:
    """Extract final answer after think tags."""
    if not completion_text:
        return ""
    if "<think>" in completion_text and "</think>" in completion_text:
        return re.sub(r".*?</think>", "", completion_text, flags=re.DOTALL).strip()
    return completion_text.strip()


def _build_mcq_prompt(question: str, options: list[str], answer_format: AnswerFormat) -> str:
    """Build MCQ prompt using the paper's question format."""
    opts = "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))
    instruction = ANSWER_INSTRUCTION_BOXED if answer_format == AnswerFormat.BOXED else ANSWER_INSTRUCTION_XML
    return QUESTION_TEMPLATE_MCQ.format(question=question, options=opts, answer_instruction=instruction)


def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
    """Reward based on shared multiple-choice accuracy grading."""
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    split: Split | str = Split.REVIEWED,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    category: Category | str = Category.ALL,
    judge_model: str = "gpt-5-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> vf.Environment:
    """
    PubHealthBench evaluation environment.

    A public health benchmark with questions derived from UK Health Security Agency guidance.

    Args:
        split: Dataset split:
            - "full" (MCQ, 7929 questions)
            - "validation" (MCQ, 161 questions)
            - "reviewed" (MCQ, 760 questions)
            - "freeform" (LLM-as-judge, 760 questions)
            - "freeform_valid" (LLM-as-judge, 161 questions)
        system_prompt: Custom system prompt (default: paper's system prompt).
        shuffle_answers: Whether to shuffle answer options (MCQ only).
        shuffle_seed: Seed for deterministic answer shuffling (MCQ only).
        answer_format: Answer format - "xml" (default) or "boxed" (MCQ only).
        category: Filter rows by one category value (exact match).
        judge_model: Model for LLM-as-judge evaluation (freeform only).
        judge_base_url: Base URL for judge API (freeform only).
        judge_api_key: API key for judge (freeform only).

    Returns:
        A vf.Environment configured for PubHealthBench evaluation.
    """
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    split = Split(split) if isinstance(split, str) else split
    category = Category(category) if isinstance(category, str) else category

    if split in (Split.FREEFORM, Split.FREEFORM_VALID):
        hf_split = "validation" if split == Split.FREEFORM_VALID else "reviewed"
        return _load_freeform_environment(
            hf_split=hf_split,
            system_prompt=system_prompt,
            category=category,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
        )
    else:
        return _load_mcq_environment(
            split=split,
            system_prompt=system_prompt,
            shuffle_answers=shuffle_answers,
            shuffle_seed=shuffle_seed,
            answer_format=answer_format,
            category=category,
        )


def _load_mcq_environment(
    split: Split,
    system_prompt: Optional[str],
    shuffle_answers: bool,
    shuffle_seed: int | None,
    answer_format: AnswerFormat,
    category: Category,
) -> vf.Environment:
    """Load PubHealthBench MCQ environment."""
    ds = load_dataset("Joshua-Harris/PubHealthBench")

    def _map(ex, idx=None):
        question: str = ex["question"]
        options: list[str] = ex["options"]
        answer_idx: int = ex["answer_index"]

        if shuffle_answers:
            labels = [chr(ord("A") + i) for i in range(len(options))]
            options, gold_letter, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=labels,
                seed=shuffle_seed,
                row_id=ex.get("question_id", idx),
            )
        else:
            gold_letter = chr(ord("A") + answer_idx)

        return {
            "question": _build_mcq_prompt(question, options, answer_format),
            "answer": gold_letter,
            "info": {
                "answer_text": options[answer_idx],
                "category": ex.get("category", ""),
            },
        }

    load_from_cache_file = not shuffle_answers

    if split == Split.FULL:
        hf_split = "test"
    elif split == Split.VALIDATION:
        hf_split = "validation"
    elif split == Split.REVIEWED:
        hf_split = "reviewed"

    eval_ds = ds[hf_split]
    if category != Category.ALL:
        eval_ds = eval_ds.filter(
            lambda row: row.get("category") == category.value,
            load_from_cache_file=load_from_cache_file,
        )

    eval_ds = eval_ds.map(
        _map,
        with_indices=True,
        remove_columns=ds[hf_split].column_names,
        load_from_cache_file=load_from_cache_file,
    )

    if answer_format == AnswerFormat.BOXED:
        parser = vf.Parser(extract_boxed_answer)
    else:
        parser = XMLParser(fields=["answer"], answer_field="answer")

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        eval_dataset=eval_ds,
        system_prompt=system_prompt or QUESTION_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
    )


def _load_freeform_environment(
    hf_split: str,
    system_prompt: Optional[str],
    category: Category,
    judge_model: str,
    judge_base_url: str | None,
    judge_api_key: str | None,
) -> vf.Environment:
    """Load PubHealthBench freeform environment with LLM-as-judge evaluation."""
    ds = load_dataset("Joshua-Harris/PubHealthBench")
    eval_dataset = ds[hf_split]
    if category != Category.ALL:
        eval_dataset = eval_dataset.filter(
            lambda row: row.get("category") == category.value, load_from_cache_file=False
        )

    def _map(ex):
        question: str = ex["question"]
        options: list[str] = ex["options"]
        answer_idx: int = ex["answer_index"]
        answer_text = options[answer_idx]
        context = ex.get("retrieved_context_for_judge", "")

        return {
            "question": QUESTION_TEMPLATE_FREEFORM.format(question=question),
            "answer": answer_text,
            "info": {
                "question": question,
                "context": context,
                "category": ex.get("category", ""),
            },
        }

    eval_dataset = eval_dataset.map(_map, remove_columns=eval_dataset.column_names)

    # Judge client setup
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)
    judge_parser = JSONParser(fields=["reasoning", "predicted_correct"], answer_field="predicted_correct")

    judge_rubric = vf.JudgeRubric(
        parser=judge_parser,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    async def accuracy(judge, prompt, completion, answer, state: State, info: Info) -> float:
        """Evaluate using LLM-as-judge."""
        completion_text = completion if isinstance(completion, str) else str(completion)
        response = extract_answer_section(completion_text)

        judge_prompt = JUDGE_TEMPLATE.format(
            question=info.get("question", ""),
            context=info.get("context", ""),
            ground_truth_answer=answer,
            given_answer=response,
        )
        try:
            judge_response = await judge_rubric.judge(judge_prompt, "", "", state)
            parsed = judge_parser.parse(judge_response)
        except AttributeError:
            judge_response = await judge_rubric.judge(judge_prompt, "", "", state)
            parsed = judge_parser.parse(judge_response)

        predicted_correct = getattr(parsed, "predicted_correct", False)
        if isinstance(predicted_correct, str):
            predicted_correct = predicted_correct.lower().strip()
            predicted_correct = "true" in predicted_correct and "false" not in predicted_correct

        reasoning = getattr(parsed, "reasoning", "")

        info.setdefault("judge_feedback", []).append(
            {
                "predicted_correct": predicted_correct,
                "reasoning": reasoning,
                "raw_judge": str(judge_response),
            }
        )
        return 1.0 if predicted_correct else 0.0

    judge_rubric.add_reward_func(accuracy, weight=1.0)

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt or QUESTION_SYSTEM_PROMPT,
        rubric=judge_rubric,
    )
