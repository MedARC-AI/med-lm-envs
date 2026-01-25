import re
from enum import Enum
from typing import Optional

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.parsers.xml_parser import XMLParser
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from openai import AsyncOpenAI
from verifiers.types import Info, State
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer


class CareQASplit(Enum):
    """Mode selector for CareQA environment."""

    EN = "en"
    OPEN = "open"


# --- MCQ Helpers ---


def _build_mcq_prompt(question: str, options: dict[str, str]) -> str:
    """Create an MCQ prompt."""
    formatted_opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"Question: {question}\nChoices:\n{formatted_opts}\nAnswer:"


def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
    """Reward based on shared multiple-choice accuracy grading."""
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


# --- Open-Ended Helpers ---


JUDGE_TEMPLATE = """You are grading an AI assistant's answer to a medical/science exam questions.

Input:
- <question>: The exam question.
- <reference_answer>: The correct answer.
- <assistant_answer>: The AI's response to grade.

Task: Determine if the assistant's answer is correct or incorrect by comparing it to the reference answer and output your grade in <grade>...</grade> tags.

Grading Rules:
- Assume the reference answer is correct and reflects the expected exam solution.
- Focus on factual content and meaning, not style, length, or confidence.

Correct if the assistant's answer conveys the same essential fact(s) as the reference, including:
- Synonyms, acronyms (expanded or abbreviated), or rephrasing with equivalent meaning
- Slightly more general/specific phrasing that captures the key concept
- Shorter or longer answers that express the tested fact without contradictions
- Additional supporting details that don't contradict the reference

Incorrect if any of these apply:
- Different main concept, mechanism, structure, or relationship
- Contradicts the reference on key points (wrong organ, drug, effect, process, etc.)
- Contains clearly incorrect information
- Too vague/incomplete to match the reference
- Merely repeats question words without the core information from the reference

Be strict: clear mismatches on main concepts or incorrect claims = Incorrect.

<question>{question}</question>
<reference_answer>{answer}</reference_answer>
<assistant_answer>{response}</assistant_answer>

Briefly explain whether the assistant's answer matches or conflicts with the reference. Then output your grade as:

<grade>[Correct or Incorrect]</grade>
""".strip()


def extract_answer_section(completion_text: str) -> str:
    """Extract final answer after think tags."""
    if not completion_text:
        return ""
    if "<think>" in completion_text and "</think>" in completion_text:
        return re.sub(r".*?</think>", "", completion_text, flags=re.DOTALL).strip()
    return completion_text.strip()


def load_environment(
    split: str | CareQASplit,
    system_prompt: Optional[str] = None,
    # MCQ-specific options
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    # Open-ended specific options
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs,
) -> vf.Environment:
    """
    CareQA evaluation environment supporting both MCQ and Open-Ended modes.

    Args:
        split: CareQASplit.EN for multiple-choice or CareQASplit.OPEN for open-ended QA.
        system_prompt: Custom system prompt (uses mode-appropriate default if None).
        shuffle_answers: Shuffle MCQ answer options (MCQ mode only).
        shuffle_seed: Seed for answer shuffling (MCQ mode only).
        judge_model: Model to use for LLM-as-judge evaluation (Open-ended mode only).
        judge_base_url: Base URL for judge API (Open-ended mode only).
        judge_api_key: API key for judge (Open-ended mode only).

    Returns:
        A vf.Environment configured for the selected mode.
    """
    split = CareQASplit(split) if isinstance(split, str) else split
    if split == CareQASplit.EN:
        return _load_mcq_environment(
            system_prompt=system_prompt,
            shuffle_answers=shuffle_answers,
            shuffle_seed=shuffle_seed,
        )
    elif split == CareQASplit.OPEN:
        return _load_open_ended_environment(
            system_prompt=system_prompt,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
        )
    else:
        raise ValueError(f"Invalid mode: {split}")


def _load_mcq_environment(
    system_prompt: Optional[str],
    shuffle_answers: bool,
    shuffle_seed: int | None,
) -> vf.Environment:
    """Load CareQA multiple-choice environment."""
    eval_dataset = load_dataset("HPAI-BSC/CareQA", "CareQA_en", split="test")

    def _map(ex, idx=None):
        options = {"A": ex["op1"], "B": ex["op2"], "C": ex["op3"], "D": ex["op4"]}
        gold_letter = ["A", "B", "C", "D"][ex["cop"] - 1]

        if shuffle_answers and gold_letter in options:
            options, gold_letter, _ = randomize_multiple_choice(
                options=options,
                answer_choice=gold_letter,
                seed=shuffle_seed,
                row_id=ex.get("id", idx),
            )

        return {
            "question": _build_mcq_prompt(ex["question"], options),
            "answer": gold_letter,
            "info": {
                "answer_text": options.get(gold_letter, None),
                **({"options": options} if shuffle_answers else {}),
            },
        }

    load_from_cache_file = not shuffle_answers
    eval_dataset = eval_dataset.map(
        _map,
        with_indices=True,
        remove_columns=eval_dataset.column_names,
        load_from_cache_file=load_from_cache_file,
    )

    parser = vf.Parser(extract_boxed_answer)
    final_system_prompt = BOXED_SYSTEM_PROMPT or system_prompt

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=final_system_prompt,
    )


def _load_open_ended_environment(
    system_prompt: Optional[str],
    judge_model: str,
    judge_base_url: str | None,
    judge_api_key: str | None,
) -> vf.Environment:
    """Load CareQA open-ended environment with LLM-as-judge evaluation."""
    eval_dataset = load_dataset("HPAI-BSC/CareQA", "CareQA_en_open", split="test")

    def _map(ex):
        info = {}
        info["question"] = ex["question"].strip()
        return {
            "question": ex["question"].strip(),
            "answer": ex.get("answer_explanation", ex.get("answer", "")),
            "task": "careqa_open",
            "info": info,
        }

    eval_dataset = eval_dataset.map(_map, remove_columns=eval_dataset.column_names)

    final_system_prompt = system_prompt or (
        "Instructions: The following text is a medical question. Answer it in the most factual, concise, and informative way possible."
    )

    # Judge client setup
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)
    judge_parser = XMLParser(fields=["grade"], answer_field="grade")

    judge_rubric = vf.JudgeRubric(
        parser=judge_parser,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    async def accuracy(judge, prompt, completion, answer, state: State, info: Info) -> float:
        """Evaluate medical equivalence using LLM-as-judge."""
        completion_text = completion if isinstance(completion, str) else str(completion)
        response = extract_answer_section(completion_text)

        try:
            judge_prompt = JUDGE_TEMPLATE.format(question=info.get("question", ""), answer=answer, response=response)
            judge_response = await judge_rubric.judge(judge_prompt, "", "", state)
            grade = judge_parser.parse_answer(judge_response).strip().lower()
        except AttributeError:
            judge_response = await judge_rubric.judge(judge_prompt, "", "", state)
            grade = judge_parser.parse_answer(judge_response).strip().lower()

        info.setdefault("judge_feedback", []).append(
            {
                "grade": grade,
                "raw_judge": str(judge_response),
            }
        )

        if "correct" in grade and "incorrect" not in grade:
            return 1.0
        else:
            return 0.0

    judge_rubric.add_reward_func(accuracy, weight=1.0)

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=final_system_prompt,
        rubric=judge_rubric,
    )
