import json
from typing import Optional

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers.xml_parser import XMLParser
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from openai import AsyncOpenAI
from verifiers.types import Info, State
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()

MCQ_QUESTION_TEMPLATE = """\
Question: {question}
Choices:
{choices}
Answer:"""

OPEN_QUESTION_TEMPLATE = """\
{question}"""

JUDGE_TEMPLATE = """\
You are evaluating an AI assistant's answer to a medical question.

<question>{question}</question>
<reference_answer>{answer}</reference_answer>
<assistant_answer>{response}</assistant_answer>

Is the assistant's answer medically equivalent to the reference answer?
Consider synonyms, paraphrasing, and reasonable generalizations as correct.
Answer [yes/no]."""


def _parse_options(options_str: str | None) -> dict[str, str] | None:
    """Parse the options field from the dataset.

    The options field can be a JSON string representing a dict or list,
    or it can be empty/None for open-ended questions.
    """
    if not options_str or options_str.strip() in ("", "None", "null", "{}"):
        return None
    try:
        parsed = json.loads(options_str)
    except (json.JSONDecodeError, TypeError):
        return None

    if isinstance(parsed, dict):
        if not parsed:
            return None
        return {str(k): str(v) for k, v in parsed.items()}
    if isinstance(parsed, list):
        if not parsed:
            return None
        labels = [chr(ord("A") + i) for i in range(len(parsed))]
        return dict(zip(labels, [str(v) for v in parsed]))
    return None


def _format_mcq_prompt(question: str, options: dict[str, str]) -> str:
    """Format a multiple-choice question prompt."""
    choices = "\n".join(f"{k}. {v}" for k, v in options.items())
    return MCQ_QUESTION_TEMPLATE.format(question=question, choices=choices)


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> vf.Environment:
    """
    MedReason medical reasoning evaluation environment.

    Supports both multiple-choice and open-ended questions from the MedReason
    dataset (UCSC-VLAA/MedReason). MCQ items are graded by accuracy; open-ended
    items use LLM-as-a-Judge evaluation.

    Args:
        use_think: Enable chain-of-thought reasoning with <think> tags.
        system_prompt: Custom system prompt override.
        shuffle_answers: Shuffle MCQ answer options.
        shuffle_seed: Seed for deterministic answer shuffling.
        answer_format: Answer format (xml or boxed).
        judge_model: Model to use for LLM-as-judge evaluation.
        judge_base_url: Base URL for judge API.
        judge_api_key: API key for judge model.
    """
    ds = load_dataset("UCSC-VLAA/MedReason", split="train")

    # Set up judge for open-ended questions
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)
    judge_rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    def _map(ex, idx=None):
        question_text = ex["question"]
        answer_text = ex["answer"]
        options = _parse_options(ex.get("options"))

        if options:
            # MCQ: find gold letter by matching answer text to options
            gold_letter = None
            for letter, opt_text in options.items():
                if opt_text.strip().lower() == answer_text.strip().lower():
                    gold_letter = letter
                    break

            if gold_letter is None:
                # Answer is the letter itself
                candidate = answer_text.strip().upper()
                if candidate in options:
                    gold_letter = candidate
                else:
                    gold_letter = "A"

            if shuffle_answers and gold_letter in options:
                options, gold_letter, _ = randomize_multiple_choice(
                    options=options,
                    answer_choice=gold_letter,
                    seed=shuffle_seed,
                    row_id=ex.get("id_in_dataset", idx),
                )

            return {
                "question": _format_mcq_prompt(question_text, options),
                "answer": gold_letter,
                "info": {
                    "is_mcq": True,
                    "answer_text": options.get(gold_letter, answer_text),
                    "dataset_name": ex.get("dataset_name", ""),
                    **({} if not shuffle_answers else {"options": options}),
                },
            }
        else:
            # Open-ended question
            return {
                "question": OPEN_QUESTION_TEMPLATE.format(question=question_text),
                "answer": answer_text,
                "info": {
                    "is_mcq": False,
                    "dataset_name": ex.get("dataset_name", ""),
                    "question_raw": question_text,
                },
            }

    load_from_cache_file = not shuffle_answers
    eval_dataset = ds.map(
        _map,
        with_indices=True,
        remove_columns=ds.column_names,
        load_from_cache_file=load_from_cache_file,
    )

    # Set up parser based on answer format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    if answer_format == AnswerFormat.XML:
        final_system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
        final_system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    async def medreason_reward_func(
        completion,
        answer,
        info: Info,
        state: State,
        **kwargs,
    ) -> float:
        """Unified reward: accuracy for MCQ, LLM judge for open-ended."""
        is_mcq = info.get("is_mcq", False)

        if is_mcq:
            parsed_answer = parser.parse_answer(completion) or ""
            answer_text_val = info.get("answer_text", None)
            is_correct = multiple_choice_accuracy(
                llm_answer=parsed_answer,
                answer_letter=answer,
                answer_text=answer_text_val,
            )
            return 1.0 if is_correct else 0.0
        else:
            # Open-ended: use LLM judge
            parsed = parser.parse(completion, last=True)
            model_answer = getattr(parsed, "answer", None)

            if model_answer is not None:
                question_raw = info.get("question_raw", "")
                judge_prompt = JUDGE_TEMPLATE.format(
                    question=question_raw,
                    answer=answer,
                    response=model_answer,
                )
                judge_response = await judge_rubric.judge(judge_prompt, model_answer, answer, state)
                judge_response_clean = judge_response.strip().lower()
            else:
                judge_response_clean = "no"
                judge_response = "no answer"

            info.setdefault("judge_feedback", []).append(
                {
                    "parsed": judge_response_clean,
                    "raw_judge": str(judge_response),
                }
            )

            if "yes" in judge_response_clean and "no" not in judge_response_clean:
                return 1.0
            else:
                return 0.0

    judge_rubric.add_reward_func(medreason_reward_func, weight=1.0)

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=final_system_prompt,
        parser=parser,
        rubric=judge_rubric,
    )
