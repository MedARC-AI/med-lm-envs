from typing import Any

import verifiers as vf
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar
from aci_bench.judge_prompts import JUDGE_DIMENSIONS, JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.prompts import XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets progress indicators

# section 4.3.1 of the ACI-Bench paper
prompt = """\
Summarize the conversation to generate a clinical note with four sections:
HISTORY OF PRESENT ILLNESS, PHYSICAL EXAM, RESULTS, ASSESSMENT AND PLAN.

The conversation is:
{conversation}
"""


def _to_vf_format(dataset: Dataset) -> Dataset:
    return dataset.map(
        lambda row: {
            "question": prompt.format(conversation=row["dialogue"]),
            "answer": row["note"],
            "task": "aci-bench",
            "info": {
                "conversation": row["dialogue"],
                "reference_response": row["note"],
                "transcript_version": row["transcript_version"],
            },
        }
    )


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _compute_normalized_reward(
    scores: dict[str, dict[str, Any]],
    min_score: float | None = None,
    max_score: float | None = None,
) -> float:
    """Accumulate per-dimension judge scores normalized from [min_score, max_score] to [0.0, 1.0]"""
    min_score = min_score if min_score is not None else 1
    max_score = max_score if max_score is not None else 5

    total_dims = len(JUDGE_DIMENSIONS)
    if total_dims == 0:
        return 0.0

    accumulated = 0.0
    for dimension in JUDGE_DIMENSIONS:
        score = _coerce_score(scores.get(dimension, {}).get("score"))
        if score is None:
            continue
        clamped = max(0.0, min(max_score, score))
        accumulated += clamped / max_score

    return max(0.0, min(min_score, accumulated / total_dims))


def _extract_completion_text(completion: Messages, parser: vf.Parser) -> str:
    # try using parser first -- so that, for example, the judge only sees the
    # final answer and not the thinking process. or could use for formatting scores.
    # completion_text = parser.parse_answer(completion)
    # if completion_text is not None:
    #     return completion_text
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def load_environment(
    subset: str = "all",
    transcript_version: str = "all",
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    judge_model: str = "gpt-5-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    # -------- load dataset and convert to vf format --------
    if subset == "all":
        subsets = ["virtassist", "virtscribe", "aci"]
        ds_dicts = [load_dataset("mkieffer/ACI-Bench-MedARC", name=s) for s in subsets]
        dataset = DatasetDict(
            {split: concatenate_datasets([d[split] for d in ds_dicts]) for split in ds_dicts[0].keys()}
        )
    else:
        dataset = load_dataset("mkieffer/ACI-Bench-MedARC", name=subset)
    if transcript_version != "all":
        dataset = dataset.filter(lambda row: row["transcript_version"] == transcript_version)
    train_ds = _to_vf_format(dataset["train"])
    # valid_ds = _to_vf_format(dataset["valid"])
    test_ds = _to_vf_format(concatenate_datasets([dataset["test1"], dataset["test2"], dataset["test3"]]))

    # -------- normalize answer_format --------
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or XML_SYSTEM_PROMPT
        parser_fields = ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        system_prompt = system_prompt or BOXED_SYSTEM_PROMPT
        parser = vf.Parser(extract_fn=extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    # -------- setup judge --------
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)

    judge_parser = JSONParser(fields=["accuracy", "completeness", "clarity"])
    judge_rubric = vf.JudgeRubric(
        judge_client=AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers),
        judge_model=judge_model,
        judge_prompt="{question}",  # gets filled in during judge_rubric.judge() call. a little hacky but means we can fill the judge template in this file
        parser=parser,
        judge_sampling_args=sampling_args,
    )

    async def judge_rubric_reward(completion: Messages, info: Info, state: State, **kwargs: Any) -> float:
        conversation = str(info.get("conversation") or "")
        gold_response = str(info.get("reference_response") or "")
        completion_text = _extract_completion_text(completion, parser)

        judge_prompt = JUDGE_TEMPLATE.format(
            conversation=conversation,
            response=completion_text,
            gold_response=gold_response,
            output_format=JUDGE_OUTPUT_JSON,
        )

        # judge_prompt assigned to question var inside judge_rubric.judge() method.
        # judge_raw returned as string.
        try:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_response, state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)
        except AttributeError:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_response, state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)
        if parsed is None:
            parsed = {dimension: {"score": None, "explanation": None, "raw": None} for dimension in JUDGE_DIMENSIONS}

        normalized = _compute_normalized_reward(parsed)

        info.setdefault("judge_feedback", []).append(
            {
                "scores": parsed,
                "raw_judge": judge_raw,
            }
        )

        return normalized

    judge_rubric.add_reward_func(judge_rubric_reward, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        rubric=judge_rubric,
        parser=parser,
        **kwargs,
    )
