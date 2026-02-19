from typing import Any

import verifiers as vf
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar
from aci_bench.judge_prompts import JUDGE_DIMENSIONS, JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.prompts import XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.judging import MultiJudge, MultiJudgeRubric
from medarc_verifiers.rewards import normalize_helm_reward
from medarc_verifiers.types import Messages
from verifiers.types import Info, State
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
    judge_model: str | list[str] = "gpt-5-mini",
    judge_base_url: str | list[str] | None = None,
    judge_api_key: str | list[str] | None = None,
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

    judge_parser = JSONParser(fields=["accuracy", "completeness", "clarity"])
    multi_judge = MultiJudge.from_env_args(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_prompt="{question}",
        completion_parser=parser,
    )
    rubric = MultiJudgeRubric(multi_judge, parser=parser)

    async def judge_rubric_reward(
        completion: Messages,
        info: Info,
        state: State,
        **kwargs: Any,
    ) -> float:
        conversation = str(info.get("conversation") or "")
        gold_response = str(info.get("reference_response") or "")
        completion_text = _extract_completion_text(completion, parser)

        judge_prompt = JUDGE_TEMPLATE.format(
            conversation=conversation,
            response=completion_text,
            gold_response=gold_response,
            output_format=JUDGE_OUTPUT_JSON,
        )

        judge_results = await rubric.judge(judge_prompt, completion_text, gold_response, state)
        judge_entries = []
        scores = []
        for result in judge_results:
            entry_error = result.error
            try:
                parsed = judge_parser.parse(result.raw, strip=True)
            except AttributeError:
                result = await rubric.rerun_judge(result, judge_prompt, completion_text, gold_response, state)
                parsed = judge_parser.parse(result.raw, strip=True)
                entry_error = result.error

            if parsed is None:
                parsed = {dimension: {"score": None, "explanation": None} for dimension in JUDGE_DIMENSIONS}

            normalized = normalize_helm_reward(parsed, dimensions=JUDGE_DIMENSIONS)
            score = normalized if result.raw is not None else None
            scores.append(score)
            judge_entries.append(
                {
                    "model": result.model,
                    "raw": result.raw,
                    "error": entry_error,
                    "scores": parsed,
                    "score": score,
                }
            )

        aggregated = rubric.multi_judge.mean(scores)
        info.setdefault("judge_feedback", []).append(
            {
                "judges": judge_entries,
                "score": aggregated,
            }
        )

        return aggregated

    rubric.add_reward_func(judge_rubric_reward, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        rubric=rubric,
        parser=parser,
        **kwargs,
    )
