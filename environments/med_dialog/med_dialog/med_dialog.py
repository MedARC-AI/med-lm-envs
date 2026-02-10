import os
from pathlib import Path
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.judging import MultiJudge, MultiJudgeRubric
from medarc_verifiers.rewards import normalize_helm_reward
from medarc_verifiers.utils import download_file
from verifiers.types import Info, Messages, State

from med_dialog.judge_prompts import JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE

disable_progress_bar()  # suppress datasets progress indicators

BASE_URL = "https://worksheets.codalab.org/rest/bundles/0x82f0c47f6d3e4462ae9ef8ea39eebe64/contents/blob"
SPLITS: Sequence[str] = ("train", "test")
SUBSETS: Sequence[str] = ("healthcaremagic", "icliniq")

PROMPT = "Generate a one sentence summary of this patient-doctor conversation."
PROMPT_THINK = "Think step-by-step inside <think>...</think> tags then generate a one sentence summary of this patient-doctor conversation."

JUDGE_RESPONSE_PARSER = JSONParser(fields=["accuracy", "completeness", "clarity"])
JUDGE_DIMENSIONS = ("accuracy", "completeness", "clarity")


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        env_override = os.getenv("MEDDIALOG_CACHE_DIR")
        if env_override:
            return Path(env_override)
        return Path.home() / ".cache" / "meddialog"
    return Path(cache_dir)


def _load_split_dataset(subsets: Sequence[str], split: str, cache_path: Path) -> Dataset:
    datasets: list[Dataset] = []

    for subset in subsets:
        json_path = cache_path / subset / f"{split}.json"
        download_file(url=f"{BASE_URL}/{subset}/{split}.json", dest=json_path, verify=False)

        dataset_dict = load_dataset("json", data_files=str(json_path), field="data")
        raw_split = dataset_dict["train"]

        def _format_row(row: dict[str, Any], *, subset: str = subset) -> dict[str, Any]:
            try:
                example_id = int(row.get("id"))
            except (TypeError, ValueError):
                example_id = int(row.get("index", 0))

            prompt = str(row.get("src", ""))
            response = str(row.get("tgt", ""))

            info = dict(row)
            info["conversation"] = prompt
            info["reference_response"] = response
            info["subset"] = subset

            info.pop("src", None)
            info.pop("tgt", None)

            return {
                "id": example_id,
                "question": prompt,
                "answer": response,
                "info": info,
            }

        formatted = raw_split.map(_format_row, remove_columns=raw_split.column_names)
        datasets.append(formatted)

    if not datasets:
        raise ValueError("No datasets were loaded for the requested MedDialog subsets.")

    return datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)


def load_environment(
    use_think: bool = False,
    cache_dir: Path | str | None = None,
    judge_model: str | list[str] = "gpt-4o-mini",
    judge_base_url: str | list[str] | None = None,
    judge_api_key: str | list[str] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """
    MedDialog summarization environment evaluated with an LLM judge.
    """
    cache_path = _resolve_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_split_dataset(subsets=SUBSETS, split="train", cache_path=cache_path)
    eval_dataset = _load_split_dataset(subsets=SUBSETS, split="test", cache_path=cache_path)

    judge_parser = JSONParser(fields=["accuracy", "completeness", "clarity"])
    completion_parser = vf.ThinkParser(extract_fn=lambda x: x) if use_think else None
    multi_judge = MultiJudge.from_env_args(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_prompt="{question}",
        completion_parser=completion_parser,
    )
    rubric = MultiJudgeRubric(multi_judge)

    async def reward_meddialog(
        prompt: Messages,
        completion: Messages,
        info: Info,
        state: State,
    ) -> float:
        conversation = str(info.get("conversation") or "")
        gold_response = str(info.get("reference_response") or "")
        completion_text = _extract_completion_text(completion)

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
                parsed = judge_parser.parse(str(result.raw), strip=True)
            except AttributeError:
                result = await rubric.rerun_judge(result, judge_prompt, completion_text, gold_response, state)
                parsed = judge_parser.parse(str(result.raw), strip=True)
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

    rubric.add_reward_func(reward_meddialog, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=PROMPT_THINK if use_think else PROMPT,
        rubric=rubric,
        **kwargs,
    )


def _extract_completion_text(completion: Messages) -> str:
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)
