import os
from pathlib import Path
from typing import Any, Sequence

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.utils import default_judge_api_key, download_file, judge_sampling_args_and_headers
from openai import AsyncOpenAI
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
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    """
    MedDialog summarization environment evaluated with an LLM judge.
    """
    cache_path = _resolve_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    train_dataset = _load_split_dataset(subsets=SUBSETS, split="train", cache_path=cache_path)
    eval_dataset = _load_split_dataset(subsets=SUBSETS, split="test", cache_path=cache_path)

    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_parser = JSONParser(fields=["accuracy", "completeness", "clarity"])

    judge_rubric = vf.JudgeRubric(
        parser=vf.ThinkParser(extract_fn=lambda x: x) if use_think else None,
        parallelize_scoring=True,
        judge_client=AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers),
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

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

        try:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_response, state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)
        except AttributeError:
            judge_raw = await judge_rubric.judge(judge_prompt, "", "", state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)

        if parsed is None:
            parsed = {dimension: {"score": None, "explanation": None, "raw": None} for dimension in JUDGE_DIMENSIONS}

        normalized = _compute_normalized_reward(parsed)

        info.setdefault("judge_feedback", []).append(
            {
                "scores": parsed,
                "raw_judge": str(judge_raw),
            }
        )

        return normalized

    judge_rubric.add_reward_func(reward_meddialog, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=PROMPT_THINK if use_think else PROMPT,
        rubric=judge_rubric,
        **kwargs,
    )


def _extract_completion_text(completion: Messages) -> str:
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _compute_normalized_reward(scores: dict[str, dict[str, Any]]) -> float:
    total_dims = len(JUDGE_DIMENSIONS)
    if total_dims == 0:
        return 0.0

    accumulated = 0.0
    for dimension in JUDGE_DIMENSIONS:
        score = _coerce_score(scores.get(dimension, {}).get("score"))
        if score is None:
            continue
        clamped = max(0.0, min(5.0, score))
        accumulated += clamped / 5.0

    return max(0.0, min(1.0, accumulated / total_dims))
