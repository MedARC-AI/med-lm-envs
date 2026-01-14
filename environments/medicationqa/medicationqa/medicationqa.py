from pathlib import Path
from typing import Any

import pandas as pd
import verifiers as vf
from datasets import Dataset, load_from_disk
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.utils import (
    default_judge_api_key,
    download_file,
    judge_sampling_args_and_headers,
    medarc_cache_dir,
)
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

from medicationqa.judge_prompts import JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE

disable_progress_bar()  # suppress datasets mapping progress bar

# 1. where the data lives upstream
DATA_URL = "https://raw.githubusercontent.com/abachaa/Medication_QA_MedInfo2019/master/MedInfo2019-QA-Medications.xlsx"

# 2. where we’ll cache locally
CACHE_SUBDIR = "medicationqa"
XLSX_FILENAME = "Medication_QA.xlsx"

# Scored dimensions must match the keys emitted by judge_prompts.JUDGE_TEMPLATE
JUDGE_DIMENSIONS = ["accuracy", "completeness", "clarity"]


def _resolve_cache_dir(cache_dir: Path | str | None = None) -> Path:
    resolved = medarc_cache_dir(cache_dir) / CACHE_SUBDIR
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_dataset(cache_dir: Path | str | None = None) -> Dataset:
    """
    Load the cached MedicationQA dataset, downloading and materializing it if needed.
    """
    cache_path = _resolve_cache_dir(cache_dir)
    dataset_info = cache_path / "dataset_info.json"
    if dataset_info.exists():
        return load_from_disk(str(cache_path))

    xlsx_path = download_file(url=DATA_URL, dest=cache_path / XLSX_FILENAME)
    df = pd.read_excel(xlsx_path)

    def _format_row(row: dict[str, Any]) -> dict[str, Any]:
        info = dict()
        info["question_type"] = str(row.get("Question Type", "")).strip()
        info["question_focus"] = str(row.get("Focus (Drug)", "")).strip()
        info["answer_section_title"] = str(row.get("Section Title", "")).strip()
        info["answer_url"] = str(row.get("URL", "")).strip()

        return {
            "question": str(row.get("Question", "")).strip(),
            "answer": str(row.get("Answer", "")).strip(),
            "info": info,
        }

    ds = Dataset.from_pandas(df).map(_format_row, remove_columns=list(df.columns))
    ds.save_to_disk(str(cache_path))
    return ds


def _extract_completion_text(completion: Messages) -> str:
    """Extract the assistant’s text content from a chat-style completion."""
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def _coerce_score(value: Any) -> float | None:
    """Best-effort conversion of a score value to a float in Python, or `None` if not possible."""

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


def _compute_normalized_reward(scores: dict[str, dict[str, Any]]) -> float:
    """Normalize per-dimension judge scores to a single value in [0.0, 1.0].

    Each dimension is expected to be on a 1–5 scale. Scores are clamped to
    [0, 5], divided by 5 to map to [0, 1], and then averaged across the
    dimensions listed in `JUDGE_DIMENSIONS`.
    """

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


def load_environment(
    cache_dir: Path | str | None = None,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """Load the MedicationQA (MedInfo 2019) evaluation environment.

    This environment:
    - downloads and caches the MedicationQA Excel source,
    - converts it to a Hugging Face dataset (single split),
    - and evaluates model completions with an LLM-as-a-judge rubric adapted
    from MedHELM / MedDialog (accuracy, completeness, clarity).

    Args:
        cache_dir: Optional override for the cache location. Defaults to `MEDARC_CACHE_DIR/medicationqa`
            (i.e., `~/.cache/medarc/medicationqa`).
        judge_model: Model identifier to use for the judge (e.g. "gpt-4o").
        judge_base_url: Optional base URL for a non-OpenAI-compatible endpoint (e.g. Ollama).
        judge_api_key: API key for the judge model. Defaults to `default_judge_api_key`, which
            checks common env vars (e.g., OpenAI/Prime `OPENAI_API_KEY`, `JUDGE_API_KEY`).
        **kwargs: Additional arguments forwarded to `vf.SingleTurnEnv`.

    Returns:
        A configured `verifiers.SingleTurnEnv` ready to be passed to `vf-eval`.
    """
    eval_dataset = _load_dataset(cache_dir)

    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)
    judge_parser = JSONParser(fields=list(JUDGE_DIMENSIONS))

    judge_rubric = vf.JudgeRubric(
        parallelize_scoring=True,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    async def reward_medicationqa(
        prompt: Messages,
        completion: Messages,
        info: Info,
        state: State,
    ) -> float:
        question = str(state.get("question") or "")
        gold_response = str(state.get("answer") or "")
        model_answer = _extract_completion_text(completion)

        judge_prompt = JUDGE_TEMPLATE.format(
            question=question,
            response=model_answer,
            gold_response=gold_response,
            output_format=JUDGE_OUTPUT_JSON,
        )

        try:
            judge_raw = await judge_rubric.judge(judge_prompt, model_answer, gold_response, state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)
        except AttributeError:
            judge_raw = await judge_rubric.judge(judge_prompt, "", "", state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)

        if parsed is None:
            parsed = {dim: {"score": None, "explanation": None, "raw": None} for dim in JUDGE_DIMENSIONS}

        normalized = _compute_normalized_reward(parsed)

        info.setdefault("judge_feedback", []).append(
            {
                "scores": parsed,
                "raw_judge": str(judge_raw),
            }
        )
        return normalized

    judge_rubric.add_reward_func(reward_medicationqa, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=eval_dataset,
        eval_dataset=eval_dataset,
        system_prompt="You are a helpful, safety-conscious medical assistant. Give a brief but accurate response to the following question.",
        rubric=judge_rubric,
        name="medicationqa",
        **kwargs,
    )
