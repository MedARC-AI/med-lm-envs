import os
import requests
from pathlib import Path
from typing import Any
from urllib.parse import quote

import verifiers as vf
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

from mtsamples_replicate.judge_prompts import JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE

disable_progress_bar()

GIT_HASH = "ebc104a4f96c5b7602242f301e081e9934a23344"
BASE_URL = f"https://raw.githubusercontent.com/raulista1997/benchmarkdata/{GIT_HASH}/mtsamples_processed"
API_URL = "https://api.github.com/repos/raulista1997/benchmarkdata/contents/mtsamples_processed"


PROMPT = """Here are information about a patient, return a reasonable treatment plan for the patient."""

PROMPT_THINK = """Here are information about a patient. Think step-by-step inside <think>...</think> tags about the key clinical details, then return a reasonable treatment plan for the patient."""

JUDGE_DIMENSIONS = ("accuracy", "completeness", "clarity")

# This follows HELM's MTSamplesReplicateScenario:
# PLAN is removed from input, SUMMARY/FINDINGS are preserved.


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        env_override = os.getenv("MTSAMPLES_replicate_CACHE_DIR")
        if env_override:
            return Path(env_override)
        return Path.home() / ".cache" / "mtsamples_replicate"
    return Path(cache_dir)


def _extract_sections(text: str) -> tuple[str | None, str | None, str | None]:
    plan, summary, findings = None, None, None
    text_upper = text.upper()

    if "PLAN:" in text_upper:
        idx = text_upper.find("PLAN:")
        after_header = text[idx + len("PLAN:") :]
        first_line = after_header.split("\n", 1)[0].strip()
        plan = first_line if first_line else None

    if "SUMMARY:" in text_upper:
        idx = text_upper.find("SUMMARY:")
        after_header = text[idx + len("SUMMARY:") :]
        first_line = after_header.split("\n", 1)[0].strip()
        summary = first_line if first_line else None

    if "FINDINGS:" in text_upper:
        idx = text_upper.find("FINDINGS:")
        after_header = text[idx + len("FINDINGS:") :]
        first_line = after_header.split("\n", 1)[0].strip()
        findings = first_line if first_line else None

    return plan, summary, findings


def _remove_plan_section(text: str) -> str:
    sections = ["PLAN:"]

    for section in sections:
        if section in text.upper():
            idx = text.upper().find(section)
            text = text[:idx].strip()

    return text


def _download_txt_files(cache_path: Path) -> list[Path]:
    txt_dir = cache_path / "txt_files"
    txt_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(txt_dir.glob("*.txt"))
    if len(existing_files) > 0:
        return existing_files

    response = requests.get(API_URL)
    response.raise_for_status()
    files_data = response.json()

    downloaded_files = []
    for file_info in files_data:
        if file_info["name"].endswith(".txt"):
            encoded_name = quote(file_info["name"])
            file_url = f"{BASE_URL}/{encoded_name}"
            dest_path = txt_dir / file_info["name"]

            file_response = requests.get(file_url)
            file_response.raise_for_status()
            dest_path.write_text(file_response.text, encoding="utf-8")
            downloaded_files.append(dest_path)

    return downloaded_files


def _load_dataset(cache_dir: Path | str | None = None) -> Dataset:
    cache_path = _resolve_cache_dir(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset_cache = cache_path / "dataset"
    if dataset_cache.exists():
        return Dataset.load_from_disk(str(dataset_cache))

    txt_files = _download_txt_files(cache_path)

    examples = []

    for idx, txt_file in enumerate(txt_files):
        text = txt_file.read_text(encoding="utf-8")

        plan, summary, findings = _extract_sections(text)

        reference = None
        extracted_section = None
        if plan:
            reference = plan
            extracted_section = "PLAN"
        elif summary:
            reference = summary
            extracted_section = "SUMMARY"
        elif findings:
            reference = findings
            extracted_section = "FINDINGS"

        if not reference:
            continue

        input_text = _remove_plan_section(text)

        examples.append(
            {
                "id": idx,
                "question": input_text,
                "answer": reference,
                "info": {
                    "filename": txt_file.name,
                    "extracted_section": extracted_section,
                    "procedure_note": input_text,
                    "reference_plan": reference,
                },
            }
        )

    dataset = Dataset.from_list(examples)

    dataset.save_to_disk(str(dataset_cache))

    return dataset


def load_environment(
    use_think: bool = False,
    cache_dir: Path | str | None = None,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs: Any,
) -> vf.Environment:
    eval_dataset = _load_dataset(cache_dir)

    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_parser = JSONParser(fields=list(JUDGE_DIMENSIONS))

    judge_rubric = vf.JudgeRubric(
        parser=vf.ThinkParser(extract_fn=lambda x: x) if use_think else None,
        parallelize_scoring=True,
        judge_client=AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers),
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    async def reward_mtsamples(
        prompt: Messages,
        completion: Messages,
        info: Info,
        state: State,
    ) -> float:
        procedure_note = str(info.get("procedure_note") or "")
        gold_plan = str(info.get("reference_plan") or "")
        completion_text = _extract_completion_text(completion)

        judge_prompt = JUDGE_TEMPLATE.format(
            procedure_note=procedure_note,
            response=completion_text,
            gold_plan=gold_plan,
            output_format=JUDGE_OUTPUT_JSON,
        )

        try:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_plan, state)
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

    judge_rubric.add_reward_func(reward_mtsamples, weight=1.0)

    system_prompt = PROMPT_THINK if use_think else PROMPT

    return vf.SingleTurnEnv(
        dataset=eval_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
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
