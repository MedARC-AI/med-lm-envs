import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

import verifiers as vf
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.judging import MultiJudge, MultiJudgeRubric
from medarc_verifiers.rewards import normalize_helm_reward
from medarc_verifiers.utils import download_file, medarc_cache_dir
from medarc_verifiers.types import Messages
from verifiers.types import Info, State

from mtsamples_procedures.judge_prompts import JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE

disable_progress_bar()

GIT_HASH = "c4c252443fa9c52afb6960f53e51be278639bea2"
BASE_URL = f"https://raw.githubusercontent.com/raulista1997/benchmarkdata/{GIT_HASH}/mtsample_procedure"
API_URL = "https://api.github.com/repos/raulista1997/benchmarkdata/contents/mtsample_procedure"


PROMPT = """Here are information about a patient, return a reasonable treatment plan for the patient."""

PROMPT_THINK = """Here are information about a patient. Think step-by-step inside <think>...</think> tags about the key clinical details, then return a reasonable treatment plan for the patient."""

JUDGE_DIMENSIONS = ("accuracy", "completeness", "clarity")


def _resolve_cache_dir(cache_dir: Path | str | None) -> Path:
    if cache_dir is None:
        env_override = os.getenv("MTSAMPLES_PROCEDURES_CACHE_DIR")
        if env_override:
            return Path(env_override)
    return medarc_cache_dir(cache_dir) / "mtsamples_procedures"


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


def _remove_sections(text: str) -> str:
    for section in ["PLAN:", "SUMMARY:", "FINDINGS:"]:
        if section in text:
            return text.split(section, 1)[0].strip()
    return text


def _download_txt_files(cache_path: Path) -> list[Path]:
    txt_dir = cache_path / "txt_files"
    txt_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(txt_dir.glob("*.txt"))
    if len(existing_files) > 0:
        return existing_files

    files_json = download_file(API_URL, cache_path / "files.json")
    files_data = json.loads(files_json.read_text(encoding="utf-8"))

    downloaded_files = []
    for file_info in files_data:
        if file_info["name"].endswith(".txt"):
            encoded_name = quote(file_info["name"])
            file_url = f"{BASE_URL}/{encoded_name}"
            dest_path = txt_dir / file_info["name"]

            download_file(file_url, dest_path)
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

        input_text = _remove_sections(text)

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
    judge_model: str | list[str] = "openai/gpt-5-mini",
    judge_base_url: str | list[str] | None = None,
    judge_api_key: str | list[str] | None = None,
    **kwargs: Any,
) -> vf.Environment:
    eval_dataset = _load_dataset(cache_dir)

    judge_parser = JSONParser(fields=list(JUDGE_DIMENSIONS))

    completion_parser = vf.ThinkParser(extract_fn=lambda x: x) if use_think else None
    multi_judge = MultiJudge.from_env_args(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_prompt="{question}",
        completion_parser=completion_parser,
    )
    rubric = MultiJudgeRubric(multi_judge)

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

        judge_results = await rubric.judge(judge_prompt, completion_text, gold_plan, state)
        judge_entries = []
        scores = []
        for result in judge_results:
            entry_error = result.error
            try:
                parsed = judge_parser.parse(str(result.raw), strip=True)
            except AttributeError:
                result = await rubric.rerun_judge(result, judge_prompt, completion_text, gold_plan, state)
                parsed = judge_parser.parse(str(result.raw), strip=True)
                entry_error = result.error

            if parsed is None:
                parsed = {
                    dimension: {"score": None, "explanation": None, "raw": None} for dimension in JUDGE_DIMENSIONS
                }

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

    rubric.add_reward_func(reward_mtsamples, weight=1.0)

    system_prompt = PROMPT_THINK if use_think else PROMPT

    return vf.SingleTurnEnv(
        dataset=None,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
        **kwargs,
    )


def _extract_completion_text(completion: Messages) -> str:
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)
