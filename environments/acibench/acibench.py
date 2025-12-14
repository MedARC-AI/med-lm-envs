import os
import zipfile
from pathlib import Path
from typing import Any

import bert_score
import bleurt.score
import numpy as np
import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.parsers import JSONParser
from openai import AsyncOpenAI
from rouge import Rouge
from verifiers.types import Info, Messages, State

from acibench_judge_prompts import JUDGE_TEMPLATE, JUDGE_OUTPUT_JSON


def medarc_cache_dir() -> Path:
    default_cache_path = Path.home() / ".cache" / "medarc"
    cache_dir = Path(os.getenv("MEDARC_CACHE_DIR", default_cache_path))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, destination: Path):
    import requests
    from tqdm import tqdm

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(destination, "wb") as f, tqdm(
        desc=destination.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)


def _get_text_from_completion(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last_item = completion[-1]
        if isinstance(last_item, dict):
            return str(last_item.get("content", ""))
        return str(last_item)
    return str(completion)


def _build_prompt(transcript: str) -> str:
    return (
        "Summarize the conversation to generate a clinical note with four sections:\n"
        "1. HISTORY OF PRESENT ILLNESS\n"
        "2. PHYSICAL EXAM\n"
        "3. RESULTS\n"
        "4. ASSESSMENT AND PLAN\n\n"
        "The conversation is:\n\n{transcript}"
    ).format(transcript=transcript)


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


def load_environment(
    system_prompt: str | None = None,
    device: str | None = None,
    num_few_shot: int = 0,
    eval_method: str = "judge",
    judge_model: str = "gpt-4o",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> vf.Environment:
    if device and "cuda" in device:
        gpu_id = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    try:
        dataset = load_dataset("harsh-c137/aci-bench-medarc-eval", split="train")
        split_ds = dataset.train_test_split(test_size=0.2, seed=42)
        train_ds = split_ds["train"]
        val_ds = split_ds["test"]
    except Exception as e:
        raise ConnectionError(f"Could not load dataset. Error: {e}")

    def _map_example(example: dict) -> dict:
        question = _build_prompt(example["transcript"])

        info = {
            "dialogue": example["transcript"],
            "reference_note": example["note"],
        }

        return {"question": question, "answer": example["note"], "info": info}

    original_columns = list(train_ds.features)
    train_mapped = train_ds.map(_map_example, remove_columns=original_columns)
    val_mapped = val_ds.map(_map_example, remove_columns=original_columns)

    few_shot_examples = []
    if num_few_shot > 0:
        few_shot_ds = train_ds.shuffle(seed=42).select(range(num_few_shot))
        for ex in few_shot_ds:
            user_content = _build_prompt(ex["transcript"])
            few_shot_examples.append({"role": "user", "content": user_content})
            few_shot_examples.append({"role": "assistant", "content": ex["note"]})

    rouge_factory = _create_rouge_reward_func()
    bertscore_factory = _create_bertscore_reward_func(device)
    bleurt_factory = _create_bleurt_reward_func()

    if eval_method == "judge":
        api_key = judge_api_key or os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
        judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key)
        
        judge_response_parser = JSONParser(fields=["accuracy", "completeness", "clarity"])
        judge_dimensions = ["accuracy", "completeness", "clarity"]

        final_rubric = vf.JudgeRubric(
            parser=None,
            parallelize_scoring=True,
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt="{question}",
        )

        async def reward_acibench(
            prompt: Messages,
            completion: Messages,
            info: Info,
            state: State,
        ) -> float:
            dialogue = str(info.get("dialogue") or "")
            reference_note = str(info.get("reference_note") or "")
            completion_text = _get_text_from_completion(completion)

            judge_prompt = JUDGE_TEMPLATE.format(
                conversation=dialogue,
                candidate_note=completion_text,
                reference_note=reference_note,
                output_format=JUDGE_OUTPUT_JSON,
            )

            judge_raw = await final_rubric.judge(
                [{"role": "user", "content": judge_prompt}], completion_text, reference_note, state
            )

            parsed = judge_response_parser.parse(str(judge_raw), strip=True)
            if parsed is None:
                parsed = {dim: {"score": None} for dim in judge_dimensions}

            accumulated = 0.0
            total_dims = len(judge_dimensions)
            for dim in judge_dimensions:
                score_val = _coerce_score(parsed.get(dim, {}).get("score"))
                if score_val is not None:
                    clamped = max(0.0, min(5.0, score_val))
                    accumulated += clamped / 5.0
            
            normalized_score = max(0.0, min(1.0, accumulated / total_dims)) if total_dims > 0 else 0.0

            state.setdefault("judge_feedback", []).append(
                {
                    "scores": parsed,
                    "raw_judge": str(judge_raw),
                }
            )
            return normalized_score

        final_rubric.add_reward_func(reward_acibench, weight=1.0, name="judge_reward")
        final_rubric.add_reward_func(rouge_factory, weight=0, name="rouge")
        final_rubric.add_reward_func(bertscore_factory, weight=0, name="bertscore")
        final_rubric.add_reward_func(bleurt_factory, weight=0, name="bleurt")

    elif eval_method == "metrics":
        final_rubric = vf.Rubric()
        final_rubric.add_reward_func(rouge_factory, weight=1.0, name="rouge")
        final_rubric.add_reward_func(bertscore_factory, weight=1.0, name="bertscore")
        final_rubric.add_reward_func(bleurt_factory, weight=1.0, name="bleurt")

    else:
        raise ValueError("eval_method must be one of 'judge' or 'metrics'")

    final_system_prompt = system_prompt or (
        "You are an expert medical scribe who generates clinical notes from doctor-patient conversations."
    )

    return vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=val_mapped,
        few_shot=few_shot_examples,
        system_prompt=final_system_prompt,
        rubric=final_rubric,
    )


def _create_rouge_reward_func():
    rouge_scorer = Rouge()

    def rouge_reward(completion: any, answer: str, **kwargs) -> float:
        clean_completion = _get_text_from_completion(completion)
        if not clean_completion.strip() or not answer.strip():
            return 0.0
        try:
            scores = rouge_scorer.get_scores(clean_completion, answer, avg=True)
            return float(np.mean([scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-l"]["f"]]))
        except Exception:
            return 0.0

    return rouge_reward


def _create_bertscore_reward_func(device: str | None):
    def bertscore_reward(completion: any, answer: str, **kwargs) -> float:
        clean_completion = _get_text_from_completion(completion)
        if not clean_completion.strip() or not answer.strip():
            return 0.0
        _, _, f1 = bert_score.score(
            [clean_completion],
            [answer],
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            device=device,
        )
        return f1.mean().item()

    return bertscore_reward


def _create_bleurt_reward_func():
    bleurt_checkpoint = medarc_cache_dir() / "acibench" / "bleurt-20"
    if not bleurt_checkpoint.exists():
        print("Downloading BLEURT-20 checkpoint (one-time setup)...")
        url = "https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip"
        zip_path = bleurt_checkpoint.parent / f"{bleurt_checkpoint.name}.zip"
        bleurt_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(bleurt_checkpoint.parent)
        zip_path.unlink()
    bleurt_scorer = bleurt.score.BleurtScorer(str(bleurt_checkpoint))

    def bleurt_reward(completion: any, answer: str, **kwargs) -> float:
        clean_completion = _get_text_from_completion(completion)
        if not clean_completion.strip() or not answer.strip():
            return 0.0
        scores = bleurt_scorer.score(references=[answer], candidates=[clean_completion])
        return np.mean(scores)

    return bleurt_reward