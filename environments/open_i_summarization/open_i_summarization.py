"""
Open-I Summarization Environment

Evaluation environment for radiology report summarization (findings → impression).
Uses LLM-as-judge evaluation plus automatic metrics (BLEU, ROUGE, BERTScore).

Dataset: medarc/open-i-summarization
"""

import re
from typing import Any

import evaluate
import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State

disable_progress_bar()  # suppress datasets progress bar

# --- Judge Prompt Template ---
# Evaluation criteria adapted from:
# Buckley T. et al., "Accuracy of a vision-language model on challenging medical cases"
# Nature Medicine (2024). https://doi.org/10.1038/s41591-024-02855-5
# See: Methods - Reader study section

JUDGE_TEMPLATE = """\
You are a radiology expert tasked with evaluating how well a model summarized radiology findings into an impression.

Your goal is to assess the quality of the model's generated impression compared to the reference impression.
You will rate the impression along three dimensions: correctness, completeness, and conciseness.

Definitions (from Nature Medicine reader study criteria):
- Correctness: Does the summary include false information? Evaluate precision—penalize any fabricated or incorrect information not supported by the findings.
- Completeness: Does the summary completely capture important information? Evaluate recall—the amount of clinically important detail retained from the input findings.
- Conciseness: Does the summary contain non-important information? Evaluate brevity—the value of a summary decreases with superfluous information. A good summary should be as brief as possible while capturing key findings.

IMPORTANT for Conciseness scoring:
- Compare the model's impression length to the reference impression length.
- If the model output is significantly longer than the reference without adding clinically important information, penalize heavily.
- Ideal summaries are similar in length to or shorter than the reference while capturing key findings.

The radiology findings will be provided in these tags:
<radiology_findings>
{findings}
</radiology_findings>

The model's generated impression will be provided in these tags:
<model_impression>
{response}
</model_impression>

The reference impression will be provided in these tags:
<reference_impression>
{reference}
</reference_impression>

For each dimension:
1. Provide a brief explanation (1–3 sentences) describing why you assigned the score.
2. Then assign a score from 1 to 5:

Correctness scoring:
- 5: No false or fabricated information
- 4: Minor inaccuracy that doesn't affect clinical interpretation
- 3: Some inaccurate information present
- 2: Multiple inaccuracies or one significant error
- 1: Major fabrications or errors

Completeness scoring:
- 5: All clinically important information captured
- 4: Most important information captured, minor omissions
- 3: Key findings present but some important details missing
- 2: Several important findings omitted
- 1: Critical findings missing

Conciseness scoring:
- 5: Similar length or shorter than reference while capturing key findings
- 4: Slightly longer (up to 1.5x) but additional content is clinically relevant
- 3: Moderately longer (1.5-2x) with some unnecessary detail
- 2: Much longer (2-3x) with substantial superfluous information
- 1: Excessively verbose (>3x reference length) or contains much irrelevant content

{output_format}
"""

JUDGE_OUTPUT_JSON = """
Output your evaluation as a single valid JSON object matching the following structure:
{
  "correctness": {
    "reason": "Brief explanation of why this score was given.",
    "score": 0
  },
  "completeness": {
    "reason": "Brief explanation of why this score was given.",
    "score": 0
  },
  "conciseness": {
    "reason": "Brief explanation of why this score was given.",
    "score": 0
  }
}

Ensure the output is valid JSON:
- Use double quotes (") for all keys and string values.
- Escape any internal quotes inside the reason fields.
- Do not include any additional text outside the JSON object.
- Do not explain your reasoning outside the JSON object; all justification must appear only in the "reason" fields.
"""

# Scored dimensions must match the keys emitted by JUDGE_TEMPLATE
# Note: "correctness" replaces "accuracy" per Nature Medicine paper terminology
JUDGE_DIMENSIONS = ["correctness", "completeness", "conciseness"]


def _extract_completion_text(completion: Messages) -> str:
    """Extract the assistant's text content from a chat-style completion."""
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def extract_answer_section(completion_text: str) -> str:
    """Extract final answer after think tags if present."""
    if not completion_text:
        return ""
    if "<think>" in completion_text and "</think>" in completion_text:
        return re.sub(r".*?</think>", "", completion_text, flags=re.DOTALL).strip()
    return completion_text.strip()


def _coerce_score(value: Any) -> float | None:
    """Best-effort conversion of a score value to a float, or None if not possible."""
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


def _compute_normalized_judge_reward(scores: dict[str, dict[str, Any]]) -> float:
    """Normalize per-dimension judge scores to a single value in [0.0, 1.0].

    Each dimension is expected to be on a 1–5 scale. Scores are clamped to
    [0, 5], divided by 5 to map to [0, 1], and then averaged across dimensions.
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
    split: str = "test",
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    compute_auto_metrics: bool = True,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """
    Load the Open-I Summarization evaluation environment.

    This environment evaluates radiology findings → impression summarization using:
    1. LLM-as-judge evaluation (accuracy, completeness, conciseness)
    2. Automatic metrics: BLEU, ROUGE, BERTScore (optional)

    Args:
        split: Dataset split to use ('train', 'validation', 'test'). Default: 'test'.
        judge_model: Model identifier for the LLM judge. Default: 'gpt-4o-mini'.
        judge_base_url: Base URL for judge API (for non-OpenAI endpoints).
        judge_api_key: API key for judge model. Falls back to env vars if not provided.
        compute_auto_metrics: Whether to compute BLEU/ROUGE/BERTScore. Default: True.
        system_prompt: Custom system prompt. Uses default if not provided.
        **kwargs: Additional arguments forwarded to vf.SingleTurnEnv.

    Returns:
        A configured vf.SingleTurnEnv for radiology summarization evaluation.
    """
    # Load dataset
    eval_dataset = load_dataset("medarc/open-i-summarization", split=split)

    def _map(ex: dict) -> dict:
        """Map dataset example to environment format."""
        return {
            "question": ex["inputs"].strip(),  # radiology findings
            "answer": ex["target"].strip(),  # reference impression
            "info": {
                "idx": ex["idx"],
                "findings": ex["inputs"].strip(),
            },
        }

    eval_dataset = eval_dataset.map(_map, remove_columns=eval_dataset.column_names)

    # Default system prompt for summarization
    final_system_prompt = system_prompt or (
        "Summarize the radiology report findings into an impression with minimal text."
    )

    # Initialize automatic metrics if enabled
    bleu_metric = None
    rouge_metric = None
    bertscore_metric = None

    if compute_auto_metrics:
        try:
            bleu_metric = evaluate.load("bleu")
            rouge_metric = evaluate.load("rouge")
            bertscore_metric = evaluate.load("bertscore")
        except Exception as e:
            print(f"Warning: Could not load automatic metrics: {e}")
            compute_auto_metrics = False

    # Judge client setup
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    # Remove extra_body as OpenAI doesn't support the usage tracking parameter
    sampling_args.pop("extra_body", None)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)
    judge_parser = JSONParser(fields=list(JUDGE_DIMENSIONS))

    judge_rubric = vf.JudgeRubric(
        parallelize_scoring=True,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    async def reward_open_i_summarization(
        prompt: Messages,
        completion: Messages,
        info: Info,
        state: State,
    ) -> float:
        """Evaluate radiology summarization using LLM-judge and automatic metrics."""
        findings = str(info.get("findings", state.get("question", "")))
        reference = str(state.get("answer", ""))
        model_response = extract_answer_section(_extract_completion_text(completion))

        # --- LLM-as-Judge Evaluation ---
        judge_prompt = JUDGE_TEMPLATE.format(
            findings=findings,
            response=model_response,
            reference=reference,
            output_format=JUDGE_OUTPUT_JSON,
        )

        try:
            judge_raw = await judge_rubric.judge(judge_prompt, model_response, reference, state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)
        except AttributeError:
            judge_raw = await judge_rubric.judge(judge_prompt, "", "", state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)

        if parsed is None:
            parsed = {dim: {"score": None, "reason": None} for dim in JUDGE_DIMENSIONS}

        judge_reward = _compute_normalized_judge_reward(parsed)

        # Store judge feedback
        info.setdefault("judge_feedback", []).append(
            {
                "scores": parsed,
                "normalized_reward": judge_reward,
                "raw_judge": str(judge_raw),
            }
        )

        # --- Automatic Metrics (BLEU, ROUGE, BERTScore) ---
        auto_metrics: dict[str, Any] = {}

        if compute_auto_metrics and model_response and reference:
            predictions = [model_response]
            references = [reference]

            # BLEU
            try:
                bleu_scores = bleu_metric.compute(predictions=predictions, references=references)
                auto_metrics["bleu"] = bleu_scores.get("bleu", 0.0)
            except Exception:
                auto_metrics["bleu"] = 0.0

            # ROUGE
            try:
                rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
                auto_metrics["rouge1"] = rouge_scores.get("rouge1", 0.0)
                auto_metrics["rouge2"] = rouge_scores.get("rouge2", 0.0)
                auto_metrics["rougeL"] = rouge_scores.get("rougeL", 0.0)
                auto_metrics["rougeLsum"] = rouge_scores.get("rougeLsum", 0.0)
            except Exception:
                auto_metrics["rouge1"] = 0.0
                auto_metrics["rouge2"] = 0.0
                auto_metrics["rougeL"] = 0.0
                auto_metrics["rougeLsum"] = 0.0

            # BERTScore
            try:
                bert_scores = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
                auto_metrics["bertscore_precision"] = (
                    bert_scores["precision"][0] if bert_scores.get("precision") else 0.0
                )
                auto_metrics["bertscore_recall"] = bert_scores["recall"][0] if bert_scores.get("recall") else 0.0
                auto_metrics["bertscore_f1"] = bert_scores["f1"][0] if bert_scores.get("f1") else 0.0
            except Exception:
                auto_metrics["bertscore_precision"] = 0.0
                auto_metrics["bertscore_recall"] = 0.0
                auto_metrics["bertscore_f1"] = 0.0

            info["auto_metrics"] = auto_metrics

        # Return the LLM-judge reward as the primary metric
        return judge_reward

    judge_rubric.add_reward_func(reward_open_i_summarization, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=eval_dataset,
        eval_dataset=eval_dataset,
        system_prompt=final_system_prompt,
        rubric=judge_rubric,
        name="open_i_summarization",
        **kwargs,
    )
