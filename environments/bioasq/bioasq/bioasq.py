from typing import Any, Optional

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import JSONParser
from medarc_verifiers.prompts import XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, Messages, State
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer
from .judge_prompts import JUDGE_DIMENSIONS, JUDGE_OUTPUT_JSON, JUDGE_TEMPLATE

disable_progress_bar()  # suppress datasets progress indicators

# System prompt aligned with BioASQ Task 1b synthesis requirement  
PROMPT = "Provide a comprehensive answer to the following biomedical question strictly based on the provided snippets."

def _parse_bioasq_hf(example: dict[str, Any]) -> dict[str, Any]:
    """Parses Hugging Face BioASQ format into the Med-LM-Env structure."""
    # Note: Structure depends on the specific HF version; 
    # BioASQ 1b/13b typically uses 'body', 'ideal_answer', and 'snippets'
    question_text = example.get("body", example.get("question", ""))
    ideal_answer = example.get("ideal_answer", "")
    snippets = example.get("snippets", [])
    
    # Handle different snippet formats (list of strings vs list of dicts)
    snippet_texts = []
    for s in snippets:
        if isinstance(s, dict):
            snippet_texts.append(s.get("text", ""))
        else:
            snippet_texts.append(str(s))

    return {
        "question": question_text,
        "answer": ideal_answer,
        "info": {
            "question_type": example.get("type", "summary"),
            "ideal_answer": ideal_answer,
            "context": "\n".join(snippet_texts),
            "documents": example.get("documents", [])
        }
    }

def _coerce_score(value: Any) -> float | None:
    """Convert score value to float or None if invalid."""
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

    return max(0.0, min(1.0, accumulated / total_dims))


def _extract_completion_text(completion: Messages, parser: vf.Parser) -> str:
    """Extract completion text, respecting parser if available."""
    if isinstance(completion, list) and completion:
        last_msg = completion[-1]
        if isinstance(last_msg, dict):
            return str(last_msg.get("content", ""))
    return str(completion)


def load_environment(
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    system_prompt: Optional[str] = None,
    **kwargs: Any,
) -> vf.Environment:
    """Load BioASQ environment for biomedical question answering evaluation.
    
    Args:
        answer_format: Format for model responses (XML or BOXED)
        judge_model: Model to use for LLM-as-judge evaluation
        judge_base_url: Base URL for judge model API
        judge_api_key: API key for judge model
        system_prompt: Custom system prompt (defaults to BioASQ-specific prompt)
        **kwargs: Additional arguments passed to SingleTurnEnv
    """
    # Load from the provided Hugging Face path
    raw_ds = load_dataset("kroshan/BioASQ", split="train")
    dataset = raw_ds.map(lambda x, idx: _parse_bioasq_hf(x), with_indices=True)

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

    judge_parser = JSONParser(fields=list(JUDGE_DIMENSIONS))
    judge_rubric = vf.JudgeRubric(
        judge_client=AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers),
        judge_model=judge_model,
        judge_prompt="{question}",  # gets filled in during judge_rubric.judge() call
        parser=parser,
        judge_sampling_args=sampling_args,
    )

    async def judge_rubric_reward(completion: Messages, info: Info, state: State, **kwargs: Any) -> float:
        question = str(info.get("question") or "")
        context = str(info.get("context") or "")
        gold_answer = str(info.get("ideal_answer") or "")
        completion_text = _extract_completion_text(completion, parser)

        judge_prompt = JUDGE_TEMPLATE.format(
            question=question,
            context=context,
            response=completion_text,
            gold_answer=gold_answer,
            output_format=JUDGE_OUTPUT_JSON,
        )

        # judge_prompt assigned to question var inside judge_rubric.judge() method
        try:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_answer, state)
            parsed = judge_parser.parse(str(judge_raw), strip=True)
        except AttributeError:
            judge_raw = await judge_rubric.judge(judge_prompt, completion_text, gold_answer, state)
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
        dataset=dataset,
        system_prompt=system_prompt,
        rubric=judge_rubric,
        parser=parser,
        **kwargs,
    )