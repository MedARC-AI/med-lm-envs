# Modified from K-QA: https://github.com/Itaymanes/K-QA
# Copyright (c) 2023 Itay Manes
# License: MIT

# ALL PROMPTS ARE FROM THE PAPER (Source: https://arxiv.org/abs/2401.14493)
# see pgs 16-18 for the prompts from the paper and github: https://github.com/Itaymanes/K-QA/tree/main/evaluation/prompts

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import verifiers as vf
from datasets import Dataset
from medarc_verifiers.parsers import JSONParser
from openai import AsyncOpenAI
from prompts import (
    _decompose_free_form_answer,
    _llm_generation_prompt,
    _prompt_for_has_contradiction,
    _prompt_for_is_entails,
    batch_eval_prompt,
)
from verifiers.utils.data_utils import extract_boxed_answer


def _get_raw_completion(completion) -> str:
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "") or ""
    return str(completion or "")


async def _extract_and_store_claims(
    extractor: vf.JudgeRubric,
    prompt,
    completion,
    info,
    state,
    semaphore: asyncio.Semaphore,
) -> float:
    # build extraction prompt and judge via the rubric
    question: str = (info or {}).get("Question", "")
    llm_answer: str = _get_raw_completion(completion)
    extraction_prompt = _decompose_free_form_answer(question, llm_answer)
    async with semaphore:
        raw = await extractor.judge(
            [{"role": "user", "content": extraction_prompt}],
            completion="",
            answer="",
            state=state,
        )

    # Parse {"claims": [...]} like format
    json_parser = JSONParser(fields=["claims"])
    parsed = json_parser.parse(str(raw)) or {}
    claims = [str(c).strip() for c in (parsed.get("claims") or []) if str(c).strip()]

    #  shared state
    state.setdefault("kqa", {})
    state["kqa"]["claims"] = claims
    state["kqa"]["raw_completion"] = llm_answer

    #  no contribution to reward
    return 0.0


# single judge call for evaluation
async def _judge_boolean(
    scorer: vf.JudgeRubric,
    prompt_str: str,
    state: Dict,
    semaphore: asyncio.Semaphore,
    verbose: bool = False,
) -> bool:
    async with semaphore:
        raw = await scorer.judge(
            [{"role": "user", "content": prompt_str}],
            completion="",
            answer="",
            state=state,
        )
    if verbose:
        print("the answer box below: ", end="\n")

    answer = extract_boxed_answer(str(raw))
    if verbose:
        print(f"answer: {answer}", end="\n")  # from the verifiers library
    return True if answer.lower() == "true" else False


# batch call for evaluation
async def _batch_eval(
    scorer: vf.JudgeRubric, question: str, info: Dict, state: Dict, semaphore: asyncio.Semaphore, verbose: bool = False
) -> None:
    if "batch_eval_results" in state.get("kqa", {}):
        return

    must_have: List[str] = (info or {}).get("Must_have", []) or []
    claims: List[str] = (state.get("kqa", {}) or {}).get("claims", []) or []
    gold_claims: List[str] = ((info or {}).get("Must_have", []) or []) + ((info or {}).get("Nice_to_have", []) or [])

    if not claims or not gold_claims:
        state.setdefault("kqa", {})["batch_eval_results"] = {
            "comprehensiveness": {"entailed_must_have_claims": []},
            "hallucination": {"contradictory_generated_claims": []},
        }
        return

    prompt_str = batch_eval_prompt(question, claims, must_have, gold_claims)
    async with semaphore:
        raw = await scorer.judge([{"role": "user", "content": prompt_str}], completion="", answer="", state=state)

    if verbose:
        print("question: ", question, end="\n")
        print("\n")
        print("claims: ", claims, end="\n")
        print("\n")
        print("must_have: ", must_have, end="\n")
        print("\n")
        print("gold_claims: ", gold_claims, end="\n")
        print("\n")
        print(raw, end="\n")

    json_parser = JSONParser(fields=["comprehensiveness", "hallucination"])

    parsed = json_parser.parse(str(raw)) or {}

    state.setdefault("kqa", {})["batch_eval_results"] = parsed


def load_environment(
    extractor_model: str = "gpt-4.1-mini",
    judge_model: str = "gpt-4.1-mini",
    extractor_base_url: str | None = None,
    extractor_api_key: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    batch: bool = False,
    verbose: bool = False,
    max_parallel_judges: int = 5,
    **kwargs,
) -> vf.Environment:
    """K-QA Environment using a FACTSCORE-style evaluation with two-phase Judge via RubricGroup:
      1) Extraction: decompose free-form answer into claims, store in state.
      2) Scoring: compute comprehensiveness and hallucination_rate using NLI judge.

    Args:
        extractor_model: LLM used to decompose answers into claims.
        judge_model: LLM used for entailment/contradiction scoring.
        extractor_base_url: Optional override for the extractor OpenAI base URL.
        extractor_api_key: Optional override for the extractor API key.
        judge_base_url: Optional override for the judge OpenAI base URL.
        judge_api_key: Optional override for the judge API key.
        batch: Whether to evaluate all claims in a single batched judge call.
        verbose: When ``True`` prints intermediate judge outputs for debugging.
        max_parallel_judges: Per-rollout cap on concurrent judge requests; the
            semaphore is shared across extraction and scoring calls.
        **kwargs: Additional loader arguments (unused).

    Returns:
        vf.Environment: Configured single-turn environment ready for evaluation.
    """
    data_fp = Path(__file__).resolve().parent / "questions_w_answers.jsonl"
    df = pd.read_json(data_fp, orient="records", lines=True)

    # Map to vf format
    def _map_to_vf_format(row: pd.Series) -> Dict[str, Any]:
        q = row.get("Question", "")
        return {
            "question": _llm_generation_prompt(q),
            "answer": "",
            "info": {
                "Question": q,
                # gold claims
                "Must_have": list(row.get("Must_have", []) or []),
                "Nice_to_have": list(row.get("Nice_to_have", []) or []),
            },
        }

    dataset = Dataset.from_list(df.apply(_map_to_vf_format, axis=1).tolist())

    extractor_key = extractor_api_key if extractor_api_key else os.getenv("JUDGE_API_KEY")
    extractor_client = AsyncOpenAI(base_url=extractor_base_url, api_key=extractor_key)

    judge_key = judge_api_key if judge_api_key else os.getenv("JUDGE_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=judge_key)

    #  Extractor rubric (stores claims in shared `state["kqa"]["claims"]`)
    extractor = vf.JudgeRubric(
        judge_client=extractor_client,
        judge_model=extractor_model,
        judge_prompt="{question}",  # prompt provided dynamically in reward
    )

    # Bound concurrent judge calls per rollout to avoid overload when batch mode fans out
    judge_semaphore = asyncio.Semaphore(max_parallel_judges)
    extractor.add_reward_func(
        lambda prompt, completion, info, state, **_: _extract_and_store_claims(
            extractor,
            prompt,
            completion,
            info,
            state,
            semaphore=judge_semaphore,
        ),
        weight=0.0,
    )

    #  Scoring rubric using NLI prompts; reads claims from state
    scorer = vf.JudgeRubric(judge_client=judge_client, judge_model=judge_model, judge_prompt="{question}")

    async def comprehensiveness_reward(prompt, completion, info, state, **_) -> float:
        # Adapted the code from the original code: https://github.com/Itaymanes/K-QA/blob/main/evaluation/metrics.py to compute comprehensiveness
        question: str = (info or {}).get("Question", "")
        must_have: List[str] = (info or {}).get("Must_have", []) or []

        if not must_have:
            return 1.0

        if batch:
            await _batch_eval(scorer, question, info, state, semaphore=judge_semaphore, verbose=verbose)

            eval_results = state.get("kqa", {}).get("batch_eval_results", {})
            comprehensiveness_results = eval_results.get("comprehensiveness", {})
            entailed_claims = comprehensiveness_results.get("entailed_must_have_claims", [])
            entailed_count = len(entailed_claims)

            return entailed_count / len(must_have)

        else:
            claims: List[str] = (state.get("kqa", {}) or {}).get("claims", []) or []
            if not claims:
                return 0.0
            covered = 0
            for must_claim in must_have:
                # True if ANY predicted claim entails the must-have claim
                entailed = False
                for pred in claims:
                    prompt_str = _prompt_for_is_entails(question, pred, must_claim)
                    if await _judge_boolean(scorer, prompt_str, state, semaphore=judge_semaphore, verbose=verbose):
                        entailed = True
                        break
                if entailed:
                    covered += 1
            return covered / len(must_have)

    async def hallucination_rate_reward(prompt, completion, info, state, **_) -> float:
        question: str = (info or {}).get("Question", "")
        claims: List[str] = (state.get("kqa", {}) or {}).get("claims", []) or []
        gold_claims: List[str] = ((info or {}).get("Must_have", []) or []) + (
            (info or {}).get("Nice_to_have", []) or []
        )

        if not gold_claims:
            return 1.0

        if not claims:
            # If no claims are generated, there are no hallucinations.
            return 1.0

        if batch:
            await _batch_eval(scorer, question, info, state, semaphore=judge_semaphore, verbose=verbose)
            eval_results = state.get("kqa", {}).get("batch_eval_results", {})
            hallucination_results = eval_results.get("hallucination", {})
            contradictory_claims = hallucination_results.get("contradictory_generated_claims", [])
            contradictory_count = len(contradictory_claims)
            return 1.0 - (contradictory_count / len(gold_claims))

        else:
            contradicted_gold = 0
            for gold in gold_claims:
                contradicted = False
                for pred in claims:
                    prompt_str = _prompt_for_has_contradiction(question, pred, gold)
                    if await _judge_boolean(
                        scorer,
                        prompt_str,
                        state,
                        semaphore=judge_semaphore,
                        verbose=verbose,
                    ):
                        contradicted = True
                        break
                if contradicted:
                    contradicted_gold += 1
            return 1.0 - (contradicted_gold / len(gold_claims))

    scorer.add_reward_func(comprehensiveness_reward, weight=0.5)
    scorer.add_reward_func(hallucination_rate_reward, weight=0.5)

    # extractor runs first, then scoring; state is shared
    rubric_group = vf.RubricGroup([extractor, scorer])

    # Simple pass-through parser; reward funcs operate directly on messages/state
    parser = vf.Parser(extract_fn=lambda x: x)

    return vf.SingleTurnEnv(eval_dataset=dataset, parser=parser, rubric=rubric_group)
