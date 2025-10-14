# ALL PROMPTS ARE FROM THE PAPER (Source: https://arxiv.org/abs/2401.14493)
# see pgs 16-18 for the prompts from the paper and github: https://github.com/Itaymanes/K-QA/tree/main/evaluation/prompts

import verifiers as vf
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from prompts import (
    _llm_generation_prompt,
    _decompose_free_form_answer,
    _prompt_for_has_contradiction,
    _prompt_for_is_entails,
)
from datasets import Dataset
from medarc_verifiers.parsers import JSONParser


def _get_raw_completion(completion) -> str:
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "") or ""
    return str(completion or "")


async def _extract_and_store_claims(
    extractor: vf.JudgeRubric, prompt, completion, info, state
) -> float:
    # Build extraction prompt and judge via the rubric
    question: str = (info or {}).get("Question", "")
    llm_answer: str = _get_raw_completion(completion)
    extraction_prompt = _decompose_free_form_answer(question, llm_answer)

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


async def _judge_boolean(scorer: vf.JudgeRubric, prompt_str: str, state: Dict) -> bool:
    raw = await scorer.judge(
        [{"role": "user", "content": prompt_str}],
        completion="",
        answer="",
        state=state,
    )
    text = str(raw).strip().lower()
    return "true" in text[-20:]


def load_environment(
    extractor_model: str = "gpt-4-mini",
    judge_model: str = "gpt-4-mini",
    **kwargs,
) -> vf.Environment:
    """
    K-QA Environment using a FACTSCORE-style evaluation with two-phase Judge via RubricGroup:
      1) Extraction: decompose free-form answer into claims, store in state.
      2) Scoring: compute comprehensiveness and hallucination_rate using NLI judge.
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

    #  Extractor rubric (stores claims in shared `state["kqa"]["claims"]`)
    extractor = vf.JudgeRubric(
        judge_model=extractor_model,
        judge_prompt="{prompt}",  # prompt provided dynamically in reward
    )
    extractor.add_reward_func(
        lambda prompt, completion, info, state, **_: vf.utils.ensure_async(
            _extract_and_store_claims
        )(extractor, prompt, completion, info, state),
        weight=0.0,
    )

    #  Scoring rubric using NLI prompts; reads claims from state
    scorer = vf.JudgeRubric(
        judge_model=judge_model,
        judge_prompt="{prompt}",
    )

    async def comprehensiveness_reward(prompt, completion, info, state, **_) -> float:
        # Adapted the code from the original code: https://github.com/Itaymanes/K-QA/blob/main/evaluation/metrics.py to compute comprehensiveness
        question: str = (info or {}).get("Question", "")
        must_have: List[str] = (info or {}).get("Must_have", []) or []
        claims: List[str] = ((state.get("kqa", {}) or {}).get("claims", []) or [])
        if not must_have:
            return 1.0
        if not claims:
            return 0.0

        covered = 0
        for must_claim in must_have:
            # True if ANY predicted claim entails the must-have claim
            entailed = False
            for pred in claims:
                prompt_str = _prompt_for_is_entails(question, pred, must_claim)
                if await _judge_boolean(scorer, prompt_str, state):
                    entailed = True
                    break
            if entailed:
                covered += 1
        return covered / len(must_have)

    async def hallucination_rate_reward(prompt, completion, info, state, **_) -> float:
        # Adapted the code from the original code: https://github.com/Itaymanes/K-QA/blob/main/evaluation/metrics.py to compute hallucination rate
        question: str = (info or {}).get("Question", "")
        claims: List[str] = ((state.get("kqa", {}) or {}).get("claims", []) or [])
        gold_claims: List[str] = (
            (info or {}).get("Must_have", []) or []
        ) + ((info or {}).get("Nice_to_have", []) or [])

        if not claims:
            return 0.0
        if not gold_claims:
            return 0.0

        contradictions = 0
        for pred in claims:
            # Count if it contradicts ANY gold claim
            contradicts = False
            for gold in gold_claims:
                prompt_str = _prompt_for_has_contradiction(question, gold, pred)
                if await _judge_boolean(scorer, prompt_str, state):
                    contradicts = True
                    break
            if contradicts:
                contradictions += 1
        return contradictions / len(claims)

    scorer.add_reward_func(comprehensiveness_reward, weight=1.0)
    scorer.add_reward_func(hallucination_rate_reward, weight=0.0)

    # extractor runs first, then scoring; state is shared
    rubric_group = vf.RubricGroup([extractor, scorer])

    # Simple pass-through parser; reward funcs operate directly on messages/state
    parser = vf.Parser(extract_fn=lambda x: x)

    return vf.SingleTurnEnv(
        eval_dataset=dataset,
        parser=parser,
        rubric=rubric_group,
    )