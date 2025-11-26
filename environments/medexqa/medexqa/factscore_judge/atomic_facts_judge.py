"""
FactScore-style judge for MedExQA explanations (reference-only, no external retrieval).

Two-step process:
1) Extract atomic medical claims from the model's explanation.
2) Verify each claim against available references: question, correct option text, exp0, exp1.

Returns support rate in [0, 1], scaled to [0, 100] for reward.
"""

import json
import re
import verifiers as vf
from .atomic_facts_generator import AtomicFactGenerator


JUDGE_TEMPLATE = """You are a medical knowledge verification expert. Evaluate if the Passage supports the Claim.

PASSAGE:
{response}

CLAIM TO VERIFY:
{answer}

INSTRUCTIONS:
1. Check if the claim is FULLY supported by the passage with explicit evidence
2. Check if the claim is PARTIALLY supported (implied/inferable but not explicit)
3. Check if the claim is NOT supported (no evidence or contradicts passage)

Respond with EXACTLY ONE of:
- "FULLY_SUPPORTED" - explicit evidence exists in passage
- "PARTIALLY_SUPPORTED" - implied/inferable from passage
- "NOT_SUPPORTED" - no evidence or contradicts passage

Your response:""".strip()


def extract_support_level(text: str) -> tuple[float, bool]:
    """
    Extract support level from LLM judge response.

    Returns:
        (score, valid): score is 0.0, 0.5, or 1.0; valid indicates if parsing succeeded
    """
    cleaned_text = (text or "").strip().upper()

    # Check for 3-level responses
    if "FULLY_SUPPORTED" in cleaned_text or "FULLY SUPPORTED" in cleaned_text:
        return (1.0, True)
    if "PARTIALLY_SUPPORTED" in cleaned_text or "PARTIALLY SUPPORTED" in cleaned_text:
        return (0.5, True)
    if "NOT_SUPPORTED" in cleaned_text or "NOT SUPPORTED" in cleaned_text:
        return (0.0, True)

    # Fallback to old binary format for backwards compatibility
    cleaned_lower = cleaned_text.lower()
    has_true = "true" in cleaned_lower
    has_false = "false" in cleaned_lower
    if has_true and not has_false:
        return (1.0, True)
    if has_false and not has_true:
        return (0.0, True)

    # Ambiguous response
    return (0.0, False)


async def explanation_factscore_reward(
    judge,
    prompt,
    completion,
    answer,
    state,
    **kwargs,
) -> float:
    # parse explanation text
    if isinstance(completion, list) and completion:
        explanation = completion[-1].get("content", "") or ""
    else:
        explanation = str(completion)

    info = kwargs.get("info", {}) or {}
    options = {k: info.get(k, "") for k in ["A", "B", "C", "D"]}
    question = info.get("question", "")
    exp0 = info.get("exp0", "")
    exp1 = info.get("exp1", "")
    correct_letter = (answer or "").strip().upper()
    correct_option_text = options.get(correct_letter, "")

    # Gate explanation to zero if predicted MCQ answer is wrong
    # Parse answer first (extracts from \boxed{} in think mode, returns raw text in normal mode)
    parser = kwargs.get("parser")
    if parser:
        parsed = parser.parse_answer(completion) or ""
    else:
        parsed = explanation

    from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy

    is_correct = multiple_choice_accuracy(
        llm_answer=parsed,
        answer_letter=correct_letter,
        answer_text=correct_option_text,
        accept_answer_text=True,
        strip_tex=False,
    )

    if not is_correct:
        return 0.0

    # Build references block
    refs = (
        f"Question: {question}\n"
        f"Correct option ({correct_letter}): {correct_option_text}\n"
        f"Reference Explanation 1: {exp0}\n"
        f"Reference Explanation 2: {exp1}"
    )

    # Initialize generator (reuse medredqa style)
    llm_client = kwargs.get("judge_client")
    llm_model = kwargs.get("judge_model", "gpt-4o-mini")
    generator = AtomicFactGenerator(llm_client, model_name=llm_model)

    # Extract atomic claims from model explanation
    try:
        if llm_client is None:
            # No client available - cannot extract claims
            return 0.0
        claims = await generator.run(explanation, state=state)
    except Exception as e:
        # Log extraction error for debugging
        import sys
        print(f"Warning: Atomic facts extraction failed: {e}", file=sys.stderr)
        claims = []
    if not claims:
        return 0.0

    # Step 2a: verify each model claim against references (support_rate)
    # One call per claim like MedRedQA approach
    support_score = 0.0
    total = 0

    for claim in claims:
        total += 1
        # Call judge like medredqa does: judge(prompt, completion, answer, state, **kwargs)
        # prompt is not used in template, completion becomes {response}, answer becomes {answer}
        judge_response = await judge(prompt, refs, str(claim), state, **kwargs)
        score, ok = extract_support_level(judge_response)
        if ok:
            support_score += score

    support_rate = (support_score / total) if total > 0 else 0.0

    # Step 2b: Coverage rate - DISABLED by default for speed
    # This measures recall: does the model explanation cover key reference concepts?
    # Enable with use_coverage=True in kwargs for balanced precision+recall evaluation
    use_coverage = kwargs.get("use_coverage", False)
    coverage_rate = 0.0

    if use_coverage and llm_client is not None:
        # Extract claims from both reference explanations
        all_ref_claims: list[str] = []

        # Extract from reference 1
        if (exp0 or "").strip():
            try:
                ref0_claims = await generator.run(exp0, state=state)
                all_ref_claims.extend(ref0_claims)
            except Exception:
                pass

        # Extract from reference 2
        if (exp1 or "").strip():
            try:
                ref1_claims = await generator.run(exp1, state=state)
                all_ref_claims.extend(ref1_claims)
            except Exception:
                pass

        # Remove duplicates while preserving order
        seen = set()
        unique_ref_claims = []
        for claim in all_ref_claims:
            if claim not in seen:
                unique_ref_claims.append(claim)
                seen.add(claim)

        # Verify each reference claim against model explanation
        coverage_score = 0.0
        coverage_total = 0

        for ref_claim in unique_ref_claims:
            coverage_total += 1
            # Check if model explanation supports this reference claim
            # Call judge: passage=explanation, claim=ref_claim
            cov_response = await judge(prompt, explanation, str(ref_claim), state, **kwargs)
            score, ok = extract_support_level(cov_response)
            if ok:
                coverage_score += score

        coverage_rate = (coverage_score / coverage_total) if coverage_total > 0 else 0.0

    # Combine support and coverage (if enabled)
    if use_coverage:
        # Use weighted combination when coverage is enabled
        w_support = float(kwargs.get("support_weight", 0.5))
        w_coverage = float(kwargs.get("coverage_weight", 0.5))
        denom = w_support + w_coverage if (w_support + w_coverage) > 0 else 1.0
        final = (w_support * support_rate + w_coverage * coverage_rate) / denom
    else:
        # Coverage disabled: use support_rate only
        final = support_rate

    # Optionally stash structured details for external loggers (if passed in kwargs)
    # Caller can access via state or judge logs; for rescore tool we return these via logs reconstruction
    state = state or {}
    try:
        state["factscore_details"] = {
            "support_rate": float(support_rate),
            "coverage_rate": float(coverage_rate),
        }
    except Exception:
        pass

    return float(final * 100.0)


def create_factscore_judge_rubric(
    parser: vf.Parser,
    judge_client,
    judge_model: str = "gpt-4o-mini",
    use_coverage: bool = False,
    explanation_weight: float = 1.0,
) -> vf.JudgeRubric:
    # Pass judge_prompt like medredqa does - uses standard {response} and {answer} placeholders
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_TEMPLATE,
        parser=parser,
        use_coverage=use_coverage,  # Pass through to reward function via kwargs
    )
    rubric.add_reward_func(explanation_factscore_reward, weight=explanation_weight)
    return rubric


