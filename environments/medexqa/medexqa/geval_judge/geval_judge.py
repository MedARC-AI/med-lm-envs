import re
import json
import verifiers as vf


GEVAL_CRITERIA = """You are evaluating a medical MCQA explanation for quality and correctness.

Assess the explanation across these dimensions:

1. MEDICAL ACCURACY: Does the explanation contain factually correct medical information that aligns with the correct option and reference explanations? Are there any medical errors or contradictions?

2. CORRECT OPTION JUSTIFICATION: Does the explanation clearly explain WHY the correct answer is medically appropriate using valid clinical/scientific reasoning?

3. DISTRACTOR ANALYSIS (when applicable): Does the explanation explain why incorrect options are wrong? Note: Not all explanations need this, but it enhances quality when present.

4. REFERENCE ALIGNMENT: Do the explanation's core medical claims align with the key concepts in both reference explanations?

5. REASONING CLARITY: Is the medical reasoning easy to follow with logical flow from evidence to conclusion?

6. COMPLETENESS: Does the explanation cover the essential medical concepts without major omissions?"""


GEVAL_EVALUATION_STEPS = [
    "Extract all medical claims from the actual explanation and list them explicitly",
    "Compare each claim against the correct option text and both reference explanations. Mark claims as: ALIGNED (matches references), CONTRADICTS (conflicts with references), or NEW_INFO (additional but not contradictory)",
    "Identify if the explanation justifies WHY the correct option is right (not just states it is correct)",
    "Check for any major medical errors, inaccuracies, or unsupported claims that could mislead",
    "Assess whether distractor refutation is present and accurate (if applicable to this question)",
    "Evaluate overall reasoning clarity, logical flow, and completeness",
    "Synthesize findings into a score using this rubric: 0.0-0.2 (major errors/irrelevant/contradicts references), 0.2-0.4 (significant gaps/multiple minor errors), 0.4-0.6 (acceptable but incomplete/some inaccuracies), 0.6-0.8 (good quality with minor issues), 0.8-1.0 (excellent: comprehensive, accurate, well-reasoned)"
]


GEVAL_PROMPT_TEMPLATE = """You are a strict medical explanation evaluator following a structured evaluation process.

CRITERIA:
{criteria}

EVALUATION STEPS (follow these in order):
{evaluation_steps}

OUTPUT FORMAT:
Respond with a JSON object containing your step-by-step analysis and final score. Use this exact structure:
{{
  "step1_claims_extracted": ["claim1", "claim2", ...],
  "step2_alignment_analysis": {{
    "aligned_claims": [...],
    "contradicting_claims": [...],
    "new_info_claims": [...]
  }},
  "step3_correct_option_justified": true/false,
  "step4_medical_errors_found": ["error description"] or [],
  "step5_distractor_refutation": "present_and_accurate" / "present_but_weak" / "absent" / "not_applicable",
  "step6_reasoning_assessment": "clear" / "somewhat_clear" / "confusing",
  "step7_final_score": 0.XX,
  "score_justification": "Brief 1-2 sentence explanation of the score"
}}

QUESTION CONTEXT:
Question: {question}
Options:
{options}
Correct Answer: {correct_answer}

REFERENCE EXPLANATIONS:
Reference 1: {ref_exp1}
Reference 2: {ref_exp2}

MODEL EXPLANATION TO EVALUATE:
{model_explanation}

Provide your evaluation as JSON:"""


def _extract_score_from_json(text: str) -> tuple[float, dict]:
    """
    Extract score from JSON response.

    Returns:
        (score, parsed_dict): score is 0.0-1.0; parsed_dict contains full evaluation
    """
    try:
        # Try to parse as JSON first
        # Find JSON object in response (may have extra text before/after)
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)

            # Extract score from step7_final_score or final_score
            score = float(data.get("step7_final_score", data.get("final_score", 0.0)))
            score = max(0.0, min(1.0, score))
            return score, data
    except Exception:
        pass

    # Fallback: try old "final_score:" pattern
    try:
        m = re.search(r"final_score\s*:\s*(\d+\.\d+|\d+)", text, flags=re.IGNORECASE)
        if m:
            score = float(m.group(1))
            return max(0.0, min(1.0, score)), {}
    except Exception:
        pass

    # Last fallback: extract any number
    try:
        m = re.search(r"(\d+\.\d+|\d+)", text.strip())
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val)), {}
    except Exception:
        pass

    return 0.0, {}


async def explanation_geval_reward(
    judge,
    prompt,
    completion,
    answer,
    state,
    **kwargs,
) -> float:
    # Extract the last assistant message content as the explanation text
    if isinstance(completion, list) and completion:
        completion_text = completion[-1].get("content", "") or ""
    else:
        completion_text = str(completion)

    info = kwargs.get("info", {}) or {}
    options = {k: info.get(k, "") for k in ["A", "B", "C", "D"]}
    question = info.get("question", "")
    exp0 = info.get("exp0", "")
    exp1 = info.get("exp1", "")
    correct_letter = (answer or "").strip().upper()

    # Gate explanation to zero if predicted MCQ answer is wrong
    # Parse answer first (extracts from \boxed{} in think mode, returns raw text in normal mode)
    parser = kwargs.get("parser")
    if parser:
        parsed = parser.parse_answer(completion) or ""
    else:
        parsed = completion_text

    from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy

    correct_option_text = options.get(correct_letter, "")
    is_correct = multiple_choice_accuracy(
        llm_answer=parsed,
        answer_letter=correct_letter,
        answer_text=correct_option_text,
        accept_answer_text=True,
        strip_tex=False,
    )

    if not is_correct:
        return 0.0

    # Format options string
    opts_str = "\n".join(f"{k}. {options.get(k, '')}" for k in ["A", "B", "C", "D"])

    # Format evaluation steps for prompt
    steps_formatted = "\n".join([f"{i+1}. {step}" for i, step in enumerate(GEVAL_EVALUATION_STEPS)])

    # Build the full prompt using the new template
    full_prompt = GEVAL_PROMPT_TEMPLATE.format(
        criteria=GEVAL_CRITERIA,
        evaluation_steps=steps_formatted,
        question=question,
        options=opts_str,
        correct_answer=f"{correct_letter} ({options.get(correct_letter, '')})",
        ref_exp1=exp0,
        ref_exp2=exp1,
        model_explanation=completion_text
    )

    # Call judge with structured prompt requesting JSON
    judge_response = await judge([
        {"role": "system", "content": "You are a strict, deterministic medical evaluator. Follow the evaluation steps carefully and output valid JSON only."},
        {"role": "user", "content": full_prompt}
    ], "", "", state, **kwargs)

    # Parse JSON response and extract score
    txt = str(judge_response)
    score, eval_details = _extract_score_from_json(txt)

    # Optionally store evaluation details in state for debugging/logging
    if state is not None and eval_details:
        try:
            state["geval_details"] = eval_details
        except Exception:
            pass

    return float(score * 100.0)


def create_geval_judge_rubric(
    parser: vf.Parser,
    judge_client,
    judge_model: str = "gpt-4o-mini",
    explanation_weight: float = 1.0,
) -> vf.JudgeRubric:
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",  # not used directly; reward builds full prompt
        parser=parser,
    )
    rubric.add_reward_func(explanation_geval_reward, weight=explanation_weight)
    return rubric


