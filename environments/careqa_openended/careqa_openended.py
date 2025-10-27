from __future__ import annotations
import os
import re
from datasets import load_dataset
from openai import AsyncOpenAI
import verifiers as vf


def load_environment(
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> vf.Environment:
    """
    CareQA Open-Ended QA environment using LLM-as-a-Judge evaluation.
    
    This environment loads the open-ended subset of the CareQA dataset and
    uses an LLM judge to assess whether a model's response matches or aligns
    medically with the reference answer.
    """

    # --- Load Dataset ---
    ds = load_dataset("HPAI-BSC/CareQA", "CareQA_en_open")
    train_dataset = ds["train"] if "train" in ds else None
    eval_dataset = ds["test"]

    def _map(ex):
        return {
            "question": ex["question"].strip(),
            "answer": ex.get("answer_explanation", ex.get("answer", "")),
            "task": "careqa_open",
        }

    if train_dataset:
        train_dataset = train_dataset.map(_map, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(_map, remove_columns=eval_dataset.column_names)

    # System Prompt
    system_prompt = (
        "You are an expert clinician answering open-ended medical questions. "
        "Read the question carefully, reason step by step, and provide a precise, "
        "clinically sound answer enclosed in <think></think> tags, followed by your conclusion."
    )

    # Judge Prompt Template
    JUDGE_TEMPLATE = """You are a clinical fact verifier.
Given:
Question: {question}
Reference (ground truth) answer: {answer}
Model’s answer: {response}

Determine if the model’s answer is medically equivalent to the reference.
- Consider medical synonyms and abbreviations equivalent.
- Ignore minor wording differences (e.g., “high blood pressure” ≈ “hypertension”).
- If the model’s answer is more general or specific but still correct, consider it equivalent.

Respond with one word only: "EQUIVALENT" or "NOT_EQUIVALENT".

""".strip()

    # Judge Client Setup
    api_key = judge_api_key or os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None

    # Reward Extraction
    def extract_answer_section(completion_text: str) -> str:
        """Extract final answer after think tags."""
        if not completion_text:
            return ""
        if "<think>" in completion_text and "</think>" in completion_text:
            return re.sub(r".*?</think>", "", completion_text, flags=re.DOTALL).strip()
        return completion_text.strip()

    async def careqa_reward_func(judge, prompt, completion, answer, state, **kwargs) -> float:
        """Evaluate medical equivalence using LLM-as-judge."""
        completion_text = completion if isinstance(completion, str) else str(completion)
        response = extract_answer_section(completion_text)

        judge_response = await judge(prompt, response, answer, state, **kwargs)
        decision = judge_response.strip().upper()

        if "EQUIVALENT" in decision and "NOT_EQUIVALENT" not in decision:
            return 1.0
        else:
            return 0.0

    # Judge Rubric
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_TEMPLATE,
    )
    rubric.add_reward_func(careqa_reward_func, weight=1.0)

    # Environment Construction
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric,
    )

    return vf_env
