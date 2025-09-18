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
    MedCaseReasoning environment using LLM-as-a-Judge evaluation.
    
    This environment loads the MedCaseReasoning dataset and uses an LLM judge
    to evaluate whether model responses are equivalent to the ground truth
    medical diagnoses.
    """
    # Load the MedCaseReasoning dataset
    full_dataset = load_dataset("zou-lab/MedCaseReasoning")
    
    # Use train split for training, val split for evaluation
    train_dataset = full_dataset["train"].map(
        lambda x: {
            "question": x["case_prompt"],
            "answer": x["final_diagnosis"],
            "task": "medcasereasoning",
        }
    )
    
    eval_dataset = full_dataset["val"].map(
        lambda x: {
            "question": x["case_prompt"],
            "answer": x["final_diagnosis"],
            "task": "medcasereasoning",
        }
    )

    # System prompt for the task
    system_prompt = (
        "You are a biomedical reasoning model. You must think step-by-step and reason carefully about "
        "the following medical case before providing your diagnosis. You should enclose your thoughts "
        "and reasoning inside <think> </think> tags, and then provide your final diagnosis concisely."
    )

    # Judge prompt template for medical diagnosis evaluation
    JUDGE_TEMPLATE = """\
Your job is to evaluate whether a medical diagnosis is equivalent to the ground truth diagnosis.

You will be given:
1. A medical case prompt (question)
2. The ground truth diagnosis (answer)  
3. A predicted diagnosis (response)

Your task is to determine if the predicted diagnosis is medically equivalent to the ground truth, even if worded differently.

Consider these guidelines:
- Medical terms that refer to the same condition should be considered equivalent
- Different levels of specificity may be acceptable (e.g., "pneumonia" vs "bacterial pneumonia")
- Spelling variations of medical terms should be considered equivalent
- The core medical meaning should match, even if additional details vary
- Consider both the primary diagnosis and any relevant differential diagnoses

Examples:
- "Acute myocardial infarction" and "heart attack" → EQUIVALENT
- "Type 2 diabetes mellitus" and "diabetes" → EQUIVALENT  
- "Upper respiratory infection" and "pneumonia" → NOT EQUIVALENT
- "Hypertension" and "high blood pressure" → EQUIVALENT

Question: {question}

Ground truth diagnosis: {answer}

Predicted diagnosis: {response}

Is the predicted diagnosis medically equivalent to the ground truth diagnosis?
Respond with either "EQUIVALENT" or "NOT_EQUIVALENT".
""".strip()

    # Initialize OpenAI client for judge
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None

    # Create JudgeRubric with custom prompt
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_TEMPLATE,
    )

    def extract_answer_section(completion_text: str) -> str:
        """
        Extract the answer section from completion text, handling think tags properly.
        """
        # Check for proper think tags
        think_tag_pairs = re.findall(
            r"<think>.*?</think>",
            completion_text,
            re.DOTALL | re.IGNORECASE
        )
        
        has_exactly_one_proper_think_tag = len(think_tag_pairs) == 1
        
        # Check for malformed tags
        has_malformed_tags = re.search(
            r"<think>(?:(?!</think>).)*$",
            completion_text,
            re.DOTALL | re.IGNORECASE
        ) is not None
        
        if has_exactly_one_proper_think_tag and not has_malformed_tags:
            # Extract everything after the properly closed </think> tag
            answer_section = re.sub(
                r".*?</think>", 
                "", 
                completion_text, 
                flags=re.DOTALL | re.IGNORECASE
            ).strip()
            return answer_section
        else:
            # If no proper think tags, return full response
            return completion_text.strip()

    async def medical_diagnosis_reward_func(
        judge, prompt, completion, answer, state, **kwargs
    ) -> float:
        """
        Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
        """
        # Extract the answer section (handling think tags)
        completion_text = completion if isinstance(completion, str) else str(completion)
        answer_section = extract_answer_section(completion_text)
        
        # Get judge response using the extracted answer
        judge_response = await judge(prompt, answer_section, answer, state, **kwargs)
        
        # Parse judge response
        judge_response_clean = judge_response.strip().upper()
        
        # Return 1.0 if equivalent, 0.0 otherwise
        if "EQUIVALENT" in judge_response_clean and "NOT_EQUIVALENT" not in judge_response_clean:
            return 1.0
        else:
            return 0.0

    # Add the reward function to the rubric
    rubric.add_reward_func(medical_diagnosis_reward_func, weight=1.0)

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric
    )
    
    return vf_env
