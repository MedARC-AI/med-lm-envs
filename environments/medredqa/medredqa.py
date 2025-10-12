import os
import sys
from datasets import load_dataset
from openai import AsyncOpenAI
import verifiers as vf
from factscore_judge.atomic_facts_judge import create_judge_rubric


def load_environment(
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> vf.Environment:
    """
    MedRedQA environment using LLM-as-a-Judge evaluation.
    
    This environment loads the MedRedQA dataset and uses an LLM judge
    to evaluate if opinion and recommendations predicted cover the inputs
    provided by a certified medical professional.
    """
    #Print debug info if verbose specified
    verbose = True if '-v' in sys.argv else False

    # Load the MedRedQA dataset
    full_dataset = load_dataset("bagga005/medredqa")
    
    # Use train split for training, val split for evaluation
    train_dataset = full_dataset["train"].map(
        lambda x: {
            "question": x["title"] + "\n" + x["body"] if x["title"] else x["body"],
            "answer": x["response"],
            "task": "medredqa",
            "info": { "judge_response": "Pending.." },
        }
    )
    
    eval_dataset = full_dataset["validation"].map(
        lambda x: {
            "question": x["title"] + "\n" + x["body"] if x["title"] else x["body"],
            "answer": x["response"],
            "task": "medredqa",
            "info": { "judge_response": "Pending.." },
        }
    )

    # Debug: Print dataset structure and sample items if in verbose mode
    if verbose:
        print("Dataset info:")
        print(f"Number of items: {len(eval_dataset)}")
        print(f"Features: {eval_dataset.features}")
        print("\nFirst few items:")
        for i, item in enumerate(eval_dataset.select(range(min(3, len(eval_dataset))))):
            print(f"Item {i}:")
            for key, value in item.items():
                print(f"  {key}: {str(value)[:50000]}{'...' if len(str(value)) > 50000 else ''}")
            print()
    

    # System prompt for the task
    system_prompt = (
        "You are a biomedical reasoning model. You must think step-by-step and reason carefully about "
        "the following medical question and then provide your opinion and recommendations."
    )

    # Initialize OpenAI client for judge
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None

    # Create JudgeRubric using the helper function from judge.py
    rubric = create_judge_rubric(
        judge_client=judge_client,
        judge_model=judge_model,
    )

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        rubric=rubric
    )
    
    return vf_env
