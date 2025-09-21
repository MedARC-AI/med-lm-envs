import os
import sys
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

    # Judge prompt template for medical diagnosis evaluation
    JUDGE_TEMPLATE = """\
Your job is to evaluate whether the provided opinions and recommendations match the opinions and recommendations of a certified medical professional.

You will be given:
1. A medical question (question)
2. The ground truth opinion and recommendations from a certified medical professional (answer)  
3. Predicted opinion and recommendations (response)

Your task is to determine how similar the predicted opinion and recommendations are to those of a certified medical professional, even if worded differently and provide a similarity score between 0 and 1.

Consider these guidelines:
- Medical terms that refer to the same condition should be considered equivalent
- Different levels of specificity may be acceptable (e.g., "pneumonia" vs "bacterial pneumonia")
- Spelling variations of medical terms should be considered equivalent
- The core medical meaning should match, even if additional details vary
- Consider if predicted opinion and recommendations are similar in implications to ground truth opinion and recommendations

Question: {question}

Certified medical professional's opinion and recommendations: {answer}

Predicted opinion and recommendations: {response}

Provide a score indicating if predicted opinion and recommendations cover the ground truth opinion and recommendations based on following:
- Consider number of unique points that are made in ground truth opinion and recommendations. For each unique point, consider if the predicted opinion and recommendations make a similar point. Use this to compute percentage of points that are made in the ground truth opinion and recommendations that are also made in the predicted opinion and recommendations.
- Respond with number between 0 and 1 based on this percentage, where 1 is the most similar and 0 is the least similar. 
- Give final score in the format **SCORE:actual_score**. For example **SCORE:0.5** or **SCORE:1.0**
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

    def extract_score(text: str) -> tuple[float, bool]:
        """
        Extracts the numeric value following 'SCORE:' in the given text. Also indicates if score was found
        """
        try:
            match = re.search(r"SCORE:([0-9]*\.?[0-9]+)", text)
            if match:
                return float(match.group(1)), True
        except Exception:
            pass
        return 0.0, False

    async def medical_diagnosis_reward_func(
        judge, prompt, completion, answer, state, info=None, **kwargs
    ) -> float:
        """
        Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
        Expects judge response to be a float between 0 and 1. Returns 0 for invalid numbers.
        """
        # Extract the answer section (handling think tags)
        completion_text = completion if isinstance(completion, str) else str(completion)
        
        # Get judge response using the extracted answer
        judge_response = await judge(prompt, completion_text, answer, state, **kwargs)

        #add judge response to info to be saved
        info["judge_response"] = judge_response

        #parse score and save parsing status
        score, passed_parse = extract_score(judge_response)

        #log to info if unable to parse
        if not passed_parse:
            info["failure_reason"] = "Unable to parse score from judge response"
        
        return score

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
