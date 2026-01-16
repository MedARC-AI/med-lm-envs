import random
from typing import Optional

import verifiers as vf
from datasets import load_dataset
from verifiers.utils.data_utils import extract_boxed_answer

from prompts import system_prompt, create_prompt_no_knowledge, create_prompt_with_knowledge

def medhallu_reward(completion: str, answer: str, parser: vf.Parser, **kwargs) -> float:
    """
    Reward function for MedHallu.
    
    - Matches Target ('0' or '1'): + 1.0
    - Model says '2' (unsure): - 0.01 (penalty)
    - Incorrect: - 1
    """
    # Extract the model's answer (expecting 0, 1, or 2 in boxed)
    extracted = parser.parse_answer(completion)
    
    if not extracted:
        return 0.0  # failed to produce a boxed answer

    cleaned_extracted = extracted.strip()

    if cleaned_extracted == answer:
        return 1.0 # correct
    elif cleaned_extracted == '2':
        return -0.01 # penalty for unsure
    else:
        return -1.0 # incorrect
    
def load_environment(
    subset: str = "pqa_labeled", 
    use_knowledge: bool = False, 
    **kwargs
) -> vf.Environment:
    """
    Loads the MedHallu evaluation environment.

    Args:
        subset: 'pqa_labeled' (1k high quality) or 'pqa_artificial' (9k generated).
        use_knowledge: If True, includes the 'Knowledge' field in the prompt.
    """
    dataset = load_dataset("UTAustin-AIHealth/MedHallu",subset,split="train")
    
    rand = random.Random()

    def _map_fn(ex):
        """
        Transforms a dataset row into an environment prompt.
        Randomly flips between showing the Factual Answer (Target 0) 
        or Hallucinated Answer (Target 1).
        """
        # randomly decide whether to show the factual or hallucinated answer
        is_factual = rand.choice([True, False])
        
        if is_factual:
            # show Ground Truth -> expect '0'
            answer_text = ex["Ground Truth"]
            target_label = "0"
        else:
            # show Hallucinated Answer -> expect '1'
            answer_text = ex["Hallucinated Answer"]
            target_label = "1"

        
        if use_knowledge:
            prompt = create_prompt_with_knowledge(
                question=ex["Question"],
                option1=answer_text,
                knowledge=ex.get("Knowledge", "")
            )
        else:
            prompt = create_prompt_no_knowledge(
                question=ex["Question"],
                option1=answer_text
            )

        return {
            "question": prompt,
            "answer": target_label,
            "info": {
                "original_type": "factual" if is_factual else "hallucinated",
                "difficulty": ex.get("Difficulty Level", "unknown"),
                "hallucination_category": ex.get("Category of Hallucination", "N/A")
            }
        }
        
    # apply mapping
    processed_dataset = dataset.map(
        _map_fn,
        remove_columns=dataset.column_names,
        load_from_cache_file=False 
    )

    # Parser and Rubric
    parser = vf.Parser(extract_boxed_answer)
    
    rubric = vf.Rubric(
        funcs=[medhallu_reward], 
        weights=[1.0], 
        parser=parser
    )

    return vf.SingleTurnEnv(
        eval_dataset=processed_dataset,
        rubric=rubric,
        parser=parser,
        system_prompt=system_prompt
    )

    
    
    
    

