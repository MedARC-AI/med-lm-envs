"""
Atomic Facts Judge for Medical Recommendations

This module implements an atomic facts-based evaluation system for medical
recommendations. It breaks down medical answers into atomic facts and evaluates
whether each fact is supported by the model's response.
"""

import sys
import verifiers as vf
from openai import AsyncOpenAI
from factscore_judge.atomic_facts_generator import AtomicFactGenerator


# Judge prompt template that asks whether a passage supports a given fact
# Used for evaluating individual atomic facts against the model's response
JUDGE_TEMPLATE = """\
Does the Passage support the Fact? Respond with True or False only

Passage: 
{response}

Fact: 
{answer}
""".strip()


def extract_boolean_score(text: str) -> tuple[bool, bool]:
    """
    Extract a boolean score from LLM judge response text.
    
    Parses the judge's response to determine if it contains "True" or "False".
    If both or neither are present, the response is considered ambiguous.
    
    Args:
        text: Input text from the LLM judge to parse
        
    Returns:
        A tuple of (parsed_value, found) where:
        - parsed_value: The boolean value extracted (True or False)
        - found: Whether a valid unambiguous boolean value was found
    """
    # Normalize text for parsing
    cleaned_text = text.strip().lower()
    
    # Check for presence of 'true' and 'false' keywords
    has_true = 'true' in cleaned_text
    has_false = 'false' in cleaned_text
    
    # If both or neither found, return not found (ambiguous response)
    if has_true == has_false:  # both True or both False
        return (False, False)
    
    # If only 'true' found, the fact is supported
    if has_true:
        return (True, True)
    
    # If only 'false' found, the fact is not supported
    return (False, True)


async def medical_recommendations_atomic_facts_reward_func(
    judge, prompt, completion, answer, state, info=None, **kwargs
) -> float:
    """
    Reward function that evaluates medical recommendations using atomic facts.
    
    This function:
    1. Breaks down the ground truth answer into atomic facts
    2. Evaluates each fact against the model's completion using an LLM judge
    3. Returns the proportion of facts that are supported by the completion
    
    Args:
        judge: The judge function to evaluate each atomic fact
        prompt: The original prompt given to the model
        completion: The model's response to evaluate
        answer: The ground truth answer to compare against
        state: Current state object
        info: Optional additional information
        **kwargs: Additional arguments including:
            - judge_client: AsyncOpenAI client for the judge
            - judge_model: Model name for judging (default: 'gpt-4o-mini')
    
    Returns:
        Score between 0.0 and 1.0 representing the proportion of atomic facts
        that are supported by the completion. Returns 0.0 if no facts are found.
    """
    # Check for verbose mode to print debug information
    verbose = '-v' in sys.argv

    # Initialize the atomic facts generator with the LLM client
    llm_client = kwargs.get('judge_client')
    llm_model = kwargs.get('judge_model', 'gpt-4o-mini')
    generator = AtomicFactGenerator(llm_client, model_name=llm_model)
    
    # Generate atomic facts from the ground truth answer
    atomic_facts = await generator.run(answer)

    # Ensure completion is a string for processing
    completion_text = completion if isinstance(completion, str) else str(completion)
    
    # Print debug information if verbose mode is enabled
    if verbose:
        print(f"completion_text: {completion_text}")
        print(f"answer: {answer}")
        print(f"prompt: {prompt}")
        print(f"atomic_facts: {atomic_facts}")

    # Evaluate each atomic fact against the completion
    total_score = 0
    total_facts = 0
    
    for fact in atomic_facts:
        total_facts += 1
        
        # Use the judge to evaluate if the completion supports this fact
        judge_response = await judge(prompt, completion_text, fact, state, **kwargs)
        score, passed_parse = extract_boolean_score(judge_response)
        
        # Only increment score if the fact was successfully parsed and supported
        if passed_parse and score:
            total_score += 1
            
        if verbose:
            print(f"Fact: {fact}, Score: {score}, Passed Parse: {passed_parse}")
    
    # Calculate final score as proportion of supported facts
    if verbose and total_facts > 0:
        print(f"Total Score: {total_score}, Total Facts: {total_facts}, Score: {total_score/total_facts}")

    # Return the proportion of facts supported (0.0 if no facts)
    score = total_score / total_facts if total_facts > 0 else 0.0
    return score


def create_atomic_facts_judge_rubric(
    judge_client: AsyncOpenAI | None,
    judge_model: str = "gpt-4o-mini",
) -> vf.JudgeRubric:
    """
    Creates and configures a JudgeRubric for atomic facts-based evaluation.
    
    This factory function sets up a judge rubric that uses the atomic facts
    reward function to evaluate medical recommendations. The rubric breaks down
    answers into atomic facts and verifies each one against the model's response.
    
    Args:
        judge_client: AsyncOpenAI client instance for making judge API calls
        judge_model: Model name to use for judging (default: "gpt-4o-mini")
        
    Returns:
        Configured JudgeRubric instance with the atomic facts reward function
    """
    # Initialize the rubric with the judge configuration
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_TEMPLATE,
    )
    
    # Register the atomic facts reward function with full weight
    rubric.add_reward_func(medical_recommendations_atomic_facts_reward_func, weight=1.0)
    
    return rubric
