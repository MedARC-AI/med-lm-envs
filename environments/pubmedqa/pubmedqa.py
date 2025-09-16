import os
import re
import json
import urllib.request
from datasets import load_dataset
import verifiers as vf
import json

SYSTEM_PROMPT="" # no system prompt given. 

# Each question prompt matching Inspect Evals TEMPLATE format
SINGLE_PROMPT_TEMPLATE = r"""
Answer the following multiple choice question about medical knowledge given the context.
The entire content of your response should be of the following format: 'ANSWER: $LETTER'
(without quotes) where LETTER is one of {letters}.

{abstract_as_context_and_question}

{choices}
""".strip()


def map_row_to_mcq_prompt(row):
    """Map dataset format for PubMedQA samples"""

    # each example is a row of the HF dataset with keys: 
    # ['pubid', 'question', 'context', 'long_answer', 'final_decision']
    
    # a string containing the question: 
    # e.g. "Must early postoperative oral intake be limited to laparoscopy?""
    question_text = row.get('question')

    # a dict containing the pubmed paper abstract split in mutliple sections, e.g.
    # context_dict = {
    #     'labels': ['PURPOSE', 'METHODS', 'RESULTS'], 
    #     'contexts': ['To explore the extent to which parent-adolescent emotional closeness ...', 
    #         'Cross-sectional survey of 7631 adolescents from 231 Australian schools. Measures included ...', 
    #         'The interaction of family factors and pubertal stage did not improve the fit of the model, ....'
    #     ], 
    #     # and some unused columns: 
    #     'meshes': ['Adolescent', 'Alcohol Drinking', ...]
    #     'reasoning_required_pred': ['y', 'e', 's'], 
    #     'reasoning_free_pred': ['n', 'o']
    # }
    context_dict = row.get('context') 
    labels = context_dict.get('labels') # list of the abstract subsection titles
    contexts = context_dict.get('contexts') # a list of the subsections contents
    
    # a string which is either "yes", "no" or "maybe"
    final_decision = row.get('final_decision', '').lower() 
    choices_map = {"yes": "A", "no": "B", "maybe": "C"} 
    correct_answer_letter = choices_map[final_decision]
    
    # Zip them together and format as label[0]: contexts[0]
    formatted_contexts = []
    for label, context in zip(labels, contexts):
        formatted_contexts.append(f"({label}) {context}")
    context_text = '\n'.join(formatted_contexts)
    
    # Format as multiple choice question
    context_and_question = f"Context:\n{context_text}\n\nQuestion: {question_text}"

    # see EXAMPLE_COMPLETE_PROMPT
    complete_prompt = SINGLE_PROMPT_TEMPLATE.format(letters="A, B, C",
            abstract_as_context_and_question=context_and_question,
            choices="A) yes\nB) no\nC) maybe")
    
    # required fields for each dataset row:
    #  question: transformed into the user-chat message forming part of the prompt
    #  answer: given to the reward scoring fucntion / rubic
    #  task (optional): task name

    return {
        # alternatively: format the prompt outselves, 
        # skiping the formatting by verifiers SingleTurnEnv
        #"prompt": [{"role": "system", "content": SYSTEM_PROMPT},
        #    {"role": "user", "content": complete_prompt}],  
        "question": complete_prompt,
        "answer": correct_answer_letter,
        "task": "pubmedqa",
    }


def extract_choice(response: str) -> str:
    """Extract choice from model response using Inspect Evals parse_answers logic"""
    # Based on inspect_ai.solver._multiple_choice.parse_answers
    
    # First check whether the string strictly ends with the expected answer
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z\d ,]+)\s*(?:$|\n|\.)",
        response,
        flags=re.MULTILINE,
    )

    # If we couldn't match the strict version, try the less strict version
    if match is None:
        match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d ,]+)(?:[^\w]|\n|$|\.)",
            response,
        )

    if match is None:
        return None  # No valid answer found - matches Inspect Evals

    matched = match.group(1)

    # Strip trailing period / full stop
    matched = matched.strip()
    matched = matched.rstrip(".")

    # For single choice (PubMedQA), match must contain a single letter in allowed choices
    allowed_options = {"A", "B", "C"}
    if matched in allowed_options:
        letter_to_choice = {"A": "yes", "B": "no", "C": "maybe"}
        return letter_to_choice.get(matched, None)

    return None  # No valid answer found - matches Inspect Evals


def classification_reward_func(prompt, completion, answer, state, **kwargs) -> float:
    """
    Classification-based reward function for PubMedQA.
    Returns 1.0 for correct classification, 0.0 otherwise.
    """
    # completition is a chat response, like:
    # {'role': 'assistant', 'content': 'ANSWER: A'}]
    # so we get the first item
    completion=completion[0]["content"]
    predicted_choice = extract_choice(completion)
    
    # Handle case where no valid answer was extracted
    if predicted_choice is None:
        return 0.0  # Incorrect if no valid answer found
    
    # Map back to A/B/C format for comparison
    choice_to_letter = {"yes": "A", "no": "B", "maybe": "C"}
    predicted_letter = choice_to_letter.get(predicted_choice, None)
    
    if predicted_letter is None:
        return 0.0  # Incorrect if mapping failed
    
    # Compare with ground truth
    return 1.0 if predicted_letter == answer else 0.0


def load_environment() -> vf.Environment:
    """
    PubMedQA environment using classification-based evaluation.
    
    This environment loads the PubMedQA dataset and uses exact match scoring
    for yes/no/maybe classification tasks.
    """
 
    # pqa_labeled is the subset they use for benchmarking. This subset only have one split 'train'
    DATASET_PATH = "qiaojin/PubMedQA"
    dataset = load_dataset(DATASET_PATH, name="pqa_labeled", split="train")

    # Read in the predefined IDs in the test split taken from https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json
    file_path = os.path.join("data", "test_ground_truth.json")
    with open(file_path) as f:
        test_ids = json.load(f)  

    # reducing the 1000k annotated to the 500 human annotated
    test_dataset = dataset.filter(
        lambda sample: str(sample["pubid"]) in test_ids
    )
    
    mapped_dataset = test_dataset.map(map_row_to_mcq_prompt, load_from_cache_file=False)

    rubric = vf.Rubric(
        funcs=[classification_reward_func], weights=[1.0]
    )
    
    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=mapped_dataset,
        eval_dataset=mapped_dataset,
        system_prompt="", # by default no-system prompt given
        rubric=rubric,
    )

    return vf_env
