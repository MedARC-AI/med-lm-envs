import os
import re
import json
import urllib.request
from datasets import load_dataset
import verifiers as vf
import json

SYSTEM_PROMPT_THINK=vf.utils.data_utils.THINK_BOXED_SYSTEM_PROMPT
SYSTEM_PROMPT_NOTHINK = vf.utils.data_utils.BOXED_SYSTEM_PROMPT

ANSWER_FORMAT = r"\\boxed{LETTER}"
SINGLE_PROMPT_TEMPLATE = r"""
Answer the following multiple choice question about medical knowledge given the context.
Your final answer should be should be of the following format: '{answer_format}'
(without quotes) where LETTER is one of {letters}. 
{abstract_as_context_and_question}

{choices}
""".strip()


def map_row_to_mcq_prompt(row):
    """Map dataset format for PubMedQA samples"""

    # each example is a row of the HF dataset with keys: 
    # ['pubid', 'question', 'context', 'long_answer', 'final_decision']
    question_text = row.get('question')

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
            choices="A) yes\nB) no\nC) maybe", answer_format=ANSWER_FORMAT)
        
    # required fields: question (for the prompt), and answer (for the scoring)
    return {
        "question": complete_prompt,
        "answer": correct_answer_letter,
        "task": "pubmedqa",
    }


def classification_reward_func(prompt, completion, answer, state, **kwargs) -> float:
    """
    Classification-based reward function for PubMedQA.
    Returns 1.0 for correct classification, 0.0 otherwise.
    """

    # completition is a chat response, like: {'role': 'assistant', 'content': 'ANSWER: A'}]
    completion=completion[0]["content"]
    
    parsed_completion = kwargs["parser"].parse(completion);
    predicted_letter = parsed_completion.strip().rstrip(".")

    if predicted_letter is None:
        return 0.0  # Incorrect if no valid answer found
    
    return 1.0 if predicted_letter == answer else 0.0


def load_environment(use_think: bool = False) -> vf.Environment:
    """
    PubMedQA environment using classification-based evaluation.
    
    This environment loads the PubMedQA dataset and uses exact match scoring
    for yes/no/maybe classification tasks.
    """
 
    # Both subsets only have a 'train' split
    DATASET_PATH = "qiaojin/PubMedQA"
    dataset_train = load_dataset(DATASET_PATH, name="pqa_artificial", split="train")
    dataset_test = load_dataset(DATASET_PATH, name="pqa_labeled", split="train")

    # Read in the predefined IDs in the test split taken from 
    # https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json
    file_path = os.path.join("data", "test_ground_truth.json")
    with open(file_path) as f:
        test_ids = json.load(f)

    # reducing the 1000k annotated to the 500 human annotated
    dataset_test = dataset_test.filter(
        lambda sample: str(sample["pubid"]) in test_ids
    )
    
    mapped_dataset_train = dataset_train.map(map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True)
    mapped_dataset_test = dataset_test.map(map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True)

    sys_prompt=SYSTEM_PROMPT_NOTHINK
    parser = vf.parsers.parser.Parser(extract_fn = vf.extract_boxed_answer) 

    if use_think:
        sys_prompt=SYSTEM_PROMPT_THINK
        parser = vf.parsers.think_parser.ThinkParser(extract_fn = vf.extract_boxed_answer)

    # parses the reponse using parser and calculates rewards
    rubric = vf.Rubric(
        funcs=[classification_reward_func], weights=[1.0], parser= parser
    )

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=mapped_dataset_train,
        eval_dataset=mapped_dataset_test,
        system_prompt=sys_prompt,
        rubric=rubric, # if a rubric is given, it needs to manually call the parser
        parser=parser, # needs to be same parser as given to rubric, otherwise raises a warning
    )

    return vf_env
