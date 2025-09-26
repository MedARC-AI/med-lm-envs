import os
import re
import json
import urllib.request
from datasets import load_dataset
import verifiers as vf
import json

# "Think step-by-step inside <think>...</think> tags. Then, give your final answer inside \\boxed{}."
SYSTEM_PROMPT_THINK=vf.utils.data_utils.THINK_BOXED_SYSTEM_PROMPT
SYSTEM_PROMPT_NOTHINK = "" # empty

# Each question prompt matching Inspect Evals TEMPLATE format
# SINGLE_PROMPT_TEMPLATE = r"""
# Answer the following multiple choice question about medical knowledge given the context.
# The entire content of your response should be of the following format: 'ANSWER: $LETTER'
# (without quotes) where LETTER is one of {letters}.
#
# {abstract_as_context_and_question}
#
# {choices}
# """.strip()

ANSWER_FORMAT = r"\\boxed{ANSWER: $LETTER}"
SINGLE_PROMPT_TEMPLATE = r"""
Answer the following multiple choice question about medical knowledge given the context.
Your final answer should be should be of the following format: '{answer_format}'
(without quotes) where LETTER is one of {letters}. 
Only provide the answer in the format specified. 
Do not provide any further content after the final answer.

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
            choices="A) yes\nB) no\nC) maybe", answer_format=ANSWER_FORMAT)
        
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

    # matched and stripped response need to correspond to any of the three allowed options 
    allowed_options = {"A", "B", "C"}

    # matches "ANSWER: [spaces][mulitple a-zA-Z, digits, spaces][end of line or period]"
    # returns the [mulitple a-zA-Z, digits, spaces] group
    match = re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z\d\s]+)(?:[^\w]|\n|$|\.)",
            response,
    )
    # if no match, return the entire response
    matched = response if match is None else match.group(1)
    # Strip trailing period / full stop
    matched = matched.strip()
    matched = matched.rstrip(".")
    
    # return only reponses that are part of the allowed options
    return matched if matched in allowed_options else None


def classification_reward_func(prompt, completion, answer, state, **kwargs) -> float:
    """
    Classification-based reward function for PubMedQA.
    Returns 1.0 for correct classification, 0.0 otherwise.
    """

    # completition is a chat response, like:
    # {'role': 'assistant', 'content': 'ANSWER: A'}]
    # so we get the first item
    completion=completion[0]["content"]
    #print("Prompt:", prompt)
    
    parsed_completion = kwargs["parser"].parse(completion);
    predicted_letter = extract_choice(parsed_completion)
    #print(f"Completion: \033[34m{completion}\033[0m -> \033[1m{predicted_letter}\033[0m")
    
    # Handle case where no valid answer was extracted
    if predicted_letter is None:
        return 0.0  # Incorrect if no valid answer found
    
    # Compare with ground truth
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

    # Read in the predefined IDs in the test split taken from https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json
    file_path = os.path.join("data", "test_ground_truth.json")
    with open(file_path) as f:
        test_ids = json.load(f)

    # reducing the 1000k annotated to the 500 human annotated
    dataset_test = dataset_test.filter(
        lambda sample: str(sample["pubid"]) in test_ids
    )
    
    # load_from_cache_file=False, keep_in_memory=True need to be specified, as otherwise datasets will default to using the cache
    # but we want to be sure that the loaded dataset reflects the most recent updates to map_row_to_mcq_prompt
    mapped_dataset_train = dataset_train.map(map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True)
    mapped_dataset_test = dataset_test.map(map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True)

    sys_prompt=SYSTEM_PROMPT_NOTHINK
    # if \\boxed{<answer>} is present, extract answer from it, otherwise return the text
    parser = vf.parsers.parser.Parser(extract_fn = vf.extract_boxed_answer) 

    if use_think:
        sys_prompt=SYSTEM_PROMPT_THINK
        # ThinkParser requires <think>...</think> tags to be present and strips everything up to </think>
        # from the remaining part it then extracts the answer using the extract_fn
        parser = vf.parsers.think_parser.ThinkParser(extract_fn = vf.extract_boxed_answer)

    # parses the reponse using parser and calculates the rewards based on the extrancted answers
    rubric = vf.Rubric(
        funcs=[classification_reward_func], weights=[1.0], parser= parser
    )

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        #dataset=mapped_dataset_train,
        dataset=mapped_dataset_train,
        eval_dataset=mapped_dataset_test,
        system_prompt=sys_prompt,
        rubric=rubric, # if a rubric is given, it needs to manually call the parser
        parser=parser, # needs to be same parser as given to rubric, otherwise raises a warning
    )

    return vf_env
