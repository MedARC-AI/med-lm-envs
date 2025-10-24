import json
import os

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

SINGLE_PROMPT_TEMPLATE = r"""Answer A for yes, B for no or C for maybe.\n\nContext: {abstract_as_context}\n\nQuestion: {question}\nAnswer: """


def map_row_to_mcq_prompt(row):
    """Map dataset format for PubMedQA samples"""

    # each example is a row of the HF dataset with keys:
    # ['pubid', 'question', 'context', 'long_answer', 'final_decision']
    question_text = row.get("question")

    context_dict = row.get("context")
    labels = context_dict.get("labels")  # list of the abstract subsection titles
    contexts = context_dict.get("contexts")  # a list of the subsections contents

    # a string which is either "yes", "no" or "maybe"
    final_decision = row.get("final_decision", "").lower()
    choices_map = {"yes": "A", "no": "B", "maybe": "C"}
    correct_answer_letter = choices_map[final_decision]

    # Zip them together and format as label[0]: contexts[0]
    formatted_contexts = []
    for label, context in zip(labels, contexts):
        formatted_contexts.append(f"{label}. {context}")
    context_text = "\n".join(formatted_contexts)

    # see EXAMPLE_COMPLETE_PROMPT
    complete_prompt = SINGLE_PROMPT_TEMPLATE.format(abstract_as_context=context_text, question=question_text)

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
    completion = completion[0]["content"]

    parsed_completion = kwargs["parser"].parse(completion)
    predicted_letter = parsed_completion.strip().rstrip(".")

    if predicted_letter is None:
        return 0.0  # Incorrect if no valid answer found

    return 1.0 if predicted_letter == answer else 0.0


def load_environment(use_think: bool = False, answer_format: AnswerFormat | str = AnswerFormat.XML) -> vf.Environment:
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
    here = os.path.dirname(__file__)
    file_path = os.path.join(here, "data", "test_ground_truth.json")
    with open(file_path) as f:
        test_ids = json.load(f)

    # reducing the 1000k annotated to the 500 human annotated
    dataset_test = dataset_test.filter(lambda sample: str(sample["pubid"]) in test_ids)

    mapped_dataset_train = dataset_train.map(map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True)
    mapped_dataset_test = dataset_test.map(map_row_to_mcq_prompt, load_from_cache_file=False, keep_in_memory=True)

    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if answer_format == AnswerFormat.XML:
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
        system_prompt = THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT
    elif answer_format == AnswerFormat.BOXED:
        parser = (
            vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think else vf.Parser(extract_fn=extract_boxed_answer)
        )
        system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    # parses the reponse using parser and calculates rewards
    rubric = vf.Rubric(funcs=[classification_reward_func], weights=[1.0], parser=parser)

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=mapped_dataset_train,
        eval_dataset=mapped_dataset_test,
        system_prompt=system_prompt,
        rubric=rubric,  # if a rubric is given, it needs to manually call the parser
        parser=parser,  # needs to be same parser as given to rubric, otherwise raises a warning
    )

    return vf_env
