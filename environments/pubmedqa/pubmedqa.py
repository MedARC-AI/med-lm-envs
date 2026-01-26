import json
import os

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets progress indicators

PROMPT_TEMPLATE = """Select the best answer.

Context: {abstract_as_context}

Question: {question}
{options_block}
Answer: """

BASE_OPTIONS = {"A": "Yes", "B": "No", "C": "Maybe"}


def map_row_to_mcq_prompt(
    row: dict,
    idx: int | None = None,
    *,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
):
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

    options = dict(BASE_OPTIONS)

    if shuffle_answers:
        row_id = row.get("pubid", idx)
        shuffled, correct_answer_letter, _ = randomize_multiple_choice(
            options=options,
            answer_choice=correct_answer_letter,
            seed=shuffle_seed,
            row_id=row_id,
        )
        options = dict(shuffled)

    # Zip them together and format as label[0]: contexts[0]
    formatted_contexts = []
    for label, context in zip(labels, contexts):
        formatted_contexts.append(f"{label}. {context}")
    context_text = "\n".join(formatted_contexts)

    options_block = "\n".join(f"{letter}. {text}" for letter, text in options.items())

    # see EXAMPLE_COMPLETE_PROMPT
    complete_prompt = PROMPT_TEMPLATE.format(
        abstract_as_context=context_text,
        question=question_text,
        options_block=options_block,
    )

    # required fields: question (for the prompt), and answer (for the scoring)
    info = {
        "answer_text": options.get(correct_answer_letter, final_decision),
    }
    if shuffle_answers:
        info["options"] = options

    return {
        "question": complete_prompt,
        "answer": correct_answer_letter,
        "task": "pubmedqa",
        "info": info,
    }


def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(
        llm_answer=parsed,
        answer_letter=answer,
        answer_text=answer_text,
    )
    return 1.0 if is_correct else 0.0


def load_environment(
    use_think: bool = False,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
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

    cache_enabled = not shuffle_answers
    fn_kwargs = {"shuffle_answers": shuffle_answers, "shuffle_seed": shuffle_seed}
    mapped_dataset_train = dataset_train.map(
        map_row_to_mcq_prompt,
        with_indices=True,
        fn_kwargs=fn_kwargs,
        load_from_cache_file=cache_enabled,
        keep_in_memory=True,
    )
    mapped_dataset_test = dataset_test.map(
        map_row_to_mcq_prompt,
        with_indices=True,
        fn_kwargs=fn_kwargs,
        load_from_cache_file=cache_enabled,
        keep_in_memory=True,
    )

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
    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=mapped_dataset_train,
        eval_dataset=mapped_dataset_test,
        system_prompt=system_prompt,
        rubric=rubric,  # if a rubric is given, it needs to manually call the parser
        parser=parser,  # needs to be same parser as given to rubric, otherwise raises a warning
    )

    return vf_env
