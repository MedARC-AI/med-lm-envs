import verifiers as vf
from datasets import load_dataset
from verifiers.utils.data_utils import extract_boxed_answer

from medarc_verifiers.prompts import AnswerFormat


def _get_system_prompt(use_think: bool, answer_format: AnswerFormat) -> str:
    if answer_format == AnswerFormat.BOXED:
        think_system_prompt = "You are a helpful medical assistant. Think step-by-step inside <think>...</think> tags. Put your final answer within \\boxed{}."
        no_think_system_prompt = (
            "You are a helpful medical assistant. Think step-by-step and put your final answer within \\boxed{}."
        )
    elif answer_format == AnswerFormat.XML:
        think_system_prompt = "You are a helpful medical assistant. Think step-by-step inside <think>...</think> tags. Put your final answer within <answer>...</answer> tags."
        no_think_system_prompt = "You are a helpful medical assistant. Think step-by-step and put your final answer within <answer>...</answer> tags."
    else:
        raise ValueError(f"Unsupported answer format: {answer_format}")
    system_prompt = think_system_prompt if use_think else no_think_system_prompt
    return system_prompt


def _format_question_with_options(question_with_options: str, options) -> str:
    """
    Rebuild the composite question string from the standalone stem and options.
    This keeps the current formatting while letting us randomize the options later.
    """
    if not options:
        return question_with_options

    if isinstance(options, dict):
        option_items = list(options.items())
    elif isinstance(options, list):
        option_items = [(chr(ord("A") + idx), value) for idx, value in enumerate(options)]
    else:
        return question_with_options

    question, sep, _ = question_with_options.partition("Answer Choices:")
    question = question.strip() if sep else question_with_options.strip()
    formatted_options = " ".join(f"({key}) {value}" for key, value in option_items)
    if not formatted_options:
        return question
    return f"{question}\nAnswer Choices: {formatted_options}"


def load_environment(
    use_think: bool = False,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    """
    MedXpertQA environment using equality with the "label" column as the eval criterion.
    This environment loads the MedXpertQA dataset and compares model responses (diagnosis) with the ground truth in "label" column.
    """
    full_dataset = load_dataset("TsinghuaC3I/MedXpertQA", "Text")
    test_dataset = full_dataset["test"].map(
        lambda x: {
            "question": x["question"],
            "answer": x["label"],
            "task": "medxpertqa",
            "options": x.get("options"),
        }
    )

    def _map(ex):
        return {
            "question": _format_question_with_options(ex["question"], ex.get("options")),
            "answer": ex["label"],
        }

    mapped = test_dataset.map(_map).filter(lambda r: r is not None)

    async def medxpertqa_reward_func(completion: str, answer: str) -> float:
        """
        Reward function for MedXpertQA environment.
        Compares the model response with the ground truth answer.
        Returns 1.0 if they match (case-insensitive), else 0.0.
        """

        final_answer = parser.parse_answer(completion).strip()
        if final_answer.lower() == answer.lower():
            return 1.0
        else:
            return 0.0

    # normalize answer_format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    system_prompt = _get_system_prompt(use_think, answer_format)

    if answer_format == AnswerFormat.XML:
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")

    rubric = vf.Rubric(funcs=[medxpertqa_reward_func], weights=[1.0])

    vf_env = vf.SingleTurnEnv(eval_dataset=mapped, system_prompt=system_prompt, rubric=rubric, parser=parser)

    return vf_env
