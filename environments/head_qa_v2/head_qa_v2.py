"""
HEAD-QA v2 environment

This script defines an evaluation environment for HEAD-QA v2 compatible with the Verifiers framework.

The prompts were adapted from the HEAD-QA v2 paper.

MIT License

Citation:

@article{correa2025head,
  title={HEAD-QA v2: Expanding a Healthcare Benchmark for Reasoning},
  author={Correa-Guillen, Alexis and Gomez-Rodriguez, Carlos and Vilares, David},
  journal={arXiv preprint arXiv:2511.15355},
  year={2025}
}
"""

from typing import Any

import verifiers as vf
from datasets import load_dataset
from medarc_verifiers.parsers.json_parser import JSONParser
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice


def zero_shot_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Build the Zero-shot prompt."""
    question_text = (example.get("qtext") or "").strip()
    answers = example.get("answers", [])
    options_text = "\n".join([f"{a['aid']}. {a['atext'].strip()}" for a in answers])

    prompt = (
        "You are an expert in specialized scientific and health disciplines. "
        "Respond to the following multiple-choice question:\n"
        'Provide the answer in the following JSON format: {"answer": [number]}\n'
        'For example, if the answer is 1, write: {"answer": 1}\n\n'
        f"{question_text}\n{options_text}\n"
    )

    correct_answer = example.get("ra", -1)

    result = {
        "question": prompt,
        "answer": str(correct_answer),
        "choices": [str(a["aid"]) for a in answers],
        "gold_index": correct_answer - 1,
        "info": {"answer_text": answers[correct_answer - 1]["atext"].strip()},
    }
    return result


def few_shot_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Build the few-shot prompt with samples."""
    question_text = (example.get("qtext") or "").strip()
    answers = example.get("answers", [])
    options_text = "\n".join([f"{a['aid']}. {a['atext'].strip()}" for a in answers])

    # Few-shot demonstrations
    demonstrations = [
        {
            "qtext": "Which neurotransmitter is primarily involved in mood regulation?",
            "answers": [
                {"aid": 1, "atext": "Dopamine"},
                {"aid": 2, "atext": "Serotonin"},
                {"aid": 3, "atext": "GABA"},
                {"aid": 4, "atext": "Acetylcholine"},
            ],
            "ra": 2,
        },
        {
            "qtext": "Which of the following is an example of a neutralization reaction in chemistry?",
            "answers": [
                {"aid": 1, "atext": "CH4 + 2O2 → CO2 + 2H2O"},
                {"aid": 2, "atext": "Na + Cl2 → 2NaCl"},
                {"aid": 3, "atext": "2H2 + O2 → 2H2O"},
                {"aid": 4, "atext": "HCl + NaOH → NaCl + H2O"},
            ],
            "ra": 4,
        },
    ]

    demonstrations_text = ""
    for demo in demonstrations:
        demo_question = (demo.get("qtext") or "").strip()
        demo_answers = demo.get("answers", [])
        demo_options = "\n".join([f"{a['aid']}. {a['atext'].strip()}" for a in demo_answers])
        demo_correct = demo.get("ra", -1)
        demonstrations_text += f'{demo_question}\n{demo_options}\n{{"answer": {demo_correct}}}\n\n'

    prompt = (
        "You are an expert in specialized scientific and health disciplines. "
        'Respond to the following multiple-choice question in the following JSON format: {"answer": [number]}. '
        "No explanations are needed.\n\n"
        f"{demonstrations_text}"
        f"{question_text}\n{options_text}\n"
    )

    correct_answer = example.get("ra", -1)

    result = {
        "question": prompt,
        "answer": str(correct_answer),
        "choices": [str(a["aid"]) for a in answers],
        "gold_index": correct_answer - 1,
        "info": {"answer_text": answers[correct_answer - 1]["atext"].strip()},
    }
    return result


def cot_prompt(example: dict[str, Any]) -> dict[str, Any]:
    """Build the Chain-of-Thought (CoT) prompt with brief reasoning."""
    question_text = (example.get("qtext") or "").strip()
    answers = example.get("answers", [])
    options_text = "\n".join([f"{a['aid']}. {a['atext'].strip()}" for a in answers])

    prompt = (
        "You are an expert in scientific and health disciplines.\n"
        "Carefully analyze the following multiple-choice question and provide the correct answer. "
        "There is one and only one correct answer. "
        'Think through each option briefly before responding in the following JSON format: {"answer": [number]}.\n\n'
        f"{question_text}\n{options_text}\n"
    )

    correct_answer = example.get("ra", -1)

    result = {
        "question": prompt,
        "answer": str(correct_answer),
        "choices": [str(a["aid"]) for a in answers],
        "gold_index": correct_answer - 1,
        "info": {"answer_text": answers[correct_answer - 1]["atext"].strip()},
    }
    return result


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text") if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    prompt_type: str = "zero_shot",
    system_prompt: str | None = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
) -> vf.Environment:
    """
    Load the HEAD-QA v2 environment.

    Args:
        prompt_type: "zero_shot", "few_shot", or "cot"
        shuffle_answers: Shuffle answer options
        shuffle_seed: Seed for shuffling
    """
    val_ds = load_dataset("alesi12/head_qa_v2", "en", split="train")

    def _map_example(example: dict[str, Any]) -> dict[str, Any] | None:
        correct_answer = example.get("ra", -1)
        question_text = (example.get("qtext") or "").strip()
        answers = example.get("answers", [])

        if not question_text or not answers or not (1 <= correct_answer <= len(answers)):
            return None

        options = [a["atext"].strip() for a in answers]
        answer_idx = correct_answer - 1

        if shuffle_answers:
            indices = [str(i + 1) for i in range(len(options))]
            shuffled_options, _, answer_idx = randomize_multiple_choice(
                options=options,
                answer_choice=answer_idx,
                labels=indices,
                seed=shuffle_seed,
                row_id=question_text,
            )
            options = shuffled_options

        temp_example = {
            "qtext": question_text,
            "answers": [{"aid": i + 1, "atext": opt} for i, opt in enumerate(options)],
            "ra": answer_idx + 1,
        }

        # Select prompt based on type
        if prompt_type == "zero_shot":
            mapped = zero_shot_prompt(temp_example)
        elif prompt_type == "few_shot":
            mapped = few_shot_prompt(temp_example)
        elif prompt_type == "cot":
            mapped = cot_prompt(temp_example)
        else:
            raise ValueError(f"Unsupported prompt_type: {prompt_type}")
        return mapped

    columns_to_remove = ["qtext", "answers", "ra"]
    # Disable the Datasets cache when shuffling answers
    load_from_cache_file = False if shuffle_answers else True
    val_mapped = val_ds.map(
        _map_example,
        remove_columns=columns_to_remove,
        load_from_cache_file=load_from_cache_file,
    )

    parser = JSONParser(fields=["answer"], answer_field="answer")

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    env = vf.SingleTurnEnv(
        dataset=None,
        eval_dataset=val_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

    return env
