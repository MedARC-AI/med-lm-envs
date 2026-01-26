"""LongHealth Environment

Lifted from https://github.com/kbressem/LongHealth/tree/main
Originally licensed Apache-2.0 license

@article{adams2024longhealth,
  title={LongHealth: A Question Answering Benchmark with Long Clinical Documents},
  author={Adams, Lisa and Busch, Felix and Han, Tianyu and Excoffier, Jean-Baptiste and Ortala, Matthieu and L{\"o}ser, Alexander and Aerts, Hugo JWL and Kather, Jakob Nikolas and Truhn, Daniel and Bressem, Keno},
  journal={arXiv preprint arXiv:2401.14490},
  year={2024}
}
"""

from enum import Enum
import json
import os
import random
from typing import Any

import verifiers as vf
from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice

disable_progress_bar()  # suppress datasets progress indicators

# Reuse the system prompt from the original LongHealth implementation
LONGHEALTH_SYSTEM_PROMPT = """You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. When you receive one or more of these letters, you will be expected to carefully review the contents and accurately answer multiple-choice questions related to these documents. 

Your answers should be:
1. Accurate: Make sure your answers are based on the information provided in the letters.
2. Concise: Provide brief and direct answers without unnecessary elaboration.
3. Contextual: Consider the context and specifics of each question to provide the most relevant information.

Remember, your job is to streamline the physician's decision-making process by providing them with accurate and relevant information from discharge summaries. Efficiency and reliability are key.
"""


class LongHealthTask(str, Enum):
    TASK1 = "task1"
    TASK2 = "task2"
    ALL = "all"


def _build_longhealth_prompt(
    documents: list[str],
    question_text: str,
    options: dict[str, str],
    separator: str = "--------------",
) -> str:
    """
    Build a LongHealth prompt with documents and multiple-choice question.

    Args:
        documents: List of document strings (already selected and ordered)
        question_text: The question text
        options: Dictionary mapping option letters to answer text
        separator: Separator between documents

    Returns:
        Formatted prompt string
    """
    # Format documents
    documents_joined = f"\n\n{separator}\n\n".join(documents)

    # Format options
    options_text = "\n".join([f"{k.upper()}: {v}" for k, v in options.items()])

    # Build full prompt
    prompt = f"""--------------BEGIN DOCUMENTS--------------

{documents_joined}

--------------END DOCUMENTS--------------

{question_text}
{options_text}

Please answer using the following format:
1. Begin your answer with the phrase "The correct answer is".
2. State the letter of the correct option (e.g., A, B, C, D, E).
3. Follow the letter with a colon and the exact text of the option you chose.
4. Make sure your answer is a single, concise sentence.

For example, if the correct answer to a question is option C, and the text for C is 'Acute Bronchitis', your answer should be: 
'The correct answer is C: Acute bronchitis.'
"""
    return prompt


def _simple_truncate_documents(
    answer_docs: list[str],
    non_answer_docs: list[str],
    max_tokens: int,
    tokens_per_char: float = 0.25,  # rough estimate
) -> list[str]:
    """
    Simplified document truncation that prioritizes answer documents.

    Args:
        answer_docs: Documents containing the answer
        non_answer_docs: Distractor documents
        max_tokens: Maximum number of tokens
        tokens_per_char: Rough token-to-character ratio

    Returns:
        List of selected documents (answer docs + as many non-answer docs as fit)
    """
    max_chars = int(max_tokens / tokens_per_char)

    selected_docs = []
    total_chars = 0

    # Add all answer docs first (prioritize them)
    for doc in answer_docs:
        doc_chars = len(doc)
        if total_chars + doc_chars <= max_chars:
            selected_docs.append(doc)
            total_chars += doc_chars
        else:
            # Truncate this doc to fit
            remaining = max_chars - total_chars
            if remaining > 500:  # only add if we have reasonable space
                selected_docs.append(doc[:remaining])
            break

    # Add non-answer docs if space permits
    for doc in non_answer_docs:
        doc_chars = len(doc)
        if total_chars + doc_chars <= max_chars:
            selected_docs.append(doc)
            total_chars += doc_chars
        else:
            break

    return selected_docs


def _maybe_shuffle_options(
    options: dict[str, str],
    answer_letter: str | None,
    *,
    shuffle_answers: bool,
    shuffle_seed: int | None,
    row_id: str | None,
) -> tuple[dict[str, str], str | None]:
    """Shuffle answer options when requested, returning updated options and answer letter."""
    if not shuffle_answers or shuffle_seed is None or not answer_letter or answer_letter not in options:
        return options, answer_letter

    cannot_answer_text = "Question cannot be answered with provided documents"
    special_label = next((lab for lab, text in options.items() if text == cannot_answer_text), None)

    base_options: dict[str, str]
    special_entry: tuple[str, str] | None = None
    if special_label is not None and special_label in options:
        base_options = {k: v for k, v in options.items() if k != special_label}
        special_entry = (special_label, options[special_label])
    else:
        base_options = dict(options)

    new_letter = answer_letter
    if base_options:
        shuffle_answer_choice = answer_letter if answer_letter in base_options else next(iter(base_options))
        randomized_options, shuffled_letter, _ = randomize_multiple_choice(
            options=base_options,
            answer_choice=shuffle_answer_choice,
            seed=shuffle_seed,
            row_id=row_id,
        )
        if answer_letter in base_options:
            new_letter = shuffled_letter
    else:
        randomized_options = {}

    ordered_options = dict(randomized_options)
    if special_entry is not None:
        ordered_options[special_entry[0]] = special_entry[1]
        new_letter = special_entry[0] if answer_letter == special_entry[0] else new_letter

    return ordered_options, new_letter


def _prepare_task1_data(
    benchmark: dict,
    max_context_tokens: int = 16000,
    shuffle_docs: bool = True,
    doc_shuffle_seed: int | None = -1,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
) -> list[dict]:
    """
    Prepare Task 1 data: Information extraction with answer documents present.

    Task 1 tests the model's ability to extract correct information from long
    clinical documents when the answer IS present in the provided context.

    Args:
        benchmark: The loaded benchmark_v5.json data
        max_context_tokens: Maximum tokens for document context
        shuffle_docs: Whether to shuffle document order
        doc_shuffle_seed: Seed for document shuffling (-1 for random each time)
        shuffle_answers: Whether to shuffle answer choices
        shuffle_seed: Seed for answer shuffling (None to disable, -1 for nondeterministic)

    Returns:
        List of examples in VF format
    """
    examples = []

    for patient_id, patient_data in benchmark.items():
        if doc_shuffle_seed is not None and doc_shuffle_seed >= 0:
            random.seed(f"{doc_shuffle_seed}::{patient_id}")

        texts = patient_data["texts"]
        questions = patient_data["questions"]

        for question in questions:
            question_text = question["question"]

            # Build options dict
            options = {
                "A": question["answer_a"],
                "B": question["answer_b"],
                "C": question["answer_c"],
                "D": question["answer_d"],
                "E": question["answer_e"],
            }

            correct_answer = question["correct"]
            # Find which letter corresponds to correct answer
            correct_letter = None
            for letter, text in options.items():
                if text == correct_answer:
                    correct_letter = letter
                    break

            if not correct_letter:
                continue  # skip if we can't find the correct answer

            # Get answer and non-answer documents
            answer_location = question.get("answer_location", {})
            answer_text_ids = list(answer_location.keys())

            answer_docs = [texts[text_id] for text_id in answer_text_ids if text_id in texts]
            non_answer_docs = [text for text_id, text in texts.items() if text_id not in answer_text_ids]

            # Shuffle non-answer docs for variety
            if shuffle_docs and len(non_answer_docs) > 1:
                random.shuffle(non_answer_docs)

            # Select and optionally shuffle documents
            selected_docs = _simple_truncate_documents(answer_docs, non_answer_docs, max_context_tokens)

            # Skip if no documents selected
            if len(selected_docs) == 0:
                continue

            if shuffle_docs and len(selected_docs) > 1:
                random.shuffle(selected_docs)

            # Randomize options if requested
            row_id = f"{patient_id}::{question.get('No') or question_text}"
            options, correct_letter = _maybe_shuffle_options(
                options,
                correct_letter,
                shuffle_answers=shuffle_answers,
                shuffle_seed=shuffle_seed,
                row_id=row_id,
            )

            # Build prompt
            prompt = _build_longhealth_prompt(selected_docs, question_text, options)

            info = {
                "patient_id": patient_id,
                "question_no": question.get("No"),
                "task": "task1",
                "correct_answer_text": correct_answer,
                "num_docs": len(selected_docs),
                "has_answer_docs": len(answer_docs) > 0,
            }
            if shuffle_answers:
                info["options"] = dict(options)
                info["answer_letter"] = correct_letter

            examples.append(
                {
                    "question": prompt,
                    "answer": correct_letter,
                    "info": info,
                }
            )

    return examples


def _sample_distraction_docs(
    current_patient_id: str,
    benchmark: dict,
    n: int = 10,
) -> list[str]:
    """Sample n texts from other patients as distractions."""
    all_other_texts = [
        text for pid, patient in benchmark.items() if pid != current_patient_id for text in patient["texts"].values()
    ]

    n_sample = min(n, len(all_other_texts))
    return random.sample(all_other_texts, n_sample)


def _prepare_task2_data(
    benchmark: dict,
    max_context_tokens: int = 16000,
    shuffle_docs: bool = True,
    doc_shuffle_seed: int | None = -1,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
) -> list[dict]:
    """
    Prepare Task 2 data: Negation detection and hallucination prevention.

    Task 2 tests the model's ability to:
    1. Identify when information is NOT available (negation)
    2. Correctly extract information when it IS available (with distractors)

    For each question, we create TWO examples:
    - One WITHOUT answer docs (should respond with F: "Cannot be answered")
    - One WITH answer docs + distractors (should respond with correct answer)

    Args:
        benchmark: The loaded benchmark_v5.json data
        max_context_tokens: Maximum tokens for document context
        shuffle_docs: Whether to shuffle document order

    Returns:
        List of examples in VF format (2x the number of questions)
    """
    examples = []

    for patient_id, patient_data in benchmark.items():
        if doc_shuffle_seed is not None and doc_shuffle_seed >= 0:
            random.seed(f"{doc_shuffle_seed}::{patient_id}")

        texts = patient_data["texts"]
        questions = patient_data["questions"]

        for question in questions:
            question_text = question["question"]

            # Build options dict with F option for "cannot be answered"
            options_base = {
                "A": question["answer_a"],
                "B": question["answer_b"],
                "C": question["answer_c"],
                "D": question["answer_d"],
                "E": question["answer_e"],
            }

            correct_answer = question["correct"]
            correct_letter = None
            for letter, text in options_base.items():
                if text == correct_answer:
                    correct_letter = letter
                    break

            if not correct_letter:
                continue

            # Get answer documents
            answer_location = question.get("answer_location", {})
            answer_text_ids = list(answer_location.keys())
            answer_docs = [texts[text_id] for text_id in answer_text_ids if text_id in texts]

            # --- Example 1: NEGATION (no answer docs, only distractions) ---
            options_with_f = {**options_base, "F": "Question cannot be answered with provided documents"}

            distraction_docs = _sample_distraction_docs(patient_id, benchmark, n=10)
            if shuffle_docs and len(distraction_docs) > 1:
                random.shuffle(distraction_docs)

            # Select subset of distraction docs (ensure at least 1)
            num_docs_neg = max(1, min(len(distraction_docs), int(max_context_tokens * 0.25 / 1000)))
            selected_docs_neg = distraction_docs[:num_docs_neg]

            # Skip if no distraction docs available
            if len(selected_docs_neg) == 0:
                continue

            row_id_neg = f"{patient_id}::{question.get('No') or question_text}::negation"
            options_neg, answer_letter_neg = _maybe_shuffle_options(
                options_with_f,
                "F",
                shuffle_answers=shuffle_answers,
                shuffle_seed=shuffle_seed,
                row_id=row_id_neg,
            )
            prompt_negation = _build_longhealth_prompt(selected_docs_neg, question_text, options_neg)

            info_neg = {
                "patient_id": patient_id,
                "question_no": question.get("No"),
                "task": "task2_negation",
                "correct_answer_text": "Question cannot be answered with provided documents",
                "num_docs": len(selected_docs_neg),
                "has_answer_docs": False,
            }
            if shuffle_answers:
                info_neg["options"] = dict(options_neg)
                info_neg["answer_letter"] = answer_letter_neg or "F"

            examples.append(
                {
                    "question": prompt_negation,
                    "answer": answer_letter_neg or "F",  # F when docs don't contain info
                    "info": info_neg,
                }
            )

            # --- Example 2: IDENTIFICATION (answer docs + distractions) ---
            all_distraction_docs = _sample_distraction_docs(patient_id, benchmark, n=10)

            selected_docs_ident = _simple_truncate_documents(answer_docs, all_distraction_docs, max_context_tokens)

            # Skip if no documents selected
            if len(selected_docs_ident) == 0:
                continue

            if shuffle_docs and len(selected_docs_ident) > 1:
                random.shuffle(selected_docs_ident)

            row_id_ident = f"{patient_id}::{question.get('No') or question_text}::identification"
            options_ident, correct_letter_ident = _maybe_shuffle_options(
                options_with_f,
                correct_letter,
                shuffle_answers=shuffle_answers,
                shuffle_seed=shuffle_seed,
                row_id=row_id_ident,
            )

            prompt_identification = _build_longhealth_prompt(selected_docs_ident, question_text, options_ident)

            info_ident = {
                "patient_id": patient_id,
                "question_no": question.get("No"),
                "task": "task2_identification",
                "correct_answer_text": correct_answer,
                "num_docs": len(selected_docs_ident),
                "has_answer_docs": True,
            }
            if shuffle_answers:
                info_ident["options"] = dict(options_ident)
                info_ident["answer_letter"] = correct_letter_ident if correct_letter_ident else correct_letter

            examples.append(
                {
                    "question": prompt_identification,
                    "answer": correct_letter_ident if correct_letter_ident else correct_letter,
                    "info": info_ident,
                }
            )

    return examples


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("correct_answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(
        llm_answer=parsed, answer_letter=answer, answer_text=answer_text, prefix="The correct answer is"
    )
    return 1.0 if is_correct else 0.0


def load_environment(
    task: str | LongHealthTask = LongHealthTask.TASK1,
    max_context_tokens: int = 16000,
    shuffle_docs: bool = True,
    doc_shuffle_seed: int | None = -1,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    max_examples: int = -1,
    **kwargs,
) -> vf.Environment:
    """
    Args:
        task: Which task(s) to load:
            - "task1": Information extraction (answer present in docs)
            - "task2": Negation detection + identification (Task 2 & 3)
            - "all": Both tasks combined
        max_context_tokens: Maximum tokens for document context (~14k for 16k models)
        shuffle_docs: Whether to shuffle document order (tests positional bias)
        doc_shuffle_seed: Seed for document shuffling (-1 for random each time)
        shuffle_answers: Whether to shuffle multiple-choice answers
        shuffle_seed: Seed for answer shuffling (None to disable, -1 for nondeterministic)
        max_examples: Limit number of examples (-1 for all)

    Returns:
        vf.Environment configured for LongHealth evaluation

    Example Usage
        >>> # Load Task 1 (information extraction)
        >>> env = vf.load_environment("longhealth", task="task1")

        >>> # Load Task 2 (negation + identification)
        >>> env = vf.load_environment("longhealth", task="task2", max_context_tokens=16000)

        >>> # Run evaluation
        >>> vf-eval longhealth -m gpt-4.1-mini -n 10 -a '{"task": "task1"}'
    """

    # Load benchmark data
    here = os.path.dirname(__file__)
    benchmark_path = os.path.join(here, "benchmark_v5.json")

    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(
            f"LongHealth benchmark data not found at {benchmark_path}. Please ensure the LongHealth data is available."
        )

    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)

    effective_doc_seed = -1 if doc_shuffle_seed is None else doc_shuffle_seed
    if effective_doc_seed == -1 and not shuffle_answers and shuffle_seed not in (None, 1618):
        effective_doc_seed = shuffle_seed

    task = LongHealthTask(task) if isinstance(task, str) else task

    # Prepare data based on task
    if task == LongHealthTask.TASK1:
        examples = _prepare_task1_data(
            benchmark, max_context_tokens, shuffle_docs, effective_doc_seed, shuffle_answers, shuffle_seed
        )
    elif task == LongHealthTask.TASK2:
        examples = _prepare_task2_data(
            benchmark, max_context_tokens, shuffle_docs, effective_doc_seed, shuffle_answers, shuffle_seed
        )
    elif task == LongHealthTask.ALL:
        task1_examples = _prepare_task1_data(
            benchmark, max_context_tokens, shuffle_docs, effective_doc_seed, shuffle_answers, shuffle_seed
        )
        task2_examples = _prepare_task2_data(
            benchmark, max_context_tokens, shuffle_docs, effective_doc_seed, shuffle_answers, shuffle_seed
        )
        examples = task1_examples + task2_examples
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'task1', 'task2', or 'all'")

    parser = vf.Parser()

    # Limit examples if requested
    if max_examples > 0:
        examples = examples[:max_examples]

    # Convert to HuggingFace Dataset
    eval_dataset = Dataset.from_list(examples)

    # Create rubric with accuracy reward
    rubric = vf.Rubric(funcs=[accuracy], parser=parser, weights=[1.0])

    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset, system_prompt=LONGHEALTH_SYSTEM_PROMPT, rubric=rubric, parser=parser, **kwargs
    )
