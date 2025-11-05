"""
Med-HALT (Reasoning) Environment

Implements a verifiers environment for the Med-HALT dataset's reasoning hallucination tests.
The dataset includes three test types:
- reasoning_FCT (False Confidence Test): Evaluates if models can assess proposed answers
- reasoning_nota (None of the Above Test): Tests if models can identify when none of the options are correct
- reasoning_fake (Fake Questions Test): Assesses if models can handle nonsensical questions

Paper: https://arxiv.org/abs/2307.15343
Dataset: https://huggingface.co/datasets/openlifescienceai/Med-HALT
"""

from __future__ import annotations

import ast
from typing import Any, Optional

import verifiers as vf
from datasets import Dataset, concatenate_datasets, load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT, AnswerFormat
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

# The three reasoning test types in Med-HALT
REASONING_TEST_TYPES = ["reasoning_FCT", "reasoning_nota", "reasoning_fake"]
LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H"]


def _parse_options(options_str: str) -> dict[str, str]:
    """
    Parse the options field which is stored as a string representation of a dict.
    
    Args:
        options_str: String representation of options dict
        
    Returns:
        Dictionary mapping option indices to option text
    """
    try:
        options_dict = ast.literal_eval(options_str)
        if isinstance(options_dict, dict):
            return options_dict
    except (ValueError, SyntaxError):
        pass
    return {}


def _build_mcq_prompt(question: str, options: dict[str, str]) -> str:
    """
    Build a multiple-choice question prompt with lettered options.
    
    Args:
        question: The question text
        options: Dictionary mapping indices to option text
        
    Returns:
        Formatted MCQ prompt string
    """
    # Sort options by their numeric keys
    sorted_items = sorted(options.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    
    # Filter out 'correct answer' key if present
    option_items = [(k, v) for k, v in sorted_items if k != "correct answer"]
    
    # Build the prompt with letter labels
    lines = [f"Question: {question}"]
    for idx, (_, option_text) in enumerate(option_items):
        letter = LETTER_INDICES[idx]
        lines.append(f"{letter}. {option_text}")
    lines.append("Answer:")
    
    return "\n".join(lines)


def _map_example(
    example: dict[str, Any],
    test_type: str,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = None,
) -> dict[str, Any] | None:
    """
    Map a Med-HALT example to verifiers format.
    
    Args:
        example: Raw example from the dataset
        test_type: The reasoning test type (FCT, nota, or fake)
        shuffle_answers: Whether to randomize answer order
        shuffle_seed: Random seed for shuffling
        
    Returns:
        Mapped example or None if invalid
    """
    question = example.get("question", "").strip()
    if not question:
        return None
    
    # Parse options from string representation
    options_str = example.get("options", "{}")
    options_dict = _parse_options(options_str)
    
    if not options_dict:
        return None
    
    # Get correct answer index and text
    correct_index = example.get("correct_index")
    correct_answer = example.get("correct_answer", "")
    
    # For reasoning_fake, there's no correct answer (testing if model handles nonsense)
    # We'll need to handle this differently
    if test_type == "reasoning_fake":
        # Fake questions don't have correct answers in the traditional sense
        # We might want to skip these or handle them specially
        # For now, we'll skip them as they need special evaluation logic
        return None
    
    # Convert numeric index to letter
    if correct_index is None:
        return None
    
    # Build options list from the dict
    sorted_items = sorted(
        [(k, v) for k, v in options_dict.items() if k != "correct answer"],
        key=lambda x: int(x[0]) if x[0].isdigit() else x[0]
    )
    
    if correct_index >= len(sorted_items):
        return None
    
    options_list = [v for _, v in sorted_items]
    answer_letter = LETTER_INDICES[correct_index]
    
    # Handle answer shuffling
    if shuffle_answers:
        shuffled_options, answer_letter, new_index = randomize_multiple_choice(
            options=options_list,
            answer_choice=correct_index,
            labels=LETTER_INDICES[:len(options_list)],
            seed=shuffle_seed,
            row_id=example.get("id", question),
        )
        options_list = shuffled_options
        correct_index = new_index
    
    # Build the options dict for the prompt
    options_for_prompt = {str(i): opt for i, opt in enumerate(options_list)}
    
    prompt = _build_mcq_prompt(question, options_for_prompt)
    
    return {
        "question": prompt,
        "answer": answer_letter,
        "info": {
            "answer_text": correct_answer or options_list[correct_index],
            "test_type": test_type,
            "dataset": example.get("dataset", ""),
            "subject_name": example.get("subject_name", ""),
            "split_type": example.get("split_type", ""),
            **({"options": options_for_prompt} if shuffle_answers else {}),
        },
    }


def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
    """Reward based on shared multiple-choice accuracy grading."""
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text", None) if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    test_types: list[str] | None = None,
) -> vf.Environment:
    """
    Load the Med-HALT (Reasoning) environment.
    
    Args:
        use_think: Enable chain-of-thought reasoning with <think> tags
        system_prompt: Custom system prompt (defaults to standard XML/BOXED prompt)
        shuffle_answers: Randomize the order of answer choices
        shuffle_seed: Random seed for reproducible shuffling
        answer_format: Answer format (XML or BOXED)
        test_types: List of test types to include (default: ["reasoning_FCT", "reasoning_nota"])
                   Available: "reasoning_FCT", "reasoning_nota", "reasoning_fake"
    
    Returns:
        A SingleTurnEnv configured for Med-HALT reasoning evaluation
    """
    # Default to FCT and nota tests (excluding fake for now as it needs special handling)
    if test_types is None:
        test_types = ["reasoning_FCT", "reasoning_nota"]
    
    # Validate test types
    invalid_types = [t for t in test_types if t not in REASONING_TEST_TYPES]
    if invalid_types:
        raise ValueError(
            f"Invalid test types: {invalid_types}. "
            f"Must be one of: {REASONING_TEST_TYPES}"
        )
    
    # Load datasets for each test type
    datasets: list[Dataset] = []
    for test_type in test_types:
        ds = load_dataset("openlifescienceai/Med-HALT", test_type)
        
        # Map the examples
        def _map(ex: dict[str, Any]) -> dict[str, Any] | None:
            return _map_example(ex, test_type, shuffle_answers, shuffle_seed)
        
        # The dataset only has a 'train' split
        if "train" in ds:
            load_from_cache_file = not shuffle_answers
            mapped = ds["train"].map(
                _map,
                remove_columns=ds["train"].column_names,
                load_from_cache_file=load_from_cache_file,
            )
            # Filter out None values
            mapped = mapped.filter(
                lambda x: x is not None and x.get("question") is not None,
                load_from_cache_file=load_from_cache_file,
            )
            datasets.append(mapped)
    
    if not datasets:
        raise ValueError(f"No valid datasets loaded for test types: {test_types}")
    
    # Concatenate all test type datasets
    combined_dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    
    # Normalize answer format
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    
    # Set up parser and system prompt based on format
    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format}")
    
    # Create rubric with accuracy reward
    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)
    
    # Med-HALT only has train split, so we use it as the main dataset
    # No separate eval_dataset
    return vf.SingleTurnEnv(
        dataset=combined_dataset,
        eval_dataset=None,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
