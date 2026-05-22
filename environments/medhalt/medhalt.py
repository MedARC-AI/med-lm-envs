"""MedHALT: Medical Domain Hallucination Test

This environment implements evaluation for the MedHALT dataset, which tests
medical language models on multiple-choice questions from various medical exams.

Dataset: https://huggingface.co/datasets/openlifescienceai/Med-HALT
Paper: https://github.com/medhalt/medhalt

Supported configurations:
- reasoning_FCT: Functional correct thinking questions
- reasoning_nota: None-of-the-above questions

"""

from datasets import load_dataset
from typing import Optional, List, Dict, Any
import verifiers as vf
import re
import json
import ast
import random

def load_environment(
    config_name: str = "reasoning_FCT",
    split: str = "train",
    num_examples: Optional[int] = None,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 42,
) -> vf.SingleTurnEnv:
    """Load MedHALT environment for medical LLM evaluation. 
    
    Args:
        config_name: One of ["reasoning_FCT", "reasoning_nota"]
        split: Dataset split to use
        num_examples: Optional limit on number of examples
        shuffle_answers: Whether to randomize answer order
        shuffle_seed: Random seed for answer shuffling
    
    Returns:
        A SingleTurnEnv ready for evaluation
    """""
    valid_configs = ["reasoning_FCT", "reasoning_nota"]
    if config_name not in valid_configs:
        raise ValueError(f"Invalid config_name: {config_name}")
    
    raw_dataset = load_dataset("openlifescienceai/Med-HALT", config_name, split=split)

    dataset = raw_dataset.map(
        lambda ex: _format_example(ex, shuffle_answers, shuffle_seed),
        remove_columns=raw_dataset.column_names,
        load_from_cache_file=False,
    )
   
    # Create rubric with simple accuracy check
    def accuracy(completion: Any, answer: str, parser: vf.Parser, info: dict[str, Any] | None = None) -> float:
        parsed = parser.parse_answer(completion) or ""
        predicted = _extract_letter(parsed, info.get("num_options", 4) if info else 4)
        return 1.0 if predicted and predicted.upper() == answer.upper() else 0.0

    if num_examples is not None:
        dataset = dataset.select(range(min(num_examples, len(dataset))))
    
    parser = vf.Parser()
    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)
    
    return vf.SingleTurnEnv(dataset=dataset, parser=parser, rubric=rubric)

def _format_example(example: dict[str, Any], shuffle_answers: bool = False, shuffle_seed: int | None = 42) -> dict[str, Any]:
    """Format a MedHALT example into verifiers format."""
    
    question = example.get('question', '').strip()
    
    # Parse options string into dict
    options_raw = example.get('options', {})
    if isinstance(options_raw, str):
        try:
            options_raw = ast.literal_eval(options_raw)
        except Exception as e:
            raise ValueError(f"Failed to parse options: {e}") from e
    
    # Build options list in order (skip the 'correct answer' key if present)
    options_list = []
    i = 0
    while str(i) in options_raw:
        text = options_raw[str(i)]
        if text and str(text).strip():
            options_list.append(str(text).strip())
        i += 1
    
    # Get correct answer index directly from data
    if 'correct_index' not in example:
        raise ValueError("Missing correct_index field")
    
    original_answer_idx = int(example['correct_index'])
    
    # Validate index is in range
    if original_answer_idx < 0 or original_answer_idx >= len(options_list):
        raise ValueError(
            f"correct_index {original_answer_idx} out of range for {len(options_list)} options. "
            f"Question: {question[:50]}..."
        )
    
    # Shuffle if requested
    if shuffle_answers:
        rng = random.Random(f"{shuffle_seed}_{question}")
        indices = list(range(len(options_list)))
        rng.shuffle(indices)
        
        options_list = [options_list[i] for i in indices]
        answer_idx = indices.index(original_answer_idx)
    else:
        answer_idx = original_answer_idx
    
    # Build option_map with letters
    option_map = {chr(65 + i): text for i, text in enumerate(options_list)}
    
    # Build prompt
    options_text = '\n'.join(f"{letter}. {text}" for letter, text in option_map.items())
    prompt_text = f"""Answer the following multiple-choice medical question.

Question: {question}

Options:
{options_text}

Provide your answer as a single letter."""
    
    correct_letter = chr(65 + answer_idx)
    
    return {
        'question': prompt_text,
        'answer': correct_letter,
        'choices': list(option_map.keys()),
        'info': {
            'question_text': question,
            'options': option_map,
            'num_options': len(option_map),
        }
    }

def _extract_letter(text: str, num_options: int = 4) -> str | None:
    """Extract answer letter (A, B, C, D, ...) from model response.
    
    Args:
        text: Model's response text
        num_options: Number of valid options (determines valid letters)
    
    Returns:
        Extracted letter or None if not found
    """
    if not text:
        return None
    
    text = text.strip().upper()
    valid_letters = [chr(65 + i) for i in range(num_options)]
    
    # Direct match (just "A" or "B")
    if len(text) == 1 and text in valid_letters:
        return text
    
    # Find first valid letter in the response
    for char in text:
        if char in valid_letters:
            return char
    
    return None