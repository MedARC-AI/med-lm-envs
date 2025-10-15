""" LongHealth Environment

Lifted from https://github.com/kbressem/LongHealth/tree/main
Originally licensed Apache-2.0 license

@article{adams2024longhealth,
  title={LongHealth: A Question Answering Benchmark with Long Clinical Documents},
  author={Adams, Lisa and Busch, Felix and Han, Tianyu and Excoffier, Jean-Baptiste and Ortala, Matthieu and L{\"o}ser, Alexander and Aerts, Hugo JWL and Kather, Jakob Nikolas and Truhn, Daniel and Bressem, Keno},
  journal={arXiv preprint arXiv:2401.14490},
  year={2024}
}
"""
from __future__ import annotations
import json
import os
import random
import re
from typing import Optional, Literal
from datasets import Dataset
import verifiers as vf
from verifiers.utils.data_utils import (
    extract_boxed_answer,
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
)

# Reuse the system prompt from the original LongHealth implementation
LONGHEALTH_SYSTEM_PROMPT = """
You are a highly skilled and detail-oriented assistant, specifically trained to assist medical professionals in interpreting and extracting key information from medical documents. Your primary responsibility will be to analyze discharge letters from hospitals. When you receive one or more of these letters, you will be expected to carefully review the contents and accurately answer multiple-choice questions related to these documents. 

Your answers should be:
1. Accurate: Make sure your answers are based on the information provided in the letters.
2. Concise: Provide brief and direct answers without unnecessary elaboration.
3. Contextual: Consider the context and specifics of each question to provide the most relevant information.

Remember, your job is to streamline the physician's decision-making process by providing them with accurate and relevant information from discharge summaries. Efficiency and reliability are key.
"""

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


def _prepare_task1_data(
    benchmark: dict,
    max_context_tokens: int = 16000,
    shuffle_docs: bool = True,
) -> list[dict]:
    """
    Prepare Task 1 data: Information extraction with answer documents present.
    
    Task 1 tests the model's ability to extract correct information from long
    clinical documents when the answer IS present in the provided context.
    
    Args:
        benchmark: The loaded benchmark_v5.json data
        max_context_tokens: Maximum tokens for document context
        shuffle_docs: Whether to shuffle document order
        
    Returns:
        List of examples in VF format
    """
    examples = []
    
    for patient_id, patient_data in benchmark.items():
        texts = patient_data["texts"]
        questions = patient_data["questions"]
        
        for question in questions:
            question_text = question["question"]
            
            # Build options dict
            options = {
                "a": question["answer_a"],
                "b": question["answer_b"],
                "c": question["answer_c"],
                "d": question["answer_d"],
                "e": question["answer_e"],
            }
            
            correct_answer = question["correct"]
            # Find which letter corresponds to correct answer
            correct_letter = None
            for letter, text in options.items():
                if text == correct_answer:
                    correct_letter = letter.upper()
                    break
            
            if not correct_letter:
                continue  # skip if we can't find the correct answer
            
            # Get answer and non-answer documents
            answer_location = question.get("answer_location", {})
            answer_text_ids = list(answer_location.keys())
            
            answer_docs = [texts[text_id] for text_id in answer_text_ids if text_id in texts]
            non_answer_docs = [
                text for text_id, text in texts.items() 
                if text_id not in answer_text_ids
            ]
            
            # Shuffle non-answer docs for variety
            if shuffle_docs and len(non_answer_docs) > 1:
                random.shuffle(non_answer_docs)
            
            # Select and optionally shuffle documents
            selected_docs = _simple_truncate_documents(
                answer_docs, non_answer_docs, max_context_tokens
            )
            
            # Skip if no documents selected
            if len(selected_docs) == 0:
                continue
            
            if shuffle_docs and len(selected_docs) > 1:
                random.shuffle(selected_docs)
            
            # Build prompt
            prompt = _build_longhealth_prompt(selected_docs, question_text, options)
            
            examples.append({
                "question": prompt,
                "answer": correct_letter,
                "info": {
                    "patient_id": patient_id,
                    "question_no": question.get("No"),
                    "task": "task1",
                    "correct_answer_text": correct_answer,
                    "num_docs": len(selected_docs),
                    "has_answer_docs": len(answer_docs) > 0,
                }
            })
    
    return examples


def _sample_distraction_docs(
    current_patient_id: str,
    benchmark: dict,
    n: int = 10,
) -> list[str]:
    """Sample n texts from other patients as distractions."""
    all_other_texts = [
        text
        for pid, patient in benchmark.items()
        if pid != current_patient_id
        for text in patient["texts"].values()
    ]
    
    n_sample = min(n, len(all_other_texts))
    return random.sample(all_other_texts, n_sample)


def _prepare_task2_data(
    benchmark: dict,
    max_context_tokens: int = 16000,
    shuffle_docs: bool = True,
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
        texts = patient_data["texts"]
        questions = patient_data["questions"]
        
        for question in questions:
            question_text = question["question"]
            
            # Build options dict with F option for "cannot be answered"
            options_base = {
                "a": question["answer_a"],
                "b": question["answer_b"],
                "c": question["answer_c"],
                "d": question["answer_d"],
                "e": question["answer_e"],
            }
            
            correct_answer = question["correct"]
            correct_letter = None
            for letter, text in options_base.items():
                if text == correct_answer:
                    correct_letter = letter.upper()
                    break
            
            if not correct_letter:
                continue
            
            # Get answer documents
            answer_location = question.get("answer_location", {})
            answer_text_ids = list(answer_location.keys())
            answer_docs = [texts[text_id] for text_id in answer_text_ids if text_id in texts]
            
            # --- Example 1: NEGATION (no answer docs, only distractions) ---
            options_with_f = {
                **options_base,
                "f": "Question cannot be answered with provided documents"
            }
            
            distraction_docs = _sample_distraction_docs(patient_id, benchmark, n=10)
            if shuffle_docs and len(distraction_docs) > 1:
                random.shuffle(distraction_docs)
            
            # Select subset of distraction docs (ensure at least 1)
            num_docs_neg = max(1, min(len(distraction_docs), int(max_context_tokens * 0.25 / 1000)))
            selected_docs_neg = distraction_docs[:num_docs_neg]
            
            # Skip if no distraction docs available
            if len(selected_docs_neg) == 0:
                continue
            
            prompt_negation = _build_longhealth_prompt(
                selected_docs_neg, question_text, options_with_f
            )
            
            examples.append({
                "question": prompt_negation,
                "answer": "F",  # Correct answer is F when docs don't contain info
                "info": {
                    "patient_id": patient_id,
                    "question_no": question.get("No"),
                    "task": "task2_negation",
                    "correct_answer_text": "Question cannot be answered with provided documents",
                    "num_docs": len(selected_docs_neg),
                    "has_answer_docs": False,
                }
            })
            
            # --- Example 2: IDENTIFICATION (answer docs + distractions) ---
            all_distraction_docs = _sample_distraction_docs(patient_id, benchmark, n=10)
            
            selected_docs_ident = _simple_truncate_documents(
                answer_docs, all_distraction_docs, max_context_tokens
            )
            
            # Skip if no documents selected
            if len(selected_docs_ident) == 0:
                continue
            
            if shuffle_docs and len(selected_docs_ident) > 1:
                random.shuffle(selected_docs_ident)
            
            prompt_identification = _build_longhealth_prompt(
                selected_docs_ident, question_text, options_with_f
            )
            
            examples.append({
                "question": prompt_identification,
                "answer": correct_letter,  # Original correct answer
                "info": {
                    "patient_id": patient_id,
                    "question_no": question.get("No"),
                    "task": "task2_identification",
                    "correct_answer_text": correct_answer,
                    "num_docs": len(selected_docs_ident),
                    "has_answer_docs": True,
                }
            })
    
    return examples


def _extract_letter_from_response(response: str) -> Optional[str]:
    """
    Extract answer letter from LongHealth-style response.
    
    Expected format: "The correct answer is C: Acute bronchitis."
    Also handles boxed format: \\boxed{C}
    
    Args:
        response: The model's response text
        
    Returns:
        Extracted letter (A-F) or None
    """
    if not response:
        return None
    
    response_upper = response.upper().strip()
    
    # Try boxed format first
    boxed_match = re.search(r'\\boxed\{([A-F])\}', response_upper)
    if boxed_match:
        return boxed_match.group(1)
    
    # Try "The correct answer is X" format
    answer_match = re.search(r'THE CORRECT ANSWER IS\s*([A-F])', response_upper)
    if answer_match:
        return answer_match.group(1)
    
    # Try to find any single letter A-F
    letter_match = re.search(r'\b([A-F])\b', response_upper)
    if letter_match:
        return letter_match.group(1)
    
    return None


def exact_match_reward(parser: vf.Parser, completion: str, answer: str, **kwargs) -> float:
    """
    Reward function for exact match on answer letter.
    
    Args:
        parser: The parser (may use boxed answer extraction)
        completion: Model's completion text
        answer: Ground truth answer letter
        
    Returns:
        1.0 if match, 0.0 otherwise
    """
    # Try parser first
    parsed = parser.parse_answer(completion)
    if parsed:
        extracted = _extract_letter_from_response(parsed)
    else:
        extracted = _extract_letter_from_response(completion)
    
    if not extracted:
        return 0.0
    
    return 1.0 if extracted.upper() == answer.upper() else 0.0


def load_environment(
    task: Literal["task1", "task2", "all"] = "task1",
    max_context_tokens: int = 16000,
    use_think: bool = False,
    shuffle_docs: bool = True,
    use_custom_system_prompt: bool = True,
    max_examples: int = -1,
    **kwargs
) -> vf.Environment:
    """   
    Args:
        task: Which task(s) to load:
            - "task1": Information extraction (answer present in docs)
            - "task2": Negation detection + identification (Task 2 & 3)
            - "all": Both tasks combined
        max_context_tokens: Maximum tokens for document context (~14k for 16k models)
        use_think: Whether to use <think></think> tags for reasoning
        shuffle_docs: Whether to shuffle document order (tests positional bias)
        use_custom_system_prompt: Use LongHealth-specific system prompt
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
    benchmark_path = os.path.join(here,"benchmark_v5.json")
    
    if not os.path.exists(benchmark_path):
        raise FileNotFoundError(
            f"LongHealth benchmark data not found at {benchmark_path}. "
            "Please ensure the LongHealth data is available."
        )
    
    with open(benchmark_path, "r", encoding="utf-8") as f:
        benchmark = json.load(f)
    
    # Prepare data based on task
    if task == "task1":
        examples = _prepare_task1_data(benchmark, max_context_tokens, shuffle_docs)
    elif task == "task2":
        examples = _prepare_task2_data(benchmark, max_context_tokens, shuffle_docs)
    elif task == "all":
        task1_examples = _prepare_task1_data(benchmark, max_context_tokens, shuffle_docs)
        task2_examples = _prepare_task2_data(benchmark, max_context_tokens, shuffle_docs)
        examples = task1_examples + task2_examples
    else:
        raise ValueError(f"Unknown task: {task}. Must be 'task1', 'task2', or 'all'")
    
    # Limit examples if requested
    if max_examples > 0:
        examples = examples[:max_examples]
    
    # Convert to HuggingFace Dataset
    eval_dataset = Dataset.from_list(examples)
    
    # Setup parser and system prompt
    parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
    
    if use_custom_system_prompt:
        system_prompt = LONGHEALTH_SYSTEM_PROMPT
    else:
        system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
    
    # Create rubric with exact match reward
    rubric = vf.Rubric(
        funcs=[exact_match_reward],
        weights=[1.0],
        parser=parser,
    )
    
    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
