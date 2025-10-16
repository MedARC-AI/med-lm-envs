from typing import Any, Optional

import verifiers as vf
from datasets import Dataset, load_dataset
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
)

_VOCAB_CHOICES = ['atc', 'icd10cm', 'icd10proc', 'icd9cm', 'icd9proc']
_LEVEL_CHOICES = ['easy', 'hard', 'medium']

def _create_few_shot_data(few_shot_set: Dataset, num_few_shot: int) -> dict[tuple, str]:
    """Create few-shot examples from the dev set, grouped by (vocab, level).
    Args:
        few_shot_set: the dev set to draw few-shot examples from
        num_few_shot: number of few-shot examples to include per (vocab, level) pair
    Returns:
        dict mapping (vocab, level) to concatenated few-shot examples
    """
    few_shot_examples = {}

    for row in few_shot_set:
        key = (row['vocab'], row['level'])
        if key not in few_shot_examples:
            few_shot_examples[key] = []
        if len(few_shot_examples[key]) < num_few_shot:
            prompt = f"{row['question']}\nAnswer:{row['answer_id']}\n\n".replace("  ", "")
            few_shot_examples[key].append(prompt)
    
    for key in few_shot_examples:
        few_shot_examples[key] = "".join(few_shot_examples[key])
    
    return few_shot_examples

def load_environment(
        use_think: bool = False,
        num_few_shot: int = 3,
        vocab: Optional[str] = None,
        level: Optional[str] = None,
    ) -> vf.Environment:
    """MedConceptsQA multiple-choice evaluation
    - Loads HF 'ofir408/MedConceptsQA' (contains only dev and test split)
    - Builds a prompt per item, with optional few-shot examples from the dev set
    - Scores accuracy by comparing the model's A/B/C/D answer to the gold answer
    - Supports reasoning (use_think=True) or non-reasoning models

    Args:
        num_few_shot: number of few-shot examples to include in the prompt (default: 3)
        use_think: whether to use a ThinkParser and reasoning system prompt (default: False)
        vocab: vocabulary to subset dataset, if `None` - choses `all` (default: None)
        level: difficulty level to subset dataset (used in conjunction with vocab).
            cannot be None if vocab is not None (default: None)
    Returns:
        vf.Environment: the single-turn evaluation environment
    """
    subset = "all"
    if vocab is not None:
        if vocab not in _VOCAB_CHOICES:
            raise ValueError(f"Invalid vocab choice {vocab}, must be one of {_VOCAB_CHOICES}")
        if level is None or level not in _LEVEL_CHOICES:
            raise ValueError(f"Invalid level choice {level}, must be one of {_LEVEL_CHOICES}")
        subset = f"{vocab}_{level}"

    # load the entire dataset, should contain dev and test
    ds = load_dataset("ofir408/MedConceptsQA", subset)
    test = ds['test']
    
    if num_few_shot > 0:
        # few-shot examples are chosen based on the `vocab` and `level`
        few_shot_data = _create_few_shot_data(ds['dev'], num_few_shot)
    
    def _map(row: dict) -> dict:
        vocab = row['vocab']
        level = row['level']
        question = row['question']
        answer = row['answer_id']
        few_shot_prompt = few_shot_data.get((vocab, level), "") if num_few_shot > 0 else ""
        
        full_question = (
            "Answer A,B,C,D according to the answer to this multiple choice question.\n" +
            few_shot_prompt +
            ("\n" if len(few_shot_prompt) > 0 else "") +
            question +
            "\nAnswer:"
        )
        return {
            "question": full_question,
            "answer": answer,
        }
        
    mapped = test.map(
        _map,
        remove_columns=test.column_names,
    )

    system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT

    parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)

    def accuracy_reward(completion: Any, answer: str) -> float:
        parsed = parser.parse_answer(completion).strip().upper()[0]  # first character
        answer = answer.upper()
        return 1.0 if parsed == answer else 0.0
    
    rubric = vf.Rubric(
        funcs=[accuracy_reward],
        weights=[1.0]
    )
    
    return vf.SingleTurnEnv(
        eval_dataset=mapped,
        rubric=rubric,
        system_prompt=system_prompt,
        parser=parser,
    )
