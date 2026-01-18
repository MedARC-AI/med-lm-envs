from enum import Enum
import hashlib
import random

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from verifiers.utils.data_utils import extract_boxed_answer

from medhallu.prompts import system_prompt, create_prompt_no_knowledge, create_prompt_with_knowledge

disable_progress_bar()  # suppress datasets progress indicators


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ALL = "all"


def reward_fn(
    completion: str,
    answer: str,
    parser: vf.Parser,
    unsure_reward: float = 0.01,
    **kwargs,
) -> float:
    """
    Reward function for MedHallu.

    - Matches Target ('0' or '1'): + 1.0
    - Model says '2' (unsure): + 0.01 (small reward)
    - Incorrect: 0
    """
    # Extract the model's answer (expecting 0, 1, or 2 in boxed)
    extracted = parser.parse_answer(completion)

    if not extracted:
        return 0.0  # failed to produce a boxed answer

    cleaned_extracted = extracted.strip()

    if cleaned_extracted == answer:
        return 1.0  # correct
    elif cleaned_extracted == "2":
        return float(unsure_reward)  # small reward/penalty for unsure
    else:
        return 0.0  # incorrect


def _seeded_flip_is_factual(seed: int, row_id: str | int) -> bool:
    """Deterministically choose factual vs hallucinated for a given row.

    This is intentionally independent of `datasets.map` execution order.
    """
    payload = f"{seed}|{row_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return (digest[0] & 1) == 0


def load_environment(
    subset: str = "pqa_labeled",
    difficulty: str | Difficulty = Difficulty.ALL,
    use_knowledge: bool = False,
    flip_seed: int | None = 1618,
    unsure_reward: float = 0.01,
    **kwargs,
) -> vf.Environment:
    """
    Loads the MedHallu evaluation environment.

    Args:
        subset: 'pqa_labeled' (1k high quality) or 'pqa_artificial' (9k generated).
        difficulty: Filter questions by difficulty level.
        use_knowledge: If True, includes the 'Knowledge' field in the prompt.
        flip_seed: If set, makes the factual-vs-hallucinated flip deterministic per row.
    """
    dataset = load_dataset("UTAustin-AIHealth/MedHallu", subset, split="train")

    difficulty = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty
    if difficulty != Difficulty.ALL:
        dataset = dataset.filter(
            lambda ex: str(ex.get("Difficulty Level", "")).strip().lower() == difficulty.value,
            load_from_cache_file=False,
        )

    rand = random.Random() if flip_seed is None else None

    def _map_fn(ex, idx: int):
        """
        Transforms a dataset row into an environment prompt.
        Randomly flips between showing the Factual Answer (Target 0)
        or Hallucinated Answer (Target 1).
        """
        # randomly decide whether to show the factual or hallucinated answer
        row_id = ex.get("id", idx)
        is_factual = _seeded_flip_is_factual(flip_seed, row_id) if flip_seed is not None else rand.choice([True, False])

        if is_factual:
            # show Ground Truth -> expect '0'
            answer_text = ex["Ground Truth"]
            target_label = "0"
        else:
            # show Hallucinated Answer -> expect '1'
            answer_text = ex["Hallucinated Answer"]
            target_label = "1"

        if use_knowledge:
            prompt = create_prompt_with_knowledge(
                question=ex["Question"], option1=answer_text, knowledge=ex.get("Knowledge", "")
            )
        else:
            prompt = create_prompt_no_knowledge(question=ex["Question"], option1=answer_text)

        return {
            "question": prompt,
            "answer": target_label,
            "info": {
                "original_type": "factual" if is_factual else "hallucinated",
                "difficulty": ex.get("Difficulty Level", "unknown"),
                "hallucination_category": ex.get("Category of Hallucination", "N/A"),
            },
        }

    # apply mapping
    processed_dataset = dataset.map(
        _map_fn,
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
    )

    # Parser and Rubric
    parser = vf.Parser(extract_boxed_answer)

    def medhallu_reward(completion: str, answer: str, parser: vf.Parser, **kwargs) -> float:
        return reward_fn(completion, answer, parser, unsure_reward=unsure_reward, **kwargs)

    medhallu_reward.__name__ = "medhallu_reward"

    rubric = vf.Rubric(funcs=[medhallu_reward], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(eval_dataset=processed_dataset, rubric=rubric, parser=parser, system_prompt=system_prompt)
