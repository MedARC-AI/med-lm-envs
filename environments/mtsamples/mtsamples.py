"""
MTSamples Medical Specialty Classification Environment

This environment evaluates medical specialty classification from clinical transcriptions.
Given a medical transcription, the model must identify the correct medical specialty
from a set of options.

Dataset:
- Source: MTSamples (https://mtsamples.com)
- HuggingFace: NickyNicky/medical_mtsamples
- Size: ~5,000 medical transcription samples
- License: CC0 (Public Domain)
- Specialties: 40 medical specialties

Task: 5-way Multiple Choice Classification
- Input: Medical transcription text
- Output: Predicted specialty (from 5 options: 1 correct + 4 distractors)

MIT License
"""

import random
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar

from medarc_verifiers.parsers.json_parser import JSONParser
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice

disable_progress_bar()

# HuggingFace dataset
DATASET_NAME = "NickyNicky/medical_mtsamples"

# All 40 medical specialties in the dataset
ALL_SPECIALTIES = [
    "Allergy / Immunology",
    "Autopsy",
    "Bariatrics",
    "Cardiovascular / Pulmonary",
    "Chiropractic",
    "Consult - History and Phy.",
    "Cosmetic / Plastic Surgery",
    "Dentistry",
    "Dermatology",
    "Diets and Nutritions",
    "Discharge Summary",
    "ENT - Otolaryngology",
    "Emergency Room Reports",
    "Endocrinology",
    "Gastroenterology",
    "General Medicine",
    "Hematology - Oncology",
    "Hospice - Palliative Care",
    "IME-QME-Work Comp etc.",
    "Lab Medicine - Pathology",
    "Letters",
    "Nephrology",
    "Neurology",
    "Neurosurgery",
    "Obstetrics / Gynecology",
    "Office Notes",
    "Ophthalmology",
    "Orthopedic",
    "Pain Management",
    "Pediatrics - Neonatal",
    "Physical Medicine - Rehab",
    "Podiatry",
    "Psychiatry / Psychology",
    "Radiology",
    "Rheumatology",
    "SOAP / Chart / Progress Notes",
    "Sleep Medicine",
    "Speech - Language",
    "Surgery",
    "Urology",
]


def _get_distractors(correct_specialty: str, num_distractors: int = 4, seed: int | None = None) -> list[str]:
    """Get random distractor specialties excluding the correct one."""
    rng = random.Random(seed)
    available = [s for s in ALL_SPECIALTIES if s != correct_specialty]
    return rng.sample(available, min(num_distractors, len(available)))


def _build_prompt(transcription: str, options: dict[str, str], use_description: bool = False) -> str:
    """Build the classification prompt."""
    opts_text = "\n".join(f"{k}. {v}" for k, v in options.items())

    prompt = (
        "You are a medical expert. Based on the following clinical transcription, "
        "identify the most appropriate medical specialty.\n\n"
        f"Transcription:\n{transcription}\n\n"
        "Options:\n"
        f"{opts_text}\n\n"
        'Provide your answer in JSON format: {"answer": "X"} where X is the letter (A, B, C, D, or E).'
    )
    return prompt


def accuracy(
    completion: Any,
    answer: str,
    parser: vf.Parser,
    info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> float:
    """Reward based on multiple-choice accuracy."""
    parsed = parser.parse_answer(completion) or ""
    answer_text = info.get("answer_text") if info else None
    is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
    return 1.0 if is_correct else 0.0


def load_environment(
    split: str = "test",
    num_options: int = 5,
    shuffle_answers: bool = True,
    shuffle_seed: int | None = 1618,
    use_description: bool = False,
    max_transcription_length: int = 4000,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.SingleTurnEnv:
    """
    Load the MTSamples medical specialty classification environment.

    This environment tests whether models can correctly classify medical transcriptions
    into their appropriate specialty from a set of multiple choice options.

    Args:
        split: Dataset split. Since MTSamples has no official split, we create one:
               "train" = first 80%, "test" = last 20%
        num_options: Number of MCQ options (1 correct + N-1 distractors). Default: 5
        shuffle_answers: Whether to shuffle answer options. Default: True
        shuffle_seed: Random seed for shuffling. Default: 1618
        use_description: Use description instead of full transcription. Default: False
        max_transcription_length: Maximum length of transcription text. Default: 4000
        system_prompt: Optional system prompt override.
        **kwargs: Additional arguments forwarded to `vf.SingleTurnEnv`.

    Returns:
        A configured SingleTurnEnv for medical specialty classification.

    Example:
        >>> env = load_environment(split="test", num_options=5)
        >>> # Run with: medarc-eval mtsamples -m gpt-4.1-mini -n 10 -s
    """
    # Load the dataset
    ds = load_dataset(DATASET_NAME, split="train")

    # Create train/test split (80/20)
    total_size = len(ds)
    train_size = int(total_size * 0.8)

    if split == "train":
        ds = ds.select(range(train_size))
    elif split == "test":
        ds = ds.select(range(train_size, total_size))
    else:
        raise ValueError(f"Unknown split: {split}. Use 'train' or 'test'.")

    def _map_example(example: dict[str, Any], idx: int) -> dict[str, Any]:
        """Map a single example to the evaluation format."""
        # Get the correct specialty (strip whitespace)
        raw_specialty = example.get("medical_specialty")
        if not raw_specialty:
            return None
        correct_specialty = raw_specialty.strip()

        if not correct_specialty or correct_specialty not in ALL_SPECIALTIES:
            return None

        # Get text (description or transcription)
        if use_description:
            raw_text = example.get("description")
        else:
            raw_text = example.get("transcription")

        if not raw_text:
            return None
        text = raw_text.strip()

        if not text:
            return None

        # Truncate if too long
        if len(text) > max_transcription_length:
            text = text[:max_transcription_length] + "..."

        # Generate distractors
        distractor_seed = (shuffle_seed or 0) + idx if shuffle_seed is not None else None
        distractors = _get_distractors(correct_specialty, num_options - 1, seed=distractor_seed)

        # Create options dict
        all_options = [correct_specialty] + distractors
        labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"][:num_options]

        # Initially, correct answer is at index 0 (label A)
        correct_label = "A"

        # Shuffle if requested
        if shuffle_answers:
            shuffled_options, correct_label, _ = randomize_multiple_choice(
                options=all_options,
                answer_choice=0,  # Index of correct answer
                labels=labels,
                seed=shuffle_seed,
                row_id=idx,
            )
            all_options = shuffled_options

        # Build options dict
        options = {labels[i]: all_options[i] for i in range(len(all_options))}

        # Build prompt
        question = _build_prompt(text, options, use_description)

        return {
            "question": question,
            "answer": correct_label,
            "info": {
                "answer_text": correct_specialty,
                "specialty": correct_specialty,
                "options": options,
            },
        }

    # Map the dataset
    columns_to_remove = ds.column_names
    load_from_cache_file = not shuffle_answers

    mapped_ds = ds.map(
        _map_example,
        with_indices=True,
        remove_columns=columns_to_remove,
        load_from_cache_file=load_from_cache_file,
    )

    # Filter out None values (invalid examples)
    mapped_ds = mapped_ds.filter(lambda x: x.get("question") is not None)

    # Create parser and rubric
    parser = JSONParser(fields=["answer"], answer_field="answer")
    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    # Default system prompt
    if system_prompt is None:
        system_prompt = (
            "You are a medical expert specializing in clinical documentation. "
            "Analyze medical transcriptions and identify the appropriate specialty."
        )

    return vf.SingleTurnEnv(
        dataset=mapped_ds if split == "train" else None,
        eval_dataset=mapped_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        name="mtsamples",
        **kwargs,
    )
