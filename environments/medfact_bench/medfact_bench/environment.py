"""MedFact-Bench zero-shot classification environment."""

from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

DATASET_ID = "ncbi/MedFact-Bench"
DATASET_REVISION = "249028caf7ad5a3e63331269a606f4b2696693ed"

LABELS = ("SUPPORT", "NEI", "CONTRADICT")
INVALID_LABEL = "INVALID"
VALID_SCORES = frozenset({-2, -1, 0, 1, 2})

EXPECTED_DATASET_COUNTS = {
    "scifact": 340,
    "healthver": 903,
    "medaesqa": 9_106,
    "pubmedqa-fact": 500,
    "bioasq-fact": 3_425,
}

REQUIRED_COLUMNS = (
    "dataset",
    "claim",
    "source",
    "label",
    "system_prompt",
    "user_prompt",
)

_SCORE_PATTERN = re.compile(r"<score>(.*?)</score>", re.DOTALL)
_STRICT_FORMAT_PATTERN = re.compile(
    r"\A\s*<think>.*?</think>\s*<score>(.*?)</score>\s*\Z",
    re.DOTALL,
)


class MedFactSubset(str, Enum):
    """Supported MedFact-Bench component datasets."""

    ALL = "all"
    SCIFACT = "scifact"
    HEALTHVER = "healthver"
    MEDAESQA = "medaesqa"
    PUBMEDQA_FACT = "pubmedqa-fact"
    BIOASQ_FACT = "bioasq-fact"


def _parse_score_value(value: str) -> int | None:
    try:
        score = int(value.strip())
    except (TypeError, ValueError):
        return None
    return score if score in VALID_SCORES else None


class MedFactScoreParser(vf.Parser):
    """Parse the first paper-format integer score from a completion."""

    def __init__(self) -> None:
        super().__init__(extract_fn=self.parse_score)

    @staticmethod
    def parse_score(text: str) -> int | None:
        """Return the first valid score enclosed in a score tag."""
        match = _SCORE_PATTERN.search(text)
        if match is None:
            return None
        return _parse_score_value(match.group(1))

    def completion_text(self, completion: Any) -> str | None:
        """Return the final assistant text using Verifiers message semantics."""
        if isinstance(completion, str):
            return completion
        if not isinstance(completion, list):
            return None
        assistant_messages = self.get_assistant_messages(completion)
        if not assistant_messages:
            return None
        content = self._message_field(assistant_messages[-1], "content", "") or ""
        return self._content_to_text(content)

    def parse_completion(self, completion: Any) -> int | None:
        """Parse a string or chat-message completion."""
        text = self.completion_text(completion)
        return self.parse(text) if text is not None else None

    def is_strict_format(self, completion: Any) -> bool:
        """Check exact two-block Med-V1 formatting independently of parsing."""
        text = self.completion_text(completion)
        if text is None:
            return False
        if any(text.count(tag) != 1 for tag in ("<think>", "</think>", "<score>", "</score>")):
            return False
        match = _STRICT_FORMAT_PATTERN.fullmatch(text)
        return match is not None and _parse_score_value(match.group(1)) is not None


def score_to_label(score: int | None) -> str:
    """Map the five-point Med-V1 score to the three benchmark labels."""
    if score in (1, 2):
        return "SUPPORT"
    if score == 0:
        return "NEI"
    if score in (-1, -2):
        return "CONTRADICT"
    return INVALID_LABEL


def prediction_label(completion: Any, parser: MedFactScoreParser) -> str:
    """Return the benchmark label predicted by a completion."""
    return score_to_label(parser.parse_completion(completion))


def accuracy(
    completion: Any,
    answer: str,
    parser: MedFactScoreParser,
    **_: Any,
) -> float:
    """Score exact three-way classification accuracy."""
    return float(prediction_label(completion, parser) == answer)


def parseable_score(
    completion: Any,
    parser: MedFactScoreParser,
    **_: Any,
) -> float:
    """Measure whether a valid score can be parsed."""
    return float(parser.parse_completion(completion) is not None)


def strict_format(
    completion: Any,
    parser: MedFactScoreParser,
    **_: Any,
) -> float:
    """Measure exact compliance with the requested think and score blocks."""
    return float(parser.is_strict_format(completion))


def _resolve_subset(subset: str | MedFactSubset) -> MedFactSubset:
    if isinstance(subset, MedFactSubset):
        return subset
    try:
        return MedFactSubset(subset)
    except ValueError as exc:
        choices = ", ".join(member.value for member in MedFactSubset)
        raise ValueError(f"Unsupported MedFact-Bench subset {subset!r}. Choose one of: {choices}.") from exc


def _load_source_dataset(dataset_path: str | None, cache_dir: str | None) -> Dataset:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"MedFact-Bench dataset path does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"MedFact-Bench dataset path must be a Parquet file: {path}")
        try:
            dataset = load_dataset(
                "parquet",
                data_files=str(path),
                split="train",
                cache_dir=cache_dir,
            )
        except Exception as exc:
            raise ValueError(f"Failed to load the local MedFact-Bench Parquet file at {path}: {exc}") from exc
        return dataset

    try:
        dataset = load_dataset(
            DATASET_ID,
            split="train",
            revision=DATASET_REVISION,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load {DATASET_ID} at revision {DATASET_REVISION}: {exc}") from exc
    return dataset


def _validate_dataset(dataset: Dataset) -> str:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(dataset.column_names))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"MedFact-Bench dataset is missing required columns: {missing}.")
    if len(dataset) == 0:
        raise ValueError("MedFact-Bench dataset is empty.")

    for column in REQUIRED_COLUMNS:
        values = dataset[column]
        null_count = sum(value is None for value in values)
        if null_count:
            raise ValueError(f"MedFact-Bench column {column!r} contains {null_count} null value(s).")
        non_string_count = sum(not isinstance(value, str) for value in values)
        if non_string_count:
            raise ValueError(f"MedFact-Bench column {column!r} contains {non_string_count} non-string value(s).")
        empty_count = sum(not value.strip() for value in values)
        if empty_count:
            raise ValueError(f"MedFact-Bench column {column!r} contains {empty_count} empty value(s).")

    dataset_values = set(dataset["dataset"])
    unexpected_datasets = sorted(dataset_values - set(EXPECTED_DATASET_COUNTS))
    if unexpected_datasets:
        values = ", ".join(unexpected_datasets)
        raise ValueError(f"MedFact-Bench dataset contains unsupported component values: {values}.")

    label_values = set(dataset["label"])
    unexpected_labels = sorted(label_values - set(LABELS))
    if unexpected_labels:
        values = ", ".join(unexpected_labels)
        raise ValueError(f"MedFact-Bench dataset contains unsupported labels: {values}.")

    system_prompts = set(dataset["system_prompt"])
    if len(system_prompts) != 1:
        raise ValueError(
            f"MedFact-Bench dataset must contain exactly one distinct system prompt; found {len(system_prompts)}."
        )
    return next(iter(system_prompts))


def _prepare_eval_dataset(dataset: Dataset, subset: MedFactSubset) -> Dataset:
    selected = dataset
    if subset is not MedFactSubset.ALL:
        selected = dataset.filter(
            lambda row: row["dataset"] == subset.value,
            load_from_cache_file=False,
        )
        if len(selected) == 0:
            raise ValueError(f"MedFact-Bench subset {subset.value!r} contains no rows.")

    return selected.map(
        lambda row: {
            "question": row["user_prompt"],
            "answer": row["label"],
            "info": {"dataset": row["dataset"]},
        },
        remove_columns=selected.column_names,
        load_from_cache_file=False,
    )


def load_environment(
    subset: str | MedFactSubset = MedFactSubset.ALL,
    dataset_path: str | None = None,
    cache_dir: str | None = None,
) -> vf.Environment:
    """Load the evaluation-only MedFact-Bench environment.

    Args:
        subset: Component dataset to evaluate, or ``all`` for the full benchmark.
        dataset_path: Optional local Parquet path. When provided, it takes precedence over Hugging Face.
        cache_dir: Optional Hugging Face datasets cache directory.

    Returns:
        A single-turn environment with only an evaluation dataset.
    """
    resolved_subset = _resolve_subset(subset)
    source_dataset = _load_source_dataset(dataset_path, cache_dir)
    system_prompt = _validate_dataset(source_dataset)
    eval_dataset = _prepare_eval_dataset(source_dataset, resolved_subset)

    parser = MedFactScoreParser()
    rubric = vf.Rubric(
        funcs=[accuracy, parseable_score, strict_format],
        weights=[1.0, 0.0, 0.0],
        parser=parser,
    )
    return vf.SingleTurnEnv(
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
