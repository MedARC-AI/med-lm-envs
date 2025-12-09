"""
Patches Environment.format_dataset to always pass load_from_cache_file=False to HF dataset.map calls.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def patch_format_dataset_disable_cache() -> bool:
    """
    Patch verifiers Environment.format_dataset to disable HF map caching.

    Returns:
        bool: True if the patch was applied, False if already patched or unavailable.
    """
    try:
        from verifiers.envs.environment import Environment
    except Exception as exc:  # pragma: no cover - defensive, exercised in runtime
        logger.warning("Could not import verifiers Environment: %s", exc)
        return False

    if getattr(Environment, "_medarc_format_dataset_patched", False):
        return False

    original_format_dataset = Environment.format_dataset

    def patched_format_dataset(
        self,
        dataset: Any,
        system_prompt: str | None = None,
        few_shot: list[dict] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
    ):
        # if "id" column is present and not int, rename it to "src_id"
        if "example_id" in dataset.column_names and not isinstance(dataset["example_id"][0], int):
            dataset = dataset.rename_column("example_id", "src_id")
        if "example_id" not in dataset.column_names:
            dataset = dataset.add_column("example_id", range(len(dataset)))  # type: ignore[arg-type]

        def format_prompt_fn(prompt_str: str) -> list[dict]:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if few_shot:
                messages.extend(few_shot)
            messages.append({"role": "user", "content": prompt_str})
            return messages

        if "prompt" not in dataset.column_names:
            if answer_key == "answer":
                dataset = dataset.map(
                    lambda x: {
                        "prompt": format_prompt_fn(x[question_key]),
                    },
                    load_from_cache_file=False,
                )
            else:
                dataset = dataset.map(
                    lambda x: {
                        "prompt": format_prompt_fn(x[question_key]),
                        "answer": x[answer_key],
                    },
                    load_from_cache_file=False,
                )
        assert "example_id" in dataset.column_names
        assert "prompt" in dataset.column_names
        return dataset

    Environment._original_format_dataset_medarc = original_format_dataset
    Environment.format_dataset = patched_format_dataset
    Environment._medarc_format_dataset_patched = True
    logger.debug("Patched verifiers Environment.format_dataset (load_from_cache_file=False)")
    return True
