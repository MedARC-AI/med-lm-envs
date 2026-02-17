"""Shared helpers for verifiers resume path handling and diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from verifiers.utils.path_utils import find_latest_incomplete_eval_results_path, is_valid_eval_results_path

RESUME_MISMATCH_FIELDS: tuple[str, ...] = (
    "env_id",
    "model",
    "rollouts_per_example",
    "num_examples",
)


def is_valid_resume_results_path(path: str | Path) -> bool:
    """Return True when path points to a valid verifiers eval results directory."""
    return is_valid_eval_results_path(Path(path).expanduser())


def resolve_resume_path(
    *,
    resume_arg: str | bool | None,
    env_id: str,
    model: str,
    num_examples: int,
    rollouts_per_example: int,
    env_dir_path: str | Path,
) -> Path | None:
    """Resolve explicit/auto resume arguments into a concrete results path."""
    if resume_arg in (None, False):
        return None

    if isinstance(resume_arg, str):
        candidate = Path(resume_arg).expanduser()
        if not is_valid_eval_results_path(candidate):
            raise ValueError(
                f"Resume path {candidate} is not a valid evaluation results path "
                "(expected a directory containing results.jsonl and metadata.json)."
            )
        return candidate

    if resume_arg is True:
        return find_latest_incomplete_eval_results_path(
            env_id=env_id,
            model=model,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            env_dir_path=str(Path(env_dir_path).expanduser()),
        )

    raise ValueError(f"Invalid value for --resume: {resume_arg!r}")


def is_resume_metadata_mismatch_error(error: BaseException) -> bool:
    """Return True when the exception chain contains a verifiers resume mismatch."""
    for exc in _iter_exception_chain(error):
        message = str(exc)
        if "Cannot resume from" in message and "metadata mismatch" in message:
            return True
    return False


def load_resume_metadata_values(
    resume_path: Path | None,
    *,
    fields: tuple[str, ...] = RESUME_MISMATCH_FIELDS,
) -> dict[str, Any]:
    """Load selected metadata values from a resume path for diagnostics."""
    values: dict[str, Any] = {field: "<missing>" for field in fields}
    if resume_path is None:
        return values

    metadata_path = resume_path / "metadata.json"
    try:
        raw_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        marker = f"<missing metadata: {metadata_path}>"
        return {field: marker for field in fields}
    except (OSError, json.JSONDecodeError) as exc:
        marker = f"<unreadable metadata: {exc}>"
        return {field: marker for field in fields}

    if not isinstance(raw_payload, Mapping):
        marker = f"<invalid metadata payload: {type(raw_payload).__name__}>"
        return {field: marker for field in fields}

    for field in fields:
        values[field] = raw_payload.get(field, "<missing>")
    return values


def format_resume_mismatch_lines(
    *,
    saved_values: Mapping[str, Any],
    current_values: Mapping[str, Any],
    fields: tuple[str, ...] = RESUME_MISMATCH_FIELDS,
) -> list[str]:
    """Format saved/current metadata value comparisons for logging."""
    lines: list[str] = []
    for field in fields:
        saved = saved_values.get(field, "<missing>")
        current = current_values.get(field, "<missing>")
        detail = ""
        if field == "num_examples" and isinstance(saved, int) and isinstance(current, int) and current < saved:
            detail = " (current must be >= saved)"
        lines.append(f"{field}: saved={saved!r}, current={current!r}{detail}")
    return lines


def _iter_exception_chain(error: BaseException) -> list[BaseException]:
    chain: list[BaseException] = []
    queue: list[BaseException] = [error]
    seen: set[int] = set()

    while queue:
        current = queue.pop(0)
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        chain.append(current)
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            queue.append(cause)
        if isinstance(context, BaseException):
            queue.append(context)

    return chain


__all__ = [
    "RESUME_MISMATCH_FIELDS",
    "format_resume_mismatch_lines",
    "is_resume_metadata_mismatch_error",
    "is_valid_resume_results_path",
    "load_resume_metadata_values",
    "resolve_resume_path",
]
