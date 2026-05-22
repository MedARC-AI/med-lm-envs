"""Shared naming helpers for orchestrator task artifacts and bench runs."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


_TASK_DIR_ALLOWED = re.compile(r"[^a-zA-Z0-9_.-]+")


def sanitize_task_dirname(task_id: str, *, max_len: int = 120) -> str:
    cleaned = _TASK_DIR_ALLOWED.sub("-", task_id).strip("-.")
    if not cleaned:
        cleaned = "task"
    if cleaned != task_id:
        suffix = hashlib.sha1(task_id.encode("utf-8")).hexdigest()[:8]  # noqa: S324
        cleaned = f"{cleaned}-{suffix}"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-.")
    return cleaned or "task"


def task_root_for_id(output_root: Path, task_id: str) -> Path:
    task_dir = output_root / "tasks" / sanitize_task_dirname(task_id)
    if task_dir.exists():
        return task_dir
    raw = output_root / task_id
    if raw.exists():
        return raw
    sanitized = output_root / sanitize_task_dirname(task_id)
    if sanitized.exists():
        return sanitized
    return task_dir


__all__ = ["sanitize_task_dirname", "task_root_for_id"]
