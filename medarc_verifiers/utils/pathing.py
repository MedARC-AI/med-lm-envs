"""Shared filesystem helpers for locating and relativizing project paths."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Best-effort detection of the repository root (directory containing pyproject.toml)."""
    current = Path(__file__).resolve()
    for candidate in (current,) + tuple(current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    # Fallback to current working directory if no project marker is found.
    return Path.cwd().resolve()


def to_project_relative(path: Path | str, *, default_base: Path | None = None) -> str:
    """Convert an absolute path to a string relative to the project root when possible.

    If `path` is relative, treat it as rooted at `default_base` when provided.
    """
    resolved = _resolve_path(path, default_base=default_base)
    root = project_root()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def from_project_relative(path: Path | str) -> Path:
    """Convert a stored manifest path back into an absolute path under the project root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (project_root() / candidate).resolve()


def _resolve_path(path: Path | str, *, default_base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if default_base is not None:
        return (default_base / candidate).resolve()
    return candidate.resolve()


__all__ = ["project_root", "to_project_relative", "from_project_relative"]
