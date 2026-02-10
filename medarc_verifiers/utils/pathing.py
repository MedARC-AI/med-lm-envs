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


def normalize_results_dir_for_manifest(value: str | Path, *, run_dir: Path) -> str:
    """Normalize results_dir entries before storing them in a manifest."""
    candidate = Path(value)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == "runs":
            candidate = (project_root() / candidate).resolve()
        else:
            candidate = (run_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return to_project_relative(candidate)


def resolve_results_dir_from_manifest(value: str | None, *, job_id: str, run_dir: Path) -> Path:
    """Resolve manifest results_dir entries into concrete paths."""
    raw = "" if value is None else str(value)
    name = raw.strip() or job_id
    candidate = Path(name)
    if candidate.is_absolute():
        return candidate
    if candidate.parts and candidate.parts[0] == "runs":
        return from_project_relative(candidate)
    return (run_dir / candidate).resolve()


def _resolve_path(path: Path | str, *, default_base: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if default_base is not None:
        return (default_base / candidate).resolve()
    return candidate.resolve()


def resolve_under(base_dir: Path, rel_path: str | Path) -> Path | None:
    """Join rel_path under base_dir, rejecting obvious traversal."""
    raw = str(rel_path).strip()
    if not raw:
        return None
    raw = raw.replace("\\", "/")
    if len(raw) >= 3 and raw[1] == ":" and raw[2] == "/" and raw[0].isalpha():
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return None
    if ".." in candidate.parts:
        return None
    return base_dir / candidate


__all__ = [
    "project_root",
    "to_project_relative",
    "from_project_relative",
    "resolve_under",
    "normalize_results_dir_for_manifest",
    "resolve_results_dir_from_manifest",
]
