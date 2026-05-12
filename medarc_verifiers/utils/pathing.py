"""Shared filesystem helpers for safe relative paths."""

from __future__ import annotations

from pathlib import Path


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
    "resolve_under",
]
