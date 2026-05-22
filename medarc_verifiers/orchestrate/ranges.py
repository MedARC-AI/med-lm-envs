"""Small range parsing helpers for worker GPU environment values."""

from __future__ import annotations


def parse_index_range(expr: str, *, max_index: int | None = None) -> list[int]:
    """Parse a range expression like ``0-3,5`` into sorted indices."""
    if not expr:
        return []
    indices: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if max_index is not None:
        indices = {idx for idx in indices if 0 <= idx <= max_index}
    return sorted(indices)


__all__ = ["parse_index_range"]
