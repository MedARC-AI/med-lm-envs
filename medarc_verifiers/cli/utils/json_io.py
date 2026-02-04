"""Shared JSON serialization helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def dumps_json(payload: Any, *, newline: bool = False) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if newline and not text.endswith("\n"):
        text += "\n"
    return text


def write_json(path: Path, payload: Mapping[str, Any], newline: bool = False) -> bool:
    text = dumps_json(payload)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == text:
                return False
        except Exception:
            pass
    tmp_path = path.with_name(f".{path.name}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return True


__all__ = ["dumps_json", "write_json"]
