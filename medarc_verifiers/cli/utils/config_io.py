"""Shared config file loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml


def load_mapping_file(path: Path, *, label: str) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported {label} format: {path} (expected .yaml/.yml/.json)")
    raw = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(raw)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a mapping at top level: {path}")
    return dict(payload)


__all__ = ["load_mapping_file"]
