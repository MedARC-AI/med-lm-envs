"""Internal mapping loaders for orchestrator-owned bundle files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf


class InternalFormatError(ValueError):
    """Raised when an internal orchestrator file is malformed."""


def load_internal_mapping(path: Path, *, label: str) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    if resolved.suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(f"Unsupported {label} format: {resolved} (expected .yaml/.yml/.json)")
    try:
        if resolved.suffix == ".json":
            data = json.loads(resolved.read_text(encoding="utf-8"))
        else:
            data = OmegaConf.to_container(OmegaConf.load(resolved), resolve=True)
    except Exception as exc:  # pragma: no cover - parser error types vary
        raise InternalFormatError(f"Failed to load {label}: {resolved}") from exc
    if not isinstance(data, Mapping):
        raise InternalFormatError(f"{label} must be a mapping at top level: {resolved}")
    return data


__all__ = ["InternalFormatError", "load_internal_mapping"]
