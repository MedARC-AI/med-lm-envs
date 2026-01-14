"""Helpers for reading env_index.json inventories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class EnvIndexInventory:
    """Resolved inventory of processed datasets."""

    env_paths: dict[str, list[Path]]
    version: int


def load_env_index(path: Path) -> Mapping[str, Any]:
    """Load env_index.json payload."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _resolve_path(base_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _inventory_from_v2(payload: Mapping[str, Any], base_dir: Path) -> EnvIndexInventory:
    files = payload.get("files") if isinstance(payload.get("files"), Mapping) else {}
    env_paths: dict[str, list[Path]] = {}
    for path_str, entry in (files or {}).items():
        if not isinstance(entry, Mapping):
            continue
        env_id = entry.get("env_id") or entry.get("base_env_id")
        if not env_id:
            continue
        resolved = _resolve_path(base_dir, str(path_str))
        if not resolved:
            continue
        env_paths.setdefault(str(env_id), []).append(resolved)
    return EnvIndexInventory(env_paths=env_paths, version=2)


def read_env_index_inventory(processed_dir: Path) -> EnvIndexInventory:
    """Read env_index.json and return a dataset inventory."""
    index_path = processed_dir / "env_index.json"
    payload = load_env_index(index_path)
    version = payload.get("version") if isinstance(payload, Mapping) else None
    if version == 2:
        return _inventory_from_v2(payload, processed_dir)
    return EnvIndexInventory(env_paths={}, version=int(version or 1))


def read_env_index_runs(processed_dir: Path) -> tuple[int, dict[str, Mapping[str, Any]]]:
    """Return env_index version and run metadata map."""
    index_path = processed_dir / "env_index.json"
    payload = load_env_index(index_path)
    version = int(payload.get("version") or 1) if isinstance(payload, Mapping) else 1
    runs = payload.get("runs") if isinstance(payload, Mapping) else None
    if version != 2 or not isinstance(runs, Mapping):
        return version, {}
    return version, {str(k): v for k, v in runs.items() if isinstance(v, Mapping)}


def read_env_index_files(processed_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Return env_index file metadata map keyed by relative path."""
    index_path = processed_dir / "env_index.json"
    payload = load_env_index(index_path)
    if not isinstance(payload, Mapping) or int(payload.get("version") or 1) != 2:
        return {}
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return {}
    return {str(k): v for k, v in files.items() if isinstance(v, Mapping)}


def read_env_index_models(processed_dir: Path) -> set[str]:
    """Return model ids listed in env_index.json (v2 only)."""
    payload = load_env_index(processed_dir / "env_index.json")
    if not isinstance(payload, Mapping) or int(payload.get("version") or 1) != 2:
        return set()
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return set()
    models: set[str] = set()
    for entry in files.values():
        if not isinstance(entry, Mapping):
            continue
        model_id = entry.get("model_id")
        if model_id:
            models.add(str(model_id))
    return models


__all__ = [
    "EnvIndexInventory",
    "read_env_index_inventory",
    "read_env_index_runs",
    "read_env_index_files",
    "read_env_index_models",
]
