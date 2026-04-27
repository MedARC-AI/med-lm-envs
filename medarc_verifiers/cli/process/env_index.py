"""Helpers for reading env_index.json inventories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from medarc_verifiers.utils.pathing import resolve_under


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
    return resolve_under(base_dir, raw_path)


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
    if isinstance(payload, Mapping) and int(payload.get("version") or 0) == 2:
        return _inventory_from_v2(payload, processed_dir)
    return EnvIndexInventory(env_paths={}, version=0)


def read_env_index_files(processed_dir: Path) -> dict[str, Mapping[str, Any]]:
    """Return env_index file metadata map keyed by relative path."""
    index_path = processed_dir / "env_index.json"
    payload = load_env_index(index_path)
    if not isinstance(payload, Mapping) or int(payload.get("version") or 1) != 2:
        return {}
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return {}
    safe_files: dict[str, Mapping[str, Any]] = {}
    for path_str, entry in files.items():
        if not isinstance(entry, Mapping):
            continue
        resolved = resolve_under(processed_dir, str(path_str))
        if resolved is None:
            continue
        try:
            rel_key = resolved.relative_to(processed_dir).as_posix()
        except ValueError:
            continue
        safe_files[rel_key] = entry
    return safe_files


def read_env_index_models(processed_dir: Path) -> set[str]:
    """Return model ids listed in env_index.json (v2 only)."""
    payload = load_env_index(processed_dir / "env_index.json")
    if not isinstance(payload, Mapping) or int(payload.get("version") or 1) != 2:
        return set()
    files = payload.get("files")
    if not isinstance(files, Mapping):
        return set()
    models: set[str] = set()
    for path_str, entry in files.items():
        if resolve_under(processed_dir, str(path_str)) is None:
            continue
        if not isinstance(entry, Mapping):
            continue
        model_id = entry.get("model_id")
        if model_id:
            models.add(str(model_id))
    return models


__all__ = [
    "EnvIndexInventory",
    "read_env_index_inventory",
    "read_env_index_files",
    "read_env_index_models",
]
