"""Shared runtime environment loading for orchestrator phases."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values

from medarc_verifiers.orchestrate.runtime import RuntimeLaunchError


def load_explicit_runtime_env(
    *,
    env_file: Path | None = None,
    container_env_file: Path | None = None,
    container_env_base_dir: Path | None = None,
    env_overrides: Mapping[str, str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    root = repo_root or Path(__file__).resolve().parents[2]
    if env_file is not None:
        env.update(load_env_file(env_file, base_dir=root))
    else:
        default_env = root / ".env"
        if default_env.exists():
            env.update(load_env_file(default_env, base_dir=root))
    if container_env_file is not None:
        env.update(load_env_file(container_env_file, base_dir=container_env_base_dir or root))
    if env_overrides is not None:
        env.update(dict(env_overrides))
    if not env.get("HF_TOKEN"):
        token = load_hf_token_from_login()
        if token:
            env["HF_TOKEN"] = token
    return env


def apply_env(values: Mapping[str, str]) -> None:
    import os

    for key, value in values.items():
        os.environ[str(key)] = str(value)


def load_hf_token_from_login() -> str | None:
    try:
        module = importlib.import_module("huggingface_hub")
    except ImportError:
        return None
    get_token = getattr(module, "get_token", None)
    if not callable(get_token):
        return None
    try:
        token = get_token()
    except Exception:
        return None
    if token is None:
        return None
    text = str(token).strip()
    return text or None


def load_env_file(path: object, *, base_dir: Path) -> dict[str, str]:
    env_path = Path(str(path)).expanduser()
    if not env_path.is_absolute():
        env_path = (base_dir / env_path).resolve()
    if not env_path.exists():
        raise RuntimeLaunchError(f"env_file not found: {env_path}")
    values = dotenv_values(env_path)
    return {key: value for key, value in values.items() if value is not None}


__all__ = ["apply_env", "load_env_file", "load_explicit_runtime_env", "load_hf_token_from_login"]
