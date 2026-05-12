"""Small shared constants for CLI modules.

This module exists to avoid circular imports between `main.py`, `_single_run.py`,
and process helpers that only need the command strings for messaging.
"""

from __future__ import annotations

from pathlib import Path

COMMAND = "medarc-eval"
BENCH_COMMAND = "bench"
PROCESS_COMMAND = "process"
WINRATE_COMMAND = "winrate"

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_API_KEY_VAR = "OPENAI_API_KEY"
DEFAULT_ENDPOINTS_PATH = Path("configs") / "endpoints.toml"
DEFAULT_ENV_DIR = Path("environments")
DEFAULT_ENV_CONFIG_ROOT = Path("configs") / "envs"
DEFAULT_EVALS_DIR = Path("runs") / "evals"
DEFAULT_PROCESSED_DIR = Path("runs") / "processed"
DEFAULT_WINRATE_DIR = DEFAULT_PROCESSED_DIR / "winrate"
