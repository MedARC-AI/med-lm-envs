"""Benchmark command rendering and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import shlex
import subprocess
import time


@dataclass(frozen=True)
class BenchResult:
    exit_code: int
    duration_s: float


def render_command(template: str, context: Mapping[str, str]) -> list[str]:
    rendered = template.format(**context)
    return shlex.split(rendered)


def run_benchmark(
    command: Sequence[str] | str,
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
) -> BenchResult:
    if isinstance(command, str):
        command = shlex.split(command)
    start = time.monotonic()
    with open(stdout_path, "w", encoding="utf-8") as stdout, open(stderr_path, "w", encoding="utf-8") as stderr:
        result = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=dict(env) if env else None,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    duration = time.monotonic() - start
    return BenchResult(exit_code=result.returncode, duration_s=duration)


__all__ = ["BenchResult", "render_command", "run_benchmark"]
