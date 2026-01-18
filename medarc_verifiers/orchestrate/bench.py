"""Benchmark command rendering and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Mapping, Sequence
import asyncio
import os
import shlex
import signal
import subprocess
import time


@dataclass(frozen=True)
class BenchResult:
    exit_code: int
    duration_s: float
    terminated: bool = False


@dataclass
class BenchProcess:
    command: list[str]
    process: asyncio.subprocess.Process
    start_time: float
    stdout_handle: IO[str]
    stderr_handle: IO[str]
    terminated: bool = False


def render_command(template: str, context: Mapping[str, str]) -> list[str]:
    rendered = template.format(**context)
    return shlex.split(rendered)


async def start_benchmark(
    command: Sequence[str] | str,
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    stdout_path: Path,
    stderr_path: Path,
) -> BenchProcess:
    if isinstance(command, str):
        command = shlex.split(command)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = open(stdout_path, "w", encoding="utf-8")
    stderr_handle = open(stderr_path, "w", encoding="utf-8")
    try:
        kwargs: dict[str, object] = {}
        if os.name == "posix":
            kwargs["start_new_session"] = True
        elif os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        process = await asyncio.create_subprocess_exec(
            *list(command),
            cwd=str(cwd),
            env=dict(env) if env else None,
            stdout=stdout_handle,
            stderr=stderr_handle,
            **kwargs,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    return BenchProcess(
        command=list(command),
        process=process,
        start_time=time.monotonic(),
        stdout_handle=stdout_handle,
        stderr_handle=stderr_handle,
    )


async def wait_benchmark(proc: BenchProcess) -> BenchResult:
    try:
        await proc.process.wait()
    finally:
        proc.stdout_handle.close()
        proc.stderr_handle.close()
    duration = time.monotonic() - proc.start_time
    exit_code = proc.process.returncode if proc.process.returncode is not None else 0
    return BenchResult(exit_code=exit_code, duration_s=duration, terminated=proc.terminated)


async def terminate_benchmark(proc: BenchProcess, *, term_timeout_s: float = 5.0) -> None:
    if proc.process.returncode is not None:
        return
    proc.terminated = True
    pid = proc.process.pid
    if pid is None:
        return
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            proc.process.terminate()
    else:
        try:
            proc.process.terminate()
        except ProcessLookupError:
            return
        except Exception:
            pass
    try:
        await asyncio.wait_for(proc.process.wait(), timeout=term_timeout_s)
        return
    except asyncio.TimeoutError:
        pass
    if os.name == "posix":
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            proc.process.kill()
    else:
        try:
            proc.process.kill()
        except ProcessLookupError:
            return
        except Exception:
            pass
    try:
        await proc.process.wait()
    except Exception:
        return


__all__ = [
    "BenchProcess",
    "BenchResult",
    "render_command",
    "start_benchmark",
    "terminate_benchmark",
    "wait_benchmark",
]
