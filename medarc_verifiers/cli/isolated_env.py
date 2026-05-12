"""Temporary virtual environment helpers for isolated TOML bench evals."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse


class IsolatedEnvError(RuntimeError):
    """Raised when an isolated bench environment cannot be prepared."""


@dataclass(frozen=True)
class MedarcInstallSpec:
    editable: bool
    version: str
    checkout_root: Path | None = None


def venv_python_path(venv_path: Path) -> Path:
    posix_path = venv_path / "bin" / "python"
    if posix_path.exists():
        return posix_path
    return venv_path / "Scripts" / "python.exe"


def current_medarc_install_spec() -> MedarcInstallSpec:
    try:
        dist = metadata.distribution("medarc-verifiers")
    except metadata.PackageNotFoundError as exc:
        raise IsolatedEnvError("Cannot auto-install isolated envs because medarc-verifiers is not installed.") from exc

    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError as exc:
            raise IsolatedEnvError("Installed medarc-verifiers has malformed direct_url.json metadata.") from exc
        if direct_url.get("dir_info", {}).get("editable"):
            url = direct_url.get("url")
            parsed = urlparse(url) if isinstance(url, str) else None
            if parsed is None or parsed.scheme != "file":
                raise IsolatedEnvError("Editable medarc-verifiers install does not point at a local file:// checkout.")
            checkout_root = Path(unquote(parsed.path)).expanduser().resolve()
            _validate_editable_checkout(checkout_root)
            return MedarcInstallSpec(editable=True, version=dist.version, checkout_root=checkout_root)

    return MedarcInstallSpec(editable=False, version=dist.version)


@contextmanager
def temporary_bench_venv(repo_root: Path | None = None) -> Iterator[Path]:
    temp_root = Path(tempfile.mkdtemp(prefix="medarc-bench-venv-"))
    try:
        python_executable = _create_venv(temp_root)
        install_medarc_into_venv(python_executable, repo_root=repo_root)
        yield python_executable
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def install_medarc_into_venv(python_executable: Path, *, repo_root: Path | None = None) -> None:
    spec = current_medarc_install_spec()
    if spec.editable:
        checkout_root = repo_root or spec.checkout_root
        if checkout_root is None:
            raise IsolatedEnvError("Editable medarc-verifiers checkout path could not be resolved.")
        _validate_editable_checkout(checkout_root)
        command = ["uv", "pip", "install", "--python", str(python_executable), "-e", str(checkout_root)]
        _run_uv(command, "install editable medarc-verifiers into isolated venv")
        return

    requirement = f"medarc-verifiers=={spec.version}"
    command = ["uv", "pip", "install", "--python", str(python_executable), requirement]
    try:
        _run_uv(command, f"install {requirement} into isolated venv")
    except IsolatedEnvError as exc:
        raise IsolatedEnvError(
            f"Could not resolve {requirement} for isolated auto-install. Run from an editable checkout, "
            "or preinstall environment packages and pass --no-auto-install."
        ) from exc


def install_env_package(python_executable: Path, env_path: Path) -> None:
    _run_uv(
        ["uv", "pip", "install", "--python", str(python_executable), "-e", str(env_path)],
        f"install environment package {env_path} into isolated venv",
    )


def _create_venv(venv_path: Path) -> Path:
    _run_uv(["uv", "venv", "--python", sys.executable, str(venv_path)], "create isolated bench venv")
    python_executable = venv_python_path(venv_path)
    if not python_executable.exists():
        raise IsolatedEnvError(f"uv created {venv_path}, but no Python executable was found in it.")
    return python_executable


def _validate_editable_checkout(checkout_root: Path) -> None:
    if not (checkout_root / "pyproject.toml").is_file() or not (checkout_root / "medarc_verifiers").is_dir():
        raise IsolatedEnvError(
            f"Editable medarc-verifiers checkout at {checkout_root} is invalid; expected pyproject.toml "
            "and medarc_verifiers/."
        )


def _run_uv(command: list[str], action: str) -> None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise IsolatedEnvError(f"Cannot {action}: uv is not installed or not on PATH.") from exc
    if completed.returncode != 0:
        stderr_tail = _tail(completed.stderr)
        stdout_tail = _tail(completed.stdout)
        detail = "\n".join(part for part in (stderr_tail, stdout_tail) if part)
        raise IsolatedEnvError(f"Failed to {action} with exit code {completed.returncode}.\n{detail}".rstrip())


def _tail(text: str, *, lines: int = 20) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return "\n".join(stripped.splitlines()[-lines:])


__all__ = [
    "IsolatedEnvError",
    "MedarcInstallSpec",
    "current_medarc_install_spec",
    "install_env_package",
    "install_medarc_into_venv",
    "temporary_bench_venv",
    "venv_python_path",
]
