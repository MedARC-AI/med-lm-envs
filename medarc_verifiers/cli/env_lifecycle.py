"""Local environment package lifecycle helpers for TOML bench subprocesses."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifiers.utils.import_utils import load_toml


@dataclass(frozen=True)
class EnvPackageRef:
    env_id: str
    module_name: str
    project_name: str
    env_path: Path
    loader: str | None = None


@dataclass(frozen=True)
class EnvInstallState:
    ref: EnvPackageRef
    installed_by_child: bool
    distribution_preexisting: bool
    module_preexisting: bool


def upstream_module_name(env_id: str) -> str:
    return env_id.replace("-", "_").split("/")[-1]


def resolve_env_package(env_id: str, env_dir: str | Path) -> EnvPackageRef:
    module_name = upstream_module_name(env_id)
    env_root = Path(env_dir).expanduser() / module_name
    pyproject_path = env_root / "pyproject.toml"
    if not env_root.exists():
        raise FileNotFoundError(
            f"Environment {env_id!r} is not installed and no local package was found at {env_root}. "
            "Install it manually or pass --env-dir."
        )
    if not pyproject_path.is_file():
        raise FileNotFoundError(f"Environment {env_id!r} local package at {env_root} is missing pyproject.toml.")

    with pyproject_path.open("rb") as handle:
        pyproject_data: dict[str, Any] = load_toml(handle)

    project_name = pyproject_data.get("project", {}).get("name")
    if not isinstance(project_name, str) or not project_name:
        raise ValueError(f"Environment {env_id!r} pyproject.toml must define [project].name.")

    loader = pyproject_data.get("tool", {}).get("prime", {}).get("environment", {}).get("loader")
    if loader is not None and not isinstance(loader, str):
        loader = None

    return EnvPackageRef(
        env_id=env_id,
        module_name=module_name,
        project_name=project_name,
        env_path=env_root,
        loader=loader,
    )


def inspect_install_state(ref: EnvPackageRef) -> EnvInstallState:
    distribution_preexisting = _distribution_exists(ref.project_name)
    module_preexisting = _module_importable(ref.module_name)

    if distribution_preexisting and not module_preexisting:
        loader_note = f" Loader metadata is {ref.loader!r}." if ref.loader else ""
        raise ModuleNotFoundError(
            f"Distribution {ref.project_name!r} is installed, but upstream module "
            f"{ref.module_name!r} is not importable.{loader_note}"
        )

    return EnvInstallState(
        ref=ref,
        installed_by_child=False,
        distribution_preexisting=distribution_preexisting,
        module_preexisting=module_preexisting,
    )


def ensure_installed(ref: EnvPackageRef) -> EnvInstallState:
    state = inspect_install_state(ref)
    if state.distribution_preexisting or state.module_preexisting:
        return state

    subprocess.run(
        ["uv", "pip", "install", "--python", sys.executable, "-e", str(ref.env_path)],
        check=True,
    )
    importlib.invalidate_caches()
    if not _module_importable(ref.module_name):
        subprocess.run(
            ["uv", "pip", "uninstall", "--python", sys.executable, "-y", ref.project_name],
            check=False,
        )
        loader_note = f" Loader metadata is {ref.loader!r}." if ref.loader else ""
        raise ModuleNotFoundError(
            f"Installed {ref.project_name!r} from {ref.env_path}, but upstream module "
            f"{ref.module_name!r} is still not importable.{loader_note}"
        )
    return EnvInstallState(
        ref=ref,
        installed_by_child=True,
        distribution_preexisting=False,
        module_preexisting=False,
    )


def uninstall_if_child_installed(state: EnvInstallState) -> None:
    if not state.installed_by_child:
        return
    subprocess.run(
        ["uv", "pip", "uninstall", "--python", sys.executable, "-y", state.ref.project_name],
        check=True,
    )
    importlib.invalidate_caches()
    sys.modules.pop(state.ref.module_name, None)


def _distribution_exists(project_name: str) -> bool:
    try:
        importlib.metadata.distribution(project_name)
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def _module_importable(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


__all__ = [
    "EnvInstallState",
    "EnvPackageRef",
    "ensure_installed",
    "inspect_install_state",
    "resolve_env_package",
    "uninstall_if_child_installed",
    "upstream_module_name",
]
