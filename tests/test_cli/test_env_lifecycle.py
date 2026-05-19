from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import env_lifecycle


def _write_env(root: Path, folder: str, *, project_name: str, loader: str | None = None) -> Path:
    env_path = root / folder
    env_path.mkdir(parents=True)
    loader_block = ""
    if loader is not None:
        loader_block = f'\n[tool.prime.environment]\nloader = "{loader}"\n'
    (env_path / "pyproject.toml").write_text(
        f'[project]\nname = "{project_name}"\n{loader_block}',
        encoding="utf-8",
    )
    return env_path


def test_resolve_env_package_uses_upstream_module_and_project_name(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path, "head_qa_v2", project_name="head-qa-v2", loader="other:load_environment")

    ref = env_lifecycle.resolve_env_package("owner/head-qa-v2", tmp_path)

    assert ref.env_id == "owner/head-qa-v2"
    assert ref.module_name == "head_qa_v2"
    assert ref.project_name == "head-qa-v2"
    assert ref.env_path == env_path
    assert ref.loader == "other:load_environment"


def test_resolve_env_package_errors_for_missing_pyproject(tmp_path: Path) -> None:
    (tmp_path / "medqa").mkdir()

    with pytest.raises(FileNotFoundError, match="missing pyproject.toml"):
        env_lifecycle.resolve_env_package("medqa", tmp_path)


def test_inspect_install_state_rejects_installed_distribution_without_module(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = env_lifecycle.EnvPackageRef("medqa", "medqa", "medqa", Path("envs/medqa"), None)
    monkeypatch.setattr(env_lifecycle, "_distribution_exists", lambda name: True)
    monkeypatch.setattr(env_lifecycle, "_module_importable", lambda name: False)

    with pytest.raises(ModuleNotFoundError, match="upstream module 'medqa' is not importable"):
        env_lifecycle.inspect_install_state(ref)


def test_ensure_installed_installs_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = env_lifecycle.EnvPackageRef("medqa", "medqa", "medqa", Path("envs/medqa"), None)
    calls: list[list[str]] = []
    importable = [False, True]

    monkeypatch.setattr(env_lifecycle, "_distribution_exists", lambda name: False)
    monkeypatch.setattr(env_lifecycle, "_module_importable", lambda name: importable.pop(0))
    monkeypatch.setattr(env_lifecycle.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(
        env_lifecycle.subprocess,
        "run",
        lambda cmd, check: calls.append(cmd) or SimpleNamespace(returncode=0),
    )

    state = env_lifecycle.ensure_installed(ref)

    assert state.installed_by_child is True
    assert calls[0][:4] == ["uv", "pip", "install", "--python"]


def test_uninstall_only_child_installed_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    ref = env_lifecycle.EnvPackageRef("medqa", "medqa", "medqa", Path("envs/medqa"), None)
    state = env_lifecycle.EnvInstallState(ref, True, False, False)
    calls: list[list[str]] = []

    monkeypatch.setattr(env_lifecycle.importlib, "invalidate_caches", lambda: None)
    monkeypatch.setattr(
        env_lifecycle.subprocess,
        "run",
        lambda cmd, check: calls.append(cmd) or SimpleNamespace(returncode=0),
    )

    env_lifecycle.uninstall_if_child_installed(state)

    assert calls[0][:4] == ["uv", "pip", "uninstall", "--python"]
