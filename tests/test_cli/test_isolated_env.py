from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import isolated_env


class FakeDistribution:
    def __init__(self, *, version: str = "1.2.3", direct_url: dict | None = None) -> None:
        self.version = version
        self._direct_url = direct_url

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json" or self._direct_url is None:
            return None
        return json.dumps(self._direct_url)


def test_current_medarc_install_spec_detects_editable_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'medarc-verifiers'\n", encoding="utf-8")
    (tmp_path / "medarc_verifiers").mkdir()
    direct_url = {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}

    monkeypatch.setattr(isolated_env.metadata, "distribution", lambda name: FakeDistribution(direct_url=direct_url))

    spec = isolated_env.current_medarc_install_spec()

    assert spec.editable is True
    assert spec.checkout_root == tmp_path


def test_current_medarc_install_spec_rejects_invalid_editable_checkout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    direct_url = {"url": tmp_path.as_uri(), "dir_info": {"editable": True}}
    monkeypatch.setattr(isolated_env.metadata, "distribution", lambda name: FakeDistribution(direct_url=direct_url))

    with pytest.raises(isolated_env.IsolatedEnvError, match="invalid"):
        isolated_env.current_medarc_install_spec()


def test_install_medarc_non_editable_uses_pinned_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(isolated_env, "current_medarc_install_spec", lambda: isolated_env.MedarcInstallSpec(False, "9.8.7"))
    monkeypatch.setattr(isolated_env, "_run_uv", lambda command, action: commands.append(command))

    isolated_env.install_medarc_into_venv(tmp_path / "python")

    assert commands == [["uv", "pip", "install", "--python", str(tmp_path / "python"), "medarc-verifiers==9.8.7"]]


def test_install_medarc_non_editable_resolution_failure_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(isolated_env, "current_medarc_install_spec", lambda: isolated_env.MedarcInstallSpec(False, "9.8.7"))

    def fail(command: list[str], action: str) -> None:
        raise isolated_env.IsolatedEnvError("resolver failed")

    monkeypatch.setattr(isolated_env, "_run_uv", fail)

    with pytest.raises(isolated_env.IsolatedEnvError, match="preinstall environment packages"):
        isolated_env.install_medarc_into_venv(tmp_path / "python")


def test_temporary_bench_venv_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    venv_path = tmp_path / "venv"
    python_path = venv_path / "bin" / "python"
    created: list[Path] = []

    def fake_mkdtemp(prefix: str) -> str:
        venv_path.mkdir(parents=True)
        python_path.parent.mkdir(parents=True)
        python_path.write_text("", encoding="utf-8")
        return str(venv_path)

    monkeypatch.setattr(isolated_env.tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(isolated_env, "_create_venv", lambda path: created.append(path) or python_path)
    monkeypatch.setattr(isolated_env, "install_medarc_into_venv", lambda python, repo_root=None: None)

    with isolated_env.temporary_bench_venv() as python:
        assert python == python_path
        assert venv_path.exists()

    assert created == [venv_path]
    assert not venv_path.exists()


def test_run_uv_reports_missing_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_uv(*args, **kwargs):
        raise FileNotFoundError("uv")

    monkeypatch.setattr(isolated_env.subprocess, "run", missing_uv)

    with pytest.raises(isolated_env.IsolatedEnvError, match="uv is not installed"):
        isolated_env._run_uv(["uv", "venv"], "create venv")


def test_run_uv_reports_failing_command(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        isolated_env.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2, stderr="bad\nerror", stdout=""),
    )

    with pytest.raises(isolated_env.IsolatedEnvError, match="error"):
        isolated_env._run_uv(["uv", "venv"], "create venv")


@pytest.mark.skipif(
    os.environ.get("MEDARC_RUN_ISOLATED_ENV_SMOKE") != "1",
    reason="set MEDARC_RUN_ISOLATED_ENV_SMOKE=1 to run the real uv isolated-env smoke",
)
def test_temporary_bench_venv_real_helper_imports_bench_child() -> None:
    with isolated_env.temporary_bench_venv() as python:
        completed = subprocess.run(
            [str(python), "-m", "medarc_verifiers.cli.bench_child", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

    assert completed.returncode == 0
    assert "Run one TOML bench eval child payload" in completed.stdout
