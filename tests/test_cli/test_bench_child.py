from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from medarc_verifiers.cli import bench_child
from medarc_verifiers.cli.env_lifecycle import EnvInstallState, EnvPackageRef


def _payload(tmp_path: Path) -> dict:
    return {
        "raw_config": {"env_id": "medqa", "model": "parent-model"},
        "overrides": {},
        "env_dir": str(tmp_path / "envs"),
        "resume_path": str(tmp_path / "runs" / "evals" / "parent-model" / "medqa" / "base"),
        "status_path": str(tmp_path / "status.json"),
        "expected_env_id": "medqa",
        "expected_model": "parent-model",
    }


def _state(installed_by_child: bool) -> EnvInstallState:
    ref = EnvPackageRef("medqa", "medqa", "medqa", Path("envs/medqa"), None)
    return EnvInstallState(ref, installed_by_child, False, False)


def test_child_installs_builds_runs_and_cleans_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    config = SimpleNamespace(
        env_id="medqa",
        model="parent-model",
        model_copy=lambda update: SimpleNamespace(env_id="medqa", model="parent-model", **update),
    )

    monkeypatch.setattr(bench_child, "resolve_env_package", lambda env_id, env_dir: object())
    monkeypatch.setattr(bench_child, "ensure_installed", lambda ref: calls.append("install") or _state(True))
    monkeypatch.setattr(bench_child, "build_eval_config", lambda raw, overrides: calls.append("build") or config)
    monkeypatch.setattr(bench_child, "uninstall_if_child_installed", lambda state: calls.append("cleanup"))

    async def fake_run_evaluation(run_config):
        calls.append(f"run:{run_config.resume_path}")

    monkeypatch.setattr(bench_child, "run_evaluation", fake_run_evaluation)

    status = bench_child._run_payload(_payload(tmp_path))

    assert status["exit_code"] == 0
    assert status["installed_by_child"] is True
    assert calls == [
        "install",
        "build",
        f"run:{tmp_path / 'runs' / 'evals' / 'parent-model' / 'medqa' / 'base'}",
        "cleanup",
    ]


def test_child_cleanup_env_package_false_skips_uninstall(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    config = SimpleNamespace(
        env_id="medqa",
        model="parent-model",
        model_copy=lambda update: SimpleNamespace(env_id="medqa", model="parent-model", **update),
    )

    monkeypatch.setattr(bench_child, "resolve_env_package", lambda env_id, env_dir: object())
    monkeypatch.setattr(bench_child, "ensure_installed", lambda ref: calls.append("install") or _state(True))
    monkeypatch.setattr(bench_child, "build_eval_config", lambda raw, overrides: calls.append("build") or config)
    monkeypatch.setattr(bench_child, "uninstall_if_child_installed", lambda state: calls.append("cleanup"))

    async def fake_run_evaluation(run_config):
        calls.append(f"run:{run_config.resume_path}")

    monkeypatch.setattr(bench_child, "run_evaluation", fake_run_evaluation)

    payload = _payload(tmp_path)
    payload["cleanup_env_package"] = False
    status = bench_child._run_payload(payload)

    assert status["exit_code"] == 0
    assert status["installed_by_child"] is True
    assert calls == ["install", "build", f"run:{tmp_path / 'runs' / 'evals' / 'parent-model' / 'medqa' / 'base'}"]


def test_child_env_preinstalled_skips_install_and_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    config = SimpleNamespace(
        env_id="medqa",
        model="parent-model",
        model_copy=lambda update: SimpleNamespace(env_id="medqa", model="parent-model", **update),
    )

    monkeypatch.setattr(bench_child, "resolve_env_package", lambda env_id, env_dir: calls.append("resolve"))
    monkeypatch.setattr(bench_child, "ensure_installed", lambda ref: calls.append("install") or _state(True))
    monkeypatch.setattr(bench_child, "build_eval_config", lambda raw, overrides: calls.append("build") or config)
    monkeypatch.setattr(bench_child, "uninstall_if_child_installed", lambda state: calls.append("cleanup"))

    async def fake_run_evaluation(run_config):
        calls.append(f"run:{run_config.resume_path}")

    monkeypatch.setattr(bench_child, "run_evaluation", fake_run_evaluation)

    payload = _payload(tmp_path)
    payload["env_preinstalled"] = True
    payload["cleanup_env_package"] = False
    status = bench_child._run_payload(payload)

    assert status["exit_code"] == 0
    assert status["installed_by_child"] is False
    assert calls == ["build", f"run:{tmp_path / 'runs' / 'evals' / 'parent-model' / 'medqa' / 'base'}"]


def test_child_install_failure_does_not_build_or_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []

    monkeypatch.setattr(bench_child, "resolve_env_package", lambda env_id, env_dir: object())

    def fail_install(ref):
        calls.append("install")
        raise RuntimeError("install failed")

    monkeypatch.setattr(bench_child, "ensure_installed", fail_install)
    monkeypatch.setattr(bench_child, "build_eval_config", lambda raw, overrides: calls.append("build"))
    monkeypatch.setattr(bench_child, "uninstall_if_child_installed", lambda state: calls.append("cleanup"))

    status = bench_child._run_payload(_payload(tmp_path))

    assert status["exit_code"] == 1
    assert status["exit_reason"] == "eval_failed"
    assert "install failed" in status["primary_error"]
    assert calls == ["install"]


def test_child_cleanup_failure_after_success_is_fatal(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = SimpleNamespace(
        env_id="medqa",
        model="parent-model",
        model_copy=lambda update: SimpleNamespace(env_id="medqa", model="parent-model", **update),
    )

    monkeypatch.setattr(bench_child, "resolve_env_package", lambda env_id, env_dir: object())
    monkeypatch.setattr(bench_child, "ensure_installed", lambda ref: _state(True))
    monkeypatch.setattr(bench_child, "build_eval_config", lambda raw, overrides: config)

    async def fake_run_evaluation(run_config):
        return None

    monkeypatch.setattr(bench_child, "run_evaluation", fake_run_evaluation)

    def fail_cleanup(state):
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(bench_child, "uninstall_if_child_installed", fail_cleanup)

    status = bench_child._run_payload(_payload(tmp_path))

    assert status["eval_ok"] is True
    assert status["cleanup_ok"] is False
    assert status["exit_code"] == 1
    assert status["exit_reason"] == "cleanup_failed"
