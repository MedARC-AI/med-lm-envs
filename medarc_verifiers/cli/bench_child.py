"""Private subprocess runner for one TOML bench eval with env lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from verifiers.utils.eval_utils import run_evaluation

from medarc_verifiers.cli.bench_health import (
    DEFAULT_VLLM_HEALTH_CHECK_FAILURES,
    DEFAULT_VLLM_HEALTH_CHECK_INTERVAL_SECONDS,
    DEFAULT_VLLM_HEALTH_CHECK_TIMEOUT_SECONDS,
    resolve_vllm_health_check_config,
    run_with_vllm_health_check,
)
from medarc_verifiers.cli.env_lifecycle import (
    EnvInstallState,
    ensure_installed,
    resolve_env_package,
    uninstall_if_child_installed,
)
from medarc_verifiers.cli.upstream_eval import EvalConfigOverrides, build_eval_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one TOML bench eval child payload.")
    parser.add_argument("payload", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.payload.read_text(encoding="utf-8"))
    status = _run_payload(payload)
    status_path = Path(payload["status_path"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, sort_keys=True), encoding="utf-8")
    return int(status["exit_code"])


def _run_payload(payload: dict[str, Any]) -> dict[str, Any]:
    installed_state: EnvInstallState | None = None
    eval_failed = False
    cleanup_failed = False
    status: dict[str, Any] = {
        "env_id": payload.get("expected_env_id"),
        "model": payload.get("expected_model"),
        "installed_by_child": False,
        "eval_ok": False,
        "cleanup_ok": True,
        "primary_error": None,
        "cleanup_error": None,
        "exit_code": 1,
        "exit_reason": "not_started",
    }

    try:
        if payload.get("env_preinstalled", False):
            status["installed_by_child"] = False
        else:
            ref = resolve_env_package(payload["raw_config"]["env_id"], payload["env_dir"])
            installed_state = ensure_installed(ref)
            status["installed_by_child"] = installed_state.installed_by_child

        config = build_eval_config(payload["raw_config"], overrides=_overrides_from_payload(payload["overrides"]))
        planned_resume_path = Path(payload["resume_path"])
        if config.env_id != payload["expected_env_id"]:
            raise ValueError(f"Child resolved env_id {config.env_id!r}, expected {payload['expected_env_id']!r}.")
        if config.model != payload["expected_model"]:
            raise ValueError(f"Child resolved model {config.model!r}, expected {payload['expected_model']!r}.")
        config = config.model_copy(update={"resume_path": planned_resume_path, "save_results": True})
        health_config = resolve_vllm_health_check_config(
            config,
            provider=_raw_provider(payload),
            **_health_check_options(payload),
        )
        asyncio.run(run_with_vllm_health_check(lambda: run_evaluation(config), health_config))
        status["eval_ok"] = True
        status["exit_code"] = 0
        status["exit_reason"] = "success"
    except Exception as exc:  # noqa: BLE001
        eval_failed = True
        status["primary_error"] = _format_exception(exc)
        status["exit_reason"] = "eval_failed"
    finally:
        try:
            if installed_state is not None and payload.get("cleanup_env_package", True):
                uninstall_if_child_installed(installed_state)
        except Exception as exc:  # noqa: BLE001
            cleanup_failed = True
            status["cleanup_ok"] = False
            status["cleanup_error"] = _format_exception(exc)

    if cleanup_failed and not eval_failed:
        status["exit_code"] = 1
        status["exit_reason"] = "cleanup_failed"
    elif eval_failed:
        status["exit_code"] = 1
    return status


def _overrides_from_payload(payload: dict[str, Any]) -> EvalConfigOverrides:
    return EvalConfigOverrides(
        model=payload.get("model"),
        provider=payload.get("provider"),
        api_base_url=payload.get("api_base_url"),
        api_key_var=payload.get("api_key_var"),
        api_client_type=payload.get("api_client_type"),
        endpoints_path=payload.get("endpoints_path"),
        max_concurrent=payload.get("max_concurrent"),
        env_args=payload.get("env_args"),
        sampling_args=payload.get("sampling_args"),
    )


def _health_check_options(payload: dict[str, Any]) -> dict[str, Any]:
    options = payload.get("vllm_health_check")
    if not isinstance(options, dict):
        options = {}
    return {
        "mode": options.get("mode", "auto"),
        "interval_seconds": float(options.get("interval_seconds", DEFAULT_VLLM_HEALTH_CHECK_INTERVAL_SECONDS)),
        "timeout_seconds": float(options.get("timeout_seconds", DEFAULT_VLLM_HEALTH_CHECK_TIMEOUT_SECONDS)),
        "failure_threshold": int(options.get("failure_threshold", DEFAULT_VLLM_HEALTH_CHECK_FAILURES)),
    }


def _raw_provider(payload: dict[str, Any]) -> str | None:
    overrides = payload.get("overrides")
    if isinstance(overrides, dict) and overrides.get("provider") is not None:
        return str(overrides["provider"])
    raw_config = payload.get("raw_config")
    if isinstance(raw_config, dict) and raw_config.get("provider") is not None:
        return str(raw_config["provider"])
    return None


def _format_exception(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
