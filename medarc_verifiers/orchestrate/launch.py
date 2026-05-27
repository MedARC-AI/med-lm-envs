"""Canonical launch resolution for Slurm orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from medarc_verifiers.orchestrate.config import (
    PlanConfig,
    PrepareConfig,
    TaskSpec,
    TeardownConfig,
    expand_tasks,
    load_plan,
    make_plan,
    resolve_default_endpoints_path,
)
from medarc_verifiers.utils.run_naming import generate_run_id


@dataclass(frozen=True)
class LaunchRequest:
    plan: Path | None = None
    suite: Path | None = None
    endpoints: tuple[str, ...] = ()
    name: str | None = None
    env_file: Path | None = None
    run_id: str | None = None
    bundle_dir: Path | None = None
    output_dir: Path | None = None
    readiness_timeout_s: int | None = None
    prune_logs_on_success: bool = False
    eval_images_config: Path | None = None
    endpoints_path: Path | None = None


@dataclass(frozen=True)
class LaunchPlan:
    tasks: list[TaskSpec]
    run_id: str
    output_root: Path
    env_file: Path | None
    readiness_timeout_s: int
    prune_logs_on_success: bool
    prepare: PrepareConfig
    teardown: TeardownConfig


@dataclass(frozen=True)
class LaunchStatusTarget:
    run_id: str | None
    output_root: Path


def resolve_launch_plan(request: LaunchRequest, *, cwd: Path) -> LaunchPlan:
    plan, base_dir = _resolve_plan_config(request, cwd=cwd)
    if request.env_file is not None:
        plan = plan.model_copy(update={"env_file": _resolve_path(request.env_file, base_dir=base_dir)})
    if request.output_dir is not None:
        plan = plan.model_copy(update={"output_dir": _resolve_path(request.output_dir, base_dir=base_dir)})
    if request.prune_logs_on_success:
        plan = plan.model_copy(update={"prune_logs_on_success": True})

    default_endpoint_path = None if plan.endpoints_path is not None else resolve_default_endpoints_path(cwd)
    tasks = expand_tasks(plan, default_endpoints_path=default_endpoint_path)
    run_id, output_root = resolve_output_root(request, plan=plan)
    readiness_timeout_s = (
        request.readiness_timeout_s if request.readiness_timeout_s is not None else plan.readiness_timeout_s
    )
    return LaunchPlan(
        tasks=tasks,
        run_id=run_id,
        output_root=output_root,
        env_file=plan.env_file,
        readiness_timeout_s=int(readiness_timeout_s or 1800),
        prune_logs_on_success=plan.prune_logs_on_success,
        prepare=plan.prepare,
        teardown=plan.teardown,
    )


def resolve_output_root(request: LaunchRequest, *, plan: PlanConfig) -> tuple[str, Path]:
    configured_run_id = request.run_id or plan.run_id
    run_id = configured_run_id or generate_run_id(plan.name)
    output_root = request.bundle_dir or plan.bundle_dir or Path("outputs") / "orchestrate" / run_id
    return run_id, output_root.expanduser().resolve()


def resolve_status_target(*, run_id: str | None, bundle_dir: Path | None, cwd: Path) -> LaunchStatusTarget:
    root_dir = cwd.expanduser().resolve()
    if bundle_dir is not None:
        root = bundle_dir.expanduser().resolve()
    elif run_id:
        root = (root_dir / "outputs" / "orchestrate" / run_id).expanduser().resolve()
    else:
        root = (root_dir / "outputs" / "orchestrate").expanduser().resolve()
    return LaunchStatusTarget(run_id=run_id, output_root=root)


def _resolve_plan_config(request: LaunchRequest, *, cwd: Path) -> tuple[PlanConfig, Path]:
    base_dir = cwd.expanduser().resolve()
    if request.plan is not None:
        plan_path = request.plan.expanduser().resolve()
        plan = load_plan(plan_path)
        base_dir = plan_path.parent
        updates: dict[str, object] = {}
        if plan.name is None:
            updates["name"] = plan_path.stem
        if request.eval_images_config is not None:
            updates["eval_images_config"] = _resolve_path(request.eval_images_config, base_dir=base_dir)
        elif plan.eval_images_config is None:
            updates["eval_images_config"] = _resolve_eval_images_config_path(None, cwd=cwd)
        if request.endpoints_path is not None:
            updates["endpoints_path"] = _resolve_path(request.endpoints_path, base_dir=base_dir)
        return (plan.model_copy(update=updates) if updates else plan), base_dir

    if request.suite is None or not request.endpoints:
        raise ValueError("medarc-orchestrate run requires --plan or --suite with at least one --endpoint.")
    name = request.name
    if name is None:
        name = request.suite.expanduser().stem
    plan = make_plan(
        suite=request.suite,
        targets=list(request.endpoints),
        base_dir=base_dir,
        name=name,
        eval_images_config=request.eval_images_config or _resolve_eval_images_config_path(None, cwd=cwd),
        endpoints_path=request.endpoints_path,
    )
    return plan, base_dir


def _resolve_eval_images_config_path(path: Path | None, *, cwd: Path) -> Path | None:
    if path is not None:
        return path.expanduser().resolve()
    default = (cwd.expanduser().resolve() / "configs" / "eval_images.toml").resolve()
    return default if default.exists() else None


def _resolve_path(path: Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


__all__ = [
    "LaunchPlan",
    "LaunchRequest",
    "LaunchStatusTarget",
    "resolve_launch_plan",
    "resolve_output_root",
    "resolve_status_target",
]
