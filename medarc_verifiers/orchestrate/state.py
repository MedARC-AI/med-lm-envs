"""State tracking and artifact persistence for orchestrator runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import os
import uuid


class JobState:
    pending = "pending"
    allocating = "allocating"
    launching = "launching"
    loading = "loading"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


@dataclass
class TaskManifest:
    task_id: str
    config_path: str
    model_key: str
    model_id: str
    state: str = JobState.pending
    updated_at: str = field(default_factory=_now)
    state_entered_at: str | None = field(default_factory=_now)
    started_at: str | None = None
    completed_at: str | None = None
    gpu_ids: list[int] | None = None
    port: int | None = None
    container_id: str | None = None
    container_name: str | None = None
    container_exit_code: int | None = None
    image: str | None = None
    readiness: Mapping[str, Any] | None = None
    bench_command: str | None = None
    bench_exit_code: int | None = None
    bench_duration_s: float | None = None
    failure_reason: str | None = None
    error: str | None = None

    def touch(self) -> None:
        self.updated_at = _now()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskPaths:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def result_path(self) -> Path:
        return self.root / "result.json"

    @property
    def serve_dir(self) -> Path:
        return self.root / "serve"

    @property
    def bench_dir(self) -> Path:
        return self.root / "bench"

    @property
    def stdout_path(self) -> Path:
        return self.bench_dir / "stdout.txt"

    @property
    def stderr_path(self) -> Path:
        return self.bench_dir / "stderr.txt"

    @property
    def container_request_path(self) -> Path:
        return self.serve_dir / "container_create_request.json"

    @property
    def container_logs_path(self) -> Path:
        return self.serve_dir / "container_logs.txt"

    @property
    def readiness_path(self) -> Path:
        return self.serve_dir / "readiness.json"


def write_task_manifest(paths: TaskPaths, manifest: TaskManifest) -> None:
    manifest.touch()
    _write_json_atomic(paths.manifest_path, manifest.to_dict())


def write_task_result(paths: TaskPaths, payload: Mapping[str, Any]) -> None:
    _write_json_atomic(paths.result_path, payload)


def write_summary(path: Path, tasks: list[TaskManifest]) -> None:
    payload = {
        "updated_at": _now(),
        "tasks": [
            {
                "task_id": task.task_id,
                "state": task.state,
                "model_key": task.model_key,
                "model_id": task.model_id,
                "config_path": task.config_path,
                "failure_reason": task.failure_reason,
                "error": task.error,
            }
            for task in tasks
        ],
    }
    _write_json_atomic(path, payload)


def load_summary(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Summary file is not a mapping: {path}")
    return payload


def filter_tasks_for_resume(
    tasks: list["TaskSpec"], summary: Mapping[str, Any], *, rerun_failed: bool
) -> list["TaskSpec"]:
    entries = summary.get("tasks")
    if not isinstance(entries, list):
        return tasks
    completed = {entry.get("task_id") for entry in entries if entry.get("state") == JobState.completed}
    failed = {entry.get("task_id") for entry in entries if entry.get("state") == JobState.failed}
    filtered: list["TaskSpec"] = []
    for task in tasks:
        if task.task_id in completed:
            continue
        if not rerun_failed and task.task_id in failed:
            continue
        filtered.append(task)
    return filtered


class TaskSpec:  # pragma: no cover - imported from config at runtime
    task_id: str


__all__ = [
    "JobState",
    "TaskManifest",
    "TaskPaths",
    "filter_tasks_for_resume",
    "load_summary",
    "write_summary",
    "write_task_manifest",
    "write_task_result",
    "write_text",
]
