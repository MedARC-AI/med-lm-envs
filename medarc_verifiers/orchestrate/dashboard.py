"""Rich live dashboard for orchestrator progress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from rich.console import Console
from rich.live import Live
from rich.table import Table

from medarc_verifiers.orchestrate.state import TaskManifest


ACTIVE_STATES = {"allocating", "launching", "loading", "running"}


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _format_elapsed(started_at: str | None, completed_at: str | None) -> str:
    start = _parse_time(started_at)
    if not start:
        return "-"
    end = _parse_time(completed_at) or datetime.now(timezone.utc)
    elapsed = end - start
    total_seconds = int(elapsed.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def build_table(tasks: Iterable[TaskManifest], *, caption: str | None = None) -> Table:
    table = Table(title="Current Running Jobs", caption=caption, expand=True)
    table.add_column("Task", no_wrap=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("State Elapsed", no_wrap=True)
    table.add_column("Total Elapsed", no_wrap=True)
    table.add_column("GPUs", no_wrap=True)
    table.add_column("Port", no_wrap=True)
    table.add_column("Note")
    for task in tasks:
        if task.state not in ACTIVE_STATES:
            continue
        gpu_text = ",".join(str(gpu) for gpu in task.gpu_ids or []) or "-"
        port_text = str(task.port) if task.port is not None else "-"
        state_elapsed = _format_elapsed(task.state_entered_at, None)
        total_elapsed = _format_elapsed(task.started_at, None)
        note = task.error or task.failure_reason or ""
        table.add_row(
            task.task_id,
            task.model_key,
            task.state,
            state_elapsed,
            total_elapsed,
            gpu_text,
            port_text,
            note,
        )
    return table


@dataclass
class OrchestratorDashboard:
    refresh_hz: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        self._console = Console()
        self._live = Live(
            build_table([]),
            refresh_per_second=self.refresh_hz,
            transient=False,
            console=self._console,
        )

    def start(self) -> None:
        if self.enabled:
            self._live.start()

    def update(self, tasks: Iterable[TaskManifest], *, caption: str | None = None) -> None:
        if self.enabled:
            self._live.update(build_table(tasks, caption=caption))

    def stop(self) -> None:
        if self.enabled:
            self._live.stop()

    def log(self, message: str) -> None:
        self._console.log(message)


__all__ = ["ACTIVE_STATES", "OrchestratorDashboard", "build_table"]
