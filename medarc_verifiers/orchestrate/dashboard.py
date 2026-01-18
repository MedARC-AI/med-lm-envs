"""Rich live dashboard for orchestrator progress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from rich.live import Live
from rich.table import Table

from medarc_verifiers.orchestrate.state import TaskManifest


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


def build_table(tasks: Iterable[TaskManifest]) -> Table:
    table = Table(title="vLLM Orchestrator", expand=True)
    table.add_column("Model", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("GPUs", no_wrap=True)
    table.add_column("Port", no_wrap=True)
    table.add_column("Elapsed", no_wrap=True)
    table.add_column("Last Error")
    for task in tasks:
        gpu_text = ",".join(str(gpu) for gpu in task.gpu_ids or []) or "-"
        port_text = str(task.port) if task.port is not None else "-"
        elapsed = _format_elapsed(task.started_at, task.completed_at)
        last_error = task.error or task.failure_reason or ""
        table.add_row(task.model_key, task.state, gpu_text, port_text, elapsed, last_error)
    return table


@dataclass
class OrchestratorDashboard:
    refresh_hz: float = 1.0

    def __post_init__(self) -> None:
        self._live = Live(build_table([]), refresh_per_second=self.refresh_hz, transient=False)

    def start(self) -> None:
        self._live.start()

    def update(self, tasks: Iterable[TaskManifest]) -> None:
        self._live.update(build_table(tasks))

    def stop(self) -> None:
        self._live.stop()


__all__ = ["OrchestratorDashboard", "build_table"]
