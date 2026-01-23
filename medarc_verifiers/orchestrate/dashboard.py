"""Rich live dashboard for orchestrator progress."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from medarc_verifiers.orchestrate.state import TaskManifest


ACTIVE_STATES = {"allocating", "launching", "loading", "running"}

STATE_STYLES: dict[str, str] = {
    "pending": "dim",
    "allocating": "cyan",
    "launching": "blue",
    "loading": "magenta",
    "running": "yellow",
    "completed": "green",
    "failed": "bold red",
    "cancelled": "magenta",
}

LOG_PREFIX_STYLES: dict[str, str] = {
    "RUN": "bold cyan",
    "JOB": "bold",
    "SHUTDOWN": "bold red",
}

LOG_EVENT_STYLES: dict[str, str] = {
    "started": "cyan",
    "start": "cyan",
    "state": "blue",
    "ready": "green",
    "bench-start": "yellow",
    "bench-ok": "bold green",
    "bench-failed": "bold red",
    "bench-terminated": "bold magenta",
    "complete": "bold green",
    "failed": "bold red",
    "cancelled": "magenta",
}


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


def build_table(
    tasks: Iterable[TaskManifest],
    *,
    caption: str | None = None,
    include_summary: bool = False,
) -> Table:
    task_list = list(tasks)
    pending_count = sum(1 for task in task_list if task.state == "pending")
    completed_count = sum(1 for task in task_list if task.state == "completed")

    table = Table(title=Text("Orchestrator", style="bold cyan"), caption=caption, expand=True)
    table.add_column("Task", no_wrap=True, style="bold")
    table.add_column("Model", no_wrap=True, style="dim")
    table.add_column("State", no_wrap=True)
    table.add_column("State Elapsed", no_wrap=True, style="dim")
    table.add_column("Total Elapsed", no_wrap=True, style="dim")
    table.add_column("GPUs", no_wrap=True, style="cyan")
    table.add_column("Port", no_wrap=True, style="cyan")
    table.add_column("Note")
    for task in task_list:
        if task.state not in ACTIVE_STATES:
            continue
        gpu_text = ",".join(str(gpu) for gpu in task.gpu_ids or []) or "-"
        port_text = str(task.port) if task.port is not None else "-"
        state_elapsed = _format_elapsed(task.state_entered_at, None)
        total_elapsed = _format_elapsed(task.started_at, None)
        note = task.error or task.failure_reason or ""
        note_style = "red" if note else "dim"
        table.add_row(
            Text(task.task_id),
            Text(task.model_key, style="dim"),
            Text(task.state, style=STATE_STYLES.get(task.state, "")),
            state_elapsed,
            total_elapsed,
            gpu_text,
            port_text,
            Text(note, style=note_style),
        )

    if include_summary:
        table.add_row(
            Text("PENDING (all)", style="dim"),
            Text("-", style="dim"),
            Text("pending", style=STATE_STYLES["pending"]),
            Text("-", style="dim"),
            Text("-", style="dim"),
            Text("-", style="dim"),
            Text("-", style="dim"),
            Text(f"count={pending_count}", style="dim"),
        )
        table.add_row(
            Text("COMPLETED (all)", style="dim"),
            Text("-", style="dim"),
            Text("completed", style=STATE_STYLES["completed"]),
            Text("-", style="dim"),
            Text("-", style="dim"),
            Text("-", style="dim"),
            Text("-", style="dim"),
            Text(f"count={completed_count}", style="dim"),
        )
    return table


def format_log_message(message: str) -> Text:
    text = Text(message)
    parts = message.split(" ", maxsplit=2)
    if not parts:
        return text
    prefix = parts[0]
    prefix_style = LOG_PREFIX_STYLES.get(prefix)
    if prefix_style:
        text.stylize(prefix_style, 0, len(prefix))
    if len(parts) >= 2:
        event = parts[1]
        event_style = LOG_EVENT_STYLES.get(event)
        if event_style:
            start = len(prefix) + 1
            end = start + len(event)
            text.stylize(event_style, start, end)
    return text


@dataclass
class OrchestratorDashboard:
    refresh_hz: float = 1.0
    enabled: bool = True

    def __post_init__(self) -> None:
        # Keep logs human-readable: no source file/line prefixes and no automatic syntax highlighting.
        self._console = Console(log_path=False, highlight=False)
        self._live = Live(
            build_table([], include_summary=True),
            refresh_per_second=self.refresh_hz,
            transient=False,
            console=self._console,
        )

    def start(self) -> None:
        if self.enabled:
            self._live.start()

    def update(self, tasks: Iterable[TaskManifest], *, caption: str | None = None) -> None:
        if self.enabled:
            self._live.update(build_table(tasks, caption=caption, include_summary=True))

    def stop(self) -> None:
        if self.enabled:
            self._live.stop()

    def log(self, message: str) -> None:
        self._console.log(format_log_message(message))


__all__ = ["ACTIVE_STATES", "OrchestratorDashboard", "build_table"]
