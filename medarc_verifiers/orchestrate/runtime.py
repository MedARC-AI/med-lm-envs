"""Runtime adapter contracts for orchestrator backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol


class RuntimeLaunchError(RuntimeError):
    """Raised when a serve runtime fails to launch."""


@dataclass(frozen=True)
class RuntimeHandle:
    """Opaque handle returned after launching a serve runtime."""

    base_url: str
    identifier: str


class LogStreamer(Protocol):
    def start(self) -> None: ...

    def stop(self, *, timeout: float = 2.0) -> None: ...

    def is_alive(self) -> bool: ...


class RuntimeAdapter(Protocol):
    def launch(
        self,
        *,
        task_id: str,
        model_id: str,
        container_args: list[str],
        image: str,
        container_port: int,
        volume_mounts: list[str],
        gpus_required: int,
        gpu_ids: list[int],
        server_port: int,
        env: Mapping[str, str],
        labels: Mapping[str, str],
        **runtime_kwargs,
    ) -> RuntimeHandle: ...

    def stream_logs(self, handle: RuntimeHandle, sink: Path) -> LogStreamer: ...

    def teardown(self, handle: RuntimeHandle) -> None: ...


__all__ = ["LogStreamer", "RuntimeAdapter", "RuntimeHandle", "RuntimeLaunchError"]
