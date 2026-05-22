"""Runtime adapter contracts for orchestrator backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Protocol

RuntimeName = Literal["docker", "podman", "pyxis"]


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


def normalize_runtime(value: str) -> RuntimeName:
    runtime = str(value).strip().lower()
    if runtime not in {"docker", "podman", "pyxis"}:
        raise ValueError(f"Unsupported runtime {value!r}; expected 'docker', 'podman', or 'pyxis'.")
    return runtime  # type: ignore[return-value]


def build_runtime_adapter(runtime: RuntimeName) -> RuntimeAdapter:
    if runtime == "docker":
        from medarc_verifiers.orchestrate.docker_vllm import DockerRuntimeAdapter

        return DockerRuntimeAdapter()
    if runtime == "podman":
        from medarc_verifiers.orchestrate.podman_vllm import PodmanRuntimeAdapter

        return PodmanRuntimeAdapter()
    if runtime == "pyxis":
        from medarc_verifiers.orchestrate.pyxis_vllm import PyxisRuntimeAdapter

        return PyxisRuntimeAdapter()
    raise ValueError(f"Unsupported runtime {runtime!r}.")


__all__ = [
    "LogStreamer",
    "RuntimeAdapter",
    "RuntimeHandle",
    "RuntimeLaunchError",
    "RuntimeName",
    "build_runtime_adapter",
    "normalize_runtime",
]
