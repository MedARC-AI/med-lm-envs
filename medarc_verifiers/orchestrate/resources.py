"""Resource discovery and reservation primitives for the orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import socket


class ResourceError(RuntimeError):
    """Raised when resources cannot be discovered or allocated."""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    total_gb: float
    free_gb: float


def parse_index_range(expr: str, *, max_index: int | None = None) -> list[int]:
    """Parse a range expression like "0-3,5" into sorted indices."""
    if not expr:
        return []
    indices: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    if max_index is not None:
        indices = {idx for idx in indices if 0 <= idx <= max_index}
    return sorted(indices)


def discover_gpus() -> list[GpuInfo]:
    """Return GPU info via NVML, or raise ResourceError."""
    try:
        import pynvml
    except Exception as exc:  # pragma: no cover - dependency import varies
        raise ResourceError("pynvml is required for GPU discovery.") from exc
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GpuInfo] = []
        for idx in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GpuInfo(
                    index=idx,
                    total_gb=mem.total / (1024**3),
                    free_gb=mem.free / (1024**3),
                )
            )
        return gpus
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


class ResourceManager:
    """In-process GPU/port reservation map."""

    def __init__(self, gpu_indices: Sequence[int] | None = None, port_range: tuple[int, int] | None = None):
        self._gpu_indices = list(gpu_indices) if gpu_indices is not None else None
        self._port_range = port_range
        self._gpu_reservations: dict[int, str] = {}
        self._port_reservations: dict[int, str] = {}

    def available_gpus(self, *, min_free_gb: float | None = None) -> list[GpuInfo]:
        gpus = discover_gpus()
        if self._gpu_indices is not None:
            allowed = set(self._gpu_indices)
            gpus = [gpu for gpu in gpus if gpu.index in allowed]
        if min_free_gb is not None:
            gpus = [gpu for gpu in gpus if gpu.free_gb >= min_free_gb]
        return gpus

    def reserve_gpus(
        self,
        task_id: str,
        *,
        count: int,
        min_free_gb: float | None = None,
        require_contiguous: bool = False,
    ) -> list[int]:
        gpus = self.available_gpus(min_free_gb=min_free_gb)
        free = [gpu.index for gpu in gpus if gpu.index not in self._gpu_reservations]
        if len(free) < count:
            raise ResourceError("Insufficient free GPUs for reservation.")
        selection = _select_contiguous(free, count)
        if selection is None:
            if require_contiguous and count > 1:
                raise ResourceError("No contiguous GPUs available for reservation.")
            selection = free[:count]
        for idx in selection:
            self._gpu_reservations[idx] = task_id
        return selection

    def release_gpus(self, indices: Iterable[int]) -> None:
        for idx in indices:
            self._gpu_reservations.pop(idx, None)

    def reserve_port(self, task_id: str) -> int:
        if not self._port_range:
            raise ResourceError("Port range is not configured.")
        start, end = self._port_range
        for port in range(start, end + 1):
            if port in self._port_reservations:
                continue
            if not _port_is_available(port):
                continue
            self._port_reservations[port] = task_id
            return port
        raise ResourceError("No free ports available in range.")

    def release_port(self, port: int) -> None:
        self._port_reservations.pop(port, None)


def _port_is_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _select_contiguous(indices: Sequence[int], count: int) -> list[int] | None:
    if count <= 1:
        return list(indices[:count])
    sorted_indices = sorted(indices)
    start = 0
    for end in range(len(sorted_indices)):
        if end > start and sorted_indices[end] != sorted_indices[end - 1] + 1:
            start = end
        if end - start + 1 == count:
            return sorted_indices[start : end + 1]
    return None


__all__ = [
    "GpuInfo",
    "ResourceError",
    "ResourceManager",
    "discover_gpus",
    "parse_index_range",
]
