"""Podman-backed vLLM launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping

from medarc_verifiers.orchestrate.runtime import RuntimeHandle, RuntimeLaunchError

ORCHESTRATOR_LABEL_KEY = "orchestrator.managed"


class PodmanLaunchError(RuntimeLaunchError):
    """Raised when Podman container launch fails."""


class PodmanLogStreamer:
    def __init__(self, container_id: str, sink_path: str) -> None:
        self._container_id = container_id
        self._sink_path = sink_path
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._process is not None and self.is_alive():
            return
        sink = open(self._sink_path, "wb")
        self._process = subprocess.Popen(
            ["podman", "logs", "-f", self._container_id],
            stdout=sink,
            stderr=subprocess.STDOUT,
        )

    def stop(self, *, timeout: float = 2.0) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=timeout)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None


class PodmanRuntimeAdapter:
    """Runtime adapter that launches vLLM in Podman containers."""

    def __init__(self) -> None:
        self._containers: dict[str, str] = {}

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
    ) -> RuntimeHandle:
        del task_id, model_id, gpus_required
        command = [
            "podman",
            "run",
            "--detach",
            "--replace",
            "--name",
            str(runtime_kwargs.get("name", "")),
            "--publish",
            f"127.0.0.1:{server_port}:{container_port}",
        ]
        ipc_mode = runtime_kwargs.get("ipc_mode")
        if ipc_mode:
            command.extend(["--ipc", str(ipc_mode)])
        for mount in volume_mounts:
            command.extend(["--volume", mount])
        for key, value in env.items():
            command.extend(["--env", f"{key}={value}"])
        for key, value in labels.items():
            command.extend(["--label", f"{key}={value}"])
        if gpu_ids:
            command.extend(["--env", f"NVIDIA_VISIBLE_DEVICES={','.join(str(gpu) for gpu in gpu_ids)}"])
            command.extend(["--device", f"nvidia.com/gpu={','.join(str(gpu) for gpu in gpu_ids)}"])
        command.append(image)
        command.extend(container_args)
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise PodmanLaunchError(completed.stderr.strip() or completed.stdout.strip() or "podman run failed")
        container_id = completed.stdout.strip()
        if not container_id:
            raise PodmanLaunchError("podman run did not return a container id")
        handle = RuntimeHandle(base_url=f"http://127.0.0.1:{server_port}/v1", identifier=container_id)
        self._containers[handle.identifier] = container_id
        return handle

    def stream_logs(self, handle: RuntimeHandle, sink: Path) -> PodmanLogStreamer:
        return PodmanLogStreamer(self._containers.get(handle.identifier, handle.identifier), str(sink))

    def teardown(self, handle: RuntimeHandle) -> None:
        container_id = self._containers.pop(handle.identifier, None)
        if container_id is None:
            return
        subprocess.run(["podman", "rm", "-f", container_id], check=False, capture_output=True, text=True)


def cleanup_orphan_containers(run_id: str | None = None) -> list[str]:
    command = [
        "podman",
        "ps",
        "-a",
        "--filter",
        f"label={ORCHESTRATOR_LABEL_KEY}=true",
        "--format",
        "{{.ID}}\t{{.Names}}",
    ]
    if run_id:
        command.extend(["--filter", f"label=orchestrator.run_id={run_id}"])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise PodmanLaunchError(completed.stderr.strip() or completed.stdout.strip() or "podman ps failed")

    removed: list[str] = []
    for line in completed.stdout.splitlines():
        parts = line.strip().split("\t", maxsplit=1)
        if len(parts) != 2:
            continue
        container_id, name = parts
        if not container_id:
            continue
        rm_result = subprocess.run(["podman", "rm", "-f", container_id], check=False, capture_output=True, text=True)
        if rm_result.returncode == 0:
            removed.append(name or container_id)
    return removed


__all__ = [
    "ORCHESTRATOR_LABEL_KEY",
    "PodmanLaunchError",
    "PodmanLogStreamer",
    "PodmanRuntimeAdapter",
    "cleanup_orphan_containers",
]
