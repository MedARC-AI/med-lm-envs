"""Docker-backed vLLM launcher and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import asyncio
import json
import re
import threading
import time

import httpx

from medarc_verifiers.orchestrate.runtime import RuntimeHandle, RuntimeLaunchError
from medarc_verifiers.orchestrate.vllm_args import build_container_args, normalize_volume_mounts


class DockerLaunchError(RuntimeLaunchError):
    """Raised when container launch fails."""


class ReadinessError(RuntimeError):
    """Raised when readiness checks fail."""


@dataclass(frozen=True)
class ReadinessResult:
    ready: bool
    elapsed_s: float
    attempts: int
    last_error: str | None = None


ORCHESTRATOR_LABEL_KEY = "orchestrator.managed"


def normalize_volumes(volumes: object) -> dict[str, dict[str, str]]:
    mounts: dict[str, dict[str, str]] = {}
    try:
        normalized_mounts = normalize_volume_mounts(volumes)
    except ValueError as exc:
        raise DockerLaunchError(str(exc)) from exc
    for entry in normalized_mounts:
        host, container_path, mode = entry.split(":")
        mounts[host] = {"bind": container_path, "mode": mode}
    return mounts


def sanitize_container_name(value: str, *, max_len: int = 128) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    if not cleaned:
        cleaned = "task"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-.")
    return cleaned


def create_and_start_container(
    *,
    image: str,
    name: str,
    container_port: int,
    host_port: int,
    env: Mapping[str, str],
    volumes: object,
    ipc_mode: str | None,
    gpu_ids: Iterable[int],
    command: list[str],
    labels: Mapping[str, str],
):
    try:
        import docker
    except Exception as exc:  # pragma: no cover - dependency import varies
        raise DockerLaunchError("docker package is required for container launch.") from exc
    try:
        from docker.types import DeviceRequest
    except Exception as exc:  # pragma: no cover - dependency import varies
        raise DockerLaunchError("docker.types.DeviceRequest is required for GPU requests.") from exc

    # Docker-py uses requests with a default read timeout of 60s; under heavy daemon load
    # container creation can exceed that and still succeed server-side, leaving a container
    # behind that a retry then conflicts with.
    client = docker.from_env(timeout=600)

    def remove_existing_if_safe() -> bool:
        try:
            existing = client.containers.get(name)
        except Exception:
            return False
        try:
            existing.reload()
        except Exception:
            pass
        existing_labels = getattr(existing, "labels", None) or {}
        if existing_labels.get(ORCHESTRATOR_LABEL_KEY) != "true":
            return False
        for key, value in labels.items():
            if existing_labels.get(key) != value:
                return False
        status = getattr(existing, "status", None)
        if status == "running":
            raise DockerLaunchError(f"Container name {name!r} is already running (id={getattr(existing, 'id', '?')}).")
        try:
            existing.remove(v=True, force=True)
            return True
        except Exception:
            return False

    def get_existing_if_owned():
        try:
            existing = client.containers.get(name)
        except Exception:
            return None
        try:
            existing.reload()
        except Exception:
            pass
        existing_labels = getattr(existing, "labels", None) or {}
        if existing_labels.get(ORCHESTRATOR_LABEL_KEY) != "true":
            return None
        for key, value in labels.items():
            if existing_labels.get(key) != value:
                return None
        return existing

    gpu_id_list = [int(gpu) for gpu in gpu_ids]
    device_request = DeviceRequest(
        device_ids=[str(gpu) for gpu in gpu_id_list],
        capabilities=[["gpu"]],
    )
    container_create_kwargs = {
        "image": image,
        "name": name,
        "command": command,
        "ports": {f"{container_port}/tcp": ("127.0.0.1", host_port)},
        "environment": dict(env),
        "volumes": normalize_volumes(volumes),
        "ipc_mode": ipc_mode,
        "labels": {ORCHESTRATOR_LABEL_KEY: "true", **dict(labels)},
        "device_requests": [device_request],
        "detach": True,
    }
    remove_existing_if_safe()
    try:
        container = client.containers.create(**container_create_kwargs)
    except Exception as exc:
        message = str(exc)
        lower_message = message.lower()
        if "read timed out" in lower_message or "timeout" in lower_message:
            existing = get_existing_if_owned()
            if existing is not None:
                container = existing
            else:
                raise DockerLaunchError(message) from exc
        elif "already in use" in message.lower() or "conflict" in message.lower():
            if remove_existing_if_safe():
                container = client.containers.create(**container_create_kwargs)
            else:
                raise DockerLaunchError(message) from exc
        elif "No such image" in message or "not found" in message.lower():
            try:
                client.images.pull(image)
            except Exception as pull_exc:
                raise DockerLaunchError(f"Failed to pull image {image!r}: {pull_exc}") from pull_exc
            container = client.containers.create(**container_create_kwargs)
        else:
            raise
    try:
        container.start()
    except Exception:
        try:
            try:
                container.reload()
            except Exception:
                pass
            if getattr(container, "status", None) == "running":
                return container
            container.remove(v=True, force=True)
        except Exception:
            pass
        raise
    return container


def stream_container_logs(container, sink_path: str) -> None:
    with open(sink_path, "w", encoding="utf-8") as handle:
        for chunk in container.logs(stream=True, follow=True):
            handle.write(chunk.decode("utf-8", errors="replace"))
            handle.flush()


class ContainerLogStreamer:
    def __init__(self, container, sink_path: str) -> None:
        self._container = container
        self._sink_path = sink_path
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="container-log-streamer", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop_event.set()
        self._close_stream()
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        try:
            self._stream = self._container.logs(stream=True, follow=True)
            with open(self._sink_path, "a", encoding="utf-8") as handle:
                for chunk in self._stream:
                    if self._stop_event.is_set():
                        break
                    handle.write(chunk.decode("utf-8", errors="replace"))
                    handle.flush()
        finally:
            self._close_stream()

    def _close_stream(self) -> None:
        stream = self._stream
        if stream is None:
            return
        close = getattr(stream, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


class DockerRuntimeAdapter:
    """Runtime adapter that launches vLLM in Docker containers."""

    def __init__(self) -> None:
        self._containers: dict[str, object] = {}

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
        container = create_and_start_container(
            image=image,
            name=str(runtime_kwargs.get("name", "")),
            container_port=container_port,
            host_port=server_port,
            env=env,
            volumes=volume_mounts,
            ipc_mode=runtime_kwargs.get("ipc_mode"),
            gpu_ids=gpu_ids,
            command=container_args,
            labels=labels,
        )
        handle = RuntimeHandle(base_url=f"http://127.0.0.1:{server_port}/v1", identifier=str(container.id))
        self._containers[handle.identifier] = container
        return handle

    def stream_logs(self, handle: RuntimeHandle, sink: Path) -> ContainerLogStreamer:
        container = self._containers.get(handle.identifier)
        if container is None:
            raise DockerLaunchError(f"Unknown Docker runtime handle: {handle.identifier}")
        return ContainerLogStreamer(container, str(sink))

    def teardown(self, handle: RuntimeHandle) -> None:
        container = self._containers.pop(handle.identifier, None)
        if container is None:
            return
        try:
            container.wait(timeout=1)
        except Exception:
            pass
        try:
            container.stop(timeout=10)
        finally:
            container.remove(v=True, force=True)


def wait_for_readiness(
    base_url: str,
    *,
    model_id: str | None = None,
    timeout_s: float = 1800,
    poll_interval_s: float = 5.0,
) -> ReadinessResult:
    start = time.monotonic()
    attempts = 0
    last_error: str | None = None
    with httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        while True:
            attempts += 1
            try:
                resp = client.get(f"{base_url}/models")
                if resp.status_code == 200:
                    payload = resp.json()
                    if _models_ok(payload, model_id=model_id):
                        if _warmup(client, base_url, model_id=model_id):
                            elapsed = time.monotonic() - start
                            return ReadinessResult(ready=True, elapsed_s=elapsed, attempts=attempts)
                else:
                    last_error = f"GET /models {resp.status_code}"
            except Exception as exc:
                last_error = str(exc)
            if time.monotonic() - start > timeout_s:
                return ReadinessResult(
                    ready=False,
                    elapsed_s=time.monotonic() - start,
                    attempts=attempts,
                    last_error=last_error,
                )
            time.sleep(poll_interval_s)


def _models_ok(payload: object, *, model_id: str | None) -> bool:
    if not isinstance(payload, Mapping):
        return False
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False
    if model_id is None:
        return True
    for entry in data:
        if isinstance(entry, Mapping) and entry.get("id") == model_id:
            return True
    return False


def _warmup(client: httpx.Client, base_url: str, *, model_id: str | None) -> bool:
    payload = {"model": model_id or "unknown", "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    try:
        resp = client.post(f"{base_url}/chat/completions", json=payload)
        return resp.status_code == 200
    except Exception:
        return False


async def wait_for_readiness_async(
    base_url: str,
    *,
    model_id: str | None = None,
    timeout_s: float = 1800,
    poll_interval_s: float = 5.0,
) -> ReadinessResult:
    start = time.monotonic()
    attempts = 0
    last_error: str | None = None
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while True:
            attempts += 1
            try:
                resp = await client.get(f"{base_url}/models")
                if resp.status_code == 200:
                    payload = resp.json()
                    if _models_ok(payload, model_id=model_id):
                        if await _warmup_async(client, base_url, model_id=model_id):
                            elapsed = time.monotonic() - start
                            return ReadinessResult(ready=True, elapsed_s=elapsed, attempts=attempts)
                else:
                    last_error = f"GET /models {resp.status_code}"
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            if time.monotonic() - start > timeout_s:
                return ReadinessResult(
                    ready=False,
                    elapsed_s=time.monotonic() - start,
                    attempts=attempts,
                    last_error=last_error,
                )
            await asyncio.sleep(poll_interval_s)


async def _warmup_async(client: httpx.AsyncClient, base_url: str, *, model_id: str | None) -> bool:
    payload = {"model": model_id or "unknown", "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    try:
        resp = await client.post(f"{base_url}/chat/completions", json=payload)
        return resp.status_code == 200
    except Exception:  # noqa: BLE001
        return False


def write_container_request(path: str, payload: Mapping[str, object]) -> None:
    from pathlib import Path

    request_path = Path(path)
    request_path.parent.mkdir(parents=True, exist_ok=True)
    with open(request_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def cleanup_orphan_containers(run_id: str | None = None) -> list[str]:
    try:
        import docker
    except Exception as exc:  # pragma: no cover - dependency import varies
        raise DockerLaunchError("docker package is required for container cleanup.") from exc
    client = docker.from_env()
    labels = [f"{ORCHESTRATOR_LABEL_KEY}=true"]
    if run_id:
        labels.append(f"orchestrator.run_id={run_id}")
    containers = client.containers.list(all=True, filters={"label": labels})
    removed: list[str] = []
    for container in containers:
        try:
            if container.status == "running":
                container.stop(timeout=10)
            container.remove(v=True, force=True)
            removed.append(container.name)
        except Exception:
            continue
    return removed


__all__ = [
    "DockerLaunchError",
    "DockerRuntimeAdapter",
    "ReadinessError",
    "ReadinessResult",
    "ORCHESTRATOR_LABEL_KEY",
    "build_container_args",
    "ContainerLogStreamer",
    "create_and_start_container",
    "cleanup_orphan_containers",
    "normalize_volume_mounts",
    "normalize_volumes",
    "sanitize_container_name",
    "stream_container_logs",
    "wait_for_readiness",
    "wait_for_readiness_async",
    "write_container_request",
]
