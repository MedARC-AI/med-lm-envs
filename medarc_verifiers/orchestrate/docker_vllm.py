"""Docker-backed vLLM launcher and readiness checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping
import asyncio
import json
import re
import threading
import time

import httpx


class DockerLaunchError(RuntimeError):
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


def build_container_args(model_id: str, *, tensor_parallel_size: int | None, serve: Mapping[str, object]) -> list[str]:
    _validate_serve_config(serve)
    args = ["--model", model_id]
    if tensor_parallel_size and tensor_parallel_size > 1:
        args.extend(["--tensor-parallel-size", str(tensor_parallel_size)])
    args.extend(_render_serve_flags(serve))
    return args


def _render_serve_flags(serve: Mapping[str, object]) -> list[str]:
    flags: list[str] = []
    scalar_map = {
        "dtype": "--dtype",
        "max_model_len": "--max-model-len",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_num_seqs": "--max-num-seqs",
        "max_num_batched_tokens": "--max-num-batched-tokens",
        "tokenizer_mode": "--tokenizer_mode",
        "config_format": "--config_format",
        "load_format": "--load_format",
        "reasoning_parser": "--reasoning-parser",
        "reasoning_parser_plugin": "--reasoning-parser-plugin",
        "tool_call_parser": "--tool-call-parser",
        "mamba_ssm_cache_dtype": "--mamba_ssm_cache_dtype",
        "quantization": "--quantization",
        "chat_template": "--chat-template",
    }
    for key, flag in scalar_map.items():
        if key in serve and serve[key] is not None:
            flags.extend([flag, str(serve[key])])
    bool_map = {
        "async_scheduling": "--async-scheduling",
        "enable_prefix_caching": "--enable-prefix-caching",
        "enable_chunked_prefill": "--enable-chunked-prefill",
        "trust_remote_code": "--trust-remote-code",
        "enable_expert_parallel": "--enable-expert-parallel",
        "enable_auto_tool_choice": "--enable-auto-tool-choice",
    }
    for key, flag in bool_map.items():
        if serve.get(key) is True:
            flags.append(flag)
    limit_mm = serve.get("limit_mm_per_prompt")
    if isinstance(limit_mm, Mapping):
        for sub_key in ("image", "video"):
            if sub_key in limit_mm and limit_mm[sub_key] is not None:
                flags.extend([f"--limit-mm-per-prompt.{sub_key}", str(limit_mm[sub_key])])
    return flags


def _validate_serve_config(serve: Mapping[str, object]) -> None:
    scalar_keys = {
        "dtype",
        "max_model_len",
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "tokenizer_mode",
        "config_format",
        "load_format",
        "reasoning_parser",
        "reasoning_parser_plugin",
        "tool_call_parser",
        "mamba_ssm_cache_dtype",
        "quantization",
        "chat_template",
    }
    bool_keys = {
        "async_scheduling",
        "enable_prefix_caching",
        "enable_chunked_prefill",
        "trust_remote_code",
        "enable_expert_parallel",
        "enable_auto_tool_choice",
    }
    allowed = scalar_keys | bool_keys | {"limit_mm_per_prompt"}
    unknown = sorted(set(serve.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown vLLM serve keys: {unknown}")
    limit_mm = serve.get("limit_mm_per_prompt")
    if limit_mm is None:
        return
    if not isinstance(limit_mm, Mapping):
        raise ValueError("limit_mm_per_prompt must be a mapping.")
    unknown_subkeys = sorted(set(limit_mm.keys()) - {"image", "video"})
    if unknown_subkeys:
        raise ValueError(f"Unknown limit_mm_per_prompt keys: {unknown_subkeys}")


def normalize_volumes(volumes: object) -> dict[str, dict[str, str]]:
    if volumes is None:
        return {}
    if isinstance(volumes, Mapping):
        return dict(volumes)
    if not isinstance(volumes, list):
        raise DockerLaunchError("orchestrate.vllm-docker.volumes must be a list of mount strings or a mapping.")
    mounts: dict[str, dict[str, str]] = {}
    for entry in volumes:
        if not entry:
            continue
        if not isinstance(entry, str):
            raise DockerLaunchError("orchestrate.vllm-docker.volumes entries must be strings like host:container[:mode].")
        parts = entry.split(":")
        if len(parts) < 2 or len(parts) > 3:
            raise DockerLaunchError(f"Invalid volume mount: {entry!r} (expected host:container[:mode])")
        host = parts[0].strip()
        container_path = parts[1].strip()
        mode = parts[2].strip() if len(parts) == 3 else "rw"
        if not host or not container_path:
            raise DockerLaunchError(f"Invalid volume mount: {entry!r} (host and container path required)")
        if mode not in {"ro", "rw"}:
            raise DockerLaunchError(f"Invalid volume mount mode: {entry!r} (expected ro/rw)")
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
            raise DockerLaunchError(
                f"Container name {name!r} is already running (id={getattr(existing, 'id', '?')})."
            )
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
        if "already in use" in message.lower() or "conflict" in message.lower():
            if remove_existing_if_safe():
                container = client.containers.create(**container_create_kwargs)
            else:
                raise DockerLaunchError(message) from exc
        if "No such image" in message or "not found" in message.lower():
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
            with open(self._sink_path, "w", encoding="utf-8") as handle:
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
    "ReadinessError",
    "ReadinessResult",
    "ORCHESTRATOR_LABEL_KEY",
    "build_container_args",
    "ContainerLogStreamer",
    "create_and_start_container",
    "cleanup_orphan_containers",
    "normalize_volumes",
    "sanitize_container_name",
    "stream_container_logs",
    "wait_for_readiness",
    "wait_for_readiness_async",
    "write_container_request",
]
