"""Pyxis-backed vLLM launcher for Slurm allocations."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterable, Mapping

from medarc_verifiers.orchestrate.runtime import RuntimeHandle, RuntimeLaunchError


class PyxisLaunchError(RuntimeLaunchError):
    """Raised when Pyxis launch fails."""


class ProcessLogStreamer:
    def __init__(self, process: subprocess.Popen[bytes], sink_path: str) -> None:
        self._process = process
        self._sink_path = sink_path
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="pyxis-log-streamer", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _run(self) -> None:
        stream = self._process.stdout
        if stream is None:
            return
        with open(self._sink_path, "ab") as handle:
            while not self._stop_event.is_set():
                chunk = stream.readline()
                if not chunk:
                    break
                handle.write(chunk)
                handle.flush()


class PyxisRuntimeAdapter:
    """Runtime adapter that launches vLLM via `srun --container-image ...`."""

    _DEFAULT_HYGIENE_FLAGS = {
        "--container-entrypoint": "--no-container-entrypoint",
        "--container-mount-home": "--no-container-mount-home",
    }

    def __init__(self) -> None:
        self._processes: dict[str, subprocess.Popen[bytes]] = {}

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
        del task_id, model_id, container_port, gpu_ids, labels
        command = self._build_srun_command(
            image=image,
            volume_mounts=volume_mounts,
            gpus_required=gpus_required,
            server_port=server_port,
            container_args=container_args,
            env=env,
            srun_extra_args=runtime_kwargs.get("srun_extra_args") or [],
        )
        proc_env = os.environ.copy()
        proc_env.update(env)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=proc_env,
            )
        except FileNotFoundError as exc:
            raise PyxisLaunchError("srun not found; Pyxis runtime requires Slurm client tools.") from exc
        except Exception as exc:
            raise PyxisLaunchError(str(exc)) from exc
        time.sleep(0.2)
        return_code = process.poll()
        if return_code is not None and return_code != 0:
            output = b""
            if process.stdout is not None:
                try:
                    output = process.stdout.read() or b""
                except Exception:
                    output = b""
            message = _classify_pyxis_error(output.decode("utf-8", errors="replace").strip())
            raise PyxisLaunchError(message)
        handle = RuntimeHandle(base_url=f"http://127.0.0.1:{server_port}/v1", identifier=str(process.pid))
        self._processes[handle.identifier] = process
        return handle

    def stream_logs(self, handle: RuntimeHandle, sink: Path) -> ProcessLogStreamer:
        process = self._processes.get(handle.identifier)
        if process is None:
            raise PyxisLaunchError(f"Unknown Pyxis runtime handle: {handle.identifier}")
        return ProcessLogStreamer(process, str(sink))

    def teardown(self, handle: RuntimeHandle) -> None:
        process = self._processes.pop(handle.identifier, None)
        if process is None:
            return
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def _build_srun_command(
        self,
        *,
        image: str,
        volume_mounts: list[str],
        gpus_required: int,
        server_port: int,
        container_args: list[str],
        env: Mapping[str, str],
        srun_extra_args: Iterable[str],
    ) -> list[str]:
        extra_args = [str(arg) for arg in srun_extra_args]
        command = [
            "srun",
            "--nodes=1",
            "--ntasks=1",
            f"--gpus-per-task={gpus_required}",
            f"--container-image={image}",
        ]
        if volume_mounts:
            command.append(f"--container-mounts={_render_container_mounts(volume_mounts)}")
        command.extend(_render_hygiene_flags(extra_args))
        # --container-env overrides container-image env with host values.
        # Vars from .env files are already in proc_env (srun inherits them).
        # We only need --container-env for vars also present on the host,
        # where the image might have a conflicting default.
        container_env_vars = sorted(
            name
            for name in env
            if os.environ.get(name) is not None or name in {"HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_TOKEN"}
        )
        if container_env_vars:
            command.append(f"--container-env={','.join(container_env_vars)}")
        command.extend(extra_args)
        command.extend(
            [
                "vllm",
                "serve",
                "--host",
                "127.0.0.1",
                "--port",
                str(server_port),
                *container_args,
            ]
        )
        return command


def _render_container_mounts(volume_mounts: list[str]) -> str:
    return ",".join(volume_mounts)


def _render_hygiene_flags(extra_args: list[str]) -> list[str]:
    extras = set(extra_args)
    defaults: list[str] = []
    for positive, negative in PyxisRuntimeAdapter._DEFAULT_HYGIENE_FLAGS.items():
        if positive in extras or negative in extras:
            continue
        defaults.append(negative)
    return defaults


def _classify_pyxis_error(output: str) -> str:
    lowered = output.lower()
    if "container support is not enabled" in lowered or "unknown option --container-image" in lowered:
        return "Pyxis container support is unavailable on this Slurm cluster."
    if "no such file" in lowered and ("squashfs" in lowered or "container image" in lowered):
        return f"Pyxis container image is unavailable: {output}" if output else "Pyxis container image is unavailable."
    if "enroot" in lowered and ("failed" in lowered or "not found" in lowered):
        return f"Pyxis/Enroot image setup failed: {output}" if output else "Pyxis/Enroot image setup failed."
    if output:
        return output
    return "Pyxis launch failed before server readiness."


__all__ = ["ProcessLogStreamer", "PyxisLaunchError", "PyxisRuntimeAdapter"]
