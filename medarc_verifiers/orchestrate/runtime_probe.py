"""Fast runtime availability probes for orchestration launch resolution."""

from __future__ import annotations

import os
import shutil
import subprocess


def docker_available() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker CLI not found"
    return _probe(["docker", "info"], label="docker info")


def podman_available() -> tuple[bool, str]:
    if shutil.which("podman") is None:
        return False, "podman CLI not found"
    return _probe(["podman", "info"], label="podman info")


def pyxis_available_inside_slurm() -> tuple[bool, str]:
    if not _inside_slurm_allocation():
        return False, "not inside a Slurm allocation"
    if shutil.which("srun") is None:
        return False, "srun CLI not found"
    result = subprocess.run(["srun", "--help"], text=True, capture_output=True, timeout=5, check=False)
    text = f"{result.stdout}\n{result.stderr}".lower()
    if "container-image" in text or "pyxis" in text:
        return True, "srun appears to support Pyxis"
    return False, "srun help did not expose Pyxis container options"


def _inside_slurm_allocation() -> bool:
    allocation_keys = ("SLURM_JOB_ID", "SLURM_STEP_ID", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE")
    visible_keys = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")
    has_allocation = any(os.environ.get(key) for key in allocation_keys)
    has_visible_gpus = any(_has_visible_devices(os.environ.get(key)) for key in visible_keys + allocation_keys)
    return has_allocation and has_visible_gpus


def _has_visible_devices(value: str | None) -> bool:
    if value is None:
        return False
    text = value.strip().lower()
    return bool(text and text not in {"none", "void", "novisibledevices"})


def _probe(command: list[str], *, label: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=5, check=False)
    except Exception as exc:  # pragma: no cover - platform/runtime dependent
        return False, f"{label} failed: {exc}"
    if result.returncode == 0:
        return True, f"{label} succeeded"
    detail = (result.stderr or result.stdout or "").strip()
    return False, f"{label} exited {result.returncode}: {detail}"


__all__ = ["docker_available", "podman_available", "pyxis_available_inside_slurm"]
