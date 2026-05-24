"""Manifest persistence for Slurm submission bundles."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp-{uuid.uuid4().hex}")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


@dataclass
class SlurmTaskEntry:
    run_id: str
    task_id: str
    task_slug: str
    suite_path: str
    suite_checksum: str
    target_endpoint_id: str
    generated_eval_config_path: str
    bundled_eval_config_checksum: str
    state_path: str
    gpus: int
    allocated_gpus: int
    tensor_parallel_size: int
    data_parallel_size: int
    vllm_world_size: int
    script_path: str
    generated_dependency: str | None
    base_dependency: str | None
    submission_order: int
    job_name: str
    account: str | None = None
    slurm_job_id: str | None = None
    state: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SlurmTaskEntry":
        return cls(
            run_id=str(payload["run_id"]),
            task_id=str(payload["task_id"]),
            task_slug=str(payload["task_slug"]),
            suite_path=str(payload["suite_path"]),
            suite_checksum=str(payload["suite_checksum"]),
            target_endpoint_id=str(payload["target_endpoint_id"]),
            generated_eval_config_path=str(payload["generated_eval_config_path"]),
            bundled_eval_config_checksum=str(payload["bundled_eval_config_checksum"]),
            state_path=str(payload["state_path"]),
            gpus=int(payload["gpus"]),
            allocated_gpus=int(payload["allocated_gpus"]),
            tensor_parallel_size=int(payload["tensor_parallel_size"]),
            data_parallel_size=int(payload["data_parallel_size"]),
            vllm_world_size=int(payload["vllm_world_size"]),
            script_path=str(payload["script_path"]),
            generated_dependency=(
                str(payload["generated_dependency"]) if payload.get("generated_dependency") is not None else None
            ),
            base_dependency=str(payload["base_dependency"]) if payload.get("base_dependency") is not None else None,
            submission_order=int(payload["submission_order"]),
            job_name=str(payload["job_name"]),
            account=str(payload["account"]) if payload.get("account") is not None else None,
            slurm_job_id=str(payload["slurm_job_id"]) if payload.get("slurm_job_id") is not None else None,
            state=str(payload.get("state") or "pending"),
        )


@dataclass
class SlurmLifecycleEntry:
    run_id: str
    task_id: str
    task_slug: str
    phase: str
    script_path: str
    generated_dependency: str | None
    base_dependency: str | None
    submission_order: int
    job_name: str
    cpus: int
    account: str | None = None
    slurm_job_id: str | None = None
    state: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SlurmLifecycleEntry":
        return cls(
            run_id=str(payload["run_id"]),
            task_id=str(payload["task_id"]),
            task_slug=str(payload["task_slug"]),
            phase=str(payload["phase"]),
            script_path=str(payload["script_path"]),
            generated_dependency=(
                str(payload["generated_dependency"]) if payload.get("generated_dependency") is not None else None
            ),
            base_dependency=str(payload["base_dependency"]) if payload.get("base_dependency") is not None else None,
            submission_order=int(payload["submission_order"]),
            job_name=str(payload["job_name"]),
            cpus=int(payload["cpus"]),
            account=str(payload["account"]) if payload.get("account") is not None else None,
            slurm_job_id=str(payload["slurm_job_id"]) if payload.get("slurm_job_id") is not None else None,
            state=str(payload.get("state") or "pending"),
        )


@dataclass
class SlurmBundleManifest:
    run_id: str
    bundle_root: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    entries: list[SlurmTaskEntry] = field(default_factory=list)
    lifecycle_entries: list[SlurmLifecycleEntry] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now()

    def eval_entry_map(self) -> dict[str, SlurmTaskEntry]:
        return {entry.task_id: entry for entry in self.entries}

    def lifecycle_entry_map(self) -> dict[tuple[str, str], SlurmLifecycleEntry]:
        return {(entry.task_id, entry.phase): entry for entry in self.lifecycle_entries}

    def entry_map(self) -> dict[str, SlurmTaskEntry]:
        return self.eval_entry_map()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "bundle_root": self.bundle_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entries": [entry.to_dict() for entry in self.entries],
            "lifecycle_entries": [entry.to_dict() for entry in self.lifecycle_entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SlurmBundleManifest":
        return cls(
            run_id=str(payload["run_id"]),
            bundle_root=str(payload["bundle_root"]),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
            entries=[SlurmTaskEntry.from_dict(dict(entry)) for entry in payload.get("entries", [])],
            lifecycle_entries=[
                SlurmLifecycleEntry.from_dict(dict(entry)) for entry in payload.get("lifecycle_entries", [])
            ],
        )


def load_bundle_manifest(path: Path) -> SlurmBundleManifest:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Slurm manifest must be a mapping: {path}")
    return SlurmBundleManifest.from_dict(payload)


def write_bundle_manifest(path: Path, manifest: SlurmBundleManifest) -> None:
    manifest.touch()
    _write_json_atomic(path, manifest.to_dict())


__all__ = [
    "SlurmBundleManifest",
    "SlurmLifecycleEntry",
    "SlurmTaskEntry",
    "load_bundle_manifest",
    "write_bundle_manifest",
]
