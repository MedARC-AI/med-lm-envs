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
    original_job_config_path: str
    effective_job_config_path: str
    patched_job_config_path: str | None
    tp_size: int
    dp_size: int
    effective_gpus: int
    inner_run_id: str
    restart_source: str | None
    restart_strategy: str | None
    script_path: str
    generated_dependency: str | None
    base_dependency: str | None
    predecessor_task_id: str | None
    chain_index: int
    submission_order: int
    job_name: str
    account: str | None = None
    slurm_job_id: str | None = None
    state: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SlurmTaskEntry":
        return cls(**payload)


@dataclass
class SlurmBundleManifest:
    run_id: str
    bundle_root: str
    node_gpus: int
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    entries: list[SlurmTaskEntry] = field(default_factory=list)

    def touch(self) -> None:
        self.updated_at = _now()

    def entry_map(self) -> dict[str, SlurmTaskEntry]:
        return {entry.task_id: entry for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "bundle_root": self.bundle_root,
            "node_gpus": self.node_gpus,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SlurmBundleManifest":
        return cls(
            run_id=str(payload["run_id"]),
            bundle_root=str(payload["bundle_root"]),
            node_gpus=int(payload["node_gpus"]),
            created_at=str(payload.get("created_at") or _now()),
            updated_at=str(payload.get("updated_at") or _now()),
            entries=[SlurmTaskEntry.from_dict(dict(entry)) for entry in payload.get("entries", [])],
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
    "SlurmTaskEntry",
    "load_bundle_manifest",
    "write_bundle_manifest",
]
