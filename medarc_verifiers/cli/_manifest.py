"""Legacy run manifest schemas retained for process discovery.

The YAML benchmark runner no longer writes manifests, but `medarc-eval process`
still supports old `runs/raw/<run_id>/run_manifest.json` directories during the
transition. Keep this module to the schema pieces needed for that reader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

MANIFEST_VERSION = 3
SUPPORTED_MANIFEST_VERSIONS = {MANIFEST_VERSION}
MANIFEST_FILENAME = "run_manifest.json"


class ManifestJobEntry(BaseModel):
    """Pydantic model describing a single legacy manifest job entry."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    env_id: str | None = None
    model_id: str | None = None
    env_template_id: str
    env_variant_id: str
    env_args: dict[str, Any]
    sampling_args: dict[str, Any] | None = None
    status: str = "pending"
    reason: str | None = None
    attempt: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    results_dir: str | None = None
    results_relpath: str | None = None
    metadata_relpath: str | None = None
    row_count: int | None = None
    metrics: dict[str, Any] | None = None
    avg_reward: float | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None


class RunManifestModel(BaseModel):
    """Root legacy manifest payload persisted by the retired YAML runner."""

    model_config = ConfigDict(extra="allow")

    version: int = MANIFEST_VERSION
    run_id: str
    name: str
    config_source: str
    config_checksum: str
    created_at: str
    updated_at: str
    restart_source: str | None = None
    artifacts_root: str = "."
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_templates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    jobs: list[ManifestJobEntry] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_version(self) -> RunManifestModel:
        if self.version not in SUPPORTED_MANIFEST_VERSIONS:
            msg = (
                f"Manifest version {self.version} is not supported; "
                f"expected one of {sorted(SUPPORTED_MANIFEST_VERSIONS)}."
            )
            raise ValueError(msg)
        return self


def _require_manifest_v3(payload: Mapping[str, Any], *, path: Path | None = None) -> None:
    """Raise when a legacy manifest payload is not version 3."""
    version = payload.get("version")
    if version != MANIFEST_VERSION:
        location = f" at {path}" if path else ""
        raise ValueError(f"Unsupported legacy run manifest version {version!r}{location}; expected 3.")


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_VERSION",
    "SUPPORTED_MANIFEST_VERSIONS",
    "ManifestJobEntry",
    "RunManifestModel",
    "_require_manifest_v3",
]
