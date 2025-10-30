"""Pydantic models for MedARC export artifacts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ManifestJob(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    job_name: str | None = None
    model_id: str | None = None
    env_id: str | None = None
    env_overrides: dict[str, Any] = Field(default_factory=dict)
    sampling_overrides: dict[str, Any] = Field(default_factory=dict)
    results_dir: str | None = None


class RunManifestModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    name: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    config_source: str | None = None
    config_checksum: str | None = None
    config_snapshot: dict[str, Any] | None = None
    jobs: list[ManifestJob] = Field(default_factory=list)


class SummaryEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    job_id: str
    status: str | None = None
    duration_seconds: float | None = None
    error: str | None = None


class RunSummaryModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    jobs: list[SummaryEntry] = Field(default_factory=list)


class MetadataModel(BaseModel):
    model_config = ConfigDict(extra="allow")

    env_id: str | None = None
    model: str | None = None
    env_args: dict[str, Any] = Field(default_factory=dict)
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    sampling_args: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ManifestJob",
    "MetadataModel",
    "RunManifestModel",
    "RunSummaryModel",
    "SummaryEntry",
]
