"""Run manifest helpers for the unified CLI."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from medarc_verifiers.cli_new._job_builder import ResolvedJob
from medarc_verifiers.cli_new._schemas import ModelConfigSchema
from medarc_verifiers.cli_new.utils.shared import compute_checksum, resolve_env_identifier_or
from medarc_verifiers.utils.pathing import project_root, to_project_relative

MANIFEST_FILENAME = "run_manifest.json"
PROJECT_ROOT = project_root()


class ManifestJobEntry(BaseModel):
    """Pydantic model describing a single manifest job entry."""

    model_config = ConfigDict(extra="allow")

    job_id: str
    job_name: str | None = None
    env_id: str | None = None
    model_id: str | None = None
    status: str = "pending"
    reason: str | None = None
    attempt: int = 0
    started_at: str | None = None
    ended_at: str | None = None
    duration_seconds: float | None = None
    results_dir: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    avg_reward: float | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None
    checksum: str
    config: dict[str, Any] = Field(default_factory=dict)
    seeds: dict[str, Any] | None = None


class RunManifestModel(BaseModel):
    """Root manifest payload persisted to disk."""

    model_config = ConfigDict(extra="allow")

    version: int = 1
    run_id: str
    name: str
    config_source: str
    config_checksum: str
    created_at: str
    updated_at: str
    restart_source: str | None = None
    jobs: list[ManifestJobEntry] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)


def timestamp() -> str:
    """Return an ISO8601 timestamp in UTC."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_snapshot_checksum(snapshot: Mapping[str, Any]) -> str:
    """Public helper to compute the checksum for a config snapshot."""
    sanitized = dict(snapshot)
    models = sanitized.get("models")
    if isinstance(models, Mapping):
        sanitized_models: dict[str, Any] = {}
        for model_id, payload in models.items():
            if isinstance(payload, Mapping):
                sanitized_models[str(model_id)] = {
                    key: value for key, value in payload.items() if key not in ModelConfigSchema.resume_tolerant_fields
                }
            else:
                sanitized_models[str(model_id)] = payload
        sanitized["models"] = sanitized_models
    return compute_checksum(sanitized)


def compute_job_checksum(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
) -> str:
    """Expose job checksum calculations for resume/regeneration workflows."""
    payload = _canonicalize_job_config(job, env_args=env_args, sampling_args=sampling_args)
    return compute_checksum(_drop_resume_tolerant_fields(payload))


def _canonicalize_job_config(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a normalized payload describing how the job will run."""
    model_payload = json.loads(job.model.model_dump_json(exclude_none=True))
    env_payload = json.loads(job.env.model_dump_json(exclude_none=True))
    env_payload["env_args"] = _to_jsonable(env_args)
    model_payload["sampling_args"] = _to_jsonable(sampling_args)
    return {
        "job_id": job.job_id,
        "job_name": job.name,
        "model": model_payload,
        "env": env_payload,
    }


def _drop_resume_tolerant_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    model_payload = cleaned.get("model")
    if isinstance(model_payload, Mapping):
        cleaned["model"] = {
            key: value for key, value in model_payload.items() if key not in ModelConfigSchema.resume_tolerant_fields
        }
    return cleaned


def _relativize_results_dir(value: str | Path, *, run_dir: Path) -> str:
    """Ensure results directories are stored relative to the project root."""
    candidate = Path(value)
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == "runs":
            candidate = (PROJECT_ROOT / candidate).resolve()
        else:
            candidate = (run_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return to_project_relative(candidate)


def _extract_seeds(env_args: Mapping[str, Any], sampling_args: Mapping[str, Any]) -> dict[str, Any] | None:
    """Capture seed-like values for easier debugging."""
    seeds: dict[str, Any] = {}
    for source in (env_args, sampling_args):
        for key, value in source.items():
            if "seed" in key.lower():
                seeds[key] = value
    return seeds or None


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary data to JSON-serializable structures (default=str)."""
    return json.loads(json.dumps(value, default=str))


def _drop_nones(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _drop_nones(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nones(v) for v in value]
    return value


def _resolve_env_identifier(job: ResolvedJob) -> str:
    return resolve_env_identifier_or(job.env, job.job_id)


def _resolve_model_identifier(job: ResolvedJob) -> str:
    mid = getattr(job.model, "id", None)
    if mid:
        return mid
    if getattr(job.model, "model", None):
        return job.model.model  # type: ignore[return-value]
    return job.job_id


def build_job_entry(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
    results_dir: str,
) -> ManifestJobEntry:
    """Build the manifest entry recorded for a job."""
    config_payload = _canonicalize_job_config(job, env_args=env_args, sampling_args=sampling_args)
    checksum = compute_checksum(config_payload)
    return ManifestJobEntry(
        job_id=job.job_id,
        job_name=job.name,
        env_id=_resolve_env_identifier(job),
        model_id=_resolve_model_identifier(job),
        status="pending",
        reason=None,
        attempt=0,
        started_at=None,
        ended_at=None,
        duration_seconds=None,
        results_dir=results_dir,
        artifacts=[],
        metrics={},
        avg_reward=None,
        num_examples=None,
        rollouts_per_example=None,
        checksum=checksum,
        config=config_payload,
        seeds=_extract_seeds(env_args, sampling_args),
    )


def _summarize_jobs(entries: Sequence[ManifestJobEntry]) -> dict[str, int]:
    counter = Counter((entry.status or "pending") for entry in entries)
    skipped = sum(1 for entry in entries if entry.reason in {"up_to_date", "skipped"})
    summary = {
        "total": len(entries),
        "pending": counter.get("pending", 0),
        "running": counter.get("running", 0),
        "completed": counter.get("completed", 0),
        "failed": counter.get("failed", 0),
        "skipped": skipped,
    }
    return summary


@dataclass
class RunManifest:
    """In-memory representation of a run manifest."""

    path: Path
    model: RunManifestModel
    persist: bool = True

    def __post_init__(self) -> None:
        self._jobs: list[ManifestJobEntry] = list(self.model.jobs)
        self.model.jobs = self._jobs
        self._index: dict[str, ManifestJobEntry] = {entry.job_id: entry for entry in self._jobs if entry.job_id}
        if not self.model.summary:
            self.model.summary = _summarize_jobs(self._jobs)

    @property
    def jobs(self) -> list[ManifestJobEntry]:
        return self._jobs

    @property
    def summary(self) -> Mapping[str, Any]:
        return self.model.summary

    @property
    def payload(self) -> dict[str, Any]:
        """Dictionary representation (back-compat)."""
        return self.model.model_dump()

    def job_entry(self, job_id: str) -> ManifestJobEntry | None:
        return self._index.get(job_id)

    @property
    def run_dir(self) -> Path:
        return self.path.parent

    def ensure_job(
        self,
        job: ResolvedJob,
        *,
        env_args: Mapping[str, Any],
        sampling_args: Mapping[str, Any],
        results_dir: Path,
    ) -> ManifestJobEntry:
        entry = self._index.get(job.job_id)
        normalized_results_dir = _relativize_results_dir(results_dir, run_dir=self.run_dir)
        if entry is None:
            entry = build_job_entry(
                job,
                env_args=env_args,
                sampling_args=sampling_args,
                results_dir=normalized_results_dir,
            )
            self._jobs.append(entry)
            self._index[job.job_id] = entry
            self._refresh_summary(save=False)
            return entry

        config_payload = _canonicalize_job_config(job, env_args=env_args, sampling_args=sampling_args)
        entry.config = config_payload
        entry.checksum = compute_checksum(config_payload)
        entry.env_id = entry.env_id or _resolve_env_identifier(job)
        entry.model_id = entry.model_id or _resolve_model_identifier(job)
        entry.results_dir = entry.results_dir or normalized_results_dir
        entry.seeds = entry.seeds or _extract_seeds(env_args, sampling_args)
        return entry

    def record_job_start(self, job_id: str) -> None:
        entry = self._index.get(job_id)
        if not entry:
            return
        entry.status = "running"
        entry.reason = None
        entry.started_at = timestamp()
        entry.attempt = int(entry.attempt or 0) + 1
        self._refresh_summary()

    def record_job_completion(
        self,
        job_id: str,
        *,
        duration_seconds: float,
        results_dir: Path,
        artifacts: Sequence[str],
        avg_reward: float | None,
        metrics: Mapping[str, Any],
        num_examples: int | None,
        rollouts_per_example: int | None,
    ) -> None:
        entry = self._index.get(job_id)
        if not entry:
            return
        entry.status = "completed"
        entry.reason = None
        entry.ended_at = timestamp()
        entry.duration_seconds = duration_seconds
        entry.results_dir = _relativize_results_dir(results_dir, run_dir=self.run_dir)
        entry.artifacts = list(artifacts)
        entry.avg_reward = avg_reward
        entry.metrics = dict(metrics)
        entry.num_examples = num_examples
        entry.rollouts_per_example = rollouts_per_example
        self._refresh_summary()

    def record_job_failure(self, job_id: str, *, error: str, duration_seconds: float | None = None) -> None:
        entry = self._index.get(job_id)
        if not entry:
            return
        entry.status = "failed"
        entry.reason = error
        entry.ended_at = timestamp()
        entry.duration_seconds = duration_seconds
        self._refresh_summary()

    def record_job_skip(
        self,
        job_id: str,
        *,
        reason: str,
        results_dir: str | Path | None = None,
        source_entry: Mapping[str, Any] | None = None,
    ) -> None:
        entry = self._index.get(job_id)
        if not entry:
            return
        entry.status = "completed"
        entry.reason = reason
        entry.ended_at = entry.ended_at or timestamp()

        def _maybe_get(source: Mapping[str, Any] | ManifestJobEntry, key: str) -> Any:
            if isinstance(source, Mapping):
                return source.get(key)
            return getattr(source, key)

        if source_entry:
            is_mapping = isinstance(source_entry, Mapping)
            for key in (
                "duration_seconds",
                "avg_reward",
                "metrics",
                "num_examples",
                "rollouts_per_example",
                "artifacts",
            ):
                if is_mapping:
                    if key in source_entry:
                        setattr(entry, key, source_entry[key])
                else:
                    setattr(entry, key, getattr(source_entry, key))
        if results_dir:
            entry.results_dir = _relativize_results_dir(results_dir, run_dir=self.run_dir)
        self._refresh_summary()

    def _refresh_summary(self, *, save: bool = True) -> None:
        self.model.summary = _summarize_jobs(self._jobs)
        self.model.updated_at = timestamp()
        if save:
            self.save()

    def save(self) -> None:
        if not self.persist:
            return
        tmp_path = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.model.model_dump(exclude_none=True), handle, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    @classmethod
    def load(cls, path: Path, *, persist: bool = True) -> RunManifest:
        if not path.exists():
            raise FileNotFoundError(f"Run manifest '{path}' not found.")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        model = RunManifestModel.model_validate(payload)
        return cls(path=path, model=model, persist=persist)

    @classmethod
    def create(
        cls,
        *,
        run_dir: Path,
        run_id: str,
        run_name: str,
        config_source: Path,
        config_checksum: str,
        jobs: Sequence[ResolvedJob],
        env_args_map: Mapping[str, Mapping[str, Any]],
        sampling_args_map: Mapping[str, Mapping[str, Any]],
        persist: bool = True,
        restart_source: str | None = None,
    ) -> RunManifest:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_FILENAME
        payload: Mapping[str, Any] = {
            "version": 1,
            "run_id": run_id,
            "name": run_name,
            "config_source": str(config_source),
            "config_checksum": config_checksum,
            "created_at": timestamp(),
            "updated_at": timestamp(),
            "restart_source": restart_source,
            "jobs": [],
            "summary": {},
        }
        model = RunManifestModel.model_validate(payload)
        manifest = cls(path=path, model=model, persist=persist)
        for job in jobs:
            env_args = env_args_map[job.job_id]
            sampling_args = sampling_args_map[job.job_id]
            manifest.ensure_job(
                job,
                env_args=env_args,
                sampling_args=sampling_args,
                results_dir=(run_dir / job.job_id),
            )
        manifest._refresh_summary(save=True)
        return manifest


__all__ = [
    "MANIFEST_FILENAME",
    "RunManifest",
    "RunManifestModel",
    "ManifestJobEntry",
    "build_job_entry",
    "compute_job_checksum",
    "compute_snapshot_checksum",
    "timestamp",
]
