"""Run manifest helpers for the unified CLI."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._schemas import ModelConfigSchema
from medarc_verifiers.cli.utils.shared import compute_checksum, resolve_env_identifier_or
from medarc_verifiers.utils.pathing import project_root, to_project_relative

MANIFEST_FILENAME = "run_manifest.json"
PROJECT_ROOT = project_root()
MANIFEST_VERSION = 2


class ManifestJobEntry(BaseModel):
    """Pydantic model describing a single manifest job entry."""

    model_config = ConfigDict(extra="ignore")

    job_id: str
    job_name: str | None = None
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
    artifacts: list[str] | None = None
    metrics: dict[str, Any] | None = None
    avg_reward: float | None = None
    num_examples: int | None = None
    rollouts_per_example: int | None = None


class RunManifestModel(BaseModel):
    """Root manifest payload persisted to disk."""

    model_config = ConfigDict(extra="allow")

    version: int = MANIFEST_VERSION
    run_id: str
    name: str
    config_source: str
    config_checksum: str
    created_at: str
    updated_at: str
    restart_source: str | None = None
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)
    env_templates: dict[str, dict[str, Any]] = Field(default_factory=dict)
    jobs: list[ManifestJobEntry] = Field(default_factory=list)
    summary: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_version(self) -> RunManifestModel:
        if self.version != MANIFEST_VERSION:
            msg = f"Manifest version {self.version} is not supported; expected {MANIFEST_VERSION}."
            raise ValueError(msg)
        return self


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


def _to_jsonable(value: Any) -> Any:
    """Convert arbitrary data to JSON-serializable structures (default=str)."""
    return json.loads(json.dumps(value, default=str))


def _normalize_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    def _drop(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _drop(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [_drop(v) for v in value]
        return value

    return _drop(_to_jsonable(payload))


def _require_manifest_v2(payload: Mapping[str, Any], *, path: Path | None = None) -> None:
    version = payload.get("version")
    if version != MANIFEST_VERSION:
        location = f" '{path}'" if path else ""
        msg = f"Manifest{location} uses version {version}; expected {MANIFEST_VERSION}."
        raise ValueError(msg)


def _sanitize_model_payload(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model_payload.items() if key not in ModelConfigSchema.resume_tolerant_fields}


def _effective_sampling_args(entry: ManifestJobEntry, model_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if entry.sampling_args is not None:
        return _normalize_payload(entry.sampling_args)
    return _normalize_payload(model_payload.get("sampling_args") or {})


def manifest_job_signature(manifest: RunManifestModel, entry: ManifestJobEntry) -> dict[str, Any]:
    model_payload = _normalize_payload(manifest.models.get(entry.model_id or "", {}) or {})
    env_template = _normalize_payload(manifest.env_templates.get(entry.env_template_id, {}) or {})
    effective_env = dict(env_template)
    effective_env["module"] = entry.env_id or env_template.get("module")
    effective_env["id"] = entry.env_variant_id
    effective_env["env_args"] = entry.env_args
    signature = {
        "model": _sanitize_model_payload(model_payload),
        "env": effective_env,
        "sampling_args": _effective_sampling_args(entry, model_payload),
    }
    return _normalize_payload(signature)


def resolved_job_signature(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
) -> dict[str, Any]:
    model_payload = _normalize_payload(json.loads(job.model.model_dump_json(exclude_none=True)))
    env_payload = _normalize_payload(json.loads(job.env.model_dump_json(exclude_none=True)))
    env_template_payload = _build_env_template_payload(env_payload)
    env_id = env_template_payload.get("module") or _resolve_env_identifier(job)
    if "module" not in env_template_payload:
        env_template_payload["module"] = env_id
    env_variant_id = env_payload.get("id") or job.job_id
    sampling_override = _sampling_args_override(sampling_args=sampling_args, model_payload=model_payload)
    effective_env = {
        **env_template_payload,
        "module": env_id,
        "id": env_variant_id,
        "env_args": _normalize_payload(env_args),
    }
    signature = {
        "model": _sanitize_model_payload(model_payload),
        "env": effective_env,
        "sampling_args": sampling_override or model_payload.get("sampling_args") or {},
    }
    return _normalize_payload(signature)


def _maybe_store_results_dir(value: str | Path | None, *, run_dir: Path, job_id: str) -> str | None:
    if value is None:
        return None
    normalized = _relativize_results_dir(value, run_dir=run_dir)
    default_value = _relativize_results_dir(run_dir / job_id, run_dir=run_dir)
    if normalized == default_value:
        return None
    return normalized


def _build_env_template_payload(env_payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(env_payload)
    payload.pop("id", None)
    payload.pop("env_args", None)
    return _normalize_payload(payload)


def _env_template_id(env_id: str, env_template_payload: Mapping[str, Any]) -> str:
    digest = compute_checksum(_normalize_payload(env_template_payload))[:12]
    return f"{env_id}:{digest}"


def _sampling_args_override(
    *,
    sampling_args: Mapping[str, Any],
    model_payload: Mapping[str, Any],
) -> dict[str, Any] | None:
    normalized_sampling = _normalize_payload(sampling_args)
    model_sampling = model_payload.get("sampling_args") or {}
    normalized_model_sampling = _normalize_payload(model_sampling)
    if compute_checksum(normalized_sampling) == compute_checksum(normalized_model_sampling):
        return None
    return normalized_sampling


def _merge_unique_payload(
    container: dict[str, dict[str, Any]],
    key: str,
    payload: dict[str, Any],
    *,
    allow_mismatch: bool,
    label: str,
) -> None:
    existing = container.get(key)
    if existing is None:
        container[key] = payload
        return
    if existing != payload and not allow_mismatch:
        msg = f"Conflicting {label} payload for '{key}'."
        raise ValueError(msg)


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
    results_dir: str | None,
    models: dict[str, dict[str, Any]] | None = None,
    env_templates: dict[str, dict[str, Any]] | None = None,
    allow_model_mismatch: bool = False,
) -> ManifestJobEntry:
    """Build the manifest entry recorded for a job."""
    model_payload = _normalize_payload(json.loads(job.model.model_dump_json(exclude_none=True)))
    env_payload = _normalize_payload(json.loads(job.env.model_dump_json(exclude_none=True)))
    env_template_payload = _build_env_template_payload(env_payload)
    env_id = env_template_payload.get("module") or _resolve_env_identifier(job)
    if "module" not in env_template_payload:
        env_template_payload["module"] = env_id
    env_template_id = _env_template_id(env_id, env_template_payload)
    env_variant_id = env_payload.get("id") or job.job_id
    sampling_override = _sampling_args_override(sampling_args=sampling_args, model_payload=model_payload)
    if models is not None:
        _merge_unique_payload(
            models,
            _resolve_model_identifier(job),
            model_payload,
            allow_mismatch=allow_model_mismatch,
            label="model",
        )
    if env_templates is not None:
        _merge_unique_payload(
            env_templates,
            env_template_id,
            env_template_payload,
            allow_mismatch=False,
            label="manifest template",
        )
    return ManifestJobEntry(
        job_id=job.job_id,
        job_name=job.name,
        env_id=env_id,
        model_id=_resolve_model_identifier(job),
        env_template_id=env_template_id,
        env_variant_id=env_variant_id,
        env_args=_normalize_payload(env_args),
        sampling_args=sampling_override,
        status="pending",
        reason=None,
        attempt=0,
        started_at=None,
        ended_at=None,
        duration_seconds=None,
        results_dir=results_dir,
        artifacts=None,
        metrics=None,
        avg_reward=None,
        num_examples=None,
        rollouts_per_example=None,
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
        normalized_results_dir = _maybe_store_results_dir(results_dir, run_dir=self.run_dir, job_id=job.job_id)
        if entry is None:
            entry = build_job_entry(
                job,
                env_args=env_args,
                sampling_args=sampling_args,
                results_dir=normalized_results_dir,
                models=self.model.models,
                env_templates=self.model.env_templates,
            )
            self._jobs.append(entry)
            self._index[job.job_id] = entry
            self._refresh_summary(save=False)
            return entry

        updated = build_job_entry(
            job,
            env_args=env_args,
            sampling_args=sampling_args,
            results_dir=normalized_results_dir,
            models=self.model.models,
            env_templates=self.model.env_templates,
        )
        entry.env_id = updated.env_id
        entry.model_id = updated.model_id
        entry.env_template_id = updated.env_template_id
        entry.env_variant_id = updated.env_variant_id
        entry.env_args = updated.env_args
        entry.sampling_args = updated.sampling_args
        if entry.results_dir is None:
            entry.results_dir = updated.results_dir
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
        entry.results_dir = _maybe_store_results_dir(results_dir, run_dir=self.run_dir, job_id=job_id)
        entry.artifacts = list(artifacts) if artifacts else None
        entry.avg_reward = avg_reward
        entry.metrics = dict(metrics) if metrics else None
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
            entry.results_dir = _maybe_store_results_dir(results_dir, run_dir=self.run_dir, job_id=job_id)
        if entry.artifacts == []:
            entry.artifacts = None
        if entry.metrics == {}:
            entry.metrics = None
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
            "version": MANIFEST_VERSION,
            "run_id": run_id,
            "name": run_name,
            "config_source": str(config_source),
            "config_checksum": config_checksum,
            "created_at": timestamp(),
            "updated_at": timestamp(),
            "restart_source": restart_source,
            "models": {},
            "env_templates": {},
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
    "compute_snapshot_checksum",
    "manifest_job_signature",
    "resolved_job_signature",
    "timestamp",
]
