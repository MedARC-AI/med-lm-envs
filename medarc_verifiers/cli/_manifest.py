"""Run manifest helpers for the unified CLI."""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._schemas import ModelConfigSchema
from medarc_verifiers.cli.utils.json_io import dumps_json
from medarc_verifiers.cli.utils.shared import count_jsonl_rows, compute_checksum, resolve_env_identifier_or
from medarc_verifiers.utils.pathing import normalize_results_dir_for_manifest

MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_VERSION = 3
SUPPORTED_MANIFEST_VERSIONS = {3}

logger = logging.getLogger(__name__)


class ManifestConflictError(ValueError):
    """Raised when an existing manifest conflicts with the current config."""


def _normalize_model_slug(value: str) -> str:
    """Normalize model slugs for restart comparisons.

    Some providers expose the same model under different namespaces (e.g.
    `google/gemini-3-pro-preview` vs `gemini-3-pro-preview`). For now, we only
    normalize Gemini model slugs by stripping a single leading namespace.
    """
    if not value:
        return value
    if "/" not in value:
        return value
    candidate = value.rsplit("/", 1)[-1]
    if candidate.startswith("gemini-"):
        return candidate
    return value


class ManifestJobEntry(BaseModel):
    """Pydantic model describing a single manifest job entry."""

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


# Keep this list aligned with runtime/progress fields mutated by record_job_* methods.
# Fields here must survive ensure_job() refreshes when config/env metadata changes.
_ENSURE_JOB_RUNTIME_STATE_FIELDS = (
    "status",
    "reason",
    "attempt",
    "started_at",
    "ended_at",
    "duration_seconds",
    "row_count",
    "metrics",
    "avg_reward",
    "num_examples",
    "rollouts_per_example",
)


def _validate_ensure_job_runtime_state_fields() -> None:
    missing = set(_ENSURE_JOB_RUNTIME_STATE_FIELDS) - set(ManifestJobEntry.model_fields)
    if missing:
        msg = f"Unknown manifest fields in _ENSURE_JOB_RUNTIME_STATE_FIELDS: {sorted(missing)}"
        raise ValueError(msg)


_validate_ensure_job_runtime_state_fields()


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
    return normalize_results_dir_for_manifest(value, run_dir=run_dir)


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


def _require_manifest_v3(payload: Mapping[str, Any], *, path: Path | None = None) -> None:
    version = payload.get("version")
    if version not in SUPPORTED_MANIFEST_VERSIONS:
        location = f" '{path}'" if path else ""
        msg = f"Manifest{location} uses version {version}; expected one of {sorted(SUPPORTED_MANIFEST_VERSIONS)}."
        raise ValueError(msg)


def _sanitize_model_payload(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = {
        key: value for key, value in model_payload.items() if key not in ModelConfigSchema.resume_tolerant_fields
    }

    model_slug = sanitized.get("model")
    if isinstance(model_slug, str):
        sanitized["model"] = _normalize_model_slug(model_slug)

    # Provider quirks: OpenAI-compatible endpoints vary widely in what they accept when
    # we forward `sampling_args.extra_body`. Treat *all* of extra_body as resume-tolerant
    # for the purposes of manifest conflict detection so users can switch providers
    # without getting blocked by payload drift.
    sampling_args = sanitized.get("sampling_args")
    if isinstance(sampling_args, Mapping):
        updated_sampling_args = dict(sampling_args)
        updated_sampling_args.pop("extra_body", None)
        if updated_sampling_args:
            sanitized["sampling_args"] = updated_sampling_args
        else:
            sanitized.pop("sampling_args", None)

    return sanitized


def _sampling_extra_body(model_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    sampling_args = model_payload.get("sampling_args")
    if not isinstance(sampling_args, Mapping):
        return None
    extra_body = sampling_args.get("extra_body")
    if not isinstance(extra_body, Mapping):
        return None
    normalized = _normalize_payload(extra_body)
    return normalized or None


def _warn_extra_body_change(key: str, existing: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    existing_extra = _sampling_extra_body(existing)
    payload_extra = _sampling_extra_body(payload)
    if existing_extra is None and payload_extra is None:
        return
    if compute_checksum(existing_extra or {}) == compute_checksum(payload_extra or {}):
        return
    logger.warning(
        "Model '%s' sampling_args.extra_body changed; allowing restart, but providers may reject unknown fields.",
        key,
    )


def _sampling_args_payload(model_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    sampling_args = model_payload.get("sampling_args")
    if not isinstance(sampling_args, Mapping):
        return None
    normalized = _normalize_payload(sampling_args)
    return normalized or None


def _warn_sampling_args_change(key: str, existing: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    existing_sampling = _sampling_args_payload(existing)
    payload_sampling = _sampling_args_payload(payload)
    if existing_sampling is None and payload_sampling is None:
        return
    if compute_checksum(existing_sampling or {}) == compute_checksum(payload_sampling or {}):
        return
    logger.warning(
        "Model '%s' sampling_args changed; allowing restart, but providers may reject unsupported parameters.",
        key,
    )


def _effective_sampling_args(entry: ManifestJobEntry, model_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if entry.sampling_args is not None:
        return _normalize_payload(entry.sampling_args)
    return _normalize_payload(model_payload.get("sampling_args") or {})


def _canonical_manifest_parts(
    *,
    model_payload: Mapping[str, Any],
    env_payload: Mapping[str, Any],
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
    env_identifier: str | None,
    env_variant_id: str,
    env_payload_is_template: bool = False,
    sampling_args_already_resolved: bool = False,
) -> dict[str, Any]:
    model_normalized = _normalize_payload(model_payload)
    env_normalized = _normalize_payload(env_payload)
    if env_payload_is_template:
        env_template_payload = dict(env_normalized)
    else:
        env_template_payload = _build_env_template_payload(env_normalized)
        if "module" not in env_template_payload:
            env_template_payload["module"] = env_identifier
    if sampling_args_already_resolved:
        sampling_override = None
        sampling_payload = sampling_args
    else:
        sampling_override = _sampling_args_override(sampling_args=sampling_args, model_payload=model_normalized)
        sampling_payload = sampling_override or model_normalized.get("sampling_args") or {}
    effective_env_payload = {
        **env_template_payload,
        "module": env_identifier,
        "id": env_variant_id,
        "env_args": _normalize_payload(env_args),
    }
    return {
        "model_sanitized": _sanitize_model_payload(model_normalized),
        "env_template_payload": _normalize_payload(env_template_payload),
        "effective_env_payload": _normalize_payload(effective_env_payload),
        "sampling_payload": _normalize_payload(sampling_payload),
        "sampling_override": sampling_override,
    }


def manifest_job_signature(manifest: RunManifestModel, entry: ManifestJobEntry) -> dict[str, Any]:
    model_payload = _normalize_payload(manifest.models.get(entry.model_id or "", {}) or {})
    env_template = _normalize_payload(manifest.env_templates.get(entry.env_template_id, {}) or {})
    env_identifier = entry.env_id or env_template.get("module")
    canonical = _canonical_manifest_parts(
        model_payload=model_payload,
        env_payload=env_template,
        env_args=entry.env_args,
        sampling_args=_effective_sampling_args(entry, model_payload),
        env_identifier=env_identifier,
        env_variant_id=entry.env_variant_id,
        env_payload_is_template=True,
        sampling_args_already_resolved=True,
    )
    signature = {
        "model": canonical["model_sanitized"],
        "env": canonical["effective_env_payload"],
        "sampling_args": canonical["sampling_payload"],
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
    env_id = env_payload.get("module") or _resolve_env_identifier(job)
    env_variant_id = str(env_payload.get("id") or job.job_id)
    canonical = _canonical_manifest_parts(
        model_payload=model_payload,
        env_payload=env_payload,
        env_args=env_args,
        sampling_args=sampling_args,
        env_identifier=env_id,
        env_variant_id=env_variant_id,
    )
    signature = {
        "model": canonical["model_sanitized"],
        "env": canonical["effective_env_payload"],
        "sampling_args": canonical["sampling_payload"],
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


def _manifest_relative_artifacts(*, run_dir: Path, job_id: str, results_dir: Path | str | None) -> tuple[str, str]:
    if results_dir is None:
        base_rel = Path(job_id)
    else:
        candidate = Path(results_dir)
        if not candidate.is_absolute():
            candidate = (run_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()
        try:
            base_rel = candidate.relative_to(run_dir)
        except ValueError:
            base_rel = Path(job_id)
    base_rel = Path(base_rel.as_posix())
    return (
        (base_rel / "results.jsonl").as_posix(),
        (base_rel / "metadata.json").as_posix(),
    )


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


def _merge_unique_model_payload(
    container: dict[str, dict[str, Any]],
    key: str,
    payload: dict[str, Any],
    *,
    allow_mismatch: bool,
) -> None:
    existing = container.get(key)
    if existing is None:
        container[key] = payload
        return
    if existing == payload:
        return
    if allow_mismatch:
        container[key] = payload
        return
    sanitized_existing = _sanitize_model_payload(existing)
    sanitized_payload = _sanitize_model_payload(payload)
    if sanitized_existing == sanitized_payload:
        _warn_extra_body_change(key, existing, payload)
        container[key] = payload
        return

    stripped_existing = dict(sanitized_existing)
    stripped_payload = dict(sanitized_payload)
    stripped_existing.pop("sampling_args", None)
    stripped_payload.pop("sampling_args", None)
    if stripped_existing == stripped_payload:
        _warn_sampling_args_change(key, existing, payload)
        _warn_extra_body_change(key, existing, payload)
        container[key] = payload
        return

    all_keys = set(sanitized_existing) | set(sanitized_payload)
    diff_keys = sorted(key for key in all_keys if sanitized_existing.get(key) != sanitized_payload.get(key))
    suffix = f" (conflicting keys: {', '.join(diff_keys)})" if diff_keys else ""
    msg = f"Conflicting model payload for '{key}'{suffix}."
    raise ManifestConflictError(msg)


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
    container[key] = payload


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
    env_id = env_payload.get("module") or _resolve_env_identifier(job)
    env_variant_id = str(env_payload.get("id") or job.job_id)
    canonical = _canonical_manifest_parts(
        model_payload=model_payload,
        env_payload=env_payload,
        env_args=env_args,
        sampling_args=sampling_args,
        env_identifier=env_id,
        env_variant_id=env_variant_id,
    )
    env_template_payload = canonical["env_template_payload"]
    env_template_id = _env_template_id(env_id, env_template_payload)
    if models is not None:
        _merge_unique_model_payload(
            models,
            _resolve_model_identifier(job),
            model_payload,
            allow_mismatch=allow_model_mismatch,
        )
    if env_templates is not None:
        _merge_unique_payload(
            env_templates,
            env_template_id,
            env_template_payload,
            allow_mismatch=False,
            label="manifest template",
        )
    results_relpath, metadata_relpath = _manifest_relative_artifacts(
        run_dir=Path("."),
        job_id=job.job_id,
        results_dir=job.job_id,
    )
    return ManifestJobEntry(
        job_id=job.job_id,
        env_id=env_id,
        model_id=_resolve_model_identifier(job),
        env_template_id=env_template_id,
        env_variant_id=env_variant_id,
        env_args=_normalize_payload(env_args),
        sampling_args=canonical["sampling_override"],
        status="pending",
        reason=None,
        attempt=0,
        started_at=None,
        ended_at=None,
        duration_seconds=None,
        results_dir=results_dir,
        results_relpath=results_relpath,
        metadata_relpath=metadata_relpath,
        row_count=None,
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
        results_relpath, metadata_relpath = _manifest_relative_artifacts(
            run_dir=self.run_dir,
            job_id=job.job_id,
            results_dir=results_dir,
        )
        if entry is None:
            entry = build_job_entry(
                job,
                env_args=env_args,
                sampling_args=sampling_args,
                results_dir=normalized_results_dir,
                models=self.model.models,
                env_templates=self.model.env_templates,
            )
            entry.results_relpath = results_relpath
            entry.metadata_relpath = metadata_relpath
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
        runtime_state = {field: getattr(entry, field) for field in _ENSURE_JOB_RUNTIME_STATE_FIELDS}
        if entry.results_dir is not None:
            results_dir_value = entry.results_dir
        else:
            results_dir_value = updated.results_dir
        replacement = updated.model_copy(
            update={
                **runtime_state,
                "results_dir": results_dir_value,
                "results_relpath": results_relpath,
                "metadata_relpath": metadata_relpath,
            }
        )
        # Preserve object identity so external references to `entry` remain live.
        for field_name, value in replacement.model_dump().items():
            setattr(entry, field_name, value)
        self._index[job.job_id] = entry
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
        results_relpath, metadata_relpath = _manifest_relative_artifacts(
            run_dir=self.run_dir,
            job_id=job_id,
            results_dir=results_dir,
        )
        entry.results_relpath = results_relpath
        entry.metadata_relpath = metadata_relpath
        entry.avg_reward = avg_reward
        entry.metrics = dict(metrics) if metrics else None
        entry.num_examples = num_examples
        entry.rollouts_per_example = rollouts_per_example
        results_path = results_dir / "results.jsonl"
        entry.row_count = count_jsonl_rows(results_path)
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

        if source_entry:
            is_mapping = isinstance(source_entry, Mapping)
            for key in (
                "duration_seconds",
                "avg_reward",
                "metrics",
                "num_examples",
                "rollouts_per_example",
                "row_count",
            ):
                if is_mapping:
                    if key in source_entry:
                        setattr(entry, key, source_entry[key])
                else:
                    setattr(entry, key, getattr(source_entry, key))
        if results_dir:
            entry.results_dir = _maybe_store_results_dir(results_dir, run_dir=self.run_dir, job_id=job_id)
            results_relpath, metadata_relpath = _manifest_relative_artifacts(
                run_dir=self.run_dir,
                job_id=job_id,
                results_dir=results_dir,
            )
            entry.results_relpath = results_relpath
            entry.metadata_relpath = metadata_relpath
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
        text = dumps_json(self.model.model_dump(exclude_none=True))
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(self.path)

    @classmethod
    def load(cls, path: Path, *, persist: bool = True) -> RunManifest:
        if not path.exists():
            raise FileNotFoundError(f"Run manifest '{path}' not found.")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload, _ = _upgrade_manifest_payload(payload)
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
            "artifacts_root": ".",
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


def _upgrade_manifest_payload(payload: Any) -> tuple[Any, bool]:
    """Apply in-memory migrations for older manifest payloads."""
    if not isinstance(payload, dict):
        return payload, False

    changed = False
    version = payload.get("version")
    jobs = payload.get("jobs")
    if version == 2:
        payload["version"] = MANIFEST_VERSION
        payload.setdefault("artifacts_root", ".")
        changed = True
        if isinstance(jobs, list):
            for index, job in enumerate(jobs):
                if not isinstance(job, dict):
                    continue
                job_id = str(job.get("job_id") or "")
                base_dir = Path(str(job.get("results_dir") or job_id or ""))
                if not base_dir.as_posix():
                    base_dir = Path(job_id or f"job-{index}")
                if "results_relpath" not in job:
                    job["results_relpath"] = (base_dir / "results.jsonl").as_posix()
                    changed = True
                if "metadata_relpath" not in job:
                    job["metadata_relpath"] = (base_dir / "metadata.json").as_posix()
                    changed = True
                if "summary_relpath" in job:
                    job.pop("summary_relpath", None)
                    changed = True
                if "artifacts_checksum" in job:
                    job.pop("artifacts_checksum", None)
                    changed = True
                if "artifacts" in job:
                    job.pop("artifacts", None)
                    changed = True

    env_templates = payload.get("env_templates")
    if isinstance(env_templates, dict):
        for template_id, template in env_templates.items():
            if not isinstance(template, dict):
                continue
            if "interleave_scoring" not in template:
                continue
            interleave_value = template.pop("interleave_scoring")
            template.setdefault("independent_scoring", interleave_value)
            env_templates[template_id] = template
            changed = True

    return payload, changed
