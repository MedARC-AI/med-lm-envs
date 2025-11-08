"""Run manifest helpers for the unified CLI."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from medarc_verifiers.cli_new._job_builder import ResolvedJob
from medarc_verifiers.utils.pathing import project_root, to_project_relative

MANIFEST_FILENAME = "run_manifest.json"
PROJECT_ROOT = project_root()


def timestamp() -> str:
    """Return an ISO8601 timestamp in UTC."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_data(value: Any) -> Any:
    """Recursively convert values to JSON-serializable structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_data(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_data(item) for item in value]
    return value


def _prune_nones(value: Any) -> Any:
    """Strip None-valued fields to keep manifests compact."""
    if isinstance(value, dict):
        return {key: _prune_nones(val) for key, val in value.items() if val is not None}
    if isinstance(value, list):
        return [_prune_nones(item) for item in value]
    return value


def _compute_checksum(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic checksum for a payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def compute_snapshot_checksum(snapshot: Mapping[str, Any]) -> str:
    """Public helper to compute the checksum for a config snapshot."""
    return _compute_checksum(_normalize_data(snapshot))


def compute_job_checksum(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
) -> str:
    """Expose job checksum calculations for resume/regeneration workflows."""
    payload = _canonicalize_job_config(job, env_args=env_args, sampling_args=sampling_args)
    return _compute_checksum(payload)


def _canonicalize_job_config(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce a normalized payload describing how the job will run."""
    model_payload = _prune_nones(json.loads(job.model.model_dump_json()))
    env_payload = _prune_nones(json.loads(job.env.model_dump_json()))
    return {
        "job_id": job.job_id,
        "job_name": job.name,
        "model": _normalize_data(model_payload),
        "env": _normalize_data(env_payload),
        "env_args": _normalize_data(env_args),
        "sampling_args": _normalize_data(sampling_args),
    }


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


def _resolve_env_identifier(job: ResolvedJob) -> str:
    if job.env.id:
        return job.env.id
    if job.env.module:
        return job.env.module
    return job.job_id


def _resolve_model_identifier(job: ResolvedJob) -> str:
    if job.model.id:
        return job.model.id
    if job.model.model:
        return job.model.model
    return job.job_id


def build_job_entry(
    job: ResolvedJob,
    *,
    env_args: Mapping[str, Any],
    sampling_args: Mapping[str, Any],
    results_dir: str,
) -> dict[str, Any]:
    """Build the manifest entry recorded for a job."""
    config_payload = _canonicalize_job_config(job, env_args=env_args, sampling_args=sampling_args)
    checksum = _compute_checksum(config_payload)
    return {
        "job_id": job.job_id,
        "job_name": job.name,
        "env_id": _resolve_env_identifier(job),
        "model_id": _resolve_model_identifier(job),
        "status": "pending",
        "reason": None,
        "attempt": 0,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "results_dir": results_dir,
        "artifacts": [],
        "metrics": {},
        "avg_reward": None,
        "num_examples": None,
        "rollouts_per_example": None,
        "checksum": checksum,
        "config": config_payload,
        "seeds": _extract_seeds(env_args, sampling_args),
    }


def _summarize_jobs(entries: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counter = Counter(entry.get("status", "pending") for entry in entries)
    skipped = sum(1 for entry in entries if entry.get("reason") in {"up_to_date", "skipped"})
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
    payload: MutableMapping[str, Any]
    persist: bool = True

    def __post_init__(self) -> None:
        self.payload.setdefault("jobs", [])
        self._jobs: list[MutableMapping[str, Any]] = list(self.payload["jobs"])
        self.payload["jobs"] = self._jobs
        self._index: dict[str, MutableMapping[str, Any]] = {
            entry["job_id"]: entry for entry in self._jobs if "job_id" in entry
        }
        self.payload.setdefault("summary", _summarize_jobs(self._jobs))

    @property
    def jobs(self) -> list[MutableMapping[str, Any]]:
        return self._jobs

    @property
    def summary(self) -> Mapping[str, Any]:
        return self.payload.get("summary", {})

    def job_entry(self, job_id: str) -> MutableMapping[str, Any] | None:
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
    ) -> MutableMapping[str, Any]:
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
        entry["config"] = config_payload
        entry["checksum"] = _compute_checksum(config_payload)
        entry["env_id"] = entry.get("env_id") or _resolve_env_identifier(job)
        entry["model_id"] = entry.get("model_id") or _resolve_model_identifier(job)
        entry.setdefault("results_dir", normalized_results_dir)
        entry.setdefault("seeds", _extract_seeds(env_args, sampling_args))
        return entry

    def record_job_start(self, job_id: str) -> None:
        entry = self._index.get(job_id)
        if not entry:
            return
        entry["status"] = "running"
        entry["reason"] = None
        entry["started_at"] = timestamp()
        entry["attempt"] = int(entry.get("attempt", 0)) + 1
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
        entry["status"] = "completed"
        entry["reason"] = None
        entry["ended_at"] = timestamp()
        entry["duration_seconds"] = duration_seconds
        entry["results_dir"] = _relativize_results_dir(results_dir, run_dir=self.run_dir)
        entry["artifacts"] = list(artifacts)
        entry["avg_reward"] = avg_reward
        entry["metrics"] = dict(metrics)
        entry["num_examples"] = num_examples
        entry["rollouts_per_example"] = rollouts_per_example
        self._refresh_summary()

    def record_job_failure(self, job_id: str, *, error: str, duration_seconds: float | None = None) -> None:
        entry = self._index.get(job_id)
        if not entry:
            return
        entry["status"] = "failed"
        entry["reason"] = error
        entry["ended_at"] = timestamp()
        entry["duration_seconds"] = duration_seconds
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
        entry["status"] = "completed"
        entry["reason"] = reason
        entry["ended_at"] = entry.get("ended_at") or timestamp()
        if source_entry:
            for key in ("duration_seconds", "avg_reward", "metrics", "num_examples", "rollouts_per_example", "artifacts"):
                if key in source_entry:
                    entry[key] = source_entry[key]
        if results_dir:
            entry["results_dir"] = _relativize_results_dir(results_dir, run_dir=self.run_dir)
        self._refresh_summary()

    def _refresh_summary(self, *, save: bool = True) -> None:
        self.payload["summary"] = _summarize_jobs(self._jobs)
        self.payload["updated_at"] = timestamp()
        if save:
            self.save()

    def save(self) -> None:
        if not self.persist:
            return
        tmp_path = self.path.with_suffix(".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(self.payload, handle, indent=2, sort_keys=True)
        tmp_path.replace(self.path)

    @classmethod
    def load(cls, path: Path, *, persist: bool = True) -> RunManifest:
        if not path.exists():
            raise FileNotFoundError(f"Run manifest '{path}' not found.")
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(path=path, payload=payload, persist=persist)

    @classmethod
    def create(
        cls,
        *,
        run_dir: Path,
        run_id: str,
        run_name: str,
        config_source: Path,
        config_snapshot: Mapping[str, Any],
        jobs: Sequence[ResolvedJob],
        env_args_map: Mapping[str, Mapping[str, Any]],
        sampling_args_map: Mapping[str, Mapping[str, Any]],
        persist: bool = True,
        regen_source: str | None = None,
    ) -> RunManifest:
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / MANIFEST_FILENAME
        normalized_snapshot = _normalize_data(config_snapshot)
        compact_snapshot = _prune_nones(normalized_snapshot)
        payload: MutableMapping[str, Any] = {
            "version": 1,
            "run_id": run_id,
            "name": run_name,
            "config_source": str(config_source),
            "config_snapshot": compact_snapshot,
            "config_checksum": compute_snapshot_checksum(config_snapshot),
            "created_at": timestamp(),
            "updated_at": timestamp(),
            "regen_source": regen_source,
            "jobs": [],
            "summary": {},
        }
        manifest = cls(path=path, payload=payload, persist=persist)
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
    "build_job_entry",
    "compute_job_checksum",
    "compute_snapshot_checksum",
    "timestamp",
]
