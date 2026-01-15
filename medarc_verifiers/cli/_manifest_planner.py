"""Manifest planning helpers separating selection from runnable computation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._manifest import MANIFEST_FILENAME, RunManifest, manifest_job_signature, resolved_job_signature
from medarc_verifiers.cli.utils.shared import slugify
from medarc_verifiers.utils.pathing import from_project_relative

logger = logging.getLogger(__name__)


@dataclass
class ManifestPlan:
    manifest: RunManifest
    runnable_job_ids: set[str]
    reused_job_ids: set[str]


@dataclass
class ManifestSelection:
    manifest: RunManifest
    seed_manifest: RunManifest | None
    strategy: str


class ManifestPlanner:
    """Resolve a manifest for a run and compute runnable/reused job sets."""

    def __init__(
        self,
        *,
        output_dir: Path,
        run_id: str | None,
        run_name: str,
        config_path: Path,
        config_checksum: str,
        jobs: Sequence[ResolvedJob],
        env_args_map: Mapping[str, Mapping[str, Any]],
        sampling_args_map: Mapping[str, Mapping[str, Any]],
        restart_source: str | None,
        auto_resume: bool,
        persist: bool,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.run_name = run_name
        self.config_path = Path(config_path)
        self.config_checksum = config_checksum
        self.jobs = jobs
        self.env_args_map = env_args_map
        self.sampling_args_map = sampling_args_map
        self.restart_source = restart_source
        self.auto_resume = auto_resume
        self.persist = persist

    def plan(self, *, force_all: bool, forced_envs: set[str]) -> ManifestPlan:
        selection = self._select_manifest()
        runnable, reused = self._compute_runnable(selection, force_all=force_all, forced_envs=forced_envs)
        return ManifestPlan(manifest=selection.manifest, runnable_job_ids=runnable, reused_job_ids=reused)

    # Selection helpers
    def _select_manifest(self) -> ManifestSelection:
        if self.restart_source:
            restart = self._select_restart_manifest(self.restart_source)
            if restart:
                return restart

        if self.auto_resume:
            resumed = self._select_auto_resume_manifest()
            if resumed:
                return resumed

        manifest = self._create_fresh_manifest()
        return ManifestSelection(manifest=manifest, seed_manifest=None, strategy="fresh")

    def _select_restart_manifest(self, restart_source: str) -> ManifestSelection | None:
        persist = self.persist
        restart_path = Path(restart_source).expanduser()
        seed_dir: Path | None = None
        if restart_path.exists() and restart_path.is_dir():
            seed_dir = restart_path.resolve()
        else:
            candidate = (self.output_dir / restart_source).resolve()
            if candidate.exists() and candidate.is_dir():
                seed_dir = candidate
        if seed_dir and (seed_dir / MANIFEST_FILENAME).exists():
            seed_manifest = RunManifest.load(seed_dir / MANIFEST_FILENAME, persist=persist)
            logger.info(
                "Restart in-place: extending existing run '%s' with any new jobs from current config.",
                seed_manifest.model.run_id,
            )
            self._ensure_jobs(seed_manifest, seed_manifest.run_dir)
            return ManifestSelection(
                manifest=seed_manifest,
                seed_manifest=seed_manifest,
                strategy="restart_in_place",
            )

        if seed_dir is None:
            return None
        seed_manifest = RunManifest.load(seed_dir / MANIFEST_FILENAME, persist=False)
        dest_run_id = self.run_id or _generate_run_id(self.run_name)
        run_dir = self._run_dir_for(dest_run_id)
        manifest_path = run_dir / MANIFEST_FILENAME
        if run_dir.exists() and manifest_path.exists() and persist:
            msg = f"Run directory '{run_dir}' already exists; choose a different --run-id."
            raise ValueError(msg)
        logger.info("Restarting run '%s' from prior run '%s'.", dest_run_id, restart_source)
        manifest = RunManifest.create(
            run_dir=run_dir,
            run_id=dest_run_id,
            run_name=self.run_name,
            config_source=self.config_path,
            config_checksum=self.config_checksum,
            jobs=self.jobs,
            env_args_map=self.env_args_map,
            sampling_args_map=self.sampling_args_map,
            persist=persist,
            restart_source=restart_source,
        )
        self._ensure_jobs(manifest, run_dir)
        return ManifestSelection(manifest=manifest, seed_manifest=seed_manifest, strategy="restart_new")

    def _select_auto_resume_manifest(self) -> ManifestSelection | None:
        persist = self.persist
        if self.run_id:
            run_dir = self._run_dir_for(self.run_id)
            manifest_path = run_dir / MANIFEST_FILENAME
            if manifest_path.exists():
                manifest = RunManifest.load(manifest_path, persist=persist)
                existing_checksum = manifest.model.config_checksum
                if existing_checksum and existing_checksum != self.config_checksum:
                    msg = (
                        f"Run '{self.run_id}' was created from a different configuration. "
                        f"To start fresh, pick a different --run-id or pass --no-auto-resume. "
                        f"To reuse completed jobs from this run, pass --restart {self.run_id}."
                    )
                    raise ValueError(msg)
                self._ensure_jobs(manifest, run_dir)
                return ManifestSelection(manifest=manifest, seed_manifest=None, strategy="auto_resume")
            if run_dir.exists():
                msg = f"Run '{self.run_id}' is missing {MANIFEST_FILENAME}; cannot auto-resume."
                raise ValueError(msg)
            logger.info(
                "Auto-resume requested for run '%s', but no prior run exists. Starting a fresh run with this id.",
                self.run_id,
            )
            manifest = self._create_fresh_manifest(run_id=self.run_id)
            return ManifestSelection(manifest=manifest, seed_manifest=None, strategy="fresh")

        candidate = _find_auto_resume_candidate(self.output_dir, expected_checksum=self.config_checksum)
        if candidate is None:
            logger.info(
                "Auto-resume enabled but no matching run exists in %s; starting a fresh run. "
                "Use --no-auto-resume to always start new runs.",
                self.output_dir,
            )
            return None
        manifest = RunManifest.load(candidate / MANIFEST_FILENAME, persist=persist)
        self._ensure_jobs(manifest, manifest.run_dir)
        return ManifestSelection(manifest=manifest, seed_manifest=None, strategy="auto_resume")

    def _create_fresh_manifest(self, run_id: str | None = None) -> RunManifest:
        dest_run_id = run_id or _generate_run_id(self.run_name)
        run_dir = self._run_dir_for(dest_run_id)
        manifest = RunManifest.create(
            run_dir=run_dir,
            run_id=dest_run_id,
            run_name=self.run_name,
            config_source=self.config_path,
            config_checksum=self.config_checksum,
            jobs=self.jobs,
            env_args_map=self.env_args_map,
            sampling_args_map=self.sampling_args_map,
            persist=self.persist,
            restart_source=None,
        )
        self._ensure_jobs(manifest, run_dir)
        return manifest

    def _ensure_jobs(self, manifest: RunManifest, run_dir: Path) -> None:
        for job in self.jobs:
            manifest.ensure_job(
                job,
                env_args=self.env_args_map[job.job_id],
                sampling_args=self.sampling_args_map[job.job_id],
                results_dir=run_dir / job.job_id,
            )

    def _run_dir_for(self, run_id: str) -> Path:
        return Path(self.output_dir) / run_id

    # Runnable computation
    def _compute_runnable(
        self,
        selection: ManifestSelection,
        *,
        force_all: bool,
        forced_envs: set[str],
    ) -> tuple[set[str], set[str]]:
        manifest = selection.manifest
        strategy = selection.strategy
        if strategy in {"restart_in_place", "restart_new"} and selection.seed_manifest is not None:
            runnable, reused = _plan_regen_jobs(
                manifest=manifest,
                seed_manifest=selection.seed_manifest,
                jobs=self.jobs,
                force_all=force_all,
                forced_envs=forced_envs,
            )
            if strategy == "restart_new" and reused:
                logger.info("Reused %d completed job(s) from '%s'.", len(reused), self.restart_source)
            return runnable, reused

        if strategy == "auto_resume":
            runnable = _plan_auto_resume_jobs(
                manifest=manifest,
                jobs=self.jobs,
                env_args_map=self.env_args_map,
                sampling_args_map=self.sampling_args_map,
                force_all=force_all,
                forced_envs=forced_envs,
            )
            return runnable, set()

        runnable = {job.job_id for job in self.jobs}
        return runnable, set()


def _find_auto_resume_candidate(output_dir: Path, *, expected_checksum: str) -> Path | None:
    """Pick the best prior run directory to auto-resume for the given checksum.

    Preference order:
    1) Matching config checksum and incomplete (completed < total)
    2) Matching config checksum and most recent updated_at
    Returns the run directory Path or None if no candidates.
    """
    candidates: list[tuple[bool, float, Path]] = []
    for child in sorted(output_dir.iterdir() if output_dir.exists() else [], key=lambda p: p.name):
        if not child.is_dir():
            continue
        manifest_path = child / MANIFEST_FILENAME
        if not manifest_path.exists():
            continue
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:  # noqa: BLE001
            continue
        if payload.get("config_checksum") != expected_checksum:
            continue
        summary = payload.get("summary") or {}
        total = int(summary.get("total", 0))
        completed = int(summary.get("completed", 0))
        incomplete = completed < total if total > 0 else True
        updated_at = payload.get("updated_at") or payload.get("created_at")
        try:
            ts = _parse_iso_ts(updated_at) if isinstance(updated_at, str) else (manifest_path.stat().st_mtime)
        except Exception:  # noqa: BLE001
            ts = manifest_path.stat().st_mtime
        candidates.append((incomplete, float(ts), child))

    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1]))
    return candidates[-1][2]


def _parse_iso_ts(value: str) -> float:
    # Accept timestamps like '2025-11-07T01:23:45Z' or ISO with offset
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except Exception:  # noqa: BLE001
        return 0.0


def _plan_auto_resume_jobs(
    *,
    manifest: RunManifest,
    jobs: Sequence[ResolvedJob],
    env_args_map: Mapping[str, Mapping[str, Any]],
    sampling_args_map: Mapping[str, Mapping[str, Any]],
    force_all: bool,
    forced_envs: set[str],
) -> set[str]:
    job_lookup = {job.job_id: job for job in jobs}
    manifest_signatures: dict[str, dict[str, Any]] = {}
    resolved_signatures: dict[str, dict[str, Any]] = {}
    runnable: set[str] = set()
    manifest_job_ids = {entry.job_id for entry in manifest.jobs if entry.job_id}
    new_jobs = set(job_lookup) - manifest_job_ids
    if new_jobs:
        logger.info(
            "Auto-resume ignoring %d new job(s) not present in the manifest: %s",
            len(new_jobs),
            ", ".join(sorted(new_jobs)),
        )
    for entry in manifest.jobs:
        job_id = entry.job_id
        if not job_id:
            continue
        job = job_lookup.get(job_id)
        if job is None:
            logger.debug("Manifest contains job '%s' that is absent from the current config; skipping.", job_id)
            continue
        manifest_signature = manifest_signatures.get(job_id)
        if manifest_signature is None:
            manifest_signature = manifest_job_signature(manifest.model, entry)
            manifest_signatures[job_id] = manifest_signature
        resolved_signature = resolved_signatures.get(job_id)
        if resolved_signature is None:
            resolved_signature = resolved_job_signature(
                job,
                env_args=env_args_map[job_id],
                sampling_args=sampling_args_map[job_id],
            )
            resolved_signatures[job_id] = resolved_signature
        if manifest_signature != resolved_signature:
            msg = (
                f"Job '{job_id}' arguments changed since the manifest was recorded. "
                "Start a fresh run by choosing a different --run-id or passing --no-auto-resume. "
                "To reuse completed jobs from this run, pass --restart <run-id-or-path>."
            )
            raise ValueError(msg)
        env_id = (entry.env_id or job.env.id or job.job_id).lower()
        forced = force_all or env_id in forced_envs
        if forced or entry.status != "completed":
            runnable.add(job_id)
    return runnable


def _plan_regen_jobs(
    *,
    manifest: RunManifest,
    seed_manifest: RunManifest,
    jobs: Sequence[ResolvedJob],
    force_all: bool,
    forced_envs: set[str],
) -> tuple[set[str], set[str]]:
    runnable: set[str] = set()
    reused: set[str] = set()
    manifest_signatures: dict[str, dict[str, Any]] = {}
    seed_signatures: dict[str, dict[str, Any]] = {}
    for job in jobs:
        entry = manifest.job_entry(job.job_id)
        if entry is None:
            continue
        seed_entry = seed_manifest.job_entry(job.job_id)
        env_id = (entry.env_id or job.env.id or job.job_id).lower()
        forced = force_all or env_id in forced_envs
        if (
            not forced
            and seed_entry is not None
            and seed_entry.status == "completed"
            and _manifest_job_signature_cached(seed_manifest, seed_entry, seed_signatures)
            == _manifest_job_signature_cached(manifest, entry, manifest_signatures)
        ):
            seed_results_dir = seed_entry.results_dir
            if seed_results_dir is None:
                seed_results_dir = seed_manifest.run_dir / seed_entry.job_id
            resolved_results_dir: Path | str | None = None
            if isinstance(seed_results_dir, str):
                seed_path = Path(seed_results_dir)
                if seed_path.is_absolute():
                    resolved_results_dir = seed_path
                elif seed_path.parts and seed_path.parts[0] == "runs":
                    resolved_results_dir = from_project_relative(seed_path)
                else:
                    resolved_results_dir = (seed_manifest.run_dir / seed_path).resolve()
            elif isinstance(seed_results_dir, Path):
                resolved_results_dir = seed_results_dir
            manifest.record_job_skip(
                job.job_id,
                reason="up_to_date",
                results_dir=resolved_results_dir or seed_results_dir,
                source_entry=seed_entry,
            )
            reused.add(job.job_id)
            continue
        runnable.add(job.job_id)
    return runnable, reused


def _manifest_job_signature_cached(
    manifest: RunManifest,
    entry: Any,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    job_id = entry.job_id
    signature = cache.get(job_id)
    if signature is None:
        signature = manifest_job_signature(manifest.model, entry)
        cache[job_id] = signature
    return signature


def _generate_run_id(name: str) -> str:
    base = slugify(name or "run")
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{base}-{timestamp}"
