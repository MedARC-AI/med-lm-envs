from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._manifest import (
    _ENSURE_JOB_RUNTIME_STATE_FIELDS,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    ManifestJobEntry,
    RunManifest,
    RunManifestModel,
    build_job_entry,
    compute_snapshot_checksum,
    manifest_job_signature,
    resolved_job_signature,
)
from medarc_verifiers.cli._schemas import EnvironmentConfigSchema, ModelConfigSchema

SNAPSHOT_ENV_VAR = "UPDATE_CLI_MANIFEST_SNAPSHOT"
SNAPSHOT_PATH = Path(__file__).parent / "data" / "run_manifest_snapshot.json"


def _build_job() -> ResolvedJob:
    model = ModelConfigSchema(
        id="snapshot-model",
        model="gpt-4o-mini",
        headers={"X-Test": "one"},
        sampling_args={"max_tokens": 256, "temperature": 0.3},
        env_args={"split": "dev"},
        env_overrides={"snapshot-env": {"temperature": 0.2}},
    )
    env = EnvironmentConfigSchema(
        id="snapshot-env",
        module="environments.snapshot_env",
        num_examples=3,
        rollouts_per_example=2,
        max_concurrent=4,
        independent_scoring=False,
        state_columns=["student_answer", "score"],
        env_args={"difficulty": "easy", "runner_seed": 99},
    )
    return ResolvedJob(
        job_id="snapshot-model-snapshot-env",
        name="snapshot-eval",
        model=model,
        env=env,
        env_args={"difficulty": "easy", "runner_seed": 99, "split": "dev", "job_seed": 7},
        sampling_args={"max_tokens": 256, "temperature": 0.3, "eval_seed": 17},
    )


def _normalize_manifest(payload: Any, *, base_dir: Path) -> Any:
    base_posix = base_dir.as_posix()
    base_native = str(base_dir)

    if isinstance(payload, dict):
        return {key: _normalize_manifest(value, base_dir=base_dir) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_normalize_manifest(item, base_dir=base_dir) for item in payload]
    if isinstance(payload, str):
        return payload.replace(base_posix, "<TMP>").replace(base_native, "<TMP>")
    return payload


def test_run_manifest_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = _build_job()
    monkeypatch.setattr("medarc_verifiers.cli._manifest.timestamp", lambda: "2024-03-01T00:00:00Z")

    run_dir = tmp_path / "snapshot-run"
    snapshot_cfg = {
        "models": {"snapshot-model": {"model": "gpt-4o-mini"}},
        "envs": {"snapshot-env": {"module": "environments.snapshot_env"}},
        "jobs": [{"model": "snapshot-model", "env": "snapshot-env"}],
    }
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="snapshot-run",
        run_name="Snapshot Run",
        config_source=Path("configs/snapshot.yaml"),
        config_checksum=compute_snapshot_checksum(snapshot_cfg),
        jobs=[job],
        env_args_map={job.job_id: job.env_args},
        sampling_args_map={job.job_id: job.sampling_args},
        persist=True,
        restart_source="baseline-run",
    )

    manifest_path = manifest.path
    assert manifest_path.name == MANIFEST_FILENAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = _normalize_manifest(payload, base_dir=tmp_path)

    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(normalized, indent=2, sort_keys=True) + "\n"

    if os.environ.get(SNAPSHOT_ENV_VAR):
        SNAPSHOT_PATH.write_text(serialized, encoding="utf-8")

    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    assert normalized == expected

    loaded = RunManifest.load(manifest_path, persist=False)
    assert loaded.model.config_checksum == expected["config_checksum"]
    assert loaded.jobs[0].status == "pending"


def test_manifest_load_upgrades_interleave_scoring(tmp_path: Path) -> None:
    """Older manifests may store interleave_scoring in env_templates; load should upgrade it."""
    manifest_path = tmp_path / "run_manifest.json"
    payload = {
        "version": 2,
        "run_id": "demo",
        "name": "Demo",
        "config_source": "configs/demo.yaml",
        "config_checksum": "abc",
        "created_at": "2024-03-01T00:00:00Z",
        "updated_at": "2024-03-01T00:00:00Z",
        "models": {},
        "env_templates": {
            "env:template": {
                "module": "environments.snapshot_env",
                "num_examples": 3,
                "rollouts_per_example": 2,
                "interleave_scoring": False,
            }
        },
        "jobs": [],
        "summary": {},
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = RunManifest.load(manifest_path, persist=False)
    template = loaded.model.env_templates["env:template"]
    assert "interleave_scoring" not in template
    assert template["independent_scoring"] is False


def test_manifest_serialization_prunes_nones_and_relativizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = _build_job()
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    run_dir = fake_root / "runs" / "phase5"

    def fake_to_project_relative(path: Path | str, *, default_base: Path | None = None) -> str:
        resolved = Path(path).resolve()
        base = fake_root if default_base is None else default_base
        return resolved.relative_to(base).as_posix()

    monkeypatch.setattr("medarc_verifiers.utils.pathing.project_root", lambda: fake_root)
    monkeypatch.setattr("medarc_verifiers.utils.pathing.to_project_relative", fake_to_project_relative)

    snapshot_cfg = {
        "models": {"snapshot-model": {"model": "gpt-4o-mini"}},
        "envs": {"snapshot-env": {"module": "environments.snapshot_env"}},
        "jobs": [{"model": "snapshot-model", "env": "snapshot-env"}],
    }
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="phase5",
        run_name="Phase 5 Run",
        config_source=fake_root / "configs" / "phase5.yaml",
        config_checksum=compute_snapshot_checksum(snapshot_cfg),
        jobs=[job],
        env_args_map={job.job_id: job.env_args},
        sampling_args_map={job.job_id: job.sampling_args},
    )

    payload = json.loads(manifest.path.read_text(encoding="utf-8"))
    job_payload = payload["jobs"][0]

    assert "results_dir" not in job_payload
    assert "reason" not in job_payload
    assert "avg_reward" not in job_payload
    assert job_payload["env_args"]["job_seed"] == 7
    assert job_payload["sampling_args"]["eval_seed"] == 17


def test_manifest_job_signature_is_stable(tmp_path: Path) -> None:
    job = _build_job()
    run_dir = tmp_path / "sig-run"
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="sig-run",
        run_name="Signature Run",
        config_source=Path("configs/sig.yaml"),
        config_checksum="sig",
        jobs=[job],
        env_args_map={job.job_id: job.env_args},
        sampling_args_map={job.job_id: job.sampling_args},
        persist=False,
    )
    entry = manifest.jobs[0]

    signature = manifest_job_signature(manifest.model, entry)
    assert signature == {
        "model": {
            "id": "snapshot-model",
            "model": "gpt-4o-mini",
            "sampling_args": {"max_tokens": 256, "temperature": 0.3},
            "env_args": {"split": "dev"},
            "env_overrides": {"snapshot-env": {"temperature": 0.2}},
        },
        "env": {
            "module": "environments.snapshot_env",
            "num_examples": 3,
            "rollouts_per_example": 2,
            "max_concurrent": 4,
            "independent_scoring": False,
            "state_columns": ["student_answer", "score"],
            "print_results": False,
            "rerun": False,
            "id": "snapshot-env",
            "env_args": {"difficulty": "easy", "runner_seed": 99, "split": "dev", "job_seed": 7},
        },
        "sampling_args": {"max_tokens": 256, "temperature": 0.3, "eval_seed": 17},
    }


def test_resolved_job_signature_is_stable() -> None:
    job = _build_job()

    signature = resolved_job_signature(job, env_args=job.env_args, sampling_args=job.sampling_args)
    assert signature == {
        "model": {
            "id": "snapshot-model",
            "model": "gpt-4o-mini",
            "sampling_args": {"max_tokens": 256, "temperature": 0.3},
            "env_args": {"split": "dev"},
            "env_overrides": {"snapshot-env": {"temperature": 0.2}},
        },
        "env": {
            "module": "environments.snapshot_env",
            "num_examples": 3,
            "rollouts_per_example": 2,
            "max_concurrent": 4,
            "independent_scoring": False,
            "state_columns": ["student_answer", "score"],
            "print_results": False,
            "rerun": False,
            "id": "snapshot-env",
            "env_args": {"difficulty": "easy", "runner_seed": 99, "split": "dev", "job_seed": 7},
        },
        "sampling_args": {"max_tokens": 256, "temperature": 0.3, "eval_seed": 17},
    }


def test_build_job_entry_is_stable() -> None:
    job = _build_job()
    entry = build_job_entry(job, env_args=job.env_args, sampling_args=job.sampling_args, results_dir=None)
    assert entry.model_dump() == {
        "job_id": "snapshot-model-snapshot-env",
        "env_id": "environments.snapshot_env",
        "model_id": "snapshot-model",
        "env_template_id": "environments.snapshot_env:6ef485576891",
        "env_variant_id": "snapshot-env",
        "env_args": {"difficulty": "easy", "runner_seed": 99, "split": "dev", "job_seed": 7},
        "sampling_args": {"max_tokens": 256, "temperature": 0.3, "eval_seed": 17},
        "status": "pending",
        "reason": None,
        "attempt": 0,
        "started_at": None,
        "ended_at": None,
        "duration_seconds": None,
        "results_dir": None,
        "results_relpath": "snapshot-model-snapshot-env/results.jsonl",
        "metadata_relpath": "snapshot-model-snapshot-env/metadata.json",
        "row_count": None,
        "metrics": None,
        "avg_reward": None,
        "num_examples": None,
        "rollouts_per_example": None,
    }


def test_resolved_job_signature_ignores_resume_tolerant_fields() -> None:
    base_job = _build_job()
    model_variant = base_job.model.model_copy(update={"api_key_var": "ALT_KEY"})
    variant_job = ResolvedJob(
        job_id=base_job.job_id,
        name=base_job.name,
        model=model_variant,
        env=base_job.env,
        env_args=base_job.env_args,
        sampling_args=base_job.sampling_args,
        sleep=base_job.sleep,
    )

    base_sig = resolved_job_signature(base_job, env_args=base_job.env_args, sampling_args=base_job.sampling_args)
    variant_sig = resolved_job_signature(
        variant_job, env_args=variant_job.env_args, sampling_args=variant_job.sampling_args
    )

    assert base_sig == variant_sig


def test_ensure_job_preserves_runtime_fields_on_update(tmp_path: Path) -> None:
    seed_job = _build_job()
    run_dir = tmp_path / "runtime-run"
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="runtime-run",
        run_name="Runtime Run",
        config_source=Path("configs/runtime.yaml"),
        config_checksum="runtime",
        jobs=[seed_job],
        env_args_map={seed_job.job_id: seed_job.env_args},
        sampling_args_map={seed_job.job_id: seed_job.sampling_args},
        persist=False,
    )
    manifest.record_job_completion(
        seed_job.job_id,
        duration_seconds=3.5,
        results_dir=run_dir / seed_job.job_id,
        avg_reward=0.75,
        metrics={"pass_rate": 0.75},
        num_examples=12,
        rollouts_per_example=2,
    )
    entry_before = manifest.job_entry(seed_job.job_id)
    assert entry_before is not None
    entry_before.row_count = 4
    assert set(_ENSURE_JOB_RUNTIME_STATE_FIELDS) == {
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
    }
    before_runtime = {
        "status": entry_before.status,
        "reason": entry_before.reason,
        "attempt": entry_before.attempt,
        "started_at": entry_before.started_at,
        "ended_at": entry_before.ended_at,
        "duration_seconds": entry_before.duration_seconds,
        "row_count": entry_before.row_count,
        "metrics": entry_before.metrics,
        "avg_reward": entry_before.avg_reward,
        "num_examples": entry_before.num_examples,
        "rollouts_per_example": entry_before.rollouts_per_example,
    }

    updated_job = ResolvedJob(
        job_id=seed_job.job_id,
        name=seed_job.name,
        model=seed_job.model,
        env=seed_job.env,
        env_args={**seed_job.env_args, "job_seed": 999},
        sampling_args={**seed_job.sampling_args, "eval_seed": 999},
        sleep=seed_job.sleep,
    )
    manifest.ensure_job(
        updated_job,
        env_args=updated_job.env_args,
        sampling_args=updated_job.sampling_args,
        results_dir=run_dir / updated_job.job_id,
    )

    entry_after = manifest.job_entry(seed_job.job_id)
    assert entry_after is not None
    after_runtime = {
        "status": entry_after.status,
        "reason": entry_after.reason,
        "attempt": entry_after.attempt,
        "started_at": entry_after.started_at,
        "ended_at": entry_after.ended_at,
        "duration_seconds": entry_after.duration_seconds,
        "row_count": entry_after.row_count,
        "metrics": entry_after.metrics,
        "avg_reward": entry_after.avg_reward,
        "num_examples": entry_after.num_examples,
        "rollouts_per_example": entry_after.rollouts_per_example,
    }

    assert before_runtime == after_runtime


def test_ensure_job_preserves_entry_object_identity(tmp_path: Path) -> None:
    seed_job = _build_job()
    run_dir = tmp_path / "identity-run"
    manifest = RunManifest.create(
        run_dir=run_dir,
        run_id="identity-run",
        run_name="Identity Run",
        config_source=Path("configs/identity.yaml"),
        config_checksum="identity",
        jobs=[seed_job],
        env_args_map={seed_job.job_id: seed_job.env_args},
        sampling_args_map={seed_job.job_id: seed_job.sampling_args},
        persist=False,
    )
    entry_before = manifest.job_entry(seed_job.job_id)
    assert entry_before is not None

    updated_job = ResolvedJob(
        job_id=seed_job.job_id,
        name=seed_job.name,
        model=seed_job.model,
        env=seed_job.env,
        env_args={**seed_job.env_args, "job_seed": 111},
        sampling_args={**seed_job.sampling_args, "eval_seed": 111},
        sleep=seed_job.sleep,
    )
    manifest.ensure_job(
        updated_job,
        env_args=updated_job.env_args,
        sampling_args=updated_job.sampling_args,
        results_dir=run_dir / updated_job.job_id,
    )
    entry_after = manifest.job_entry(seed_job.job_id)
    assert entry_after is not None
    assert entry_before is entry_after
    assert entry_before.env_args["job_seed"] == 111


def test_manifest_job_signature_does_not_fallback_module_to_variant_id() -> None:
    model = RunManifestModel(
        version=MANIFEST_VERSION,
        run_id="r",
        name="n",
        config_source="cfg.yaml",
        config_checksum="x",
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        models={},
        env_templates={"template-no-module": {}},
        jobs=[],
        summary={},
    )
    entry = ManifestJobEntry(
        job_id="job-x",
        env_id=None,
        model_id="missing-model",
        env_template_id="template-no-module",
        env_variant_id="variant-x",
        env_args={},
    )
    signature = manifest_job_signature(model, entry)
    assert "module" not in signature["env"]
    assert signature["env"]["id"] == "variant-x"
