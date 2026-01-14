from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from medarc_verifiers.cli._job_builder import ResolvedJob
from medarc_verifiers.cli._manifest import (
    MANIFEST_FILENAME,
    RunManifest,
    compute_snapshot_checksum,
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
        interleave_scoring=False,
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


def test_manifest_serialization_prunes_nones_and_relativizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    job = _build_job()
    fake_root = tmp_path / "repo"
    fake_root.mkdir()
    run_dir = fake_root / "runs" / "phase5"

    def fake_to_project_relative(path: Path | str, *, default_base: Path | None = None) -> str:
        resolved = Path(path).resolve()
        base = fake_root if default_base is None else default_base
        return resolved.relative_to(base).as_posix()

    monkeypatch.setattr("medarc_verifiers.cli._manifest.PROJECT_ROOT", fake_root)
    monkeypatch.setattr("medarc_verifiers.cli._manifest.to_project_relative", fake_to_project_relative)

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
