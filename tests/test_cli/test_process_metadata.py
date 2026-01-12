from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.cli.process.discovery import RunManifestInfo, RunRecord
from medarc_verifiers.cli.process.metadata import load_normalized_metadata


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_record(
    tmp_path: Path,
    *,
    manifest_env_id: str | None = "demo-env-rollout3",
    env_args: dict | None = None,
    sampling_args: dict | None = None,
    num_examples: int | None = 10,
    rollouts_per_example: int | None = None,
    has_metadata: bool = True,
    env_config: dict | None = None,
) -> RunRecord:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-123"
    results_dir = run_dir / "job-abc"
    manifest_info = RunManifestInfo(
        job_run_id="run-123",
        run_name="Example Run",
        summary_completed=1,
        summary_total=1,
        summary_total_known=True,
        manifest_path=run_dir / "run_manifest.json",
        run_dir=run_dir,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:05:00Z",
        config_source="configs/example.yaml",
        config_checksum="deadbeef",
        run_summary_path=run_dir / "run_summary.json",
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        manifest=manifest_info,
        job_id="job-abc",
        job_name="Example Job",
        model_id="gpt-4o",
        manifest_env_id=manifest_env_id,
        results_dir_name="job-abc",
        results_dir=results_dir,
        metadata_path=results_dir / "metadata.json",
        results_path=results_dir / "results.jsonl",
        summary_path=results_dir / "summary.json",
        has_metadata=has_metadata,
        has_results=False,
        has_summary=False,
        status="completed",
        duration_seconds=12.5,
        reason=None,
        started_at="2024-01-01T00:00:10Z",
        ended_at="2024-01-01T00:00:50Z",
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        env_args=env_args or {},
        sampling_args=sampling_args or {},
        env_config=env_config or {},
        model_config={},
    )
    return record


def test_load_normalized_metadata_prefers_manifest_fields(tmp_path: Path) -> None:
    record = _make_record(
        tmp_path,
        env_args={"difficulty": "hard"},
        sampling_args={"temperature": 0.1},
        rollouts_per_example=None,
    )
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env-rollout1",
            "model": "gpt-4o-mini",
            "env_args": {"difficulty": "easy", "split": "dev"},
            "sampling_args": {"temperature": 0.9, "top_p": 0.95},
            "num_examples": 20,
            "rollouts_per_example": 2,
        },
    )

    normalized = load_normalized_metadata(record)

    assert normalized.manifest_env_id == "demo-env-rollout3"
    assert normalized.base_env_id == "demo-env"
    assert normalized.rollout_index == 3
    assert normalized.env_args == {"difficulty": "hard", "split": "dev"}
    assert normalized.sampling_args == {"temperature": 0.1, "top_p": 0.95}
    assert normalized.num_examples == 10
    assert normalized.rollouts_per_example == 2
    assert normalized.model_id == "gpt-4o"
    assert normalized.metadata_model == "gpt-4o-mini"


def test_load_normalized_metadata_without_file(tmp_path: Path) -> None:
    record = _make_record(tmp_path, has_metadata=False, manifest_env_id="demo-env")
    normalized = load_normalized_metadata(record)

    assert normalized.metadata_env_id is None
    assert normalized.raw_metadata == {}
    assert normalized.base_env_id == "demo-env"
    assert normalized.rollout_index == 0


def test_load_normalized_metadata_falls_back_to_metadata_env_id(tmp_path: Path) -> None:
    record = _make_record(tmp_path, manifest_env_id=None)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env-r7",
            "env_args": {"split": "train"},
        },
    )

    normalized = load_normalized_metadata(record)
    assert normalized.manifest_env_id == "demo-env-r7"
    assert normalized.base_env_id == "demo-env"
    assert normalized.rollout_index == 7
    assert normalized.env_args == {"split": "train"}


def test_load_normalized_metadata_prefers_env_config_variant_id(tmp_path: Path) -> None:
    record = _make_record(
        tmp_path,
        manifest_env_id="longhealth",
        has_metadata=False,
        env_config={"id": "longhealth-task1-rollout1618", "module": "longhealth"},
    )

    normalized = load_normalized_metadata(record)

    assert normalized.manifest_env_id == "longhealth-task1-rollout1618"
    assert normalized.base_env_id == "longhealth-task1"
    assert normalized.rollout_index == 1618
