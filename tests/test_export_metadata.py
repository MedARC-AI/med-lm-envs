from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.export.parquet.discovery import RunManifestInfo, RunRecord
from medarc_verifiers.export.parquet.metadata import load_normalized_metadata


def _create_run_record(
    tmp_path: Path,
    *,
    env_args: dict[str, object],
    manifest_env_id: str,
    job_run_id: str,
) -> RunRecord:
    run_dir = tmp_path / job_run_id
    results_dir = run_dir / "job-dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest = RunManifestInfo(
        job_run_id=job_run_id,
        run_name="example",
        manifest_path=run_dir / "run_manifest.json",
        run_dir=run_dir,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
        config_source=None,
        config_checksum=None,
        config_snapshot=None,
        run_summary_path=run_dir / "run_summary.json",
    )
    metadata_payload = {
        "env_id": "base-env",
        "model": "gpt-4",
        "env_args": env_args,
        "num_examples": 42,
        "rollouts_per_example": 3,
        "sampling_args": {"temperature": 0.1},
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata_payload), encoding="utf-8")

    return RunRecord(
        manifest=manifest,
        job_id="job-id",
        job_name="job",
        model_id="model-id",
        manifest_env_id=manifest_env_id,
        env_overrides={},
        sampling_overrides={},
        results_dir_name="job-dir",
        results_dir=results_dir,
        metadata_path=results_dir / "metadata.json",
        results_path=results_dir / "results.jsonl",
        summary_path=results_dir / "summary.json",
        has_metadata=True,
        has_results=False,
        has_summary=False,
        status="succeeded",
        duration_seconds=None,
        error=None,
    )


def test_load_normalized_metadata_basic(tmp_path: Path):
    record = _create_run_record(
        tmp_path,
        env_args={"difficulty": "easy", "shuffle": True, "seed": 12},
        manifest_env_id="base-env-r1",
        job_run_id="run-basic",
    )

    normalized = load_normalized_metadata(record)
    assert normalized is not None
    assert normalized.base_env_id == "base-env"
    assert normalized.env_columns == {
        "env_difficulty": "easy",
        "env_shuffle": True,
        "env_seed": 12,
    }
    assert normalized.variant_id
    assert len(normalized.variant_id) == 16
    assert normalized.num_examples == 42
    assert normalized.rollouts_per_example == 3
    assert normalized.sampling_args == {"temperature": 0.1}


def test_variant_id_changes_with_env_args(tmp_path: Path):
    first = _create_run_record(
        tmp_path,
        env_args={"seed": 1},
        manifest_env_id="base-env",
        job_run_id="run-first",
    )
    second = _create_run_record(
        tmp_path,
        env_args={"seed": 2},
        manifest_env_id="base-env",
        job_run_id="run-second",
    )

    first_variant = load_normalized_metadata(first)
    second_variant = load_normalized_metadata(second)
    assert first_variant is not None and second_variant is not None
    assert first_variant.variant_id != second_variant.variant_id


def test_variant_id_differs_when_env_args_missing_but_suffix_varies(tmp_path: Path):
    first = _create_run_record(
        tmp_path,
        env_args={},
        manifest_env_id="base-env",
        job_run_id="run-empty-1",
    )
    second = _create_run_record(
        tmp_path,
        env_args={},
        manifest_env_id="base-env-r1",
        job_run_id="run-empty-2",
    )

    first_variant = load_normalized_metadata(first)
    second_variant = load_normalized_metadata(second)
    assert first_variant is not None and second_variant is not None
    assert first_variant.variant_id != second_variant.variant_id
