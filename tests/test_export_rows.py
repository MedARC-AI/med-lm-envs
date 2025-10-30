from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.export.parquet.discovery import RunManifestInfo, RunRecord
from medarc_verifiers.export.parquet.metadata import load_normalized_metadata
from medarc_verifiers.export.parquet.rows import load_enriched_rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_run_record_with_results(
    tmp_path: Path,
    *,
    job_run_id: str,
    manifest_env_id: str,
    metadata_payload: dict,
    results_rows: list[dict],
    status: str = "succeeded",
) -> RunRecord:
    run_dir = tmp_path / job_run_id
    results_dir = run_dir / "job-dir"
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = results_dir / "metadata.json"
    _write_json(metadata_path, metadata_payload)

    results_path = results_dir / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results_rows:
            handle.write(json.dumps(row))
            handle.write("\n")

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
        metadata_path=metadata_path,
        results_path=results_path,
        summary_path=results_dir / "summary.json",
        has_metadata=True,
        has_results=True,
        has_summary=False,
        status=status,
        duration_seconds=12.5,
        error=None,
    )


def test_load_enriched_rows_basic(tmp_path: Path):
    metadata_payload = {
        "env_id": "base-env",
        "model": "gpt-4",
        "env_args": {"split": "dev"},
        "num_examples": 2,
        "rollouts_per_example": 1,
        "sampling_args": {"temperature": 0.2},
    }
    results_rows = [
        {"example_id": 0, "prompt": "prompt-0", "completion": "completion-0", "reward": 0.5},
        {"example_id": 1, "prompt": "prompt-1", "completion": "completion-1", "reward": 0.6},
    ]
    record = _create_run_record_with_results(
        tmp_path,
        job_run_id="run-basic",
        manifest_env_id="base-env-r2",
        metadata_payload=metadata_payload,
        results_rows=results_rows,
    )
    normalized = load_normalized_metadata(record)

    enriched = load_enriched_rows(normalized, drop_fields={"prompt", "completion"})
    assert len(enriched) == 2
    first = enriched[0]
    assert first["env_id"] == "base-env"
    assert first["manifest_env_id"] == "base-env-r2"
    assert first["variant_id"]
    assert first["rollout_index"] == 2
    assert first["intra_rollout_index"] is None
    assert first["job_run_id"] == "run-basic"
    assert first["model"] == "gpt-4"
    assert first["sampling_args"] == {"temperature": 0.2}
    assert first["env_split"] == "dev"
    assert "prompt" not in first and "completion" not in first


def test_load_enriched_rows_with_multiple_rollouts(tmp_path: Path):
    metadata_payload = {
        "env_id": "bench",
        "model": "model-x",
        "env_args": {},
        "num_examples": 2,
        "rollouts_per_example": 3,
        "sampling_args": {},
    }
    results_rows = [
        {"example_id": 0, "reward": 0.1},
        {"example_id": 1, "reward": 0.2},
        {"example_id": 0, "reward": 0.3},
        {"example_id": 1, "reward": 0.4},
        {"example_id": 0, "reward": 0.5},
        {"example_id": 1, "reward": 0.6},
    ]
    record = _create_run_record_with_results(
        tmp_path,
        job_run_id="run-rollouts",
        manifest_env_id="bench",
        metadata_payload=metadata_payload,
        results_rows=results_rows,
    )
    normalized = load_normalized_metadata(record)

    enriched = load_enriched_rows(normalized, drop_fields=set())
    assert len(enriched) == 6
    intra = [row["intra_rollout_index"] for row in enriched]
    assert intra == [0, 0, 1, 1, 2, 2]
