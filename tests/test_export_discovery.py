from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.export.parquet.discovery import discover_run_records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_run_records_basic(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = {
        "run_id": "job-run-123",
        "name": "example-run",
        "created_at": "2024-01-01T00:00:00Z",
        "jobs": [
            {
                "job_id": "model-env-job",
                "job_name": "demo-job",
                "model_id": "gpt-4",
                "env_id": "demo-env",
                "results_dir": "model-env-job",
                "env_overrides": {"foo": "bar"},
                "sampling_overrides": {"temperature": 0.2},
            }
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest_payload)

    _write_json(
        run_dir / "run_summary.json",
        {
            "jobs": [
                {
                    "job_id": "model-env-job",
                    "status": "succeeded",
                    "duration_seconds": 12.5,
                    "error": None,
                }
            ]
        },
    )

    _write_json(results_dir / "metadata.json", {"env_id": "demo-env", "env_args": {}})
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")
    _write_json(results_dir / "summary.json", {"env_id": "demo-env"})

    records = discover_run_records(runs_dir)
    assert len(records) == 1
    record = records[0]
    assert record.manifest.job_run_id == "job-run-123"
    assert record.job_id == "model-env-job"
    assert record.status == "succeeded"
    assert record.duration_seconds == 12.5
    assert record.has_metadata is True
    assert record.has_results is True
    assert record.has_summary is True
    assert record.manifest.config_snapshot is None
    assert record.env_overrides == {"foo": "bar"}
    assert record.sampling_overrides == {"temperature": 0.2}


def test_discover_run_records_filters_status(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = {
        "run_id": "job-run-123",
        "jobs": [
            {
                "job_id": "model-env-job",
                "model_id": "gpt-4",
            }
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest_payload)
    _write_json(
        run_dir / "run_summary.json",
        {
            "jobs": [
                {
                    "job_id": "model-env-job",
                    "status": "failed",
                    "error": "boom",
                }
            ]
        },
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")

    filtered = discover_run_records(runs_dir, filter_status=("failed",))
    assert len(filtered) == 1
    assert filtered[0].status == "failed"

    filtered_none = discover_run_records(runs_dir, filter_status=("succeeded",))
    assert filtered_none == []


def test_discover_run_records_missing_summary_defaults_to_unknown(tmp_path):
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = {
        "run_id": "job-run-123",
        "jobs": [
            {
                "job_id": "model-env-job",
            }
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest_payload)

    _write_json(results_dir / "metadata.json", {"env_id": "demo-env"})
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")

    records = discover_run_records(runs_dir)
    assert len(records) == 1
    assert records[0].status == "unknown"
    assert records[0].has_summary is False
