from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.cli.process.discovery import discover_run_records


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base_manifest(
    job_payloads: list[dict],
    *,
    models: dict | None = None,
    env_templates: dict | None = None,
) -> dict:
    return {
        "version": 2,
        "run_id": "job-run-123",
        "name": "example-run",
        "config_source": "configs/example.yaml",
        "config_snapshot": {"jobs": []},
        "config_checksum": "abc123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:05:00Z",
        "models": models or {},
        "env_templates": env_templates or {},
        "jobs": job_payloads,
        "summary": {"completed": 1},
    }


def test_discover_run_records_basic(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = _base_manifest(
        [
            {
                "job_id": "model-env-job",
                "job_name": "demo-job",
                "model_id": "gpt-4",
                "env_id": "demo-env-module",
                "env_template_id": "demo-env-template",
                "env_variant_id": "demo-env",
                "env_args": {"fold": "dev"},
                "results_dir": "model-env-job",
                "status": "completed",
                "started_at": "2024-01-01T00:00:30Z",
                "ended_at": "2024-01-01T00:01:00Z",
                "num_examples": 10,
                "rollouts_per_example": 2,
            }
        ],
        models={"gpt-4": {"sampling_args": {"temperature": 0.2}}},
        env_templates={"demo-env-template": {"module": "demo-env-module"}},
    )
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

    _write_json(results_dir / "metadata.json", {"env_id": "demo-env"})
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")
    _write_json(results_dir / "summary.json", {"env_id": "demo-env"})

    records = discover_run_records(runs_dir)
    assert len(records) == 1
    record = records[0]
    assert record.status == "succeeded"
    assert record.duration_seconds == 12.5
    assert record.has_metadata is True
    assert record.has_results is True
    assert record.has_summary is True
    assert record.env_args == {"fold": "dev"}
    assert record.sampling_args == {"temperature": 0.2}
    assert record.manifest.job_run_id == "job-run-123"


def test_discover_run_records_filters_status(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = _base_manifest(
        [
            {
                "job_id": "model-env-job",
                "model_id": "gpt-4",
                "env_id": "demo-env-module",
                "env_template_id": "demo-env-template",
                "env_variant_id": "demo-env",
                "env_args": {},
            }
        ],
        models={"gpt-4": {"sampling_args": {}}},
        env_templates={"demo-env-template": {"module": "demo-env-module"}},
    )
    _write_json(run_dir / "run_manifest.json", manifest_payload)
    _write_json(
        run_dir / "run_summary.json",
        {"jobs": [{"job_id": "model-env-job", "status": "failed", "error": "boom"}]},
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")

    filtered = discover_run_records(runs_dir, filter_status=("failed",))
    assert len(filtered) == 1
    assert filtered[0].status == "failed"

    filtered_none = discover_run_records(runs_dir, filter_status=("succeeded",))
    assert filtered_none == []


def test_discover_run_records_only_complete_runs_missing_total(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = _base_manifest(
        [
            {
                "job_id": "model-env-job",
                "model_id": "gpt-4",
                "env_id": "demo-env-module",
                "env_template_id": "demo-env-template",
                "env_variant_id": "demo-env",
                "env_args": {},
            }
        ],
        models={"gpt-4": {"sampling_args": {}}},
        env_templates={"demo-env-template": {"module": "demo-env-module"}},
    )
    _write_json(run_dir / "run_manifest.json", manifest_payload)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")

    records = discover_run_records(runs_dir, only_complete_runs=True)
    assert len(records) == 1


def test_discover_run_records_missing_summary_uses_manifest_status(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "job-run-123"
    results_dir = run_dir / "model-env-job"

    manifest_payload = _base_manifest(
        [
            {
                "job_id": "model-env-job",
                "status": "completed",
                "reason": "cached",
                "model_id": "gpt-4",
                "env_id": "demo-env-module",
                "env_template_id": "demo-env-template",
                "env_variant_id": "demo-env",
                "env_args": {},
            }
        ],
        models={"gpt-4": {"sampling_args": {}}},
        env_templates={"demo-env-template": {"module": "demo-env-module"}},
    )
    _write_json(run_dir / "run_manifest.json", manifest_payload)

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "results.jsonl").write_text("{}", encoding="utf-8")

    records = discover_run_records(runs_dir)
    assert len(records) == 1
    record = records[0]
    assert record.status == "completed"
    assert record.reason == "cached"
    assert record.has_summary is False
