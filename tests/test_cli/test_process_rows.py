from __future__ import annotations

import json
from pathlib import Path

import pytest

from medarc_verifiers.cli.process.discovery import RunManifestInfo, RunRecord
from medarc_verifiers.cli.process.metadata import load_normalized_metadata
from medarc_verifiers.cli.process.rows import load_rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_record(tmp_path: Path, *, status: str = "completed", reason: str | None = None) -> RunRecord:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    results_dir = run_dir / "job-1"
    manifest_info = RunManifestInfo(
        job_run_id="run-1",
        run_name="Example Run",
        summary_completed=1,
        summary_total=1,
        summary_total_known=True,
        manifest_path=run_dir / "run_manifest.json",
        run_dir=run_dir,
        created_at="2024-05-01T00:00:00Z",
        updated_at="2024-05-01T00:10:00Z",
        config_source="configs/example.yaml",
        config_checksum="checksum",
        run_summary_path=run_dir / "run_summary.json",
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    record = RunRecord(
        manifest=manifest_info,
        job_id="job-1",
        job_name="Job 1",
        model_id="model-alpha",
        manifest_env_id="demo-env-rollout3",
        results_dir_name="job-1",
        results_dir=results_dir,
        metadata_path=results_dir / "metadata.json",
        results_path=results_dir / "results.jsonl",
        summary_path=results_dir / "summary.json",
        has_metadata=True,
        has_results=True,
        has_summary=False,
        status=status,
        duration_seconds=12.0,
        reason=reason,
        started_at="2024-05-01T00:00:30Z",
        ended_at="2024-05-01T00:00:42Z",
        num_examples=10,
        rollouts_per_example=1,
        env_args={"split": "dev", "extra_body": {}},
        sampling_args={"temperature": 0.2},
        env_config={},
        model_config={},
    )
    return record


def _write_results(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_load_rows_basic_enrichment(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env-rollout9",
            "env_args": {"difficulty": "easy"},
            "sampling_args": {"top_p": 0.8},
        },
    )
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-1",
                "prompt": "Q?",
                "completion": "A",
                "info": {"debug": True},
                "task": "qa",
                "reward": 1.0,
            }
        ],
    )

    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata)

    assert len(rows) == 1
    row = rows[0]
    assert row["example_id"] == "ex-1"
    assert "prompt" not in row
    assert "completion" not in row
    assert "info" not in row
    assert row["env_id"] == "demo-env"
    assert row["manifest_env_id"] == "demo-env-rollout3"
    assert row["base_env_id"] == "demo-env"
    assert row["job_run_id"] == "run-1"
    assert row["run_id"] == "job-1"
    assert row["model_id"] == "model-alpha"
    assert row["rollout_index"] == 3
    assert "sampling_args" not in row
    assert "split" not in row
    assert "difficulty" not in row
    assert row["status"] == "completed"
    assert row["error"] is None
    assert "extra_body" not in row


def test_load_rows_drops_prompt_completion(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-1",
                "prompt": "Q?",
                "completion": "A",
            }
        ],
    )
    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata)

    assert "prompt" not in rows[0]
    assert "completion" not in rows[0]


def test_load_rows_missing_example_id_raises(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "task": "qa",
            }
        ],
    )
    metadata = load_normalized_metadata(record)

    with pytest.raises(ValueError) as excinfo:
        load_rows(metadata)
    assert "example_id" in str(excinfo.value)


def test_load_rows_extra_columns_populates_extras(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-keep",
                "info": {"debug": True},
            }
        ],
    )
    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata, extra_columns=("debug",))
    row = rows[0]
    assert "info" not in row
    assert row["extras"] == "{\"debug\": true}"


def test_load_rows_failed_job_preserves_error(tmp_path: Path) -> None:
    record = _build_record(tmp_path, status="failed", reason="exploded")
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-fail",
            }
        ],
    )

    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata)

    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "exploded"


def test_load_rows_does_not_add_env_args(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-struct",
            }
        ],
    )

    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata)
    row = rows[0]
    assert "split" not in row
    assert "extra_body" not in row


def test_load_rows_flattens_token_usage(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-token",
                "token_usage": {
                    "model": {"prompt": 10, "completion": 5, "total": 15, "cost": 0.02},
                    "judge": {"prompt": 3, "completion": 4, "total": 7, "cost": 0.01},
                },
            }
        ],
    )

    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata)
    row = rows[0]

    assert "token_usage" not in row
    assert row["model_cost"] == 0.02
    assert row["model_token_prompt"] == 10
    assert row["model_token_completion"] == 5
    assert row["model_token_total"] == 15
    assert row["judge_cost"] == 0.01
    assert row["judge_token_prompt"] == 3
    assert row["judge_token_completion"] == 4
    assert row["judge_token_total"] == 7


def test_load_rows_drops_non_mapping_token_usage(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-token",
                "token_usage": "n/a",
            }
        ],
    )

    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata)
    row = rows[0]

    assert "token_usage" not in row
    assert row["model_cost"] is None
    assert row["judge_cost"] is None


def test_load_rows_maps_answer_column(tmp_path: Path) -> None:
    record = _build_record(tmp_path)
    _write_json(record.metadata_path, {})
    _write_results(
        record.results_path,
        [
            {
                "example_id": "ex-1",
                "ground_truth": "42",
            }
        ],
    )
    metadata = load_normalized_metadata(record)
    rows = load_rows(metadata, answer_column="ground_truth")

    assert rows[0]["answer"] == "42"
