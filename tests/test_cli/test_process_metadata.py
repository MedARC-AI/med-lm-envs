from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from medarc_verifiers.cli.process.discovery import RunManifestInfo, RunRecord
from medarc_verifiers.cli.process.metadata import load_normalized_metadata


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_record(
    tmp_path: Path,
    *,
    manifest_env_id: str | None = "demo-env-rollout3",
    results_dir_name: str = "job-abc",
    env_args: dict | None = None,
    sampling_args: dict | None = None,
    avg_reward: float | None = None,
    num_examples: int | None = 10,
    rollouts_per_example: int | None = None,
    has_metadata: bool = True,
    env_config: dict | None = None,
) -> RunRecord:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-123"
    results_dir = run_dir / results_dir_name
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
        model_id="gpt-4o",
        manifest_env_id=manifest_env_id,
        results_dir_name=results_dir_name,
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
        avg_reward=avg_reward,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        row_count=1,
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
        avg_reward=0.8,
        rollouts_per_example=None,
    )
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env-rollout1",
            "model": "gpt-4o-mini",
            "env_args": {"difficulty": "easy", "split": "dev"},
            "sampling_args": {"temperature": 0.9, "top_p": 0.95},
            "avg_reward": 0.8,
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


def test_load_normalized_metadata_falls_back_to_results_dir_underscore_rollout(tmp_path: Path) -> None:
    record = _make_record(
        tmp_path,
        manifest_env_id="agentclinic",
        results_dir_name="baichuan-m2-agentclinic_rollout1",
    )

    normalized = load_normalized_metadata(record)

    assert normalized.manifest_env_id == "agentclinic"
    assert normalized.base_env_id == "agentclinic"
    assert normalized.rollout_index == 1


def test_load_normalized_metadata_preserves_raw_metadata_payload(tmp_path: Path) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env")
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "version_info": {
                "vf_version": "0.1.10",
                "vf_commit": "abc123",
                "env_version": "1.2.3",
                "env_commit": None,
            },
            "endpoint_id": "cluster-a",
            "base_url": "https://example.invalid/v1",
        },
    )

    normalized = load_normalized_metadata(record)

    assert normalized.raw_metadata["endpoint_id"] == "cluster-a"
    assert normalized.raw_metadata["base_url"] == "https://example.invalid/v1"
    assert normalized.raw_metadata["version_info"]["vf_version"] == "0.1.10"


def test_load_normalized_metadata_validation_failure_sanitizes_raw_metadata(tmp_path: Path) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env")
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "num_examples": "not-an-int",
            "version_info": {
                "vf_version": "0.1.10",
                "vf_commit": "abc123",
            },
            "endpoint_id": "cluster-a",
            "base_url": "https://example.invalid/v1",
            "api_key": "secret-value",
            "nested": {"keep": "out"},
        },
    )

    normalized = load_normalized_metadata(record)

    assert normalized.raw_metadata == {
        "version_info": {
            "vf_version": "0.1.10",
            "vf_commit": "abc123",
        },
        "endpoint_id": "cluster-a",
        "base_url": "https://example.invalid/v1",
    }


def test_load_normalized_metadata_keeps_zero_num_examples_from_manifest(tmp_path: Path) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env", num_examples=0, rollouts_per_example=1)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "num_examples": 20,
            "rollouts_per_example": 3,
        },
    )

    normalized = load_normalized_metadata(record)

    assert normalized.num_examples == 0
    assert normalized.rollouts_per_example == 1


def test_load_normalized_metadata_keeps_zero_rollouts_from_manifest(tmp_path: Path) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env", num_examples=10, rollouts_per_example=0)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "num_examples": 20,
            "rollouts_per_example": 3,
        },
    )

    normalized = load_normalized_metadata(record)

    assert normalized.num_examples == 10
    assert normalized.rollouts_per_example == 0


def test_load_normalized_metadata_keeps_all_examples_sentinel_from_manifest(tmp_path: Path) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env", num_examples=-1, rollouts_per_example=1)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "num_examples": 20,
            "rollouts_per_example": 3,
        },
    )

    normalized = load_normalized_metadata(record)

    assert normalized.num_examples == -1
    assert normalized.rollouts_per_example == 1


def test_load_normalized_metadata_warns_on_avg_reward_and_num_examples_mismatch(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env", avg_reward=0.8, num_examples=10)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "avg_reward": 0.7,
            "num_examples": 12,
        },
    )

    with caplog.at_level(logging.WARNING):
        normalized = load_normalized_metadata(record)

    assert normalized.num_examples == 10
    assert "Manifest/metadata result mismatch for process input" in caplog.text
    assert "avg_reward manifest=0.8 metadata=0.7" in caplog.text
    assert "num_examples manifest=10 metadata=12" in caplog.text


def test_load_normalized_metadata_does_not_warn_when_result_fields_match(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = _make_record(tmp_path, manifest_env_id="demo-env", avg_reward=0.8, num_examples=10)
    _write_json(
        record.metadata_path,
        {
            "env_id": "demo-env",
            "avg_reward": 0.8,
            "num_examples": 10,
        },
    )

    with caplog.at_level(logging.WARNING):
        load_normalized_metadata(record)

    assert "Manifest/metadata result mismatch for process input" not in caplog.text
