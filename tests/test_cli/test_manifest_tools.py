from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.cli._manifest_tools import validate_manifests_in_runs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(
    run_dir: Path,
    *,
    num_examples: int | None = None,
    rollouts_per_example: int | None = None,
) -> None:
    payload = {
        "version": 3,
        "run_id": "demo-run",
        "name": "demo",
        "config_source": "cfg.yaml",
        "config_checksum": "x",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "artifacts_root": ".",
        "models": {},
        "env_templates": {},
        "jobs": [
            {
                "job_id": "job-1",
                "model_id": "m",
                "env_id": "e",
                "env_template_id": "e:t",
                "env_variant_id": "e",
                "env_args": {},
                "results_relpath": "job-1/results.jsonl",
                "metadata_relpath": "job-1/metadata.json",
                "status": "completed",
                "num_examples": num_examples,
                "rollouts_per_example": rollouts_per_example,
            }
        ],
        "summary": {"total": 1, "completed": 1, "pending": 0, "failed": 0, "running": 0, "skipped": 0},
    }
    _write_json(run_dir / "run_manifest.json", payload)


def test_validate_manifests_reports_broken_paths(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "raw"
    run_dir = runs_dir / "demo-run"
    job_dir = run_dir / "job-1"
    _write_json(job_dir / "metadata.json", {"env_id": "demo"})
    (job_dir / "results.jsonl").write_text('{"example_id": 1}\n', encoding="utf-8")

    payload = {
        "version": 3,
        "run_id": "demo-run",
        "name": "demo",
        "config_source": "cfg.yaml",
        "config_checksum": "x",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "artifacts_root": ".",
        "models": {},
        "env_templates": {},
        "jobs": [
            {
                "job_id": "job-1",
                "model_id": "m",
                "env_id": "e",
                "env_template_id": "e:t",
                "env_variant_id": "e",
                "env_args": {},
                "results_relpath": "broken/job-1/results.jsonl",
                "status": "completed",
            }
        ],
        "summary": {"total": 1, "completed": 1, "pending": 0, "failed": 0, "running": 0, "skipped": 0},
    }
    _write_json(run_dir / "run_manifest.json", payload)

    result = validate_manifests_in_runs(runs_dir, strict=False)
    assert result.manifests_checked == 1
    assert result.jobs_checked == 1
    assert any(issue.kind == "warning" and "fallback" in issue.message.lower() for issue in result.issues)


def test_validate_manifests_accepts_partial_rollout_file(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "raw"
    run_dir = runs_dir / "demo-run"
    job_dir = run_dir / "job-1"
    _write_json(job_dir / "metadata.json", {"env_id": "demo"})
    (job_dir / "results.jsonl").write_text(
        "\n".join(
            [
                '{"example_id": 1, "rollout_index": 0}',
                '{"example_id": 2, "rollout_index": 0}',
                '{"example_id": 1, "rollout_index": 1}',
                '{"example_id": 2, "rollout_index": 1}',
                '{"example_id": 1, "rollout_index": 2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(run_dir, num_examples=2, rollouts_per_example=3)

    result = validate_manifests_in_runs(runs_dir, strict=False)

    assert result.manifests_checked == 1
    assert result.jobs_checked == 1
    assert result.issues == []


def test_validate_manifests_reports_out_of_range_rollout_index(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "raw"
    run_dir = runs_dir / "demo-run"
    job_dir = run_dir / "job-1"
    _write_json(job_dir / "metadata.json", {"env_id": "demo"})
    (job_dir / "results.jsonl").write_text(
        "\n".join(
            [
                '{"example_id": 1, "rollout_index": 0}',
                '{"example_id": 2, "rollout_index": 0}',
                '{"example_id": 1, "rollout_index": 3}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_manifest(run_dir, num_examples=2, rollouts_per_example=3)

    result = validate_manifests_in_runs(runs_dir, strict=False)

    assert any("out-of-range rollout_index" in issue.message for issue in result.issues)


def test_validate_manifests_reports_malformed_last_jsonl_row(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs" / "raw"
    run_dir = runs_dir / "demo-run"
    job_dir = run_dir / "job-1"
    _write_json(job_dir / "metadata.json", {"env_id": "demo"})
    (job_dir / "results.jsonl").write_text('{"example_id": 1}\n{"example_id": ', encoding="utf-8")
    _write_manifest(run_dir, num_examples=1, rollouts_per_example=1)

    result = validate_manifests_in_runs(runs_dir, strict=False)

    assert any("failed to parse last JSONL row" in issue.message for issue in result.issues)
