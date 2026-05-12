from __future__ import annotations

import json
from pathlib import Path

from scripts.convert_legacy_raw_runs import convert_legacy_raw_runs, main


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_manifest(
    raw_dir: Path,
    *,
    run_id: str = "run-1",
    jobs: list[dict] | None = None,
) -> Path:
    run_dir = raw_dir / run_id
    manifest = {
        "version": 3,
        "run_id": run_id,
        "name": "legacy",
        "config_source": "configs/legacy.yaml",
        "config_checksum": "abc123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:01:00Z",
        "artifacts_root": ".",
        "models": {"gpt/mini": {"sampling_args": {"temperature": 0.1}}},
        "env_templates": {},
        "summary": {"completed": 1, "total": 1},
        "jobs": jobs if jobs is not None else [_job()],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def _job(**overrides: object) -> dict:
    payload = {
        "job_id": "job-1",
        "model_id": "gpt/mini",
        "env_id": "demo/env",
        "env_template_id": "demo-template",
        "env_variant_id": "demo/env",
        "env_args": {"fold": "dev"},
        "sampling_args": {"top_p": 0.9},
        "status": "completed",
        "results_relpath": "job-1/results.jsonl",
        "metadata_relpath": "job-1/metadata.json",
        "num_examples": 2,
        "rollouts_per_example": 1,
        "avg_reward": 0.75,
    }
    payload.update(overrides)
    return payload


def _write_artifacts(run_dir: Path, *, job_id: str = "job-1") -> None:
    _write_json(
        run_dir / job_id / "metadata.json",
        {
            "env_args": {"fold": "metadata"},
            "sampling_args": {"temperature": 0.2},
            "num_examples": 2,
            "rollouts_per_example": 1,
            "avg_reward": 0.5,
        },
    )
    (run_dir / job_id / "results.jsonl").write_text('{"example_id":"ex-1","reward":1.0}\n', encoding="utf-8")


def test_dry_run_lists_jobs_and_writes_nothing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    run_dir = _write_manifest(raw_dir)
    _write_artifacts(run_dir)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir)

    assert report.would_convert == 1
    assert report.failed == 0
    assert not output_dir.exists()
    entry = report.entries[0]
    assert entry.target_dir is not None
    assert entry.target_dir.endswith("gpt-mini/demo-env/base")


def test_converts_valid_manifest_job_to_processable_eval_output(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    run_dir = _write_manifest(raw_dir)
    _write_artifacts(run_dir)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir, dry_run=False)

    assert report.converted == 1
    target = output_dir / "gpt-mini" / "demo-env" / "base"
    row = json.loads((target / "results.jsonl").read_text(encoding="utf-8"))
    assert row["is_completed"] is True
    assert row["is_truncated"] is False
    assert row["metrics"] == {}
    assert row["stop_condition"] == "max_turns_reached"
    assert row["timing"]["total"] == 0.0
    assert row["tool_defs"] == []
    metadata = json.loads((target / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == {
        "avg_error": 0.0,
        "avg_metrics": {},
        "avg_reward": 1.0,
        "base_url": "",
        "env_args": {"fold": "metadata"},
        "env_id": "demo/env",
        "model": "gpt/mini",
        "num_examples": 2,
        "pass_all_k": {},
        "pass_at_k": {},
        "pass_threshold": 0.5,
        "rollouts_per_example": 1,
        "sampling_args": {"temperature": 0.2},
        "state_columns": [],
        "time": 0.0,
        "tools": None,
        "usage": None,
        "version_info": {},
    }
    assert not (target / "bench_index.json").exists()
    assert not (target / ".medarc_eval_metadata.json").exists()
    assert (run_dir / "job-1" / "results.jsonl").exists()


def test_skips_missing_results(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    _write_manifest(raw_dir)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=tmp_path / "evals", dry_run=False)

    assert report.skipped == 1
    assert report.entries[0].reason == "missing results.jsonl"


def test_skips_non_completed_jobs(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    run_dir = _write_manifest(raw_dir, jobs=[_job(status="failed")])
    _write_artifacts(run_dir)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=tmp_path / "evals", dry_run=False)

    assert report.skipped == 1
    assert "failed" in report.entries[0].reason


def test_target_collision_fails_without_writing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    run_dir = _write_manifest(raw_dir)
    _write_artifacts(run_dir)
    target = output_dir / "gpt-mini" / "demo-env" / "base"
    target.mkdir(parents=True)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir, dry_run=False)

    assert report.failed == 1
    assert "already exists" in report.entries[0].reason
    assert not (target / "metadata.json").exists()


def test_report_includes_valid_jobs_when_another_job_fails(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    run_dir = _write_manifest(
        raw_dir,
        jobs=[
            _job(job_id="valid", results_relpath="valid/results.jsonl"),
            _job(job_id="collision", results_relpath="collision/results.jsonl", env_variant_id="demo/env::seed-1"),
        ],
    )
    _write_artifacts(run_dir, job_id="valid")
    _write_artifacts(run_dir, job_id="collision")
    (output_dir / "gpt-mini" / "demo-env" / "seed-1").mkdir(parents=True)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir, dry_run=False)

    assert report.failed == 1
    assert report.converted == 1
    by_job = {entry.job_id: entry for entry in report.entries}
    assert by_job["collision"].status == "failed"
    assert by_job["valid"].status == "converted"
    assert (output_dir / "gpt-mini" / "demo-env" / "base" / "metadata.json").exists()


def test_invalid_existing_metadata_is_skipped(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    run_dir = _write_manifest(raw_dir)
    (run_dir / "job-1").mkdir(parents=True)
    (run_dir / "job-1" / "metadata.json").write_text("not json", encoding="utf-8")
    (run_dir / "job-1" / "results.jsonl").write_text('{"example_id":"ex-1"}\n', encoding="utf-8")

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir, dry_run=False)

    assert report.skipped == 1
    assert "invalid metadata.json" in report.entries[0].reason
    assert not output_dir.exists()


def test_path_unsafe_or_ambiguous_variants_are_skipped(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    run_dir = _write_manifest(
        raw_dir,
        jobs=[
            _job(job_id="ambiguous", env_variant_id="other-env::seed-1", results_relpath="ambiguous/results.jsonl"),
            _job(job_id="unsafe", env_variant_id="demo/env::bad value", results_relpath="unsafe/results.jsonl"),
            _job(
                job_id="base-conflict", env_variant_id="demo/env::base", results_relpath="base-conflict/results.jsonl"
            ),
        ],
    )
    for job_id in ("ambiguous", "unsafe", "base-conflict"):
        _write_artifacts(run_dir, job_id=job_id)

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=tmp_path / "evals", dry_run=False)

    assert report.skipped == 3
    reasons = {entry.job_id: entry.reason for entry in report.entries}
    assert "ambiguous env_variant_id" in reasons["ambiguous"]
    assert "path-unsafe variant" in reasons["unsafe"]
    assert "reserved base" in reasons["base-conflict"]


def test_parses_relative_variant_and_cli_report_path(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    report_path = tmp_path / "report.json"
    run_dir = _write_manifest(raw_dir, jobs=[_job(env_variant_id="demo/env::shuffle_seed-1618")])
    _write_artifacts(run_dir)

    exit_code = main(
        [
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--no-dry-run",
            "--report-path",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert (output_dir / "gpt-mini" / "demo-env" / "shuffle_seed-1618" / "metadata.json").exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["converted"] == 1


def test_parses_legacy_delimited_env_variant_ids(tmp_path: Path) -> None:
    raw_dir = tmp_path / "runs" / "raw"
    output_dir = tmp_path / "runs" / "evals"
    run_dir = _write_manifest(
        raw_dir,
        jobs=[
            _job(
                job_id="longhealth-task1",
                env_id="longhealth",
                env_variant_id="longhealth-task1",
                results_relpath="longhealth-task1/results.jsonl",
            ),
            _job(
                job_id="careqa-en",
                env_id="careqa",
                env_variant_id="careqa_en",
                env_args={"split": "en"},
                results_relpath="careqa-en/results.jsonl",
            ),
            _job(
                job_id="pubhealthbench-reviewed",
                env_id="pubhealthbench",
                env_variant_id="pubhealthbench_reviewed",
                env_args={"split": "reviewed"},
                results_relpath="pubhealthbench-reviewed/results.jsonl",
            ),
        ],
    )
    _write_artifacts(run_dir, job_id="longhealth-task1")
    _write_artifacts(run_dir, job_id="careqa-en")
    _write_artifacts(run_dir, job_id="pubhealthbench-reviewed")

    report = convert_legacy_raw_runs(raw_dir=raw_dir, output_dir=output_dir, dry_run=False)

    assert report.converted == 3
    assert (output_dir / "gpt-mini" / "longhealth" / "task1" / "metadata.json").exists()
    assert (output_dir / "gpt-mini" / "careqa" / "base" / "metadata.json").exists()
    assert (output_dir / "gpt-mini" / "pubhealthbench" / "reviewed" / "metadata.json").exists()
