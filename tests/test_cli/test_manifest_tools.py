from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from medarc_verifiers.cli._manifest_tools import validate_manifests_in_runs


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_convert_manifest_v2_to_v3_script_rewrites_paths_and_drops_redundant_fields(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "raw" / "demo-run"
    job_dir = run_dir / "job-1"
    _write_json(job_dir / "metadata.json", {"env_id": "demo"})
    (job_dir / "results.jsonl").write_text('{"example_id": 1}\n{"example_id": 2}\n', encoding="utf-8")

    src = run_dir / "run_manifest.json"
    payload = {
        "version": 2,
        "run_id": "demo-run",
        "name": "demo",
        "config_source": "cfg.yaml",
        "config_checksum": "x",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "models": {},
        "env_templates": {},
        "jobs": [
            {
                "job_id": "job-1",
                "job_name": "job-1",
                "model_id": "m",
                "env_id": "e",
                "env_template_id": "e:t",
                "env_variant_id": "e",
                "env_args": {},
                "results_dir": "job-1",
                "status": "completed",
            }
        ],
        "summary": {"total": 1, "completed": 1, "pending": 0, "failed": 0, "running": 0, "skipped": 0},
    }
    _write_json(src, payload)

    out = run_dir / "run_manifest.v3.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/convert_manifest_v2_to_v3.py",
            str(src),
            "--output",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    migrated = json.loads(out.read_text(encoding="utf-8"))
    assert migrated["version"] == 3
    assert migrated["artifacts_root"] == "."
    job = migrated["jobs"][0]
    assert job["results_relpath"] == "job-1/results.jsonl"
    assert job["metadata_relpath"] == "job-1/metadata.json"
    assert "summary_relpath" not in job
    assert "results_dir" not in job
    assert "artifacts_checksum" not in job
    assert "artifacts" not in job
    assert job["row_count"] == 2
