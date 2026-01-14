from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from medarc_verifiers.cli._manifest import MANIFEST_FILENAME, RunManifestModel
from medarc_verifiers.cli.process.discovery import discover_run_records


def _job_config(job_id: str, env_id: str, *, seed: int) -> dict:
    return {
        "job_id": job_id,
        "job_name": job_id,
        "sleep": None,
        "model": {
            "id": "model-a",
            "model": "gpt-mini",
            "sampling_args": {"temperature": 0.2},
        },
        "env": {
            "id": env_id,
            "module": "env-module",
            "env_args": {"seed": seed},
        },
    }


def test_manifest_conversion_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    job1 = _job_config("job-1", "env-a", seed=1)
    job2 = _job_config("job-2", "env-b", seed=2)

    manifest_v1 = {
        "version": 1,
        "run_id": "demo-run",
        "name": "Demo Run",
        "config_source": "configs/demo.yaml",
        "config_checksum": "abc123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:10Z",
        "jobs": [
            {
                "job_id": "job-1",
                "job_name": "job-1",
                "model_id": "model-a",
                "env_id": "env-module",
                "results_dir": "job-1",
                "config": job1,
            },
            {
                "job_id": "job-2",
                "job_name": "job-2",
                "model_id": "model-a",
                "env_id": "env-module",
                "results_dir": "job-2",
                "config": job2,
            },
        ],
        "summary": {"total": 2, "completed": 0, "failed": 0, "pending": 2, "running": 0, "skipped": 0},
    }

    manifest_path = run_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest_v1, indent=2), encoding="utf-8")

    output_path = run_dir / "run_manifest.v2.json"
    subprocess.check_call(
        [
            sys.executable,
            "scripts/convert_manifest_v1_to_v2.py",
            "--no-walk-all",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )

    converted = json.loads(output_path.read_text(encoding="utf-8"))
    RunManifestModel.model_validate(converted)

    manifest_path.write_text(json.dumps(converted, indent=2), encoding="utf-8")
    records = discover_run_records(tmp_path / "runs")
    assert len(records) == 2
