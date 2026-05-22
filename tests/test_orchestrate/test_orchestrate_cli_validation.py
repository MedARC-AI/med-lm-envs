import json
from pathlib import Path

import pytest

from medarc_verifiers.orchestrate.cli import main


def test_run_rejects_missing_source() -> None:
    with pytest.raises(SystemExit, match="requires --plan"):
        main(["run"])


def test_run_parser_rejects_deleted_backend_flag() -> None:
    with pytest.raises(SystemExit):
        main(["run", "--backend", "local", "--suite", "suite.toml", "--endpoint", "foo"])


def test_run_rejects_endpoint_with_plan() -> None:
    with pytest.raises(SystemExit, match="--endpoint is only valid with --suite"):
        main(["run", "--plan", "plan.toml", "--endpoint", "foo"])


def test_cleanup_rejects_pyxis() -> None:
    with pytest.raises(SystemExit):
        main(["cleanup", "--runtime", "pyxis"])


def test_status_combines_submission_manifest_and_summary(tmp_path: Path, capsys) -> None:
    (tmp_path / "submission_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "task_id": "task-a",
                        "state": "submitted",
                        "slurm_job_id": "123",
                        "generated_dependency": "afterany:122",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_id": "task-a",
                        "state": "completed",
                        "model_id": "Foo/Bar",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["status", "--output-dir", str(tmp_path)]) == 0
    assert "task-a\tsubmitted\tcompleted\t123" in capsys.readouterr().out


def test_status_fails_when_artifacts_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No orchestrator status found"):
        main(["status", "--output-dir", str(tmp_path)])
