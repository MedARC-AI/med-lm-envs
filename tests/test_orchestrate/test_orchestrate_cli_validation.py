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


def test_status_combines_slurm_manifest_and_summary(tmp_path: Path, capsys) -> None:
    (tmp_path / "slurm_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "task_id": "task-a",
                        "state": "submitted",
                        "slurm_job_id": "123",
                        "generated_dependency": "afterany:122",
                        "target_endpoint_id": "foo",
                    }
                ],
                "lifecycle_entries": [
                    {
                        "task_id": "task-a",
                        "phase": "prepare",
                        "state": "submitted",
                        "slurm_job_id": "122",
                    },
                    {
                        "task_id": "task-a",
                        "phase": "teardown",
                        "state": "pending",
                    },
                ],
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
    output = capsys.readouterr().out
    assert "task-a\tsubmitted\tcompleted\t123" in output
    assert "\t122\t" in output
    assert "construct" not in output


def test_status_json_uses_slurm_manifest_key(tmp_path: Path, capsys) -> None:
    (tmp_path / "slurm_manifest.json").write_text(json.dumps({"entries": [], "lifecycle_entries": []}), encoding="utf-8")

    assert main(["status", "--output-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "slurm_manifest" in payload
    assert "submission_manifest" not in payload


def test_status_fails_when_artifacts_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No orchestrator status found"):
        main(["status", "--output-dir", str(tmp_path)])


def test_top_level_help_lists_direct_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "{run,prepare,launch,teardown,status,cleanup}" in help_text
