import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

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

    assert main(["status", "--bundle-dir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "task-a\tsubmitted\tcompleted\t123" in output
    assert "\t122\t" in output
    assert "construct" not in output


def test_status_json_uses_slurm_manifest_key(tmp_path: Path, capsys) -> None:
    (tmp_path / "slurm_manifest.json").write_text(json.dumps({"entries": [], "lifecycle_entries": []}), encoding="utf-8")

    assert main(["status", "--bundle-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "slurm_manifest" in payload
    assert "submission_manifest" not in payload


def test_status_json_enriches_rows_with_live_slurm_state(tmp_path: Path, capsys, monkeypatch) -> None:
    (tmp_path / "slurm_manifest.json").write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "task_id": "task-a",
                        "state": "submitted",
                        "slurm_job_id": "123",
                        "target_endpoint_id": "foo",
                    }
                ],
                "lifecycle_entries": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_run(command, **_kwargs):  # noqa: ANN001, ANN202
        if command[0] == "squeue":
            return SimpleNamespace(
                returncode=0,
                stdout="123|PENDING|Priority|00:00|UNLIMITED|1|16|gres/gpu=1|bottom|benchmarks|0|\n",
                stderr="",
            )
        if command[0] == "sacct":
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "JobID|JobName|State|ExitCode|Elapsed|Submit|Start|End\n"
                    "123|task-a|PREEMPTED|0:0|00:10:00|2026-01-01T00:00:00|"
                    "2026-01-01T00:00:01|2026-01-01T00:10:01\n"
                    "123|task-a|PENDING|0:0|00:00:00|2026-01-01T00:10:02|Unknown|Unknown\n"
                ),
                stderr="",
            )
        if command[0] == "scontrol":
            return SimpleNamespace(
                returncode=0,
                stdout="JobId=123 JobState=PENDING Reason=Priority Restarts=1 Requeue=1",
                stderr="",
            )
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert main(["status", "--bundle-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    task = payload["tasks"][0]
    assert task["eval_slurm_live_state"] == "PENDING"
    assert task["eval_slurm_reason"] == "Priority"
    assert task["eval_slurm_restarts"] == 1
    assert task["eval_slurm_preemptions"] == 1
    assert payload["slurm"]["jobs"]["123"]["attempts"][0]["state"] == "PREEMPTED"


def test_status_fails_when_artifacts_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No orchestrator status found"):
        main(["status", "--bundle-dir", str(tmp_path)])


def test_top_level_help_lists_direct_commands(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "{run,prepare,launch,teardown,status,cleanup}" in help_text
