from pathlib import Path

from medarc_verifiers.orchestrate.task_naming import sanitize_task_dirname, task_root_for_id


def test_sanitize_task_dirname_removes_colons() -> None:
    dirname = sanitize_task_dirname("job-1:foo")
    assert ":" not in dirname
    assert dirname.startswith("job-1-foo-")


def test_task_root_for_id_prefers_existing_raw(tmp_path: Path) -> None:
    raw = tmp_path / "tasks" / sanitize_task_dirname("job-1:foo")
    raw.mkdir(parents=True)
    resolved = task_root_for_id(tmp_path, "job-1:foo")
    assert resolved == raw
