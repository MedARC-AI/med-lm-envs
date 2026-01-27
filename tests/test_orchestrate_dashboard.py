from medarc_verifiers.orchestrate.dashboard import build_table
from medarc_verifiers.orchestrate.state import JobState, TaskManifest


def test_build_table_filters_to_active_states() -> None:
    tasks = [
        TaskManifest(
            task_id="task-1",
            config_path="cfg-1.yaml",
            model_key="foo",
            model_id="Foo/Bar",
            state=JobState.running,
        ),
        TaskManifest(
            task_id="task-2",
            config_path="cfg-2.yaml",
            model_key="bar",
            model_id="Bar/Baz",
            state=JobState.completed,
        ),
    ]

    table = build_table(tasks)

    assert len(table.rows) == 1
