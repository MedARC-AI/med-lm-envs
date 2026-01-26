from __future__ import annotations

from medarc_verifiers.cli.process.aggregate import (
    AggregatedEnvRows,
    aggregate_rows_by_env,
)


def test_aggregate_rows_by_env_unions_columns() -> None:
    rows = [
        {"env_id": "env-a-rollout1", "base_env_id": "env-a", "reward": 0.5, "metric_a": 1.0},
        {"env_id": "env-a-rollout2", "base_env_id": "env-a", "reward": 0.6, "metric_b": 2.0},
        {"env_id": "env-b", "base_env_id": "env-b", "reward": 0.2},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert [group.base_env_id for group in grouped] == ["env-a", "env-b"]

    env_a = grouped[0]
    assert isinstance(env_a, AggregatedEnvRows)
    assert len(env_a.rows) == 2
    assert set(env_a.column_names) == {"env_id", "base_env_id", "reward", "metric_a", "metric_b"}

    env_b = grouped[1]
    assert len(env_b.rows) == 1
    assert set(env_b.column_names) == {"env_id", "base_env_id", "reward"}


def test_aggregate_rows_by_env_groups_by_model() -> None:
    rows = [
        {"env_id": "env-a", "base_env_id": "env-a", "model_id": "m1", "job_run_id": "r1"},
        {"env_id": "env-a", "base_env_id": "env-a", "model_id": "m2", "job_run_id": "r2"},
        {"env_id": "env-a", "base_env_id": "env-a", "model_id": "m1", "job_run_id": "r3"},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert len(grouped) == 2
    by_key = {(g.model_id, g.env_id): g for g in grouped}
    assert set(by_key.keys()) == {("m1", "env-a"), ("m2", "env-a")}
    assert by_key[("m1", "env-a")].job_run_ids == ("r1", "r3")
    assert by_key[("m2", "env-a")].job_run_ids == ("r2",)


def test_aggregate_rows_keeps_env_id() -> None:
    rows = [
        {"env_id": "longhealth-task1", "base_env_id": "longhealth-task1", "reward": 0.4},
        {"env_id": "longhealth-task1", "base_env_id": "longhealth-task1", "reward": 0.5},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert len(grouped) == 1
    assert grouped[0].env_id == "longhealth-task1"
    assert grouped[0].base_env_id == "longhealth-task1"


def test_aggregate_rows_ignores_missing_env_id() -> None:
    rows = [
        {"reward": 0.5},
        {"base_env_id": "env-a", "reward": 0.6},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert len(grouped) == 1
    assert grouped[0].base_env_id == "env-a"


def test_aggregate_rows_fallback_base_env_id_is_string() -> None:
    rows = [
        {"env_id": "env-a", "reward": 0.6},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert len(grouped) == 1
    assert grouped[0].base_env_id == "env-a"
    assert isinstance(grouped[0].base_env_id, str)


def test_aggregate_rows_tracks_job_runs() -> None:
    rows = [
        {"env_id": "env-a", "base_env_id": "env-a", "job_run_id": "r2"},
        {"env_id": "env-a", "base_env_id": "env-a", "job_run_id": "r1"},
        {"env_id": "env-a", "base_env_id": "env-a", "job_run_id": "r2"},
    ]
    grouped = aggregate_rows_by_env(rows)
    assert grouped[0].job_run_ids == ("r1", "r2")


def test_aggregate_rows_normalizes_rollout_index() -> None:
    rows = [
        {"env_id": "env-a", "base_env_id": "env-a", "rollout_index": 7},
        {"env_id": "env-a", "base_env_id": "env-a", "rollout_index": 3},
        {"env_id": "env-a", "base_env_id": "env-a", "rollout_index": 7},
    ]
    grouped = aggregate_rows_by_env(rows)
    assert grouped[0].rows[0]["rollout_index"] in {0, 1}
    assert sorted({row["rollout_index"] for row in grouped[0].rows}) == [0, 1]
