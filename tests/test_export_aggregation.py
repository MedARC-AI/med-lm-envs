from __future__ import annotations

from medarc_verifiers.export.parquet.aggregation import AggregatedEnvRows, aggregate_rows_by_env


def test_aggregate_rows_by_env_unions_columns():
    rows = [
        {"env_id": "env-a", "reward": 0.5, "metric_a": 1.0},
        {"env_id": "env-a", "reward": 0.6, "metric_b": 2.0},
        {"env_id": "env-b", "reward": 0.2},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert [group.env_id for group in grouped] == ["env-a", "env-b"]

    env_a = grouped[0]
    assert isinstance(env_a, AggregatedEnvRows)
    assert len(env_a.rows) == 2
    assert set(env_a.column_names) == {"env_id", "reward", "metric_a", "metric_b"}

    env_b = grouped[1]
    assert len(env_b.rows) == 1
    assert set(env_b.column_names) == {"env_id", "reward"}


def test_aggregate_rows_by_env_with_partitions():
    rows = [
        {"env_id": "env-a", "model": "m1", "job_run_id": "r1"},
        {"env_id": "env-a", "model": "m2", "job_run_id": "r2"},
        {"env_id": "env-a", "model": "m1", "job_run_id": "r3"},
    ]

    grouped = aggregate_rows_by_env(rows, partition_by=("model",))
    assert len(grouped) == 1
    partitions = grouped[0].partitions
    assert set(partitions.keys()) == {("m1",), ("m2",)}
    assert len(partitions[("m1",)]) == 2
    assert len(partitions[("m2",)]) == 1


def test_aggregate_rows_ignores_missing_env_id():
    rows = [
        {"reward": 0.5},
        {"env_id": "env-a", "reward": 0.6},
    ]

    grouped = aggregate_rows_by_env(rows)
    assert len(grouped) == 1
    assert grouped[0].env_id == "env-a"
