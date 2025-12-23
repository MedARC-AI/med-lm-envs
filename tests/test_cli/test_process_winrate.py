from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from medarc_verifiers.cli_new.process import winrate


def _write_dataset(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(
        rows,
        schema={
            "example_id": pl.Utf8,
            "model_id": pl.Utf8,
            "reward": pl.Float64,
        },
    )
    df.write_parquet(path)


def test_compute_winrates_two_datasets(tmp_path: Path) -> None:
    ds_one = tmp_path / "dataset_one.parquet"
    _write_dataset(
        ds_one,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
            {"example_id": "q1", "model_id": "model_b", "reward": 0.0},
            {"example_id": "q2", "model_id": "model_a", "reward": 0.0},
            {"example_id": "q2", "model_id": "model_b", "reward": 1.0},
            {"example_id": "q3", "model_id": "model_a", "reward": 0.8},
            {"example_id": "q3", "model_id": "model_b", "reward": 0.6},
        ],
    )
    ds_two = tmp_path / "dataset_two.parquet"
    _write_dataset(
        ds_two,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 0.2},
            {"example_id": "q1", "model_id": "model_c", "reward": 0.1},
            {"example_id": "q2", "model_id": "model_a", "reward": None},
            {"example_id": "q2", "model_id": "model_c", "reward": 0.3},
            {"example_id": "q3", "model_id": "model_a", "reward": 0.5},
            {"example_id": "q3", "model_id": "model_c", "reward": 0.5},
        ],
    )

    cfg = winrate.WinrateConfig()
    result = winrate.compute_winrates(
        [
            ("dataset_one", ds_one),
            ("dataset_two", ds_two),
        ],
        cfg,
    )
    payload = winrate.to_json(result)
    models = payload["models"]

    assert set(models) == {"model_a", "model_b", "model_c"}

    model_a = models["model_a"]
    assert model_a["mean_winrate"]["n_datasets"] == 2
    assert model_a["mean_winrate"]["simple_mean"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert model_a["mean_winrate"]["weighted_mean"] == pytest.approx((2 / 3 + 0.5) / 2)
    assert model_a["vs"]["model_b"]["per_dataset"]["dataset_one"] == pytest.approx(2 / 3)
    assert model_a["vs"]["model_c"]["per_dataset"]["dataset_two"] == pytest.approx(0.5)
    assert model_a["avg_reward_per_dataset"]["dataset_one"] == pytest.approx(0.6)
    assert model_a["avg_reward_per_dataset"]["dataset_two"] == pytest.approx(0.35)

    model_b = models["model_b"]
    assert model_b["mean_winrate"]["n_datasets"] == 1
    assert model_b["mean_winrate"]["simple_mean"] == pytest.approx(1 / 3)
    assert model_b["vs"]["model_a"]["per_dataset"]["dataset_one"] == pytest.approx(1 / 3)
    assert model_b["avg_reward_per_dataset"]["dataset_one"] == pytest.approx(0.5333333333)

    model_c = models["model_c"]
    assert model_c["mean_winrate"]["n_datasets"] == 1
    assert model_c["mean_winrate"]["simple_mean"] == pytest.approx(0.5)
    assert model_c["vs"]["model_a"]["per_dataset"]["dataset_two"] == pytest.approx(0.5)
    assert model_c["avg_reward_per_dataset"]["dataset_two"] == pytest.approx(0.3)


def test_read_dataset_lazy_supports_model_id(tmp_path: Path) -> None:
    dataset = tmp_path / "model_id.parquet"
    df = pl.DataFrame(
        {
            "example_id": ["ex-1", "ex-2"],
            "model_id": ["m1", "m2"],
            "reward": [1.0, 0.5],
        }
    )
    df.write_parquet(dataset)

    lf = winrate.read_dataset_lazy(dataset)
    df_avg, _ = winrate.average_rollouts(lf)

    assert "model_id" in df_avg.columns
    assert sorted(df_avg["model_id"].unique().to_list()) == ["m1", "m2"]
