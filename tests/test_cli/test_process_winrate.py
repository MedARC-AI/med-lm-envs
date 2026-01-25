from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from medarc_verifiers.cli import winrate


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


def test_unknown_include_models_error(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_dataset(
        dataset,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    cfg = winrate.WinrateConfig(include_models=("model_a", "model_missing"))
    with pytest.raises(ValueError, match="Unknown include model ids"):
        winrate.compute_winrates([("dataset", dataset)], cfg, known_models=["model_a"])


def test_unknown_exclude_models_warns(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_dataset(
        dataset,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    cfg = winrate.WinrateConfig(exclude_models=("model_missing",))
    result = winrate.compute_winrates([("dataset", dataset)], cfg, known_models=["model_a"])
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "model_missing" in captured.out
    payload = winrate.to_json(result)
    assert set(payload["models"]) == {"model_a"}


def test_partial_datasets_strict_drops_missing_models(tmp_path: Path) -> None:
    ds_one = tmp_path / "dataset_one.parquet"
    _write_dataset(
        ds_one,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
            {"example_id": "q1", "model_id": "model_b", "reward": 0.0},
        ],
    )
    ds_two = tmp_path / "dataset_two.parquet"
    _write_dataset(
        ds_two,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 0.5},
        ],
    )
    cfg = winrate.WinrateConfig(
        include_models=("model_a", "model_b"),
        partial_datasets="strict",
    )
    result = winrate.compute_winrates(
        [("dataset_one", ds_one), ("dataset_two", ds_two)],
        cfg,
    )
    payload = winrate.to_json(result)
    assert set(payload["datasets"]) == {"dataset_one"}
    assert payload["models"]["model_a"]["mean_winrate"]["n_datasets"] == 1


def test_partial_datasets_include_adds_missing_models(tmp_path: Path) -> None:
    ds_one = tmp_path / "dataset_one.parquet"
    _write_dataset(
        ds_one,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
            {"example_id": "q1", "model_id": "model_b", "reward": 0.0},
        ],
    )
    ds_two = tmp_path / "dataset_two.parquet"
    _write_dataset(
        ds_two,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 0.5},
        ],
    )
    cfg = winrate.WinrateConfig(
        include_models=("model_a", "model_b"),
        partial_datasets="include",
    )
    result = winrate.compute_winrates(
        [("dataset_one", ds_one), ("dataset_two", ds_two)],
        cfg,
    )
    payload = winrate.to_json(result)
    assert set(payload["datasets"]) == {"dataset_one", "dataset_two"}
    per_dataset = payload["models"]["model_a"]["vs"]["model_b"]["per_dataset"]
    assert per_dataset["dataset_two"] == pytest.approx(1.0)


def test_dataset_failure_errors(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.parquet"
    df = pl.DataFrame({"example_id": ["q1"], "reward": [1.0]})
    df.write_parquet(dataset)
    cfg = winrate.WinrateConfig()
    with pytest.raises(ValueError, match="Failed to process dataset"):
        winrate.compute_winrates([("bad", dataset)], cfg)


def test_pairwise_epsilon_boundary() -> None:
    df_wide = pl.DataFrame(
        {
            "example_id": ["q1", "q2"],
            "model_a": [0.1, 0.3],
            "model_b": [0.0, 0.0],
        }
    )
    wr, n_used = winrate.pairwise_win_rate(df_wide, "model_a", "model_b", epsilon=0.1)
    assert n_used == 2
    assert wr == pytest.approx(0.75)


def test_pairwise_missing_policy_affects_outcome() -> None:
    df_wide = pl.DataFrame(
        {
            "example_id": ["q1"],
            "model_a": [None],
            "model_b": [-1.0],
        }
    )
    wr_neg_inf, _ = winrate.pairwise_win_rate(df_wide, "model_a", "model_b", missing_policy="neg-inf")
    wr_zero, _ = winrate.pairwise_win_rate(df_wide, "model_a", "model_b", missing_policy="zero")
    assert wr_neg_inf == pytest.approx(0.0)
    assert wr_zero == pytest.approx(1.0)


def test_pairwise_both_missing_excluded_and_min_common() -> None:
    df_wide = pl.DataFrame(
        {
            "example_id": ["q1", "q2"],
            "model_a": [None, 1.0],
            "model_b": [None, None],
        }
    )
    wr, n_used = winrate.pairwise_win_rate(df_wide, "model_a", "model_b")
    assert n_used == 1
    assert wr == pytest.approx(1.0)

    wr_min_common, n_used_min = winrate.pairwise_win_rate(df_wide, "model_a", "model_b", min_common=2)
    assert wr_min_common is None
    assert n_used_min == 0


def test_rollout_averaging_sets_n_questions(tmp_path: Path) -> None:
    dataset = tmp_path / "rollouts.parquet"
    df = pl.DataFrame(
        {
            "example_id": ["q1", "q1", "q1", "q1"],
            "model_id": ["model_a", "model_a", "model_b", "model_b"],
            "reward": [1.0, 0.0, 0.5, 0.5],
            "rollout_index": [0, 1, 0, 1],
        }
    )
    df.write_parquet(dataset)
    cfg = winrate.WinrateConfig()
    result = winrate.compute_winrates([("rollouts", dataset)], cfg)
    payload = winrate.to_json(result)
    assert payload["datasets"]["rollouts"]["n_questions"] == 1


def test_include_exclude_remove_all_models_errors(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_dataset(
        dataset,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    cfg = winrate.WinrateConfig(include_models=("model_a",), exclude_models=("model_a",))
    with pytest.raises(ValueError, match="No models remain"):
        winrate.compute_winrates([("dataset", dataset)], cfg)


def test_partial_datasets_noop_without_include(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_dataset(
        dataset,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    cfg = winrate.WinrateConfig(partial_datasets="strict")
    result = winrate.compute_winrates([("dataset", dataset)], cfg)
    payload = winrate.to_json(result)
    assert set(payload["datasets"]) == {"dataset"}


def test_unknown_include_models_error_without_known_models(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_dataset(
        dataset,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    cfg = winrate.WinrateConfig(include_models=("model_missing",))
    with pytest.raises(ValueError, match="Unknown include model ids"):
        winrate.compute_winrates([("dataset", dataset)], cfg)


def test_unknown_exclude_models_warns_without_known_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset.parquet"
    _write_dataset(
        dataset,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    cfg = winrate.WinrateConfig(exclude_models=("model_missing",))
    result = winrate.compute_winrates([("dataset", dataset)], cfg)
    captured = capsys.readouterr()
    assert "Warning" in captured.out
    assert "model_missing" in captured.out
    payload = winrate.to_json(result)
    assert set(payload["models"]) == {"model_a"}


def test_dataset_failure_multi_file_error_includes_paths(tmp_path: Path) -> None:
    valid = tmp_path / "valid.parquet"
    invalid = tmp_path / "invalid.parquet"
    _write_dataset(
        valid,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    df = pl.DataFrame({"example_id": ["q2"], "reward": [0.5]})
    df.write_parquet(invalid)
    cfg = winrate.WinrateConfig()
    with pytest.raises(ValueError) as excinfo:
        winrate.compute_winrates([("multi", [valid, invalid])], cfg)
    message = str(excinfo.value)
    assert "Failed to process dataset multi" in message
    assert str(valid) in message
    assert str(invalid) in message


def test_run_winrate_validates_known_models_from_env_index(tmp_path: Path) -> None:
    processed_dir = tmp_path / "processed"
    output_dir = tmp_path / "out"
    dataset_path = processed_dir / "envs" / "env-a.parquet"
    _write_dataset(
        dataset_path,
        [
            {"example_id": "q1", "model_id": "model_a", "reward": 1.0},
        ],
    )
    env_index = {
        "version": 2,
        "processed_at": "2024-01-01T00:00:00Z",
        "schema_version": 1,
        "processed_with_args": {},
        "runs": {},
        "files": {
            "envs/env-a.parquet": {
                "env_id": "env-a",
                "model_id": "model_a",
                "row_count": 1,
            }
        },
    }
    index_path = processed_dir / "env_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(env_index), encoding="utf-8")

    cfg = winrate.WinrateConfig(include_models=("model_missing",))
    with pytest.raises(ValueError, match="Unknown include model ids"):
        winrate.run_winrate(
            processed_dir=processed_dir,
            output_dir=output_dir,
            output_path=None,
            output_name=None,
            config=cfg,
        )
