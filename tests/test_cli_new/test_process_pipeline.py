from __future__ import annotations

import json
from pathlib import Path

import pytest

from medarc_verifiers.cli_new._schemas import EnvironmentExportConfig
from medarc_verifiers.cli_new.process import ProcessOptions, run_process
from medarc_verifiers.cli_new.process.winrate import WinrateConfig
from medarc_verifiers.cli_new.process.winrate_runner import discover_datasets, run_winrate
from medarc_verifiers.cli_new.process.hf_sync import HFSyncConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setup_run(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    results_dir = run_dir / "demo-job"
    manifest = {
        "run_id": "run-1",
        "name": "demo",
        "config_source": "configs/demo.yaml",
        "config_snapshot": {"jobs": []},
        "config_checksum": "abc123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "jobs": [
            {
                "job_id": "demo-job",
                "env_id": "demo-env-rollout3",
                "model_id": "gpt-mini",
                "results_dir": "demo-job",
                "checksum": "deadbeef",
            }
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    metadata = {
        "env_id": "demo-env-rollout3",
        "env_args": {},
        "sampling_args": {},
    }
    _write_json(results_dir / "metadata.json", metadata)
    results = [
        {
            "example_id": "ex-1",
            "prompt": "Question?",
            "completion": "Answer",
            "info": {"debug": True},
            "reward": 1.0,
        }
    ]
    results_path = results_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row) + "\n")
    return runs_dir


def test_run_process_respects_env_export_defaults(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exporter_version="0.1.0",
        dry_run=True,
        max_workers=1,
    )
    env_export = {
        "demo-env": EnvironmentExportConfig(
            keep_columns=["info"],
            include_prompt_completion=True,
        )
    }

    result = run_process(options, env_export_map=env_export)

    assert result.records_processed == 1
    assert result.rows_processed == 1
    group = result.env_groups[0]
    row = group.rows[0]
    assert row["prompt"] == "Question?"
    assert row["completion"] == "Answer"
    assert row["info"] == {"debug": True}
    # env_id now resolves to the base environment id; rollout info remains in base_env_id/derivation
    assert group.env_id == "demo-env"
    assert group.base_env_id == "demo-env"


def test_run_process_cli_overrides_env_export(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exporter_version="0.1.0",
        dry_run=True,
        include_prompt_completion=False,
        keep_columns=("reward",),
        max_workers=1,
    )
    env_export = {
        "demo-env": EnvironmentExportConfig(
            keep_columns=["info"],
            include_prompt_completion=True,
        )
    }

    result = run_process(options, env_export_map=env_export)
    row = result.env_groups[0].rows[0]
    assert "prompt" not in row
    assert "completion" not in row
    assert "info" not in row
    assert row["reward"] == 1.0


def test_run_process_respects_combine_rollouts_override(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exporter_version="0.1.0",
        dry_run=True,
        max_workers=1,
    )
    env_export = {
        "demo-env-rollout3": EnvironmentExportConfig(
            keep_columns=[],
            combine_rollouts=True,
        )
    }

    result = run_process(options, env_export_map=env_export)
    group = result.env_groups[0]
    assert group.env_id == "demo-env"
    assert group.base_env_id == "demo-env"


def test_run_winrate_from_processed_outputs(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    process_opts = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        exporter_version="0.1.0",
        dry_run=False,
        processed_at="2024-01-01T00:00:00Z",
        max_workers=1,
    )

    run_process(process_opts)

    cfg = WinrateConfig()
    result = run_winrate(
        processed_dir=output_dir,
        output_path=None,
        config=cfg,
        processed_at="2024-01-01T00:00:00Z",
    )

    assert result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["models"]
    model_payload = payload["models"]["gpt-mini"]
    assert model_payload["vs"] == {}
    assert model_payload["mean_winrate"]["n_datasets"] == 0
    assert model_payload["mean_winrate"]["simple_mean"] is None
    assert model_payload["mean_winrate"]["weighted_mean"] is None
    avg_rewards = model_payload["avg_reward_per_dataset"]
    assert len(avg_rewards) == 1
    assert list(avg_rewards.values())[0] == pytest.approx(1.0)


def test_run_winrate_from_hf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Prepare a fake HF split on disk
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    parquet_path = hf_dir / "demo-env.parquet"
    payload = [
        {"example_id": "ex-1", "model_id": "alpha", "reward": 0.8},
        {"example_id": "ex-1", "model_id": "beta", "reward": 0.2},
    ]
    import pandas as pd  # type: ignore[import-not-found]

    pd.DataFrame(payload).to_parquet(parquet_path, index=False)

    def _fake_download_hf_repo(config: HFSyncConfig) -> Path:
        return hf_dir

    monkeypatch.setattr("medarc_verifiers.cli_new.process.winrate_runner._download_hf_repo", _fake_download_hf_repo)

    cfg = WinrateConfig()
    result = run_winrate(
        processed_dir=tmp_path / "processed",
        output_path=None,
        config=cfg,
        processed_at="2024-01-01T00:00:00Z",
        hf_config=HFSyncConfig(repo_id="owner/ds", merge_strategy="append", branch=None, token=None, private=False),
    )

    assert result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert sorted(payload["models"].keys()) == ["alpha", "beta"]


def test_discover_datasets_handles_project_relative_paths(tmp_path: Path) -> None:
    processed_dir = tmp_path / "runs" / "processed"
    processed_dir.mkdir(parents=True)
    parquet_path = processed_dir / "demo.parquet"
    parquet_path.write_text("", encoding="utf-8")
    env_index = {
        "processed_at": "2024-01-01T00:00:00Z",
        "environments": [
            {
                "env_id": "demo",
                "base_env_id": "demo",
                "path": "runs/processed/demo.parquet",
                "row_count": 0,
                "job_run_ids": [],
                "exporter_metadata": {},
            }
        ],
    }
    index_path = processed_dir / "env_index.json"
    index_path.write_text(json.dumps(env_index), encoding="utf-8")

    datasets = discover_datasets(processed_dir)

    assert datasets == [("demo", parquet_path)]


def test_run_process_propagates_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure ctrl+c stops processing promptly."""
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exporter_version="0.1.0",
        dry_run=False,
        max_workers=1,
    )

    call_count = {"count": 0}

    def _boom(*args: object, **kwargs: object) -> None:
        call_count["count"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr("medarc_verifiers.cli_new.process.rows.load_rows", _boom)

    with pytest.raises(KeyboardInterrupt):
        run_process(options)

    assert call_count["count"] == 1


def test_run_process_parallel_workers(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exporter_version="0.1.0",
        dry_run=True,
        max_workers=2,
    )

    result = run_process(options)

    assert result.records_processed == 1
    assert result.rows_processed == 1
    assert result.env_summaries[0].row_count == 1
