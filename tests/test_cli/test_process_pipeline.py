from __future__ import annotations

import json
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from medarc_verifiers.cli._schemas import EnvironmentExportConfig
from medarc_verifiers.cli.process import ProcessOptions, run_process
from medarc_verifiers.cli.winrate import WinrateConfig
from medarc_verifiers.cli.winrate import discover_datasets, run_winrate
from medarc_verifiers.cli.hf import HFSyncConfig
from medarc_verifiers.cli.process.writer import ALLOWED_COLUMNS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _setup_run(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "run-1"
    results_dir = run_dir / "demo-job"
    manifest = {
        "version": 2,
        "run_id": "run-1",
        "name": "demo",
        "config_source": "configs/demo.yaml",
        "config_snapshot": {"jobs": []},
        "config_checksum": "abc123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "models": {"gpt-mini": {"sampling_args": {}}},
        "env_templates": {"demo-env-template": {"module": "demo-env-rollout3"}},
        "summary": {
            "total": 1,
            "completed": 1,
            "pending": 0,
            "running": 0,
            "failed": 0,
            "skipped": 0,
        },
        "jobs": [
            {
                "job_id": "demo-job",
                "model_id": "gpt-mini",
                "env_id": "demo-env-rollout3",
                "env_template_id": "demo-env-template",
                "env_variant_id": "demo-env-rollout3",
                "env_args": {},
                "results_dir": "demo-job",
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


def _write_run(
    tmp_path: Path,
    *,
    run_id: str,
    updated_at: str,
    reward: float,
    env_id: str = "demo-env-rollout3",
    model_id: str = "gpt-mini",
) -> Path:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / run_id
    results_dir = run_dir / "demo-job"
    manifest = {
        "version": 2,
        "run_id": run_id,
        "name": "demo",
        "config_source": "configs/demo.yaml",
        "config_snapshot": {"jobs": []},
        "config_checksum": "abc123",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": updated_at,
        "models": {model_id: {"sampling_args": {}}},
        "env_templates": {"demo-env-template": {"module": env_id}},
        "summary": {
            "total": 1,
            "completed": 1,
            "pending": 0,
            "running": 0,
            "failed": 0,
            "skipped": 0,
        },
        "jobs": [
            {
                "job_id": "demo-job",
                "model_id": model_id,
                "env_id": env_id,
                "env_template_id": "demo-env-template",
                "env_variant_id": env_id,
                "env_args": {},
                "results_dir": "demo-job",
            }
        ],
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    metadata = {
        "env_id": env_id,
        "env_args": {},
        "sampling_args": {},
    }
    _write_json(results_dir / "metadata.json", metadata)
    results_path = results_dir / "results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"example_id": f"ex-{run_id}", "reward": reward}
    results_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return runs_dir


def test_run_process_respects_env_export_defaults(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=1,
    )
    env_export = {
        "demo-env": EnvironmentExportConfig(
            extra_columns=["debug"],
        )
    }

    result = run_process(options, env_export_map=env_export)

    assert result.records_processed == 1
    assert result.rows_processed == 1
    group = result.env_groups[0]
    assert group.rows == []
    # env_id now resolves to the base environment id; rollout info remains in base_env_id/derivation
    assert group.env_id == "demo-env"
    assert group.base_env_id == "demo-env"
    assert group.model_id == "gpt-mini"


def test_run_process_resolves_base_env_id(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)
    group = result.env_groups[0]
    assert group.env_id == "demo-env"
    assert group.base_env_id == "demo-env"
    assert group.model_id == "gpt-mini"


def test_run_winrate_from_processed_outputs(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    process_opts = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=False,
        processed_at="2024-01-01T00:00:00Z",
        max_workers=1,
    )

    result_process = run_process(process_opts)
    index_path = output_dir / "env_index.json"
    assert index_path.exists()
    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_payload["version"] == 2
    rel_path = result_process.env_summaries[0].output_path.relative_to(output_dir).as_posix()
    assert rel_path in index_payload["files"]
    # Parquet schema should be trimmed to the fixed allowed columns for HF loading
    summary = result_process.env_summaries[0]
    schema = pq.read_schema(summary.output_path)
    assert schema.names == list(ALLOWED_COLUMNS)
    table = pq.read_table(summary.output_path)
    assert table.column("answer").to_pylist() == [None]

    cfg = WinrateConfig()
    result = run_winrate(
        processed_dir=output_dir,
        output_dir=tmp_path / "winrate",
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
    latest_csv = (tmp_path / "winrate" / "latest.csv").read_text(encoding="utf-8").splitlines()
    assert latest_csv
    header = latest_csv[0].split(",")
    assert header[0] == "model"
    assert header[1] == "weighted_winrate"
    assert header[2] == "simple_winrate"
    assert header[-1] == "num_datasets"


def test_run_winrate_from_hf(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Prepare a fake HF snapshot on disk
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    parquet_path = hf_dir / "demo-env.parquet"
    payload = [
        {"example_id": "ex-1", "model_id": "alpha", "reward": 0.8},
        {"example_id": "ex-1", "model_id": "beta", "reward": 0.2},
    ]
    import pandas as pd  # type: ignore[import-not-found]

    pd.DataFrame(payload).to_parquet(parquet_path, index=False)
    env_index = {
        "version": 2,
        "processed_at": "2024-01-01T00:00:00Z",
        "schema_version": 1,
        "processed_with_args": {},
        "runs": {},
        "files": {
            "demo-env.parquet": {
                "env_id": "demo-env",
                "model_id": "alpha",
                "row_count": 2,
            }
        },
    }
    (hf_dir / "env_index.json").write_text(json.dumps(env_index), encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_download_hf_repo(*_args, **_kwargs) -> Path:
        captured["kwargs"] = dict(_kwargs)
        return hf_dir

    monkeypatch.setattr("medarc_verifiers.cli.process.workspace.download_hf_repo", _fake_download_hf_repo)

    cfg = WinrateConfig()
    result = run_winrate(
        processed_dir=tmp_path / "processed",
        output_dir=tmp_path / "winrate",
        output_path=None,
        config=cfg,
        processed_at="2024-01-01T00:00:00Z",
        hf_config=HFSyncConfig(repo_id="owner/ds", branch=None, token=None, private=False),
    )

    kwargs = captured.get("kwargs")
    assert isinstance(kwargs, dict)
    assert "allow_patterns" in kwargs
    patterns = kwargs["allow_patterns"]
    if isinstance(patterns, str):
        patterns = [patterns]
    assert "env_index.json" in patterns
    assert any("parquet" in str(item) for item in patterns)

    assert result.output_path.exists()
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert sorted(payload["models"].keys()) == ["alpha", "beta"]


def test_discover_datasets_handles_project_relative_paths(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    processed_dir = tmp_path / "runs" / "processed"
    process_opts = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=processed_dir,
        dry_run=False,
        processed_at="2024-01-01T00:00:00Z",
        max_workers=1,
    )

    run_process(process_opts)

    datasets = discover_datasets(processed_dir)

    assert len(datasets) == 1
    env_id, splits = datasets[0]
    assert env_id == "demo-env"
    assert splits and isinstance(splits[0], Path)


def test_run_process_propagates_keyboard_interrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ensure ctrl+c stops processing promptly."""
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=False,
        max_workers=1,
    )

    call_count = {"count": 0}

    def _boom(*args: object, **kwargs: object) -> None:
        call_count["count"] += 1
        raise KeyboardInterrupt

    monkeypatch.setattr("medarc_verifiers.cli.process.rows.load_rows", _boom)

    with pytest.raises(KeyboardInterrupt):
        run_process(options)

    assert call_count["count"] == 1


def test_run_process_parallel_workers(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=2,
    )

    result = run_process(options)

    assert result.records_processed == 1
    assert result.rows_processed == 1


def test_run_process_empty_runs_returns_result(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)
    assert result.records_processed == 0
    assert result.rows_processed == 0
    assert result.env_groups == []
    assert result.env_summaries == []
    assert result.hf_summary is None


def test_process_latest_only_selects_latest_and_delta_skips(tmp_path: Path) -> None:
    runs_dir = _write_run(tmp_path, run_id="run-1", updated_at="2024-01-01T00:00:00Z", reward=0.1)
    _write_run(tmp_path, run_id="run-2", updated_at="2024-01-02T00:00:00Z", reward=0.9)
    output_dir = tmp_path / "processed"

    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=False,
        processed_at="2024-01-03T00:00:00Z",
        max_workers=1,
    )
    result = run_process(options)

    assert result.env_summaries
    out_path = result.env_summaries[0].output_path
    table = pq.read_table(out_path)
    assert set(table.column("job_run_id").to_pylist()) == {"run-2"}
    assert table.column("reward").to_pylist() == [0.9]

    result_repeat = run_process(options)
    assert result_repeat.env_summaries == []
    assert result_repeat.rows_processed == 0


def test_process_clean_clears_outputs(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    sentinel = output_dir / "stale.txt"
    sentinel.write_text("stale", encoding="utf-8")

    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=False,
        processed_at="2024-01-01T00:00:00Z",
        clean=True,
        assume_yes=True,
        max_workers=1,
    )
    run_process(options)

    assert not sentinel.exists()
    assert (output_dir / "env_index.json").exists()


def test_run_process_ignores_legacy_run_output_path(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    run_dir = runs_dir / "run-1"
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["updated_at"] = "2024-01-01T00:10:00Z"
    _write_json(manifest_path, manifest)

    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    env_index = {
        "version": 2,
        "processed_at": "2024-01-01T00:00:00Z",
        "schema_version": 1,
        "processed_with_args": {},
        "runs": {
            "run-1": {
                "updated_at": "2024-01-01T00:00:00Z",
                "output_path": "gpt-mini/old-env.parquet",
            }
        },
        "files": {},
    }
    _write_json(output_dir / "env_index.json", env_index)

    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)
    assert result.records_processed == 1


def test_run_process_ignores_legacy_index_and_writes_v2(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_index = {
        "version": 1,
        "env_groups": [
            {
                "env_id": "legacy-env",
                "paths": [{"path": "legacy/legacy.parquet"}],
            }
        ],
    }
    (output_dir / "env_index.json").write_text(json.dumps(legacy_index), encoding="utf-8")

    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=False,
        processed_at="2024-01-01T00:00:00Z",
        max_workers=1,
    )
    run_process(options)

    payload = json.loads((output_dir / "env_index.json").read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert all(not Path(path).is_absolute() for path in payload["files"].keys())
