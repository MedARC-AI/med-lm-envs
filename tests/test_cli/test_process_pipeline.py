from __future__ import annotations

import json
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from medarc_verifiers.cli._manifest import MANIFEST_VERSION
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
        "version": MANIFEST_VERSION,
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
        "version_info": {
            "vf_version": "0.1.10",
            "vf_commit": "abc123",
            "env_version": "1.0.0",
            "env_commit": None,
        },
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
    status: str = "completed",
    results_text: str | None = None,
) -> Path:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / run_id
    results_dir = run_dir / "demo-job"
    manifest = {
        "version": MANIFEST_VERSION,
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
            "completed": 1 if status == "completed" else 0,
            "pending": 0,
            "running": 0,
            "failed": 1 if status == "failed" else 0,
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
                "status": status,
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
    if results_text is None:
        row = {"example_id": f"ex-{run_id}", "reward": reward}
        results_text = json.dumps(row) + "\n"
    results_path.write_text(results_text, encoding="utf-8")
    return runs_dir


def _remove_model_id(tmp_path: Path, run_id: str) -> None:
    manifest_path = tmp_path / "runs" / run_id / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["jobs"][0]["model_id"] = None
    manifest["models"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    metadata_path = tmp_path / "runs" / run_id / "demo-job" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("model", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


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


def test_run_process_writes_version_info_column(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=False,
        max_workers=1,
    )

    result = run_process(options)
    summary = result.env_summaries[0]
    table = pq.read_table(summary.output_path)

    assert "version_info" in table.column_names
    encoded = table.column("version_info").to_pylist()[0]
    assert isinstance(encoded, str)
    payload = json.loads(encoded)
    assert payload["vf_version"] == "0.1.10"


def test_run_process_backward_compat_without_version_info(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-no-version",
        updated_at="2024-01-01T00:01:00Z",
        reward=1.0,
        env_id="demo-env-rollout3",
        model_id="gpt-mini",
    )
    output_dir = tmp_path / "processed"
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=output_dir,
        dry_run=False,
        max_workers=1,
    )

    result = run_process(options)
    summary = result.env_summaries[0]
    table = pq.read_table(summary.output_path)

    assert "version_info" in table.column_names
    assert table.column("version_info").to_pylist() == [None]


def test_run_process_excludes_datasets(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_id="run-keep",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        env_id="keep-env-rollout3",
    )
    runs_dir = _write_run(
        tmp_path,
        run_id="run-skip",
        updated_at="2024-01-01T00:01:00Z",
        reward=0.0,
        env_id="skip-env",
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exclude_datasets=("skip-env",),
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 1
    assert len(result.env_groups) == 1
    assert result.env_groups[0].base_env_id == "keep-env"


def test_run_process_excludes_models(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_id="run-keep",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        env_id="demo-env-rollout3",
        model_id="keep-model",
    )
    runs_dir = _write_run(
        tmp_path,
        run_id="run-skip",
        updated_at="2024-01-01T00:01:00Z",
        reward=0.0,
        env_id="demo-env-rollout3",
        model_id="skip-model",
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exclude_models=("skip-model",),
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 1
    assert len(result.env_groups) == 1
    assert result.env_groups[0].model_id == "keep-model"


def test_run_process_excludes_models_case_insensitive(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_id="run-keep",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        env_id="demo-env-rollout3",
        model_id="keep-model",
    )
    runs_dir = _write_run(
        tmp_path,
        run_id="run-skip",
        updated_at="2024-01-01T00:01:00Z",
        reward=0.0,
        env_id="demo-env-rollout3",
        model_id="SKIP-MODEL",
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exclude_models=("skip-model",),
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 1
    assert len(result.env_groups) == 1
    assert result.env_groups[0].model_id == "keep-model"


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


def test_process_latest_only_selects_latest_and_skips_existing_outputs(tmp_path: Path) -> None:
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

    _write_run(tmp_path, run_id="run-3", updated_at="2024-01-04T00:00:00Z", reward=0.4)
    result_newer_raw = run_process(options)
    assert result_newer_raw.env_summaries == []
    assert result_newer_raw.rows_processed == 0


def test_process_replace_model_rebuilds_existing_output(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-1",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.1,
        env_id="demo-env",
        model_id="model-a",
    )
    _write_run(
        tmp_path,
        run_id="run-2",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.2,
        env_id="demo-env",
        model_id="model-b",
    )
    output_dir = tmp_path / "processed"

    run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))
    _write_run(
        tmp_path,
        run_id="run-3",
        updated_at="2024-01-03T00:00:00Z",
        reward=0.9,
        env_id="demo-env",
        model_id="model-a",
    )
    _write_run(
        tmp_path,
        run_id="run-4",
        updated_at="2024-01-03T00:00:00Z",
        reward=0.8,
        env_id="demo-env",
        model_id="model-b",
    )

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            replace_models=("model-a",),
            dry_run=False,
            max_workers=1,
        )
    )

    rebuilt = {summary.model_id for summary in result.env_summaries}
    assert rebuilt == {"model-a"}
    model_a_table = pq.read_table(output_dir / "model-a" / "demo-env.parquet")
    model_b_table = pq.read_table(output_dir / "model-b" / "demo-env.parquet")
    assert model_a_table.column("reward").to_pylist() == [0.9]
    assert model_b_table.column("reward").to_pylist() == [0.2]


def test_process_replace_model_and_env_rebuild_only_intersection(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-1",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.1,
        env_id="env-a",
        model_id="model-a",
    )
    _write_run(
        tmp_path,
        run_id="run-2",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.2,
        env_id="env-b",
        model_id="model-a",
    )
    _write_run(
        tmp_path,
        run_id="run-3",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.3,
        env_id="env-a",
        model_id="model-b",
    )
    output_dir = tmp_path / "processed"
    run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))

    _write_run(
        tmp_path,
        run_id="run-4",
        updated_at="2024-01-03T00:00:00Z",
        reward=0.7,
        env_id="env-a",
        model_id="model-a",
    )
    _write_run(
        tmp_path,
        run_id="run-5",
        updated_at="2024-01-03T00:00:00Z",
        reward=0.8,
        env_id="env-b",
        model_id="model-a",
    )
    _write_run(
        tmp_path,
        run_id="run-6",
        updated_at="2024-01-03T00:00:00Z",
        reward=0.9,
        env_id="env-a",
        model_id="model-b",
    )

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            replace_models=("model-a",),
            replace_envs=("env-a",),
            dry_run=False,
            max_workers=1,
        )
    )

    assert {(summary.model_id, summary.env_id) for summary in result.env_summaries} == {("model-a", "env-a")}
    assert pq.read_table(output_dir / "model-a" / "env-a.parquet").column("reward").to_pylist() == [0.7]
    assert pq.read_table(output_dir / "model-a" / "env-b.parquet").column("reward").to_pylist() == [0.2]
    assert pq.read_table(output_dir / "model-b" / "env-a.parquet").column("reward").to_pylist() == [0.3]


def test_process_fails_fast_on_existing_row_count_mismatch(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    result = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))
    summary = result.env_summaries[0]
    rel_path = summary.output_path.relative_to(output_dir).as_posix()
    payload = json.loads((output_dir / "env_index.json").read_text(encoding="utf-8"))
    payload["files"][rel_path]["row_count"] = summary.row_count + 1
    (output_dir / "env_index.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="env_index.json records"):
        run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))


def test_process_ignores_invalid_superseded_run(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-1",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.1,
        results_text='{"example_id": ',
    )
    _write_run(tmp_path, run_id="run-2", updated_at="2024-01-02T00:00:00Z", reward=0.9)
    output_dir = tmp_path / "processed"

    result = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))

    assert result.env_summaries
    table = pq.read_table(result.env_summaries[0].output_path)
    assert table.column("reward").to_pylist() == [0.9]


def test_process_ignores_superseded_run_missing_model_id(tmp_path: Path) -> None:
    runs_dir = _write_run(tmp_path, run_id="run-1", updated_at="2024-01-01T00:00:00Z", reward=0.1)
    _remove_model_id(tmp_path, "run-1")
    _write_run(tmp_path, run_id="run-2", updated_at="2024-01-02T00:00:00Z", reward=0.9)

    result = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=tmp_path / "processed", dry_run=False, max_workers=1))

    table = pq.read_table(result.env_summaries[0].output_path)
    assert table.column("reward").to_pylist() == [0.9]


def test_process_latest_missing_model_id_fails_clearly(tmp_path: Path) -> None:
    runs_dir = _write_run(tmp_path, run_id="run-1", updated_at="2024-01-01T00:00:00Z", reward=0.1)
    _write_run(tmp_path, run_id="run-2", updated_at="2024-01-02T00:00:00Z", reward=0.9)
    _remove_model_id(tmp_path, "run-2")

    with pytest.raises(RuntimeError, match=r"Missing model_id for run \(job_run_id=run-2, job_id=demo-job,"):
        run_process(ProcessOptions(runs_dir=runs_dir, output_dir=tmp_path / "processed", dry_run=False, max_workers=1))


def test_process_ignores_invalid_incomplete_run_by_default(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-1",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.1,
        status="running",
        results_text='{"example_id": ',
    )
    _write_run(tmp_path, run_id="run-2", updated_at="2024-01-02T00:00:00Z", reward=0.9, env_id="other-env")
    output_dir = tmp_path / "processed"

    result = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))

    assert {summary.env_id for summary in result.env_summaries} == {"other-env"}


def test_process_selected_invalid_results_still_fail(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-1",
        updated_at="2024-01-01T00:00:00Z",
        reward=0.1,
        results_text='{"example_id": ',
    )

    with pytest.raises(ValueError, match="Failed to parse JSONL line 1"):
        run_process(ProcessOptions(runs_dir=runs_dir, output_dir=tmp_path / "processed", dry_run=False, max_workers=1))


def test_process_selected_missing_results_still_fail(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    missing_results = runs_dir / "run-1" / "demo-job" / "results.jsonl"
    missing_results.unlink()

    with pytest.raises(FileNotFoundError, match="Missing results.jsonl"):
        run_process(ProcessOptions(runs_dir=runs_dir, output_dir=tmp_path / "processed", dry_run=False, max_workers=1))


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


def test_run_process_reads_local_index_after_workspace_prep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"
    observed: list[str] = []

    def fake_prepare_output_workspace(**kwargs):
        observed.append("workspace")
        model_dir = kwargs["output_dir"] / "gpt-mini"
        model_dir.mkdir(parents=True, exist_ok=True)
        existing_path = model_dir / "demo-env.parquet"
        existing_path.write_text("placeholder", encoding="utf-8")
        (kwargs["output_dir"] / "env_index.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "files": {
                        "gpt-mini/demo-env.parquet": {
                            "env_id": "demo-env",
                            "model_id": "gpt-mini",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def fake_read_env_index_files(processed_dir: Path):
        observed.append("index")
        assert observed == ["workspace", "index"]
        return {"gpt-mini/demo-env.parquet": {"env_id": "demo-env", "model_id": "gpt-mini"}}

    monkeypatch.setattr("medarc_verifiers.cli.process.workspace.prepare_output_workspace", fake_prepare_output_workspace)
    monkeypatch.setattr("medarc_verifiers.cli.process.env_index.read_env_index_files", fake_read_env_index_files)
    monkeypatch.setattr(
        "medarc_verifiers.cli.process.pipeline._validate_existing_output_integrity",
        lambda *_args, **_kwargs: None,
    )

    result = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))

    assert observed == ["workspace", "index"]
    assert result.env_summaries == []


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
