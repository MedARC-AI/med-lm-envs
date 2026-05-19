from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from medarc_verifiers.cli._schemas import EnvironmentExportConfig
from medarc_verifiers.cli.hf import HFSyncConfig
from medarc_verifiers.cli.process import ProcessOptions, run_process
from medarc_verifiers.cli.process import workspace
from medarc_verifiers.cli.process.discovery import RunManifestInfo, RunRecord, discover_run_records
from medarc_verifiers.cli.process.pipeline import select_work_items
from medarc_verifiers.cli.winrate import WinrateConfig
from medarc_verifiers.cli.winrate import discover_datasets, run_winrate
from medarc_verifiers.cli.process.writer import ALLOWED_COLUMNS


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _set_mtime(path: Path, updated_at: str) -> None:
    timestamp = datetime.fromisoformat(updated_at.replace("Z", "+00:00")).timestamp()
    os.utime(path, (timestamp, timestamp))


def _manifest_info(
    *,
    run_id: str,
    completed: int,
    total: int,
    total_known: bool = True,
    updated_at: str = "2024-01-01T00:00:00Z",
) -> RunManifestInfo:
    run_dir = Path("/tmp") / run_id
    return RunManifestInfo(
        job_run_id=run_id,
        run_name=run_id,
        summary_completed=completed,
        summary_total=total,
        summary_total_known=total_known,
        manifest_path=run_dir / "run_manifest.json",
        run_dir=run_dir,
        created_at="2024-01-01T00:00:00Z",
        updated_at=updated_at,
        config_source="configs/demo.yaml",
        config_checksum="abc123",
        run_summary_path=run_dir / "run_summary.json",
    )


def _run_record(
    *,
    run_id: str,
    job_id: str,
    env_id: str,
    model_id: str = "gpt-mini",
    completed: int = 1,
    total: int = 1,
    total_known: bool = True,
    updated_at: str = "2024-01-01T00:00:00Z",
    row_count: int | None = 1,
    num_examples: int | None = 1,
    rollouts_per_example: int | None = 1,
) -> RunRecord:
    run_dir = Path("/tmp") / run_id
    results_dir = run_dir / job_id
    return RunRecord(
        manifest=_manifest_info(
            run_id=run_id,
            completed=completed,
            total=total,
            total_known=total_known,
            updated_at=updated_at,
        ),
        job_id=job_id,
        model_id=model_id,
        manifest_env_id=env_id,
        results_dir_name=job_id,
        results_dir=results_dir,
        metadata_path=results_dir / "metadata.json",
        results_path=results_dir / "results.jsonl",
        summary_path=results_dir / "summary.json",
        has_metadata=False,
        has_results=True,
        has_summary=True,
        status="completed",
        duration_seconds=1.0,
        reason=None,
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T00:00:01Z",
        avg_reward=1.0,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        row_count=row_count,
        env_args={},
        sampling_args={},
        env_config={"id": env_id, "module": env_id},
        model_config={},
    )


def _setup_run(tmp_path: Path) -> Path:
    runs_dir = tmp_path / "runs" / "evals"
    results_dir = runs_dir / "demo-env-rollout3--gpt-mini" / "run-1"
    metadata = {
        "env_id": "demo-env-rollout3",
        "model": "gpt-mini",
        "env_args": {},
        "sampling_args": {},
        "num_examples": 1,
        "rollouts_per_example": 1,
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
    _set_mtime(results_dir / "metadata.json", "2024-01-01T00:00:00Z")
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
    row_count: int | None = 1,
    num_examples: int | None = 1,
    rollouts_per_example: int | None = 1,
    write_results: bool = True,
    job_id: str = "demo-job",
) -> Path:
    runs_dir = tmp_path / "runs" / "evals"
    results_dir = runs_dir / f"{env_id}--{model_id}" / run_id
    metadata = {
        "env_id": env_id,
        "model": model_id,
        "env_args": {},
        "sampling_args": {},
        "num_examples": num_examples,
        "rollouts_per_example": rollouts_per_example,
    }
    _write_json(results_dir / "metadata.json", metadata)
    _set_mtime(results_dir / "metadata.json", updated_at)
    results_path = results_dir / "results.jsonl"
    if write_results:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        if results_text is None:
            result_rows = 1 if row_count is None else max(int(row_count), 0)
            results_text = "".join(
                json.dumps({"example_id": f"ex-{run_id}-{index}", "reward": reward}) + "\n"
                for index in range(result_rows)
            )
        results_path.write_text(results_text, encoding="utf-8")
        _set_mtime(results_path, updated_at)
    return runs_dir


def _write_deterministic_eval(
    tmp_path: Path,
    *,
    model_id: str = "gpt-mini",
    env_id: str = "demo-env",
    variant_id: str | None = None,
    env_args: dict[str, object] | None = None,
    result_row: dict[str, object] | None = None,
) -> Path:
    runs_dir = tmp_path / "runs" / "evals"
    results_dir = runs_dir / model_id / env_id / (variant_id or "base")
    resolved_env_args = env_args or {}
    metadata = {
        "env_id": env_id,
        "model": model_id,
        "env_args": resolved_env_args,
        "sampling_args": {},
        "num_examples": 1,
        "rollouts_per_example": 1,
    }
    _write_json(results_dir / "metadata.json", metadata)
    row = result_row or {"example_id": "ex-1", "reward": 1.0}
    (results_dir / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
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


def test_run_process_processes_deterministic_eval_outputs(tmp_path: Path) -> None:
    runs_dir = _write_deterministic_eval(tmp_path)

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=tmp_path / "processed",
            dry_run=True,
            max_workers=1,
        )
    )

    assert result.records_processed == 1
    assert result.rows_processed == 1
    group = result.env_groups[0]
    assert group.env_id == "demo-env"
    assert group.model_id == "gpt-mini"


def test_run_process_preserves_deterministic_eval_variants(tmp_path: Path) -> None:
    _write_deterministic_eval(tmp_path, variant_id="env_args.shuffle_seed-1618")
    runs_dir = _write_deterministic_eval(tmp_path, variant_id="env_args.shuffle_seed-9331")
    output_dir = tmp_path / "processed"

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
        )
    )

    assert result.records_processed == 2
    rel_paths = sorted(summary.output_path.relative_to(output_dir).as_posix() for summary in result.env_summaries)
    assert rel_paths == [
        "gpt-mini/demo-env__variants/env_args.shuffle_seed-1618.parquet",
        "gpt-mini/demo-env__variants/env_args.shuffle_seed-9331.parquet",
    ]
    index_payload = json.loads((output_dir / "env_index.json").read_text(encoding="utf-8"))
    assert sorted(index_payload["files"]) == rel_paths
    assert {entry["variant_id"] for entry in index_payload["files"].values()} == {
        "env_args.shuffle_seed-1618",
        "env_args.shuffle_seed-9331",
    }


def test_run_process_excludes_specific_deterministic_eval_variant(tmp_path: Path) -> None:
    _write_deterministic_eval(tmp_path, variant_id="env_args.shuffle_seed-1618")
    runs_dir = _write_deterministic_eval(tmp_path, variant_id="env_args.shuffle_seed-9331")

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=tmp_path / "processed",
            exclude_datasets=("demo-env::env_args.shuffle_seed-1618",),
            dry_run=True,
            max_workers=1,
        )
    )

    assert result.records_processed == 1
    assert result.env_groups[0].variant_id == "env_args.shuffle_seed-9331"


def test_run_process_excludes_deterministic_eval_variants_by_base_env(tmp_path: Path) -> None:
    _write_deterministic_eval(tmp_path, variant_id="env_args.shuffle_seed-1618")
    runs_dir = _write_deterministic_eval(tmp_path, variant_id="env_args.shuffle_seed-9331")

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=tmp_path / "processed",
            exclude_datasets=("demo-env",),
            dry_run=True,
            max_workers=1,
        )
    )

    assert result.records_processed == 0
    assert result.env_groups == []


def test_run_process_applies_variant_export_overrides_to_deterministic_eval(tmp_path: Path) -> None:
    variant_id = "env_args.add_calculator_tool-true__env_args.add_python_tool-true__env_args.version-verified"
    env_args = {
        "version": "verified",
        "add_python_tool": True,
        "add_calculator_tool": True,
    }
    runs_dir = _write_deterministic_eval(
        tmp_path,
        env_id="medcalc_bench",
        variant_id=variant_id,
        env_args=env_args,
        result_row={
            "example_id": "ex-1",
            "ground_truth": "42",
            "lower_bound": 40,
            "upper_bound": 44,
            "reward": 1.0,
        },
    )
    output_dir = tmp_path / "processed"

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
        ),
        env_export_map={
            f"medcalc_bench::{variant_id}": EnvironmentExportConfig(
                extra_columns=["lower_bound", "upper_bound"],
                answer_column="ground_truth",
            )
        },
    )

    table = pq.read_table(result.env_summaries[0].output_path)
    row = table.to_pylist()[0]
    assert row["answer"] == "42"
    assert json.loads(row["extras"]) == {"lower_bound": 40, "upper_bound": 44}
    assert "ground_truth" not in row


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


def test_run_process_preserves_string_example_id_in_parquet(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)
    output_dir = tmp_path / "processed"

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
        )
    )

    table = pq.read_table(result.env_summaries[0].output_path)
    assert table.column("example_id").to_pylist() == ["ex-1"]
    assert str(table.schema.field("example_id").type) == "large_string"


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


def test_process_allows_results_missing_pct_within_threshold(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-98pct",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=98,
        num_examples=100,
        rollouts_per_example=1,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        max_results_missing_pct=2.5,
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 1
    assert result.rows_processed == 98


def test_process_rejects_results_missing_pct_above_threshold(tmp_path: Path) -> None:
    results_text = "".join(json.dumps({"example_id": f"ex-{index}", "reward": 1.0}) + "\n" for index in range(90))
    runs_dir = _write_run(
        tmp_path,
        run_id="run-90pct",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=90,
        num_examples=100,
        rollouts_per_example=1,
        results_text=results_text,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        max_results_missing_pct=2.5,
        dry_run=True,
        max_workers=1,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_process(options)

    message = str(excinfo.value)
    assert "run-90pct" in message
    assert "expected_rows=100" in message
    assert "observed_rows=90" in message
    assert "missing_pct=10.00" in message
    assert "threshold=2.5" in message


def test_process_allows_ungateable_record_when_expected_rows_unknown(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-unknown-expected",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=10,
        num_examples=None,
        rollouts_per_example=1,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 1


def test_process_allows_ungateable_record_when_expected_rows_unknown_even_with_observed_rows(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-unknown-observed",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=None,
        num_examples=None,
        rollouts_per_example=1,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 1


def test_process_latest_record_that_fails_gate_does_not_fall_back(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        run_id="run-older-ok",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=100,
        num_examples=100,
        rollouts_per_example=1,
    )
    runs_dir = _write_run(
        tmp_path,
        run_id="run-newer-bad",
        updated_at="2024-01-02T00:00:00Z",
        reward=0.0,
        row_count=90,
        num_examples=100,
        rollouts_per_example=1,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        max_results_missing_pct=2.5,
        dry_run=True,
        max_workers=1,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_process(options)

    message = str(excinfo.value)
    assert "run-newer-bad" in message
    assert "run-older-ok" not in message


def test_process_ignores_metadata_only_output_without_results_jsonl(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-missing-results",
        updated_at="2024-01-02T00:00:00Z",
        reward=1.0,
        row_count=100,
        num_examples=100,
        rollouts_per_example=1,
        write_results=False,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 0
    assert result.rows_processed == 0


def test_process_gate_ignores_excluded_record(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-excluded-bad",
        updated_at="2024-01-02T00:00:00Z",
        reward=1.0,
        env_id="skip-env",
        row_count=90,
        num_examples=100,
        rollouts_per_example=1,
    )
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        exclude_datasets=("skip-env",),
        max_results_missing_pct=2.5,
        dry_run=True,
        max_workers=1,
    )

    result = run_process(options)

    assert result.records_processed == 0


def test_process_stale_delta_output_does_not_mask_newer_incomplete_run(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-initial",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=100,
        num_examples=100,
        rollouts_per_example=1,
    )
    output_dir = tmp_path / "processed"
    initial = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))
    assert initial.records_processed == 1

    results_text = "".join(json.dumps({"example_id": f"ex-{index}", "reward": 0.0}) + "\n" for index in range(90))
    _write_run(
        tmp_path,
        run_id="run-newer-bad",
        updated_at="2024-01-02T00:00:00Z",
        reward=0.0,
        row_count=90,
        num_examples=100,
        rollouts_per_example=1,
        results_text=results_text,
    )

    with pytest.raises(RuntimeError) as excinfo:
        run_process(
            ProcessOptions(
                runs_dir=runs_dir,
                output_dir=output_dir,
                max_results_missing_pct=2.5,
                dry_run=False,
                max_workers=1,
            )
        )

    message = str(excinfo.value)
    assert "run-newer-bad" in message
    assert "missing_pct=10.00" in message


def test_process_emits_single_warning_for_ungateable_selected_records(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-unknown-observed",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=None,
        num_examples=None,
        rollouts_per_example=1,
    )
    caplog.set_level("WARNING")

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=tmp_path / "processed",
            dry_run=True,
            max_workers=1,
        )
    )

    assert result.records_processed == 1
    warnings = [
        record for record in caplog.records if "Results row completeness gate could not be applied" in record.msg
    ]
    assert len(warnings) == 1


def test_process_uses_discovered_actual_results_rows_for_completeness_gate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    results_text = "".join(json.dumps({"example_id": f"ex-{index}", "reward": 1.0}) + "\n" for index in range(100))
    runs_dir = _write_run(
        tmp_path,
        run_id="run-stale-row-count",
        updated_at="2024-01-01T00:00:00Z",
        reward=1.0,
        row_count=90,
        num_examples=100,
        rollouts_per_example=1,
        results_text=results_text,
    )
    caplog.set_level("WARNING")

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=tmp_path / "processed",
            dry_run=True,
            max_workers=1,
        )
    )

    assert result.records_processed == 1
    assert "row_count mismatch" not in caplog.text


def test_select_work_items_rollout_gate_error_includes_output_and_manifest_ids(tmp_path: Path) -> None:
    runs_dir = _write_run(
        tmp_path,
        run_id="run-rollout-bad",
        updated_at="2024-01-02T00:00:00Z",
        reward=1.0,
        env_id="demo-env-rollout3",
        row_count=90,
        num_examples=100,
        rollouts_per_example=1,
    )
    discovered = discover_run_records(runs_dir, filter_status=("completed",))
    options = ProcessOptions(
        runs_dir=runs_dir,
        output_dir=tmp_path / "processed",
        max_results_missing_pct=2.5,
        dry_run=True,
        max_workers=1,
    )

    with pytest.raises(RuntimeError) as excinfo:
        select_work_items(discovered, options=options, env_export_map={}, index_files={})

    message = str(excinfo.value)
    assert "output_env_id=demo-env" in message
    assert "manifest_env_id=demo-env-rollout3" in message
    assert "job_id=run-rollout-bad" in message


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
    assert result_newer_raw.env_summaries
    newer_table = pq.read_table(result_newer_raw.env_summaries[0].output_path)
    assert set(newer_table.column("job_run_id").to_pylist()) == {"run-3"}
    assert newer_table.column("reward").to_pylist() == [0.4]


def test_run_process_continue_upload_syncs_pending_parquets_without_new_deltas(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_dir = _write_run(tmp_path, run_id="run-1", updated_at="2024-01-01T00:00:00Z", reward=0.1, env_id="demo-env")
    output_dir = tmp_path / "processed"

    first_result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
        )
    )
    pending_path = first_result.env_summaries[0].output_path.relative_to(output_dir).as_posix()
    captured: dict[str, object] = {}

    def fake_prepare_output_workspace(**_kwargs: object) -> workspace.WorkspacePreparationResult:
        return workspace.WorkspacePreparationResult(
            baseline_result=workspace.BaselineResult(
                policy="continue-upload",
                pending_parquet_uploads={pending_path},
            )
        )

    def fake_sync_to_hub(
        env_summaries,
        config,
        *,
        output_dir,
        metadata_paths=None,
        files=None,
        **_kwargs,
    ):
        captured["env_summaries"] = list(env_summaries)
        captured["files"] = list(files or [])
        return None

    monkeypatch.setattr(
        "medarc_verifiers.cli.process.pipeline.workspace.prepare_output_workspace", fake_prepare_output_workspace
    )
    monkeypatch.setattr("medarc_verifiers.cli.process.pipeline.hf_sync.sync_to_hub", fake_sync_to_hub)

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
            hf_config=HFSyncConfig(repo_id="demo/repo"),
            hf_pull_policy="continue-upload",
        )
    )

    assert result.env_summaries == []
    assert captured["env_summaries"] == []
    assert captured["files"] == [pending_path]


def test_run_process_continue_upload_unions_pending_and_current_touched_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runs_dir = _write_run(tmp_path, run_id="run-1", updated_at="2024-01-01T00:00:00Z", reward=0.1, env_id="demo-env")
    output_dir = tmp_path / "processed"
    first_result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
        )
    )
    current_path = first_result.env_summaries[0].output_path.relative_to(output_dir).as_posix()
    pending_path = "stale-model/stale-env.parquet"
    stale_path = output_dir / pending_path
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("stale", encoding="utf-8")
    _write_run(tmp_path, run_id="run-2", updated_at="2024-01-02T00:00:00Z", reward=0.9, env_id="demo-env")

    captured: dict[str, object] = {}

    def fake_prepare_output_workspace(**_kwargs: object) -> workspace.WorkspacePreparationResult:
        return workspace.WorkspacePreparationResult(
            baseline_result=workspace.BaselineResult(
                policy="continue-upload",
                pending_parquet_uploads={pending_path},
            )
        )

    def fake_sync_to_hub(
        env_summaries,
        config,
        *,
        output_dir,
        metadata_paths=None,
        files=None,
        **_kwargs,
    ):
        captured["files"] = list(files or [])
        return None

    monkeypatch.setattr(
        "medarc_verifiers.cli.process.pipeline.workspace.prepare_output_workspace", fake_prepare_output_workspace
    )
    monkeypatch.setattr("medarc_verifiers.cli.process.pipeline.hf_sync.sync_to_hub", fake_sync_to_hub)

    result = run_process(
        ProcessOptions(
            runs_dir=runs_dir,
            output_dir=output_dir,
            dry_run=False,
            max_workers=1,
            hf_config=HFSyncConfig(repo_id="demo/repo"),
            hf_pull_policy="continue-upload",
        )
    )

    assert result.env_summaries
    assert set(captured["files"]) == {pending_path, current_path, "dataset_infos.json", "env_index.json"}


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
    assert rebuilt == {"model-a", "model-b"}
    model_a_table = pq.read_table(output_dir / "model-a" / "demo-env.parquet")
    model_b_table = pq.read_table(output_dir / "model-b" / "demo-env.parquet")
    assert model_a_table.column("reward").to_pylist() == [0.9]
    assert model_b_table.column("reward").to_pylist() == [0.8]


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

    assert {(summary.model_id, summary.env_id) for summary in result.env_summaries} == {
        ("model-a", "env-a"),
        ("model-a", "env-b"),
        ("model-b", "env-a"),
    }
    assert pq.read_table(output_dir / "model-a" / "env-a.parquet").column("reward").to_pylist() == [0.7]
    assert pq.read_table(output_dir / "model-a" / "env-b.parquet").column("reward").to_pylist() == [0.8]
    assert pq.read_table(output_dir / "model-b" / "env-a.parquet").column("reward").to_pylist() == [0.9]


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

    monkeypatch.setattr(
        "medarc_verifiers.cli.process.workspace.prepare_output_workspace", fake_prepare_output_workspace
    )
    monkeypatch.setattr("medarc_verifiers.cli.process.env_index.read_env_index_files", fake_read_env_index_files)
    monkeypatch.setattr(
        "medarc_verifiers.cli.process.pipeline._read_existing_output_metadata",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli.process.pipeline._validate_existing_output_integrity",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "medarc_verifiers.cli.process.pipeline._existing_output_matches_selected_runs",
        lambda *_args, **_kwargs: True,
    )

    result = run_process(ProcessOptions(runs_dir=runs_dir, output_dir=output_dir, dry_run=False, max_workers=1))

    assert observed == ["workspace", "index"]
    assert result.env_summaries == []


def test_run_process_ignores_legacy_run_output_path(tmp_path: Path) -> None:
    runs_dir = _setup_run(tmp_path)

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
