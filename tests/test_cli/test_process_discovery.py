from __future__ import annotations

import json
from pathlib import Path

from medarc_verifiers.cli.process.discovery import discover_run_records
from medarc_verifiers.cli.process.metadata import load_normalized_metadata


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_eval_output(path: Path, metadata: dict | None = None, *, rows: list[dict] | None = None) -> None:
    _write_json(
        path / "metadata.json",
        {
            "env_id": "medqa",
            "model": "gpt-5-mini",
            "env_args": {"split": "test"},
            "sampling_args": {"temperature": 0},
            "num_examples": 1,
            "rollouts_per_example": 1,
            "avg_reward": 1.0,
            **(metadata or {}),
        },
    )
    result_rows = rows if rows is not None else [{"example_id": "ex-1", "reward": 1.0}]
    with (path / "results.jsonl").open("w", encoding="utf-8") as handle:
        for row in result_rows:
            handle.write(json.dumps(row) + "\n")


def test_discover_run_records_includes_deterministic_base_layout(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    output_dir = evals_dir / "gpt-5-mini" / "medqa" / "base"
    _write_eval_output(output_dir)
    _write_json(output_dir / "summary.json", {"env_id": "medqa"})

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 1
    record = records[0]
    assert record.status == "completed"
    assert record.model_id == "gpt-5-mini"
    assert record.manifest_env_id == "medqa"
    assert record.results_dir == output_dir
    assert record.has_metadata is True
    assert record.has_results is True
    assert record.has_summary is True
    assert record.env_args == {"split": "test"}
    assert record.sampling_args == {"temperature": 0}
    assert record.avg_reward == 1.0
    assert record.row_count == 1
    assert record.manifest.job_run_id == "gpt-5-mini::medqa::base"
    normalized = load_normalized_metadata(record)
    assert normalized.variant_id == "base"


def test_discover_run_records_includes_deterministic_eval_variants(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    variant_id = "env_args.shuffle_seed-1618"
    output_dir = evals_dir / "gpt-5-mini" / "medqa" / variant_id
    _write_eval_output(output_dir)

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 1
    normalized = load_normalized_metadata(records[0])
    assert normalized.variant_id == variant_id
    assert normalized.variant_payload is None
    assert normalized.medarc_config_fingerprint is None
    assert normalized.medarc_config_fingerprint_payload is None


def test_discover_run_records_includes_nested_orchestrator_task_bench_outputs(tmp_path: Path) -> None:
    run_root = tmp_path / "outputs" / "orchestrate" / "run-1"
    output_dir = run_root / "tasks" / "task-1" / "bench" / "gpt-5-mini" / "medqa" / "base"
    _write_eval_output(output_dir)

    records = discover_run_records(run_root, filter_status=("completed",))

    assert len(records) == 1
    record = records[0]
    assert record.results_dir == output_dir
    assert record.model_id == "gpt-5-mini"
    assert record.manifest_env_id == "medqa"
    assert record.manifest.job_run_id == "gpt-5-mini::medqa::base"
    normalized = load_normalized_metadata(record)
    assert normalized.variant_id == "base"


def test_discover_run_records_preserves_path_safe_variant_identity(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    variant_id = "name.with-safe_chars-123"
    output_dir = evals_dir / "gpt-5-mini" / "foo--bar" / variant_id
    _write_eval_output(output_dir, {"env_id": "foo--bar", "model": "gpt-5-mini"})

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 1
    record = records[0]
    assert record.model_id == "gpt-5-mini"
    assert record.manifest_env_id == "foo--bar"
    normalized = load_normalized_metadata(record)
    assert normalized.variant_id == variant_id


def test_discover_run_records_includes_direct_upstream_uuid_outputs(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    run_id = "016f4b4a-92a4-4a5b-a7c1-853af3318c52"
    upstream_dir = evals_dir / "medqa--gpt-5-mini" / run_id
    _write_eval_output(upstream_dir)

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 1
    record = records[0]
    assert record.model_id == "gpt-5-mini"
    assert record.manifest_env_id == "medqa"
    assert record.manifest.job_run_id == run_id
    normalized = load_normalized_metadata(record)
    assert normalized.variant_id is None


def test_discover_run_records_skips_missing_metadata(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    output_dir = evals_dir / "gpt-5-mini" / "medqa" / "base"
    output_dir.mkdir(parents=True)
    (output_dir / "results.jsonl").write_text('{"example_id":"ex-1"}\n', encoding="utf-8")

    assert discover_run_records(evals_dir, filter_status=("completed",)) == []


def test_discover_run_records_skips_invalid_metadata(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    output_dir = evals_dir / "gpt-5-mini" / "medqa" / "base"
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text("not json", encoding="utf-8")
    (output_dir / "results.jsonl").write_text('{"example_id":"ex-1"}\n', encoding="utf-8")

    assert discover_run_records(evals_dir, filter_status=("completed",)) == []


def test_discover_run_records_skips_metadata_only_directory(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    _write_json(
        evals_dir / "gpt-5-mini" / "medqa" / "base" / "metadata.json",
        {"env_id": "medqa", "model": "gpt-5-mini"},
    )

    assert discover_run_records(evals_dir, filter_status=("completed",)) == []


def test_discover_run_records_counts_empty_results_candidate(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    output_dir = evals_dir / "gpt-5-mini" / "medqa" / "base"
    _write_json(output_dir / "metadata.json", {"env_id": "medqa", "model": "gpt-5-mini"})
    (output_dir / "results.jsonl").write_text("", encoding="utf-8")

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 1
    assert records[0].row_count == 0


def test_discover_run_records_counts_invalid_jsonl_candidate_for_later_row_validation(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    output_dir = evals_dir / "gpt-5-mini" / "medqa" / "base"
    _write_json(output_dir / "metadata.json", {"env_id": "medqa", "model": "gpt-5-mini"})
    (output_dir / "results.jsonl").write_text("{not json}\n", encoding="utf-8")

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 1
    assert records[0].row_count == 1


def test_discover_run_records_filters_current_output_status(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    _write_eval_output(evals_dir / "gpt-5-mini" / "medqa" / "base")

    assert len(discover_run_records(evals_dir, filter_status=("completed",))) == 1
    assert discover_run_records(evals_dir, filter_status=("failed",)) == []


def test_discover_run_records_parent_baseline_and_child_variant_once(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    baseline_dir = evals_dir / "gpt-5-mini" / "medqa"
    variant_dir = baseline_dir / "env_args.shuffle_seed-1618"
    _write_eval_output(baseline_dir)
    _write_eval_output(variant_dir)

    records = discover_run_records(evals_dir, filter_status=("completed",))

    assert len(records) == 2
    assert {record.results_dir for record in records} == {baseline_dir, variant_dir}


def test_discover_run_records_scans_only_provided_root(tmp_path: Path) -> None:
    evals_dir = tmp_path / "runs" / "evals"
    raw_dir = tmp_path / "runs" / "raw"
    _write_eval_output(evals_dir / "gpt-5-mini" / "medqa" / "base")

    assert discover_run_records(raw_dir, filter_status=("completed",)) == []
