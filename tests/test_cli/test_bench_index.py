from __future__ import annotations

import json
from pathlib import Path

import pytest

from medarc_verifiers.cli.bench_index import BenchIndexError, validate_bench_index


def _write_eval(path: Path, *, model: str = "gpt-5-mini", env_id: str = "medqa") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(json.dumps({"model": model, "env_id": env_id}), encoding="utf-8")
    (path / "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n", encoding="utf-8")


def _index_entry(path: Path, *, model: str = "gpt-5-mini", env_id: str = "medqa", variant_id: str | None = None):
    return {
        "index": 1,
        "results_path": str(path),
        "model": model,
        "env_id": env_id,
        "variant_id": variant_id,
        "variant_payload": None,
        "env_args": {},
        "sampling_args": {"unknown_provider_arg": True},
        "num_examples": 1,
        "rollouts_per_example": 1,
        "plan_digest": "sha256:test",
    }


def test_validate_bench_index_accepts_unknown_sampling_args(tmp_path: Path) -> None:
    results_path = tmp_path / "gpt-5-mini" / "medqa"
    _write_eval(results_path)

    validate_bench_index(
        {"version": 1, "evals": [_index_entry(results_path)]},
        output_root=tmp_path,
        require_artifacts=True,
    )


def test_validate_bench_index_rejects_stale_results_path(tmp_path: Path) -> None:
    with pytest.raises(BenchIndexError, match="required artifact is missing"):
        validate_bench_index(
            {"version": 1, "evals": [_index_entry(tmp_path / "missing" / "medqa")]},
            output_root=tmp_path,
            require_artifacts=True,
        )


def test_validate_bench_index_rejects_duplicate_results_path(tmp_path: Path) -> None:
    results_path = tmp_path / "gpt-5-mini" / "medqa"
    _write_eval(results_path)

    with pytest.raises(BenchIndexError, match="duplicate results_path"):
        validate_bench_index(
            {"version": 1, "evals": [_index_entry(results_path), _index_entry(results_path)]},
            output_root=tmp_path,
            require_artifacts=True,
        )


def test_validate_bench_index_rejects_metadata_identity_mismatch(tmp_path: Path) -> None:
    results_path = tmp_path / "gpt-5-mini" / "medqa"
    _write_eval(results_path, model="other-model")

    with pytest.raises(BenchIndexError, match="identity mismatch"):
        validate_bench_index(
            {"version": 1, "evals": [_index_entry(results_path)]},
            output_root=tmp_path,
            require_artifacts=True,
        )


def test_validate_bench_index_rejects_duplicate_model_env_without_variant(tmp_path: Path) -> None:
    first = tmp_path / "gpt-5-mini" / "medqa" / "first"
    second = tmp_path / "gpt-5-mini" / "medqa" / "second"
    _write_eval(first)
    _write_eval(second)

    with pytest.raises(BenchIndexError, match="require explicit variant_id"):
        validate_bench_index(
            {"version": 1, "evals": [_index_entry(first), _index_entry(second)]},
            output_root=tmp_path,
            require_artifacts=True,
        )
