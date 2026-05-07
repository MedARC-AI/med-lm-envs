from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from medarc_verifiers.cli import main


def _write_config(path: Path, text: str) -> None:
    path.write_text(dedent(text).strip(), encoding="utf-8")


def test_toml_bench_writes_bench_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        num_examples = 1
        rollouts_per_example = 1
        """,
    )

    async def fake_run(config, **_kwargs):
        results_path = Path(config.resume_path)
        (results_path / "results.jsonl").write_text(json.dumps({"example_id": "0", "reward": 1.0}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    results_path = output_dir / "gpt-5-mini" / "medqa"
    bench_index = json.loads((output_dir / "bench_index.json").read_text())
    assert bench_index["version"] == 1
    assert bench_index["source_config"] == str(config_path)
    assert bench_index["evals"][0]["results_path"] == str(results_path)
    assert bench_index["evals"][0]["plan_digest"].startswith("sha256:")


def test_toml_bench_failed_eval_does_not_create_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        num_examples = 1
        rollouts_per_example = 1
        """,
    )

    async def fake_run(config, **_kwargs):  # noqa: ARG001
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 1

    results_path = output_dir / "gpt-5-mini" / "medqa"
    bench_index = json.loads((output_dir / "bench_index.json").read_text())
    assert bench_index["evals"] == []
    assert not (results_path / "metadata.json").exists()
    assert not (results_path / "results.jsonl").exists()


def test_toml_bench_continue_on_error_omits_failed_sidecar_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"

        [[eval]]
        env_id = "pubmedqa"
        """,
    )

    async def fake_run(config, **_kwargs):
        results_path = Path(config.resume_path)
        if results_path.name == "pubmedqa":
            raise RuntimeError("upstream failure")
        (results_path / "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert (
        main.main(
            [
                "bench",
                "--config",
                str(config_path),
                "--output-dir",
                str(output_dir),
                "--continue-on-error",
            ]
        )
        == 1
    )

    bench_index = json.loads((output_dir / "bench_index.json").read_text())
    assert [entry["env_id"] for entry in bench_index["evals"]] == ["medqa"]
    assert (output_dir / "gpt-5-mini" / "medqa" / "results.jsonl").exists()
    assert not (output_dir / "gpt-5-mini" / "pubmedqa" / "results.jsonl").exists()


def test_toml_bench_force_failure_removes_archived_sidecar_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )

    async def successful_run(config, **_kwargs):
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", successful_run)
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    async def failing_run(config, **_kwargs):  # noqa: ARG001
        raise RuntimeError("upstream failure")

    monkeypatch.setattr(main, "run_evaluation", failing_run)
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--force"]) == 1

    bench_index = json.loads((output_dir / "bench_index.json").read_text())
    assert bench_index["evals"] == []
    assert list((output_dir / "gpt-5-mini").glob("medqa__old_*"))
    assert not (output_dir / "gpt-5-mini" / "medqa" / "results.jsonl").exists()


def test_toml_bench_refuses_existing_output_without_bench_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    results_path = output_dir / "gpt-5-mini" / "medqa"
    results_path.mkdir(parents=True)
    (results_path / "metadata.json").write_text(json.dumps({"env_id": "medqa", "model": "gpt-5-mini"}))

    async def fake_run(config, **_kwargs):  # noqa: ARG001
        raise AssertionError("bench should fail before execution")

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 1


def test_toml_bench_force_archives_existing_output_without_bench_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    results_path = output_dir / "gpt-5-mini" / "medqa"
    results_path.mkdir(parents=True)
    (results_path / "metadata.json").write_text(json.dumps({"env_id": "medqa", "model": "gpt-5-mini"}))
    (results_path / "sentinel.txt").write_text("old")

    async def fake_run(config, **_kwargs):
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--force"]) == 0
    assert not (results_path / "sentinel.txt").exists()
    assert list((output_dir / "gpt-5-mini").glob("medqa__old_*"))


def test_toml_bench_refuses_existing_output_missing_from_bench_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )
    results_path = output_dir / "gpt-5-mini" / "medqa"
    results_path.mkdir(parents=True)
    (results_path / "metadata.json").write_text(json.dumps({"env_id": "medqa", "model": "gpt-5-mini"}))
    (output_dir / "bench_index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "evals": [
                    {
                        "index": 1,
                        "results_path": str(output_dir / "other-model" / "medqa"),
                        "model": "other-model",
                        "env_id": "medqa",
                        "variant_id": None,
                        "variant_payload": None,
                        "env_args": {},
                        "sampling_args": {},
                        "num_examples": 1,
                        "rollouts_per_example": 1,
                        "plan_digest": "sha256:old",
                    }
                ],
            }
        )
    )

    async def fake_run(config, **_kwargs):  # noqa: ARG001
        raise AssertionError("bench should fail before execution")

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 1


def test_toml_bench_refuses_stale_metadata_even_when_bench_index_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"
        """,
    )

    async def fake_run(config, **_kwargs):
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0

    metadata_path = output_dir / "gpt-5-mini" / "medqa" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["medarc_config_fingerprint"] = "stale"
    metadata_path.write_text(json.dumps(metadata))

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 1


def test_toml_bench_selected_runs_merge_bench_index(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "bench.toml"
    output_dir = tmp_path / "evals"
    _write_config(
        config_path,
        """
        model = "gpt-5-mini"

        [[eval]]
        env_id = "medqa"

        [[eval]]
        env_id = "pubmedqa"
        """,
    )

    async def fake_run(config, **_kwargs):
        Path(config.resume_path, "results.jsonl").write_text(json.dumps({"example_id": "0"}) + "\n")
        return {"outputs": [], "metadata": {}}

    monkeypatch.setattr(main, "run_evaluation", fake_run)

    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--eval-index", "1"]) == 0
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir), "--eval-index", "2"]) == 0

    bench_index = json.loads((output_dir / "bench_index.json").read_text())
    assert [entry["env_id"] for entry in bench_index["evals"]] == ["medqa", "pubmedqa"]
    assert main.main(["bench", "--config", str(config_path), "--output-dir", str(output_dir)]) == 0
