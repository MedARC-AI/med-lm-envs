from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_ROOT = REPO_ROOT / "environments" / "medfact_bench"
if str(ENVIRONMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVIRONMENT_ROOT))

from medfact_bench.reporting import build_report, main, write_report  # noqa: E402

SMALL_COMPLETE_COUNTS = {
    "scifact": 1,
    "healthver": 1,
    "medaesqa": 1,
    "pubmedqa-fact": 1,
    "bioasq-fact": 1,
}


def _row(dataset: str, answer: str, completion: object) -> dict[str, object]:
    return {
        "example_id": dataset,
        "answer": answer,
        "info": {"dataset": dataset},
        "completion": completion,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _imbalanced_rows() -> list[dict[str, object]]:
    return [
        _row("scifact", "SUPPORT", "<think>Strong evidence</think><score>2</score>"),
        _row("healthver", "NEI", "<think>Insufficient evidence</think><score>0</score>"),
        _row("medaesqa", "CONTRADICT", "Reasoning <score>-1</score> trailing text"),
        _row("pubmedqa-fact", "SUPPORT", "<think>Malformed score</think><score>9</score>"),
        _row("bioasq-fact", "NEI", "<score>1</score>"),
    ]


def test_report_calculates_hand_checked_metrics_and_complete_coverage(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(results_path, _imbalanced_rows())

    report = build_report(results_path, expected_dataset_counts=SMALL_COMPLETE_COUNTS)

    assert report["total_predictions"] == 5
    assert report["valid_predictions"] == 4
    assert report["invalid_predictions"] == 1
    assert report["parse_rate"] == pytest.approx(0.8)
    assert report["strict_format_rate"] == pytest.approx(0.4)
    assert report["micro_accuracy"] == pytest.approx(0.6)
    assert report["macro_accuracy"] == pytest.approx(0.6)
    assert report["paper_macro_accuracy"] == pytest.approx(0.6)
    assert report["paper_comparable"] is True
    assert report["coverage"]["status"] == "complete"
    assert report["per_dataset"]["pubmedqa-fact"] == {"correct": 0, "total": 1, "accuracy": 0.0}

    assert report["per_class"]["SUPPORT"]["precision"] == pytest.approx(0.5)
    assert report["per_class"]["SUPPORT"]["recall"] == pytest.approx(0.5)
    assert report["per_class"]["SUPPORT"]["f1"] == pytest.approx(0.5)
    assert report["per_class"]["NEI"]["f1"] == pytest.approx(2 / 3)
    assert report["per_class"]["CONTRADICT"]["f1"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 1.0) / 3)

    confusion = report["confusion_matrix"]["rows"]
    assert report["confusion_matrix"]["labels"] == ["SUPPORT", "NEI", "CONTRADICT", "INVALID"]
    assert confusion["SUPPORT"] == {"SUPPORT": 1, "NEI": 0, "CONTRADICT": 0, "INVALID": 1}
    assert confusion["NEI"] == {"SUPPORT": 1, "NEI": 1, "CONTRADICT": 0, "INVALID": 0}
    assert confusion["CONTRADICT"] == {"SUPPORT": 0, "NEI": 0, "CONTRADICT": 1, "INVALID": 0}


def test_report_marks_partial_coverage_as_not_paper_comparable(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    _write_jsonl(results_path, _imbalanced_rows()[:2])

    report = build_report(results_path)

    assert report["coverage"]["status"] == "partial"
    assert report["paper_comparable"] is False
    assert report["paper_macro_accuracy"] is None
    assert report["macro_accuracy"] == pytest.approx(1.0)
    assert report["coverage"]["missing_datasets"] == ["medaesqa", "pubmedqa-fact", "bioasq-fact"]


def test_report_handles_chat_message_completion(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    completion = [{"role": "assistant", "content": "<think>Strong evidence</think><score>2</score>"}]
    _write_jsonl(results_path, [_row("scifact", "SUPPORT", completion)])

    report = build_report(results_path, expected_dataset_counts={"scifact": 1})

    assert report["valid_predictions"] == 1
    assert report["strict_format_rate"] == 1.0
    assert report["micro_accuracy"] == 1.0
    assert report["paper_macro_accuracy"] == 1.0


def test_write_report_preserves_metadata_and_avg_reward(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    metadata_path = tmp_path / "metadata.json"
    report_path = tmp_path / "custom-report.json"
    _write_jsonl(results_path, _imbalanced_rows())
    metadata_path.write_text(
        json.dumps(
            {
                "avg_reward": 0.123,
                "avg_metrics": {"accuracy": 0.123, "preserved_metric": 9.0},
                "custom": {"keep": True},
            }
        ),
        encoding="utf-8",
    )

    report = write_report(
        results_path,
        output_path=report_path,
        metadata_path=metadata_path,
        expected_dataset_counts=SMALL_COMPLETE_COUNTS,
    )
    written_report = json.loads(report_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert written_report == report
    assert metadata["avg_reward"] == 0.123
    assert metadata["avg_metrics"]["accuracy"] == 0.123
    assert metadata["avg_metrics"]["preserved_metric"] == 9.0
    assert metadata["avg_metrics"]["micro_accuracy"] == pytest.approx(0.6)
    assert metadata["avg_metrics"]["paper_macro_accuracy"] == pytest.approx(0.6)
    assert metadata["custom"] == {"keep": True}


def test_write_report_removes_partial_paper_metric_without_touching_reward(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    metadata_path = tmp_path / "metadata.json"
    _write_jsonl(results_path, _imbalanced_rows()[:2])
    metadata_path.write_text(
        json.dumps({"avg_reward": 0.9, "avg_metrics": {"paper_macro_accuracy": 0.7, "keep": 1.0}}),
        encoding="utf-8",
    )

    write_report(results_path, metadata_path=metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert metadata["avg_reward"] == 0.9
    assert metadata["avg_metrics"]["keep"] == 1.0
    assert "paper_macro_accuracy" not in metadata["avg_metrics"]


@pytest.mark.parametrize(
    "content",
    [
        "not json\n",
        json.dumps({"answer": "SUPPORT", "info": {"dataset": "scifact"}}) + "\n",
        json.dumps({"completion": "<score>1</score>", "answer": "SUPPORT", "info": {}}) + "\n",
    ],
)
def test_report_rejects_corrupt_or_incomplete_results(tmp_path: Path, content: str) -> None:
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError):
        build_report(results_path)


def test_write_report_does_not_write_partial_output_for_late_invalid_row(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    report_path = tmp_path / "medfact_report.json"
    metadata_path = tmp_path / "metadata.json"
    original_metadata = json.dumps({"avg_reward": 0.5, "custom": {"keep": True}})
    results_path.write_text(json.dumps(_imbalanced_rows()[0]) + "\nnot json\n", encoding="utf-8")
    metadata_path.write_text(original_metadata, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        write_report(results_path, output_path=report_path, metadata_path=metadata_path)

    assert not report_path.exists()
    assert metadata_path.read_text(encoding="utf-8") == original_metadata


def test_reporting_cli_can_skip_metadata_updates(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "report.json"
    _write_jsonl(results_path, _imbalanced_rows()[:2])

    exit_code = main([str(results_path), "--output", str(output_path), "--no-update-metadata"])

    assert exit_code == 0
    assert output_path.exists()
    assert not (tmp_path / "metadata.json").exists()
