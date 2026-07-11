"""Benchmark-specific reporting for raw MedFact-Bench evaluation results."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .environment import (
    EXPECTED_DATASET_COUNTS,
    INVALID_LABEL,
    LABELS,
    MedFactScoreParser,
    score_to_label,
)

REPORT_SCHEMA_VERSION = 1
CONFUSION_LABELS = (*LABELS, INVALID_LABEL)


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _iter_results(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"MedFact-Bench results file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"MedFact-Bench results path must be a JSONL file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in MedFact-Bench results at {path}:{line_number}: {exc.msg}.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in MedFact-Bench results at {path}:{line_number}.")
            _validate_result_row(row, path, line_number)
            yield row


def _validate_result_row(row: dict[str, Any], path: Path, line_number: int) -> None:
    location = f"{path}:{line_number}"
    if "completion" not in row:
        raise ValueError(f"MedFact-Bench result at {location} is missing the completion field.")

    answer = row.get("answer")
    if answer not in LABELS:
        raise ValueError(f"MedFact-Bench result at {location} has invalid or missing answer label: {answer!r}.")

    info = row.get("info")
    if not isinstance(info, Mapping):
        raise ValueError(f"MedFact-Bench result at {location} is missing the info object.")
    dataset_name = info.get("dataset")
    if dataset_name not in EXPECTED_DATASET_COUNTS:
        raise ValueError(f"MedFact-Bench result at {location} has invalid or missing info.dataset: {dataset_name!r}.")


def _empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in CONFUSION_LABELS} for actual in CONFUSION_LABELS}


def _class_metrics(
    confusion: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, dict[str, float | int]], float]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in LABELS:
        true_positive = confusion[label][label]
        predicted_count = sum(confusion[actual][label] for actual in CONFUSION_LABELS)
        support = sum(confusion[label].values())
        precision = _safe_divide(true_positive, predicted_count)
        recall = _safe_divide(true_positive, support)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted_count,
        }
    macro_f1 = sum(metrics["f1"] for metrics in per_class.values()) / len(LABELS)
    return per_class, macro_f1


def build_report(
    results_path: str | Path,
    *,
    expected_dataset_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build a deterministic report from raw Verifiers results."""
    path = Path(results_path)
    parser = MedFactScoreParser()
    expected_counts = dict(expected_dataset_counts or EXPECTED_DATASET_COUNTS)
    expected_total = sum(expected_counts.values())

    total_predictions = 0
    valid_predictions = 0
    strict_predictions = 0
    correct_predictions = 0
    dataset_totals: Counter[str] = Counter()
    dataset_correct: Counter[str] = Counter()
    confusion = _empty_confusion_matrix()

    for row in _iter_results(path):
        answer = row["answer"]
        dataset_name = row["info"]["dataset"]
        completion_text = parser.completion_text(row["completion"])
        score = parser.parse(completion_text) if completion_text is not None else None
        predicted = score_to_label(score)

        is_valid = predicted != INVALID_LABEL
        is_correct = predicted == answer
        total_predictions += 1
        valid_predictions += int(is_valid)
        strict_predictions += int(completion_text is not None and parser.is_strict_format(completion_text))
        correct_predictions += int(is_correct)
        dataset_totals[dataset_name] += 1
        dataset_correct[dataset_name] += int(is_correct)
        confusion[answer][predicted] += 1

    if total_predictions == 0:
        raise ValueError(f"MedFact-Bench results file contains no result rows: {path}")

    per_dataset: dict[str, dict[str, float | int | None]] = {}
    observed_accuracies: list[float] = []
    for dataset_name in EXPECTED_DATASET_COUNTS:
        total = dataset_totals[dataset_name]
        correct = dataset_correct[dataset_name]
        dataset_accuracy = correct / total if total else None
        if dataset_accuracy is not None:
            observed_accuracies.append(dataset_accuracy)
        per_dataset[dataset_name] = {
            "correct": correct,
            "total": total,
            "accuracy": dataset_accuracy,
        }

    invalid_predictions = total_predictions - valid_predictions
    macro_accuracy = sum(observed_accuracies) / len(observed_accuracies)
    per_class, macro_f1 = _class_metrics(confusion)

    observed_counts = {name: dataset_totals[name] for name in expected_counts}
    missing_datasets = [name for name, count in observed_counts.items() if count == 0]
    count_mismatches = {
        name: {"expected": expected_counts[name], "observed": observed_counts[name]}
        for name in expected_counts
        if observed_counts[name] != expected_counts[name]
    }
    complete_coverage = not count_mismatches and total_predictions == expected_total

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "benchmark": "medfact_bench",
        "total_predictions": total_predictions,
        "valid_predictions": valid_predictions,
        "invalid_predictions": invalid_predictions,
        "parse_rate": valid_predictions / total_predictions,
        "strict_format_rate": strict_predictions / total_predictions,
        "micro_accuracy": correct_predictions / total_predictions,
        "macro_accuracy": macro_accuracy,
        "paper_macro_accuracy": macro_accuracy if complete_coverage else None,
        "macro_f1": macro_f1,
        "per_dataset": per_dataset,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(CONFUSION_LABELS),
            "rows": confusion,
        },
        "coverage": {
            "status": "complete" if complete_coverage else "partial",
            "complete": complete_coverage,
            "expected_total": expected_total,
            "observed_total": total_predictions,
            "expected_dataset_counts": expected_counts,
            "observed_dataset_counts": observed_counts,
            "missing_datasets": missing_datasets,
            "count_mismatches": count_mismatches,
        },
        "paper_comparable": complete_coverage,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _update_metadata(path: Path, report: Mapping[str, Any]) -> None:
    if not path.exists():
        raise FileNotFoundError(f"MedFact-Bench metadata file does not exist: {path}")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in MedFact-Bench metadata file {path}: {exc.msg}.") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"MedFact-Bench metadata file must contain a JSON object: {path}")

    avg_metrics = metadata.get("avg_metrics")
    if avg_metrics is None:
        avg_metrics = {}
        metadata["avg_metrics"] = avg_metrics
    if not isinstance(avg_metrics, dict):
        raise ValueError(f"MedFact-Bench metadata avg_metrics must be a JSON object: {path}")

    avg_metrics.update(
        {
            "parse_rate": report["parse_rate"],
            "strict_format_rate": report["strict_format_rate"],
            "micro_accuracy": report["micro_accuracy"],
            "macro_accuracy": report["macro_accuracy"],
            "macro_f1": report["macro_f1"],
            "invalid_predictions": report["invalid_predictions"],
        }
    )
    if report["paper_macro_accuracy"] is not None:
        avg_metrics["paper_macro_accuracy"] = report["paper_macro_accuracy"]
    else:
        avg_metrics.pop("paper_macro_accuracy", None)
    _write_json(path, metadata)


def write_report(
    results_path: str | Path,
    *,
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    update_metadata: bool = True,
    expected_dataset_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build and write a MedFact-Bench report, optionally updating metadata."""
    results = Path(results_path)
    report = build_report(results, expected_dataset_counts=expected_dataset_counts)
    output = Path(output_path) if output_path is not None else results.with_name("medfact_report.json")
    _write_json(output, report)

    if update_metadata:
        metadata = Path(metadata_path) if metadata_path is not None else results.with_name("metadata.json")
        _update_metadata(metadata, report)
    return report


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate benchmark-specific metrics from raw MedFact-Bench results.jsonl.",
    )
    parser.add_argument("results_jsonl", help="Path to the raw MedFact-Bench results.jsonl file.")
    parser.add_argument(
        "--output",
        default=None,
        help="Report output path. Defaults to medfact_report.json beside the results file.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Metadata path. Defaults to metadata.json beside the results file.",
    )
    parser.add_argument(
        "--no-update-metadata",
        action="store_true",
        help="Write the report without updating metadata.json.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the MedFact-Bench reporting CLI."""
    args = _build_argument_parser().parse_args(argv)
    results_path = Path(args.results_jsonl)
    output_path = Path(args.output) if args.output is not None else results_path.with_name("medfact_report.json")
    write_report(
        results_path,
        output_path=output_path,
        metadata_path=args.metadata,
        update_metadata=not args.no_update_metadata,
    )
    print(f"Wrote MedFact-Bench report to {output_path}")
    if not args.no_update_metadata:
        metadata_path = Path(args.metadata) if args.metadata is not None else results_path.with_name("metadata.json")
        print(f"Updated MedFact-Bench metrics in {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
