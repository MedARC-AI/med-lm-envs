#!/usr/bin/env python3
import argparse
import json
from typing import Iterable

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from verifiers.utils.data_utils import extract_boxed_answer


def _extract_completion_text(completion_field) -> str:
    if completion_field is None:
        return ""
    if isinstance(completion_field, str):
        return completion_field
    if isinstance(completion_field, list):
        # Expected list of {"content": "..."} chunks
        parts = []
        for item in completion_field:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion_field)


def _iter_lines(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def postprocess_medhallu(path: str) -> dict:
    total = 0
    dropped_unsure = 0
    dropped_missing = 0
    y_true: list[str] = []
    y_pred: list[str] = []

    for row in _iter_lines(path):
        total += 1
        gold = str(row.get("answer", "")).strip()
        completion_text = _extract_completion_text(row.get("completion"))
        pred = extract_boxed_answer(completion_text)

        if pred is None:
            dropped_missing += 1
            continue
        if pred not in ("0", "1", "2"):
            dropped_missing += 1
            continue
        if pred == "2":
            dropped_unsure += 1
            continue

        y_true.append(gold)
        y_pred.append(pred)

    accuracy = accuracy_score(y_true, y_pred) if y_true else 0.0
    precision = precision_score(y_true, y_pred, pos_label="1", zero_division=0) if y_true else 0.0
    recall = recall_score(y_true, y_pred, pos_label="1", zero_division=0) if y_true else 0.0
    f1 = f1_score(y_true, y_pred, pos_label="1", zero_division=0) if y_true else 0.0

    kept = total - dropped_unsure - dropped_missing
    return {
        "total": total,
        "kept": kept,
        "dropped_unsure": dropped_unsure,
        "dropped_missing": dropped_missing,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute F1 for MedHallu results.jsonl, dropping unsure (boxed{2}) answers.",
    )
    parser.add_argument("results", help="Path to results.jsonl")
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.json (defaults to sibling of results.jsonl)",
    )
    args = parser.parse_args()

    stats = postprocess_medhallu(args.results)

    metadata_path = args.metadata
    if metadata_path is None:
        metadata_path = str(args.results).rsplit("/", 1)[0] + "/metadata.json"

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except FileNotFoundError:
        metadata = None

    if isinstance(metadata, dict) and isinstance(metadata.get("avg_metrics"), dict):
        metadata["avg_metrics"].update(
            {
                "accuracy": stats["accuracy"],
                "precision": stats["precision"],
                "recall": stats["recall"],
                "f1": stats["f1"],
                "kept": stats["kept"],
                "dropped_unsure": stats["dropped_unsure"],
                "dropped_missing": stats["dropped_missing"],
            }
        )
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
    else:
        print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
