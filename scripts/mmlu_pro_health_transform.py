#!/usr/bin/env python
"""
One-off helper to:
  - load the MMLU-Pro dev split and grab the five health examples
  - show their original (list-based) option format
  - load mkieffer/MMLU-Pro-Health validation examples for comparison
  - convert the MMLU-Pro examples into the dict-based option format used by mkieffer/MMLU-Pro-Health

The script prints the before/after records so you can verify the mapping.
"""

from __future__ import annotations

import argparse
import json
import os
import string
from typing import Any, Dict, Iterable, List, Tuple

from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from huggingface_hub import HfApi

disable_progress_bar()


DATASET_MMLU_PRO = "TIGER-Lab/MMLU-Pro"
DATASET_HEALTH = "mkieffer/MMLU-Pro-Health"


def _first_available_split(dataset: str, candidates: Iterable[str]):
    """Try each split name in order until one loads."""
    last_err = None
    for split in candidates:
        try:
            ds = load_dataset(dataset, split=split, download_mode="reuse_cache_if_exists")
            return split, ds
        except Exception as exc:  # pragma: no cover - this is a best-effort utility
            last_err = exc
    raise RuntimeError(f"Could not load dataset '{dataset}' with splits {list(candidates)}") from last_err


def _normalize_options(row: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """
    Take the raw options (either list or dict) from MMLU-Pro and return:
      - dict keyed by letters
      - ordered list of option texts (mirrors the incoming order for diagnostics)
    """
    raw = row.get("options") or row.get("choices") or row.get("options_list") or []
    def _keep(text: str | None) -> bool:
        return bool(text) and str(text).strip().lower() not in {"n/a", "na", "none"}

    if isinstance(raw, dict):
        # Already keyed by letters
        ordered = [raw[k] for k in sorted(raw.keys()) if _keep(raw[k])]
        return {k: v for k, v in raw.items() if _keep(v)}, ordered
    if not isinstance(raw, list):
        return {}, []
    letters = string.ascii_uppercase
    mapped = {letters[i]: opt for i, opt in enumerate(raw) if _keep(opt)}
    return mapped, [opt for opt in raw if _keep(opt)]


def _normalize_answer(row: Dict[str, Any], options: Dict[str, str]) -> str | None:
    """
    Normalize the answer into a letter.
    Handles letters, integer indices, or digit strings.
    """
    ans = row.get("answer") or row.get("label") or row.get("correct_choice")
    letters = string.ascii_uppercase
    if ans is None:
        return None
    if isinstance(ans, str):
        ans_str = ans.strip()
        if len(ans_str) == 1 and ans_str.upper() in letters:
            return ans_str.upper()
        if ans_str.isdigit():
            idx = int(ans_str)
            return letters[idx] if idx < len(letters) else None
    if isinstance(ans, int):
        return letters[ans] if ans < len(letters) else None
    # If options already dict and answer matches a key, keep it
    if isinstance(ans, str) and ans in options:
        return ans
    return None


def _filter_health(ds):
    """Filter only on the category field containing 'health'."""
    health_rows = []
    for row in ds:
        category = row.get("category")
        if isinstance(category, str) and "health" in category.lower():
            health_rows.append(row)
    return health_rows


def _print_section(title: str, rows: List[Dict[str, Any]]):
    print(f"\n=== {title} ===")
    print(json.dumps(rows, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transform MMLU-Pro health examples and optionally push to HF Hub.")
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="Optional hub dataset repo id to push the transformed health validation examples (e.g., your-username/MMLU-Pro-Health-Transformed).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN"),
        help="HF token for pushing. Defaults to HF_TOKEN or HUGGINGFACE_TOKEN env var.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.getenv("HF_DATASETS_CACHE"),
        help="Optional cache dir override for datasets loading (passed to HF_DATASETS_CACHE).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    if args.cache_dir:
        os.environ["HF_DATASETS_CACHE"] = args.cache_dir

    # Load MMLU-Pro dev (or validation if dev is unavailable) and grab health rows
    split_name, mmlu_dev = _first_available_split(DATASET_MMLU_PRO, ["dev", "validation"])
    health_rows = _filter_health(mmlu_dev)
    health_rows = health_rows[:5]  # only keep the first five

    # Load mkieffer/MMLU-Pro-Health validation few-shot examples
    _, mkieffer_val = _first_available_split(DATASET_HEALTH, ["validation"])
    mkieffer_preview = [mkieffer_val[i] for i in range(min(5, len(mkieffer_val)))]

    # Show the raw originals
    _print_section(
        f"Original {DATASET_MMLU_PRO} {split_name} health rows (first 5, raw list-based options)",
        [
            {
                "question": r.get("question"),
                "options": r.get("options") or r.get("choices"),
                "answer": r.get("answer"),
                "subject": r.get("subject") or r.get("category"),
                "extra_keys": {k: r[k] for k in r.keys() if k not in {"question", "options", "choices", "answer", "subject", "category"}},
            }
            for r in health_rows
        ],
    )

    # Show the mkieffer format for reference
    _print_section(
        f"{DATASET_HEALTH} validation preview (dict-based options)",
        [
            {
                "question": r.get("question"),
                "options": r.get("options"),
                "answer": r.get("answer"),
                "cot_content": r.get("cot_content", None),
            }
            for r in mkieffer_preview
        ],
    )

    # Transform originals into dict-based options
    transformed: List[Dict[str, Any]] = []
    for row in health_rows:
        options_dict, original_order = _normalize_options(row)
        answer_letter = _normalize_answer(row, options_dict)
        transformed.append(
            {
                "question": row.get("question"),
                "options": options_dict,
                "answer": answer_letter,
                "answer_text": options_dict.get(answer_letter),
                "cot_content": row.get("cot_content"),
                "original_order": original_order,
            }
        )

    _print_section(f"Transformed {DATASET_MMLU_PRO} health rows (dict-based options)", transformed)

    if args.push_to_hub:
        if not args.token:
            raise SystemExit("HF token required to push. Provide --token or set HF_TOKEN/HUGGINGFACE_TOKEN.")
        ds = Dataset.from_list(
            [
                {
                    "question": row["question"],
                    "options": row["options"],
                    "answer": row["answer"],
                    "cot_content": row.get("cot_content", ""),
                }
                for row in transformed
            ]
        )
        api = HfApi(token=args.token)
        repo_id = args.push_to_hub
        print(f"\nEnsuring dataset repo exists: {repo_id}")
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=args.token)
        print(f"Pushing transformed dataset to HF Hub at {repo_id} ...")
        ds.push_to_hub(repo_id, token=args.token, split="validation")
        print("Push complete.")

    print("\nDone.")


if __name__ == "__main__":
    # Default to online to allow downloading/pushing; users can set HF_DATASETS_OFFLINE=1 if needed.
    os.environ.setdefault("HF_DATASETS_OFFLINE", "0")

    # Point HF datasets cache to a local, writable path to avoid lock conflicts with shared caches
    local_cache = os.path.join(os.getcwd(), ".cache", "hf_datasets")
    os.makedirs(local_cache, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", local_cache)

    main()
