#!/usr/bin/env python3
"""
MCQ Answer Analysis Script

Analyzes incorrect answers from model completion logs for MCQ benchmarks only.
Mirrors the exact parsing logic used during evaluation:
1. Environment-specific parsing (XML <answer> tags or \\boxed{} extraction)
2. multiple_choice_accuracy parsing strategies

Features:
- Wrong answer extraction and categorization
- Positional bias detection
- Confusion matrices (which answers are confused for which)
- Cross-rollout variation analysis
- Semantic consistency analysis
- Robustness metrics
- Incremental updates (add new models/benchmarks without re-running all)

Usage:
    # Analyze all models
    python scripts/mcq_answer_analysis.py \
        --logs-dir /path/to/medmarks_raw/raw \
        --output-dir ./analysis_output \
        --all-models

    # Analyze specific model
    python scripts/mcq_answer_analysis.py \
        --logs-dir /path/to/model_results \
        --output-dir ./analysis_output \
        --model model-name

    # Add new benchmark incrementally
    python scripts/mcq_answer_analysis.py \
        --logs-dir /path/to/medmarks_raw/raw \
        --output-dir ./analysis_output \
        --benchmark new_benchmark \
        --all-models
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Add parent directory to path to import medarc_verifiers
sys.path.insert(0, str(Path(__file__).parent.parent))

from medarc_verifiers.rewards.multiple_choice_accuracy import (
    multiple_choice_accuracy,
    _remove_think_tags,
    _nfkc_casefold,
    _norm_letter,
    ANCHOR_PATTERN,
    TOKEN_PATTERN,
    LEADING_OPTION_PATTERN,
)


# =============================================================================
# ENVIRONMENT-SPECIFIC PARSING (Step 1)
# Mirrors the parsing done by vf.XMLParser and extract_boxed_answer
# =============================================================================


def extract_xml_answer(text: str) -> str | None:
    """
    Extract content from <answer>...</answer> tags.
    Mirrors vf.XMLParser.parse_answer() logic - finds the LAST match.
    """
    if not text:
        return None
    # Pattern matching XMLParser: <answer>\s*(.*?)\s*</answer>
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    matches = list(re.finditer(pattern, text, re.DOTALL | re.IGNORECASE))
    if matches:
        # Take the last match (same as XMLParser with last=True)
        return matches[-1].group(1).strip()
    return None


def extract_boxed_answer(text: str) -> str:
    """
    Extract content from \\boxed{...} format.
    Exact copy of verifiers.utils.data_utils.extract_boxed_answer
    """
    def find_matching_brace(s: str, start: int) -> int:
        count = 1
        i = start
        while i < len(s) and count > 0:
            if s[i] == "{":
                count += 1
            elif s[i] == "}":
                count -= 1
            i += 1
        return i - 1 if count == 0 else -1

    # Find last \boxed{
    boxed_start = text.rfind("\\boxed{")
    if boxed_start == -1:
        return text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)

    if closing_brace == -1:
        return text

    return text[content_start:closing_brace]


def detect_answer_format(completion_text: str, system_prompt: str = "") -> str:
    """
    Detect whether the completion uses XML or BOXED format.
    Returns: 'xml', 'boxed', or 'unknown'
    """
    # Check system prompt hints
    system_lower = system_prompt.lower()
    if "boxed" in system_lower or "\\boxed" in system_lower:
        return "boxed"
    if "<answer>" in system_lower or "xml" in system_lower:
        return "xml"

    # Check completion content
    if "<answer>" in completion_text.lower():
        return "xml"
    if "\\boxed{" in completion_text:
        return "boxed"

    # Default to XML (most common in MedARC environments)
    return "xml"


def apply_environment_parser(completion_text: str, system_prompt: str = "") -> tuple[str, str]:
    """
    Apply environment-specific parsing to extract the answer.
    This mirrors what vf.XMLParser.parse_answer() or vf.Parser(extract_boxed_answer) does.

    Returns:
        (extracted_text, format_used)
    """
    if not completion_text:
        return "", "none"

    format_type = detect_answer_format(completion_text, system_prompt)

    if format_type == "boxed":
        extracted = extract_boxed_answer(completion_text)
        # If boxed extraction returns original text, it means no boxed found
        if extracted == completion_text:
            # Fallback: try XML
            xml_extracted = extract_xml_answer(completion_text)
            if xml_extracted:
                return xml_extracted, "xml"
            return completion_text, "boxed_fallback"
        return extracted, "boxed"

    # XML format (default)
    extracted = extract_xml_answer(completion_text)
    if extracted is not None:
        return extracted, "xml"

    # No XML tags found - return original text (some models don't use tags)
    return completion_text, "no_tags"

# MCQ benchmark prefixes (exclude known open-ended/non-MCQ tasks)
# Excluded: medec (error correction), medexqa (explanation-based), medredqa (open-ended),
#           medcasereasoning (diagnosis generation), careqa_open (open-ended subset)
MCQ_BENCHMARK_PREFIXES = {
    "careqa",  # MCQ subset (CareQA_en); careqa_open is excluded below
    "head_qa_v2",
    "longhealth",
    "m_arc",
    "med_halt",
    "med_mcqa",
    "medbullets",
    "medcalc_bench",
    "medconceptsqa",
    "medqa",
    "medxpertqa",
    "metamedqa",
    "mmlu_pro_health",
    "pubmedqa",
}

# Explicit exclusions (open-ended variants that match MCQ prefixes)
MCQ_BENCHMARK_EXCLUSIONS = {
    "careqa_open",
    "careqa-open",
}


def is_mcq_benchmark(name: str) -> bool:
    """Check if benchmark name matches MCQ benchmark patterns."""
    name_lower = name.lower().strip()

    # Check explicit exclusions first
    if name_lower in MCQ_BENCHMARK_EXCLUSIONS:
        return False
    for exclusion in MCQ_BENCHMARK_EXCLUSIONS:
        if name_lower.startswith(exclusion + "-") or name_lower.startswith(exclusion + "_"):
            return False

    # Check if matches MCQ prefixes
    return any(
        name_lower == prefix or name_lower.startswith(prefix + "-") or name_lower.startswith(prefix + "_")
        for prefix in MCQ_BENCHMARK_PREFIXES
    )


def parse_mcq_options(prompt_text: str) -> dict[str, str]:
    """Extract option texts from MCQ prompt."""
    options = {}
    pattern = r'(?:^|\n)\s*[\(\[]?([A-Z])[\)\]]?[\.\):\-]\s+(.+?)(?=\n\s*[\(\[]?[A-Z][\)\]]?[\.\):\-]\s+|$)'
    try:
        matches = re.findall(pattern, prompt_text, re.DOTALL | re.MULTILINE)
        for letter, text in matches:
            cleaned = text.strip()
            cleaned = re.sub(r'\s*(?:Answer|Question):\s*$', '', cleaned, flags=re.IGNORECASE)
            options[letter] = cleaned
    except Exception:
        pass
    return options


def extract_model_choice(
    completion_text: str,
    system_prompt: str = "",
    use_logged_parsed: bool = True,
    logged_parsed_answer: str | None = None,
    logged_parsing_method: str | None = None,
) -> tuple[Optional[str], str, str]:
    """
    Extract model's answer choice using the exact same logic as during evaluation:
    1. Apply environment-specific parser (XML/BOXED extraction)
    2. Apply multiple_choice_accuracy parsing strategies

    Args:
        completion_text: Raw model completion
        system_prompt: System prompt (helps detect format)
        use_logged_parsed: If True and logged values available, use them directly
        logged_parsed_answer: Pre-parsed answer from results.jsonl info dict
        logged_parsing_method: Parsing method from results.jsonl info dict

    Returns:
        (choice_letter, parsing_method, env_format)
    """
    # If logged parsed answer is available (from new evaluation runs), use it directly
    if use_logged_parsed and logged_parsed_answer is not None:
        return logged_parsed_answer, logged_parsing_method or "logged", "logged"

    if not completion_text:
        return None, "none", "none"

    # Step 1: Apply environment-specific parsing (XML/BOXED extraction)
    # This mirrors: parsed = parser.parse_answer(completion) or ""
    extracted_text, env_format = apply_environment_parser(completion_text, system_prompt)

    if not extracted_text:
        return None, "none", env_format

    # Step 2: Apply multiple_choice_accuracy parsing logic to the extracted text
    # This mirrors what happens inside multiple_choice_accuracy()

    # Remove think tags (same as multiple_choice_accuracy)
    text = _remove_think_tags(extracted_text)
    text_lower = _nfkc_casefold(text)

    # Strategy 1: Direct single letter (response is just the option)
    normalized = _norm_letter(text.strip())
    if normalized and len(text.strip()) <= 3:
        return normalized, "direct_answer", env_format

    # Strategy 2: Leading option token like "B. answer..."
    leading_match = LEADING_OPTION_PATTERN.match(text)
    if leading_match:
        letter = _norm_letter(leading_match.group(1))
        if letter:
            return letter, "anchored_token", env_format  # Note: leading returns "anchored_token" per original code

    # Strategy 3: Anchored patterns (same as multiple_choice_accuracy)
    anchored_matches = list(ANCHOR_PATTERN.finditer(text_lower))
    if anchored_matches:
        last_match = anchored_matches[-1]
        if last_match.group("neg") is None:  # Not negated
            letter = _norm_letter(last_match.group("opt"))
            if letter:
                return letter, "anchored_token", env_format

    # Strategy 4: Last token in tail
    tail = text[-200:] if len(text) > 200 else text
    tail_tokens = list(TOKEN_PATTERN.finditer(_nfkc_casefold(tail)))
    if tail_tokens:
        for token_match in reversed(tail_tokens):
            letter = _norm_letter(token_match.group(1))
            if letter and letter.isalpha():
                return letter, "last_token", env_format

    return None, "failed", env_format


def analyze_result(result: dict) -> dict[str, Any]:
    """Analyze a single result entry."""
    analysis = {
        "example_id": result.get("example_id"),
        "correct_answer": result.get("answer"),
        "model_choice": None,
        "is_correct": result.get("reward", 0.0) == 1.0,
        "correct_answer_text": None,
        "model_choice_text": None,
        "all_options": {},
        "parsing_method": "failed",
        "env_format": "unknown",
        "options_source": "missing",
    }

    # Extract system prompt (helps detect XML vs BOXED format)
    system_prompt = ""
    prompt = result.get("prompt", [])
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_prompt = msg.get("content", "")
                break

    # Extract user prompt text
    prompt_text = ""
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                prompt_text = msg.get("content", "")
                break

    # Extract completion text
    completion_text = ""
    completion = result.get("completion", [])
    if isinstance(completion, list):
        for msg in completion:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                completion_text = msg.get("content", "")
                break

    # Get options from info or parse from prompt
    info = result.get("info") or {}
    all_options = info.get("options") or info.get("all_options")
    if not isinstance(all_options, dict) or not all_options:
        all_options = parse_mcq_options(prompt_text)
        analysis["options_source"] = "prompt" if all_options else "missing"
    else:
        analysis["options_source"] = "info"
    analysis["all_options"] = all_options

    # Get correct answer text
    analysis["correct_answer_text"] = info.get("answer_text") or info.get("correct_answer_text")
    if not analysis["correct_answer_text"] and analysis["correct_answer"] in all_options:
        analysis["correct_answer_text"] = all_options[analysis["correct_answer"]]

    # Check if we have pre-logged parsed answer (from new evaluation runs with info logging)
    logged_parsed_answer = info.get("model_parsed_answer")
    logged_parsing_method = info.get("parsing_method")

    # Extract model's choice using the full parsing pipeline:
    # 1. Environment-specific parsing (XML/BOXED extraction)
    # 2. multiple_choice_accuracy parsing strategies
    model_choice, method, env_format = extract_model_choice(
        completion_text,
        system_prompt=system_prompt,
        use_logged_parsed=True,
        logged_parsed_answer=logged_parsed_answer,
        logged_parsing_method=logged_parsing_method,
    )

    analysis["env_format"] = env_format

    # Validate choice against known options
    if model_choice and all_options and model_choice not in all_options:
        # Check if it's a numeric answer that should be mapped
        if model_choice.isdigit() and all(k.isalpha() for k in all_options.keys()):
            model_choice = None
            method = "invalid_option"

    analysis["model_choice"] = model_choice
    analysis["parsing_method"] = method

    # Get model choice text
    if model_choice and model_choice in all_options:
        analysis["model_choice_text"] = all_options[model_choice]

    return analysis


def build_confusion_matrix(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build confusion matrix: correct_answer -> model_choice."""
    all_letters = set()
    for r in results:
        if r.get("correct_answer"):
            all_letters.add(r["correct_answer"])
        if r.get("model_choice"):
            all_letters.add(r["model_choice"])

    letters = sorted(all_letters)
    matrix = pd.DataFrame(0, index=letters, columns=letters)
    matrix.index.name = "Correct Answer"
    matrix.columns.name = "Model Choice"

    for r in results:
        correct = r.get("correct_answer")
        choice = r.get("model_choice")
        if correct and choice and correct in letters and choice in letters:
            matrix.loc[correct, choice] += 1

    normalized = matrix.div(matrix.sum(axis=1), axis=0) * 100
    normalized = normalized.fillna(0)
    matrix["Total"] = matrix.sum(axis=1)

    return matrix, normalized


def compare_rollouts(results_by_rollout: dict[str, list[dict]]) -> pd.DataFrame:
    """Compare model choices across different rollouts."""
    rollout_ids = sorted(results_by_rollout.keys())
    if len(rollout_ids) < 2:
        return pd.DataFrame()

    # Build lookup maps
    rollout_maps = {}
    for rid in rollout_ids:
        rollout_maps[rid] = {r.get("example_id"): r for r in results_by_rollout[rid] if r.get("example_id") is not None}

    all_ids = set()
    for mapping in rollout_maps.values():
        all_ids.update(mapping.keys())

    comparisons = []
    base_rollout = "base" if "base" in rollout_ids else rollout_ids[0]

    for example_id in sorted(all_ids):
        base_result = rollout_maps.get(base_rollout, {}).get(example_id, {})
        comp = {
            "example_id": example_id,
            "correct_answer": base_result.get("correct_answer"),
            "correct_answer_text": base_result.get("correct_answer_text"),
        }

        for rid in rollout_ids:
            r = rollout_maps[rid].get(example_id)
            if r:
                comp[f"{rid}_choice"] = r.get("model_choice")
                comp[f"{rid}_choice_text"] = r.get("model_choice_text")
            else:
                comp[f"{rid}_choice"] = None
                comp[f"{rid}_choice_text"] = None

        # Analyze variation
        choices = [comp.get(f"{rid}_choice") for rid in rollout_ids]
        choice_texts = [comp.get(f"{rid}_choice_text") for rid in rollout_ids]
        valid_choices = [c for c in choices if c is not None]
        valid_texts = [t for t in choice_texts if t is not None]

        comp["has_comparable_data"] = len(valid_choices) >= 2

        if len(valid_choices) < 2:
            comp["has_variation"] = False
            comp["semantic_consistency"] = None
            comp["variation_type"] = "insufficient_data"
        else:
            has_letter_variation = len(set(valid_choices)) > 1
            has_semantic_info = len(valid_texts) == len(valid_choices)

            if has_semantic_info:
                normalized_texts = [t.lower().strip() if t else None for t in valid_texts]
                semantic_consistency = len(set(t for t in normalized_texts if t)) <= 1
            else:
                semantic_consistency = None

            comp["has_variation"] = has_letter_variation
            comp["semantic_consistency"] = semantic_consistency

            if not has_letter_variation:
                comp["variation_type"] = "none"
            elif semantic_consistency is True:
                comp["variation_type"] = "letter_only"
            elif semantic_consistency is False:
                comp["variation_type"] = "semantic_change"
            else:
                comp["variation_type"] = "unknown_semantic"

        comparisons.append(comp)

    return pd.DataFrame(comparisons)


def compute_distribution_stats(counts: dict[str, int]) -> dict[str, float] | None:
    """Compute entropy and bias metrics for a distribution."""
    total = sum(counts.values())
    if total == 0:
        return None
    probs = [c / total for c in counts.values()]
    k = len(probs)
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    uniform = 1.0 / k if k > 0 else 0.0
    l1_bias = sum(abs(p - uniform) for p in probs) if k > 0 else 0.0
    max_min_bias = (max(probs) - min(probs)) if k > 0 else 0.0
    return {"entropy": entropy, "l1_bias": l1_bias, "max_min_bias": max_min_bias}


def aggregate_statistics(results: list[dict], rollout_df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Compute aggregate statistics from analyzed results."""
    stats = {
        "total_questions": len(results),
        "total_correct": sum(1 for r in results if r.get("is_correct")),
        "total_incorrect": sum(1 for r in results if not r.get("is_correct")),
    }

    stats["accuracy"] = stats["total_correct"] / stats["total_questions"] if stats["total_questions"] > 0 else 0.0

    # Parsing success rate
    parsed = sum(1 for r in results if r.get("model_choice") is not None)
    stats["parsing_success_rate"] = parsed / stats["total_questions"] if stats["total_questions"] > 0 else 0.0

    # Choice distribution
    choice_counts = {}
    for r in results:
        c = r.get("model_choice")
        if c:
            choice_counts[c] = choice_counts.get(c, 0) + 1
    stats["choice_distribution"] = choice_counts
    stats["choice_distribution_stats"] = compute_distribution_stats(choice_counts)

    # Correct answer distribution (benchmark balance)
    correct_counts = {}
    for r in results:
        c = r.get("correct_answer")
        if c:
            correct_counts[c] = correct_counts.get(c, 0) + 1
    stats["correct_answer_distribution"] = correct_counts
    stats["correct_answer_distribution_stats"] = compute_distribution_stats(correct_counts)

    # Positional bias: compare choice vs correct distributions
    if choice_counts and correct_counts:
        # Compute per-position bias (choice_rate - correct_rate)
        all_positions = set(choice_counts.keys()) | set(correct_counts.keys())
        total_choices = sum(choice_counts.values())
        total_correct = sum(correct_counts.values())
        positional_bias = {}
        for pos in sorted(all_positions):
            choice_rate = choice_counts.get(pos, 0) / total_choices if total_choices > 0 else 0
            correct_rate = correct_counts.get(pos, 0) / total_correct if total_correct > 0 else 0
            positional_bias[pos] = choice_rate - correct_rate
        stats["positional_bias"] = positional_bias

    # Parsing method distribution
    method_counts = {}
    for r in results:
        m = r.get("parsing_method")
        if m:
            method_counts[m] = method_counts.get(m, 0) + 1
    stats["parsing_method_distribution"] = method_counts

    # Environment format distribution (XML vs BOXED)
    format_counts = {}
    for r in results:
        f = r.get("env_format")
        if f:
            format_counts[f] = format_counts.get(f, 0) + 1
    stats["env_format_distribution"] = format_counts

    # Options coverage
    options_present = sum(1 for r in results if r.get("all_options"))
    stats["options_coverage_rate"] = options_present / stats["total_questions"] if stats["total_questions"] > 0 else 0.0

    # Rollout statistics
    if rollout_df is not None and not rollout_df.empty:
        comparable = rollout_df[rollout_df["has_comparable_data"] == True] if "has_comparable_data" in rollout_df.columns else rollout_df
        stats["questions_with_rollouts"] = len(comparable)
        stats["rollout_example_count"] = len(rollout_df)

        if "has_variation" in rollout_df.columns and len(comparable) > 0:
            variation_count = comparable["has_variation"].sum()
            stats["variation_rate"] = variation_count / len(comparable)
        else:
            stats["variation_rate"] = None

        if "semantic_consistency" in rollout_df.columns:
            varied = comparable[comparable["has_variation"] == True]
            if not varied.empty:
                valid_sem = varied["semantic_consistency"].dropna()
                stats["semantic_consistency_rate"] = valid_sem.mean() if len(valid_sem) > 0 else None
            else:
                stats["semantic_consistency_rate"] = None
        else:
            stats["semantic_consistency_rate"] = None

        if "variation_type" in rollout_df.columns:
            stats["variation_type_distribution"] = rollout_df["variation_type"].value_counts().to_dict()

    return stats


def load_results_jsonl(filepath: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    try:
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return results


def _extract_model_prefix(model_dir_name: str) -> str:
    """Extract model prefix from model directory name like 'gpt-5_2-20251213-183612' -> 'gpt-5_2'."""
    # Remove timestamp suffix (format: -YYYYMMDD-HHMMSS)
    import re
    match = re.match(r"(.+)-\d{8}-\d{6}$", model_dir_name)
    if match:
        return match.group(1)
    return model_dir_name


def _get_model_prefix_variants(model_prefix: str) -> list[str]:
    """Generate possible variants of model prefix for flexible matching.

    E.g., 'gpt-oss-20b-med' -> ['gpt-oss-20b-med', 'gpt-oss-20b', 'gpt-oss']
    This handles cases where subdirectories use shorter prefixes.
    """
    variants = [model_prefix]

    # Common suffixes that might be in model name but not in subdirectory names
    suffix_patterns = ['-med', '-high', '-low', '-small', '-large', '-tiny', '-base', '-instruct']

    # Try removing known suffixes
    for suffix in suffix_patterns:
        if model_prefix.endswith(suffix):
            variants.append(model_prefix[:-len(suffix)])

    # Also generate progressively shorter prefixes by splitting on '-'
    parts = model_prefix.split('-')
    for i in range(len(parts) - 1, 0, -1):
        variant = '-'.join(parts[:i])
        if variant not in variants:
            variants.append(variant)

    return variants


def _extract_benchmark_name(dirname: str, model_prefix: str) -> tuple[str, str]:
    """Extract benchmark name and rollout ID from directory name.

    Handles formats:
    - {model_prefix}-{benchmark}  -> benchmark, "base"
    - {model_prefix}-{benchmark}-rollout{seed}  -> benchmark, "rollout{seed}"
    - {benchmark}  -> benchmark, "base"
    - {benchmark}-rollout{seed}  -> benchmark, "rollout{seed}"

    Also handles prefix variants where subdirectories may use shorter model prefixes
    (e.g., model dir 'gpt-oss-20b-med' but subdirs use 'gpt-oss-20b' as prefix).
    """
    # Try all prefix variants (longest first)
    if model_prefix:
        for prefix in _get_model_prefix_variants(model_prefix):
            if dirname.startswith(prefix + "-"):
                dirname = dirname[len(prefix) + 1:]
                break

    # Handle rollout suffix
    if "-rollout" in dirname:
        parts = dirname.rsplit("-rollout", 1)
        benchmark_name = parts[0]
        rollout_id = f"rollout{parts[1]}"
    else:
        benchmark_name = dirname
        rollout_id = "base"

    # Handle -- separator (vf-eval format)
    if "--" in benchmark_name:
        benchmark_name = benchmark_name.split("--")[0]

    return benchmark_name, rollout_id


def discover_benchmarks(model_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover benchmarks and rollouts in model directory.

    Supports multiple directory structures:
    1. model_dir/{model_prefix}-{benchmark}/results.jsonl (medmarks format)
    2. model_dir/{benchmark}/results.jsonl
    3. model_dir/{benchmark--model--name}/hash_id/results.jsonl (vf-eval structure)
    """
    benchmarks = {}
    if not model_dir.is_dir():
        return benchmarks

    # Extract model prefix from parent directory name
    model_prefix = _extract_model_prefix(model_dir.name)

    for item in model_dir.iterdir():
        if not item.is_dir():
            continue

        # Check if results.jsonl is directly in this directory
        results_file = item / "results.jsonl"
        if results_file.exists():
            benchmark_name, rollout_id = _extract_benchmark_name(item.name, model_prefix)

            if benchmark_name not in benchmarks:
                benchmarks[benchmark_name] = {}
            benchmarks[benchmark_name][rollout_id] = results_file
            continue

        # Check for vf-eval structure: benchmark--model--name/hash_id/results.jsonl
        for subitem in item.iterdir():
            if not subitem.is_dir():
                continue
            results_file = subitem / "results.jsonl"
            if not results_file.exists():
                continue

            benchmark_name, rollout_id = _extract_benchmark_name(item.name, model_prefix)
            # For vf-eval structure, use hash as rollout identifier if base
            if rollout_id == "base":
                rollout_id = subitem.name

            if benchmark_name not in benchmarks:
                benchmarks[benchmark_name] = {}
            benchmarks[benchmark_name][rollout_id] = results_file

    return benchmarks


def process_benchmark(
    benchmark_name: str,
    rollout_paths: dict[str, Path],
    output_dir: Path,
    model_name: str,
) -> dict[str, Any]:
    """Process a single benchmark with all its rollouts."""
    print(f"\n  Processing {benchmark_name}...")
    print(f"    Found {len(rollout_paths)} version(s): {', '.join(sorted(rollout_paths.keys()))}")

    analyzed_by_rollout = {}
    all_analyzed = []

    for rollout_id, results_path in sorted(rollout_paths.items()):
        raw_results = load_results_jsonl(results_path)
        if not raw_results:
            continue

        analyzed = []
        for raw in raw_results:
            analysis = analyze_result(raw)
            analysis["rollout_id"] = rollout_id
            analyzed.append(analysis)
            all_analyzed.append(analysis)

        analyzed_by_rollout[rollout_id] = analyzed

    if not all_analyzed:
        print(f"    Warning: No results to analyze")
        return {}

    # Save per-example analysis
    output_base = output_dir / f"{model_name}_{benchmark_name}"
    df = pd.DataFrame(all_analyzed)
    df.to_csv(output_base.with_suffix(".csv"), index=False)

    # Confusion matrix
    confusion_counts, confusion_norm = build_confusion_matrix(all_analyzed)
    confusion_counts.to_csv(output_base.with_name(f"{output_base.name}_confusion.csv"))

    # Cross-rollout comparison
    rollout_df = None
    if len(analyzed_by_rollout) > 1:
        rollout_df = compare_rollouts(analyzed_by_rollout)
        if not rollout_df.empty:
            rollout_df.to_csv(output_base.with_name(f"{output_base.name}_rollouts.csv"), index=False)

    # Aggregate statistics
    stats = aggregate_statistics(all_analyzed, rollout_df)
    stats["benchmark_name"] = benchmark_name

    print(f"    Total: {stats['total_questions']}, Accuracy: {stats['accuracy']:.1%}, Parsed: {stats['parsing_success_rate']:.1%}")
    if stats.get("variation_rate") is not None:
        print(f"    Variation: {stats['variation_rate']:.1%}, Semantic consistency: {stats.get('semantic_consistency_rate', 0):.1%}")

    return stats


def weighted_mean(stats_list: list[dict], key: str, weight_key: str) -> float | None:
    """Compute weighted mean."""
    values = [(s.get(key), s.get(weight_key)) for s in stats_list if s.get(key) is not None and s.get(weight_key, 0) > 0]
    if not values:
        return None
    total_weight = sum(w for _, w in values)
    return sum(v * w for v, w in values) / total_weight if total_weight > 0 else None


def normalize_benchmark_name(benchmark_name: str, model_name: str) -> str:
    """Normalize benchmark name by removing model prefix."""
    model_base = re.sub(r'-\d{8}-\d{6}$', '', model_name)
    for prefix in [model_name, model_base]:
        for sep in ["-", "--"]:
            if benchmark_name.startswith(prefix + sep):
                return benchmark_name[len(prefix) + len(sep):]
    return benchmark_name


def compute_win_rates(all_summaries: list[dict]) -> dict[str, dict[str, float | None]]:
    """Compute win rates across models per benchmark."""
    bench_map: dict[str, list[tuple[str, float, int]]] = {}
    for summary in all_summaries:
        model = summary.get("model_name")
        per_bench = summary.get("per_benchmark_stats", {})
        for bench, stats in per_bench.items():
            if not model or not bench:
                continue
            bench_key = normalize_benchmark_name(bench, model)
            acc = stats.get("accuracy")
            n_q = stats.get("total_questions", 0)
            if acc is None or n_q <= 0:
                continue
            bench_map.setdefault(bench_key, []).append((model, acc, n_q))

    model_win_rates: dict[str, list[tuple[float, float]]] = {}
    for bench, entries in bench_map.items():
        if len(entries) < 2:
            continue
        weight = math.log(entries[0][2]) if entries[0][2] > 1 else 0.0
        for i, (model_i, acc_i, _) in enumerate(entries):
            wins = sum(1.0 for j, (_, acc_j, _) in enumerate(entries) if i != j and acc_i > acc_j)
            ties = sum(0.5 for j, (_, acc_j, _) in enumerate(entries) if i != j and acc_i == acc_j)
            win_rate = (wins + ties) / (len(entries) - 1) if len(entries) > 1 else 0.0
            model_win_rates.setdefault(model_i, []).append((win_rate, weight))

    results = {}
    for model, rates in model_win_rates.items():
        if not rates:
            results[model] = {"mean_win_rate": None, "weighted_mean_win_rate": None}
            continue
        mean_win = sum(r for r, _ in rates) / len(rates)
        weight_sum = sum(w for _, w in rates)
        weighted_mean = sum(r * w for r, w in rates) / weight_sum if weight_sum > 0 else None
        results[model] = {"mean_win_rate": mean_win, "weighted_mean_win_rate": weighted_mean}

    return results


def process_model(
    model_dir: Path,
    output_dir: Path,
    model_name: str | None = None,
    only_benchmarks: set[str] | None = None,
    merge_existing: bool = False,
) -> dict[str, Any] | None:
    """Process all benchmarks for a single model."""
    model_name = model_name or model_dir.name

    print(f"\n{'='*70}")
    print(f"Processing model: {model_name}")
    print(f"{'='*70}")

    benchmarks = discover_benchmarks(model_dir)
    # Filter to MCQ only
    benchmarks = {k: v for k, v in benchmarks.items() if is_mcq_benchmark(k)}
    if only_benchmarks:
        benchmarks = {k: v for k, v in benchmarks.items() if k in only_benchmarks}

    if not benchmarks:
        print(f"  No MCQ benchmarks found")
        return None

    print(f"  Found {len(benchmarks)} MCQ benchmark(s)")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing stats if merging
    benchmark_stats: dict[str, dict] = {}
    summary_path = output_dir / f"{model_name}_summary.json"
    if merge_existing and summary_path.exists():
        try:
            with open(summary_path) as f:
                existing = json.load(f)
            benchmark_stats = existing.get("per_benchmark_stats", {})
        except Exception:
            pass

    # Process each benchmark
    for bench_name, rollout_paths in sorted(benchmarks.items()):
        stats = process_benchmark(bench_name, rollout_paths, output_dir, model_name)
        if stats:
            benchmark_stats[bench_name] = stats

    if not benchmark_stats:
        return None

    # Aggregate statistics
    total_q = sum(s["total_questions"] for s in benchmark_stats.values())
    total_c = sum(s["total_correct"] for s in benchmark_stats.values())

    aggregate = {
        "total_questions": total_q,
        "total_correct": total_c,
        "overall_accuracy": total_c / total_q if total_q > 0 else 0.0,
        "parsing_success_rate": weighted_mean(list(benchmark_stats.values()), "parsing_success_rate", "total_questions"),
        "options_coverage_rate": weighted_mean(list(benchmark_stats.values()), "options_coverage_rate", "total_questions"),
    }

    # Rollout aggregates
    benchmarks_with_rollouts = [s for s in benchmark_stats.values() if s.get("questions_with_rollouts", 0) > 0]
    if benchmarks_with_rollouts:
        aggregate["questions_with_rollouts"] = sum(s["questions_with_rollouts"] for s in benchmarks_with_rollouts)
        aggregate["variation_rate"] = weighted_mean(benchmarks_with_rollouts, "variation_rate", "questions_with_rollouts")
        aggregate["semantic_consistency_rate"] = weighted_mean(benchmarks_with_rollouts, "semantic_consistency_rate", "questions_with_rollouts")

    summary = {
        "model_name": model_name,
        "total_benchmarks": len(benchmark_stats),
        "per_benchmark_stats": benchmark_stats,
        "aggregate_stats": aggregate,
    }

    # Save summary
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save benchmark metrics CSV
    metrics_rows = []
    for bench, stats in benchmark_stats.items():
        row = {
            "benchmark": bench,
            "total_questions": stats.get("total_questions"),
            "accuracy": stats.get("accuracy"),
            "parsing_success_rate": stats.get("parsing_success_rate"),
            "variation_rate": stats.get("variation_rate"),
            "semantic_consistency_rate": stats.get("semantic_consistency_rate"),
        }
        metrics_rows.append(row)
    pd.DataFrame(metrics_rows).to_csv(output_dir / f"{model_name}_benchmark_metrics.csv", index=False)

    print(f"\n  Summary: {len(benchmark_stats)} benchmarks, {total_q} questions, {aggregate['overall_accuracy']:.1%} accuracy")

    return summary


def update_all_models_metrics(output_dir: Path) -> None:
    """Recompute and save cross-model metrics."""
    summary_files = list(output_dir.glob("*/*_summary.json"))
    if not summary_files:
        print("No summary files found")
        return

    all_summaries = []
    for path in summary_files:
        with open(path) as f:
            all_summaries.append(json.load(f))

    win_rates = compute_win_rates(all_summaries)

    rows = []
    for summary in all_summaries:
        agg = summary.get("aggregate_stats", {})
        model_name = summary.get("model_name")
        row = {
            "model": model_name,
            "total_benchmarks": summary.get("total_benchmarks"),
            "total_questions": agg.get("total_questions"),
            "overall_accuracy": agg.get("overall_accuracy"),
            "parsing_success_rate": agg.get("parsing_success_rate"),
            "variation_rate": agg.get("variation_rate"),
            "semantic_consistency_rate": agg.get("semantic_consistency_rate"),
        }
        if model_name in win_rates:
            row.update(win_rates[model_name])
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "all_models_metrics.csv", index=False)
    df.to_json(output_dir / "all_models_metrics.json", orient="records", indent=2)
    print(f"\nSaved all-model metrics ({len(rows)} models)")


def main():
    parser = argparse.ArgumentParser(description="Analyze MCQ benchmark results")
    parser.add_argument("--logs-dir", type=str, required=True, help="Directory containing model results")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for analysis")
    parser.add_argument("--model", type=str, help="Specific model to analyze")
    parser.add_argument("--all-models", action="store_true", help="Analyze all models in logs-dir")
    parser.add_argument("--benchmark", action="append", help="Only analyze specific benchmark(s)")

    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    if not logs_dir.exists():
        print(f"Error: {logs_dir} does not exist")
        sys.exit(1)

    only_benchmarks = set(args.benchmark) if args.benchmark else None

    if args.all_models:
        model_dirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        print(f"Found {len(model_dirs)} model directories")

        for model_dir in sorted(model_dirs):
            try:
                process_model(
                    model_dir,
                    output_dir / model_dir.name,
                    model_dir.name,
                    only_benchmarks,
                    merge_existing=bool(only_benchmarks),
                )
            except Exception as e:
                print(f"Error processing {model_dir.name}: {e}")
                import traceback
                traceback.print_exc()

        update_all_models_metrics(output_dir)
    else:
        model_name = args.model or logs_dir.name
        process_model(logs_dir, output_dir, model_name, only_benchmarks, merge_existing=bool(only_benchmarks))
        # Update aggregate metrics after single model run
        update_all_models_metrics(output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
