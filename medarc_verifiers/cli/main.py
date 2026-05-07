"""Unified MedARC evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Mapping, MutableMapping, Sequence

import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table
from verifiers.utils.eval_utils import run_evaluation
from verifiers.utils.save_utils import make_serializable

from medarc_verifiers.cli._constants import (
    BENCH_COMMAND,
    COMMAND,
    DEFAULT_ENDPOINTS_PATH,
    DEFAULT_EVALS_DIR,
    DEFAULT_ENV_CONFIG_ROOT,
    DEFAULT_ENV_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RUNS_RAW_DIR,
    PROCESS_COMMAND,
    WINRATE_COMMAND,
)
from medarc_verifiers.cli._schemas import EnvironmentConfigSchema, EnvironmentExportConfig
from medarc_verifiers.cli._single_run import run_single_mode
from medarc_verifiers.cli.eval_identity import EvalPathPlan, generate_variant_id, plan_eval_paths
from medarc_verifiers.cli.eval_identity import metadata_identity_fields
from medarc_verifiers.cli.hf import HFSyncConfig, sync_files_to_hub
from medarc_verifiers.cli.process import PROCESS_DEFAULT_STATUS_FILTER, ProcessOptions, ProcessResult, run_process
from medarc_verifiers.cli.utils.config_io import load_mapping_file
from medarc_verifiers.cli.utils.overrides import build_cli_override
from medarc_verifiers.cli.utils.shared import (
    dataset_is_excluded,
    normalize_dataset_ids,
    normalize_model_ids,
)
from medarc_verifiers.cli.upstream_eval import EvalConfigOverrides, build_eval_config, load_toml_eval_configs
from medarc_verifiers.utils.pathing import resolve_under
from medarc_verifiers.cli.winrate import (
    WinrateConfig,
    _resolve_source,
    list_models,
    print_winrate_summary_markdown,
    run_winrate,
)

logger = logging.getLogger(__name__)
HELP_FLAGS = {"-h", "--help"}


def build_batch_parser() -> argparse.ArgumentParser:
    """Construct the unified CLI parser."""
    parser = argparse.ArgumentParser(
        prog=COMMAND,
        description="Run MedARC evaluations using upstream verifiers TOML configs.",
    )
    parser.add_argument("-c", "--config", required=True, type=Path, help="Path to an upstream TOML eval config file.")
    parser.add_argument("--force", action="store_true", help="Archive existing deterministic output and rerun.")
    parser.add_argument("--output-dir", type=Path, help="Override the output directory from the configuration.")
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=DEFAULT_ENV_DIR,
        help="Directory containing environments (default: %(default)s).",
    )
    parser.add_argument(
        "--endpoints-path",
        type=Path,
        default=DEFAULT_ENDPOINTS_PATH,
        help=f"Path to the endpoints registry file (default: {DEFAULT_ENDPOINTS_PATH}).",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="Override API base URL for all TOML evals.",
    )
    parser.add_argument("--api-key-var", default=None, help="Override API key environment variable for TOML bench.")
    parser.add_argument("--provider", default=None, help="Override provider shorthand for TOML bench.")
    parser.add_argument("--model", "-m", default=None, help="Override model for every TOML eval.")
    parser.add_argument(
        "--eval-index", "--job-index", dest="eval_index", type=int, help="Run only one TOML eval by 1-based index."
    )
    parser.add_argument("--start-at", type=int, help="Start TOML execution at this 1-based eval index.")
    parser.add_argument("--stop-after", type=int, help="Stop TOML execution after this 1-based eval index.")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue TOML sequential execution after a failed eval.",
    )
    parser.add_argument(
        "--env-arg", action="append", help="Override an environment argument with KEY=VALUE (repeatable)."
    )
    parser.add_argument("--env-args", help="Override environment arguments with a JSON object.")
    parser.add_argument(
        "--sampling-arg", action="append", help="Override a sampling argument with KEY=VALUE (repeatable)."
    )
    parser.add_argument("--sampling-args", help="Override sampling arguments with a JSON object.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve jobs and report overrides without executing them."
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Override max_concurrent for every TOML eval.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override request timeout in seconds for every TOML eval.",
    )
    parser.add_argument(
        "--max-retries",
        dest="rollout_max_retries",
        type=int,
        default=None,
        help="Override upstream rollout max_retries for every TOML eval.",
    )
    parser.add_argument(
        "--sleep",
        "--sleep-seconds",
        dest="sleep",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each job (overridden by per-job sleep).",
    )
    return parser


def build_process_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{COMMAND} {PROCESS_COMMAND}",
        description="Process MedARC run outputs into Parquet datasets and optional HF uploads.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML/JSON config file providing defaults for process options (CLI flags override).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=None,
        help=f"Directory containing raw run outputs (default: {DEFAULT_RUNS_RAW_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory to store processed parquet files (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--env-config-root",
        type=Path,
        default=None,
        help=f"Directory containing environment YAMLs for export settings (default: {DEFAULT_ENV_CONFIG_ROOT}).",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Filter runs by manifest status (repeatable).",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=None,
        help="Exclude these dataset/env ids from processing (repeatable; comma-separated values allowed).",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=None,
        help="Exclude these model ids from processing (repeatable; comma-separated values allowed).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=None,
        help="Delete processed outputs in --output-dir and rebuild from --runs-dir.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=None,
        help="Skip confirmation prompts (used by --clean and HF repo creation).",
    )
    parser.add_argument("--processed-at", default=None, help="Override processed_at timestamp (ISO8601).")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Plan processing without writing outputs.")
    parser.add_argument(
        "--replace-model",
        action="append",
        default=None,
        help="Rebuild existing processed outputs for these model ids (repeatable; comma-separated values allowed).",
    )
    parser.add_argument(
        "--replace-env",
        action="append",
        default=None,
        help="Rebuild existing processed outputs for these env ids (repeatable; comma-separated values allowed).",
    )
    parser.add_argument(
        "--max-results-missing-pct",
        type=float,
        default=None,
        help=(
            "Fail if a selected latest job record is missing more than this percentage of expected results.jsonl rows "
            "based on manifest job fields (row_count, num_examples, rollouts_per_example). "
            "Computed per selected job record and enforced only on the latest selected run; does not use "
            "manifest summary.completed/summary.total or fall back to older runs (default: 2.5)."
        ),
    )
    parser.add_argument(
        "--winrate",
        type=Path,
        default=None,
        help="Run winrate after processing using the provided config file. If omitted, an embedded winrate section in --config is used.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of parallel workers for processing datasets (default: 4). Use 1 to disable multiprocessing.",
    )

    parser.add_argument("--hf-repo", default=None, help="Hugging Face repo id for dataset sync.")
    parser.add_argument(
        "--hf-pull-policy",
        choices=("prompt", "pull", "clean", "continue-upload"),
        default=None,
        help="Baseline policy when output dir is non-empty in HF mode.",
    )
    parser.add_argument("--hf-branch", default=None, help="Target HF branch.")
    parser.add_argument("--hf-token", default=None, help="Auth token for HF operations.")
    parser.add_argument(
        "--hf-private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Push dataset as private (default: false).",
    )
    parser.add_argument(
        "--hf-request-timeout",
        type=float,
        default=None,
        help="HTTP request timeout (seconds) for HF upload/create_commit (default: 300).",
    )
    parser.add_argument(
        "--hf-retries",
        type=int,
        default=None,
        help="Retry count for HF upload timeouts/transport errors (default: 3).",
    )
    parser.add_argument(
        "--hf-max-files-per-commit",
        type=int,
        default=None,
        help="Split HF uploads into multiple commits with at most this many files per commit (default: no split).",
    )

    return parser


def build_winrate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"{COMMAND} {WINRATE_COMMAND}",
        description="Compute HELM-style win rates from processed environment parquet files.",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a YAML/JSON config file providing defaults for winrate options (CLI flags override).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help=f"Directory containing processed parquet outputs (default: {DEFAULT_PROCESSED_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store winrate outputs (default: <processed-dir>/winrate).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for winrates JSON (skips writing latest.json).",
    )
    parser.add_argument(
        "--output-name",
        help="Base name for winrates JSON (timestamp appended automatically).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=None,
        help="Skip confirmation prompts (used by HF repo creation).",
    )
    parser.add_argument(
        "--processed-at",
        help="Timestamp used for default output naming (ISO8601).",
    )
    parser.add_argument(
        "--missing-policy",
        choices=("zero", "neg-inf"),
        default=None,
        help="Missing reward policy when comparing models (default: %(default)s).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Tie tolerance epsilon for pairwise comparisons (default: %(default)s).",
    )
    parser.add_argument(
        "--min-common",
        type=int,
        default=None,
        help="Minimum overlapping examples per dataset to retain a pairwise result.",
    )
    parser.add_argument(
        "--weight-policy",
        choices=("equal", "ln", "sqrt", "cap"),
        default=None,
        help="Dataset weighting policy when aggregating win rates (default: %(default)s).",
    )
    parser.add_argument(
        "--weight-cap",
        type=int,
        default=None,
        help="Cap applied when using --weight-policy=cap (default: %(default)s).",
    )
    parser.add_argument(
        "--include-model",
        action="append",
        default=None,
        help="Only include these model ids in win rate calculation (repeatable).",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        default=None,
        help="Exclude these model ids from win rate calculation (repeatable).",
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=None,
        help="Exclude these dataset/env ids from win rate calculation (repeatable; comma-separated values allowed).",
    )
    parser.add_argument(
        "--partial-datasets",
        choices=("strict", "include"),
        default=None,
        help=(
            "Dataset selection policy when --include-model is set: "
            "strict drops datasets missing any included models, "
            "include keeps them with missing models treated as all-missing."
        ),
    )
    parser.add_argument(
        "--dataset-coverage",
        choices=("all-models", "per-model"),
        default=None,
        help=(
            "Dataset coverage policy for winrate computation: "
            "all-models enforces an intersection of datasets across the compared models (default), "
            "per-model uses the legacy behavior where each model may be averaged over a different dataset set."
        ),
    )
    parser.add_argument("--hf-repo", help="Hugging Face repo id used for processed download and winrate upload.")
    parser.add_argument(
        "--hf-processed-pull",
        action="store_true",
        default=None,
        help="Pull missing processed files from HF even when --processed-dir is non-empty.",
    )
    parser.add_argument("--hf-branch", help="Target HF branch or revision for processed download.")
    parser.add_argument("--hf-token", help="Auth token for HF operations.")
    parser.add_argument(
        "--hf-winrate-dir",
        default=None,
        help="Path under the HF repo where winrate artifacts are uploaded (default: winrate).",
    )
    parser.add_argument(
        "--hf-private",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Push winrate outputs as private when uploading (default: false).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model ids in the source parquet files (local or HF) and exit.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Unified CLI entry point."""
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        _print_general_help()
        return 0

    if args_list[0] in HELP_FLAGS:
        _print_general_help()
        return 0

    if args_list[0] == BENCH_COMMAND:
        return _run_batch_mode(args_list[1:])
    if args_list[0] == PROCESS_COMMAND:
        return _run_process_mode(args_list[1:])
    if args_list[0] == WINRATE_COMMAND:
        return _run_winrate_mode(args_list[1:])

    return run_single_mode(args_list)


def _run_batch_mode(argv: Sequence[str]) -> int:
    parser = build_batch_parser()
    args = parser.parse_args(argv)
    args.endpoints_path_explicit = _option_was_provided(argv, "--endpoints-path")

    try:
        args.cli_env_args = build_cli_override(
            json_payload=args.env_args,
            pairs=args.env_arg,
            json_flag="--env-args",
            pair_flag="--env-arg",
        )
        args.cli_sampling_args = build_cli_override(
            json_payload=args.sampling_args,
            pairs=args.sampling_arg,
            json_flag="--sampling-args",
            pair_flag="--sampling-arg",
        )
        if args.rollout_max_retries is not None and args.rollout_max_retries < 0:
            raise ValueError("--max-retries must be >= 0.")
    except ValueError as exc:
        parser.error(str(exc))

    config_path = Path(args.config).expanduser()
    if config_path.suffix.lower() != ".toml":
        parser.error("medarc-eval bench now accepts upstream TOML configs only.")
    try:
        _validate_toml_selection_args(args, parser=parser)
        return _run_toml_bench(args)
    except Exception as exc:  # noqa: BLE001
        logger.exception("TOML bench failed: %s", exc)
        return 1


def _run_process_mode(argv: Sequence[str]) -> int:
    parser, args = _resolve_process_args(argv)
    winrate_args = _resolve_embedded_winrate(args, parser=parser)

    try:
        env_export_map = _load_env_export_map(args.env_config_root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load environment export configs: %s", exc)
        env_export_map = {}

    options = _build_process_options(args)

    try:
        result = run_process(options, env_export_map=env_export_map)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Process pipeline failed: %s", exc)
        return 1

    _log_process_result(result)
    return _run_process_post_steps(args, parser=parser, options=options, winrate_args=winrate_args)


def _resolve_process_args(argv: Sequence[str]) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = build_process_parser()
    args = parser.parse_args(argv)

    if args.config:
        _load_and_apply_config(args, args.config, mode="process", parser=parser)
    _finalize_config_args(args, mode="process")
    _validate_process_args(args, argv=argv, parser=parser)
    return parser, args


def _validate_process_args(
    args: argparse.Namespace,
    *,
    argv: Sequence[str],
    parser: argparse.ArgumentParser,
) -> None:
    for flag, attr in (("--replace-model", "replace_model"), ("--replace-env", "replace_env")):
        if _option_was_provided(argv, flag) and not getattr(args, attr, None):
            parser.error(f"{flag} requires at least one non-empty value.")
    try:
        if args.exclude_dataset:
            normalize_dataset_ids(args.exclude_dataset, label="process exclude dataset")
        if args.exclude_model:
            normalize_model_ids(args.exclude_model, label="process exclude model")
        if args.max_results_missing_pct is not None:
            value = float(args.max_results_missing_pct)
            if value < 0:
                parser.error("--max-results-missing-pct must be non-negative.")
    except ValueError as exc:
        parser.error(str(exc))


def _build_process_options(args: argparse.Namespace) -> ProcessOptions:
    hf_config = HFSyncConfig.from_cli(
        repo=args.hf_repo,
        branch=args.hf_branch,
        token=args.hf_token,
        private=args.hf_private,
        dry_run=args.dry_run,
        request_timeout=args.hf_request_timeout,
        retries=args.hf_retries,
        max_files_per_commit=args.hf_max_files_per_commit,
    )
    status_values = list(args.status or [])
    status_filter = tuple(status_values) if status_values else PROCESS_DEFAULT_STATUS_FILTER
    max_results_missing_pct = float(args.max_results_missing_pct) if args.max_results_missing_pct is not None else 2.5
    processed_with_args = {
        "status": list(status_filter),
        "max_results_missing_pct": max_results_missing_pct,
        "exclude_datasets": args.exclude_dataset or [],
        "exclude_models": args.exclude_model or [],
        "replace_models": args.replace_model or [],
        "replace_envs": args.replace_env or [],
        "dry_run": bool(args.dry_run),
        "clean": bool(args.clean),
        "hf_repo": args.hf_repo,
        "hf_pull_policy": args.hf_pull_policy,
        "hf_request_timeout": args.hf_request_timeout,
        "hf_retries": args.hf_retries,
        "hf_max_files_per_commit": args.hf_max_files_per_commit,
        "max_workers": args.max_workers,
    }
    return ProcessOptions(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        exclude_datasets=tuple(args.exclude_dataset or ()),
        exclude_models=tuple(args.exclude_model or ()),
        replace_models=tuple(args.replace_model or ()),
        replace_envs=tuple(args.replace_env or ()),
        processed_at=args.processed_at,
        processed_with_args=processed_with_args,
        status_filter=status_filter,
        max_results_missing_pct=max_results_missing_pct,
        dry_run=bool(args.dry_run),
        clean=bool(args.clean),
        assume_yes=bool(args.yes),
        hf_config=hf_config,
        hf_pull_policy=args.hf_pull_policy,
        max_workers=args.max_workers,
    )


def _resolve_embedded_winrate(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
) -> argparse.Namespace | None:
    embedded_winrate = False
    if args.config and args.winrate is None:
        try:
            embedded_winrate = _config_has_embedded_winrate(Path(args.config).expanduser())
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))

    if args.winrate:
        winrate_path = Path(args.winrate).expanduser()
        if not winrate_path.exists():
            parser.error(f"Winrate config path '{winrate_path}' does not exist.")
        args.winrate = winrate_path
        return _build_winrate_args_from_config(winrate_path, parser=parser)

    if embedded_winrate:
        args.winrate = Path(args.config).expanduser()
        return _build_winrate_args_from_config(Path(args.config).expanduser(), parser=parser)
    return None


def _run_process_post_steps(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    options: ProcessOptions,
    winrate_args: argparse.Namespace | None,
) -> int:
    if not args.winrate:
        return 0
    if options.dry_run:
        logger.info("Skipping winrate post-step for dry-run process.")
        return 0

    if winrate_args is None:
        winrate_args = _build_winrate_args_from_config(Path(args.winrate), parser=parser)
    winrate_args.processed_dir = options.output_dir
    if not getattr(winrate_args, "_output_dir_explicit", False):
        winrate_args.output_dir = _default_winrate_output_dir(options.output_dir)
    winrate_args.hf_repo = None
    winrate_args.hf_processed_pull = False

    winrate_cfg = WinrateConfig(
        missing_policy=winrate_args.missing_policy,
        epsilon=winrate_args.epsilon,
        min_common=winrate_args.min_common,
        weight_policy=winrate_args.weight_policy,
        weight_cap=winrate_args.weight_cap,
        dataset_coverage=winrate_args.dataset_coverage,
        include_models=tuple(winrate_args.include_model or ()),
        exclude_models=tuple(winrate_args.exclude_model or ()),
        exclude_datasets=tuple(winrate_args.exclude_dataset or ()),
        partial_datasets=winrate_args.partial_datasets,
    )
    try:
        winrate_result = run_winrate(
            processed_dir=options.output_dir,
            output_dir=winrate_args.output_dir,
            output_path=winrate_args.output,
            output_name=winrate_args.output_name,
            config=winrate_cfg,
            processed_at=winrate_args.processed_at,
            hf_config=None,
            hf_processed_pull=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Win rate computation failed: %s", exc)
        return 1

    logger.info("Computed win rates for %d dataset(s): %s", len(winrate_result.datasets), winrate_result.output_path)
    print_winrate_summary_markdown(winrate_result.result)

    if options.hf_config and options.hf_config.repo_id:
        _upload_winrate_outputs(
            output_dir=winrate_args.output_dir,
            output_paths=winrate_result.output_paths,
            repo_id=options.hf_config.repo_id,
            token=options.hf_config.token,
            branch=options.hf_config.branch,
            private=bool(options.hf_config.private),
            winrate_dir=winrate_args.hf_winrate_dir,
            assume_yes=bool(args.yes),
        )
    return 0


def _parse_config_numeric(
    value: Any,
    *,
    parser: argparse.ArgumentParser,
    mode: Literal["process", "winrate"],
    field: str,
    cast_type: type[int] | type[float],
) -> int | float:
    if isinstance(value, bool):
        parser.error(f"Invalid {mode} config value for '{field}': expected {cast_type.__name__}, got {value!r}.")
    try:
        return cast_type(value)
    except (TypeError, ValueError):
        parser.error(f"Invalid {mode} config value for '{field}': expected {cast_type.__name__}, got {value!r}.")


def _parse_config_list(value: Any) -> list[str] | None:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    if isinstance(value, Sequence):
        return [str(item).strip() for item in value if str(item).strip()]
    return None


def _is_unset(args: argparse.Namespace, attr: str) -> bool:
    return hasattr(args, attr) and getattr(args, attr) is None


def _set_if_unset(args: argparse.Namespace, attr: str, value: Any) -> None:
    if hasattr(args, attr) and getattr(args, attr) is None:
        setattr(args, attr, value)


def _resolve_config_string_value(key: str, value: Any) -> str:
    resolved = str(value)
    if key != "hf_token":
        return resolved

    trimmed = resolved.strip()
    env_var: str | None = None
    if trimmed.startswith("${") and trimmed.endswith("}") and len(trimmed) > 3:
        env_var = trimmed[2:-1].strip()
    elif trimmed.startswith("$") and len(trimmed) > 1:
        env_var = trimmed[1:].strip()

    if not env_var:
        return resolved

    env_value = os.getenv(env_var)
    if env_value is None:
        raise ValueError(f"Config field 'hf.token' references unset environment variable '{env_var}'.")
    return env_value


def _load_config_payload(path: Path, *, mode: Literal["process", "winrate"]) -> dict[str, Any]:
    label = "Process config" if mode == "process" else "Winrate config"
    raw_payload = dict(load_mapping_file(path, label=label))
    if mode == "process":
        _reject_removed_process_config_keys(raw_payload)
    return _expand_embedded_pipeline_config(raw_payload, mode=mode)


def _reject_removed_process_config_keys(payload: Mapping[str, Any]) -> None:
    if "max_run_missing_pct" in payload:
        raise ValueError("Process config field 'max_run_missing_pct' was removed; use 'max_results_missing_pct'.")
    process_section = payload.get("process")
    if isinstance(process_section, Mapping) and "max_run_missing_pct" in process_section:
        raise ValueError(
            "Process config field 'process.max_run_missing_pct' was removed; use 'process.max_results_missing_pct'."
        )


def _expand_embedded_pipeline_config(payload: dict[str, Any], *, mode: Literal["process", "winrate"]) -> dict[str, Any]:
    expanded = dict(payload)
    process_section = payload.get("process")
    if isinstance(process_section, Mapping):
        _merge_process_section(expanded, process_section, mode=mode)

    process_output_dir = _resolve_processed_dir_from_payload(expanded, mode=mode)

    winrate_section = payload.get("winrate")
    if isinstance(winrate_section, Mapping):
        if mode == "process":
            expanded.pop("winrate", None)
        if mode == "winrate":
            _merge_winrate_section(expanded, winrate_section, process_output_dir=process_output_dir)
    elif isinstance(winrate_section, bool) and mode == "process":
        expanded.pop("winrate", None)

    if mode == "winrate" and "processed_dir" not in expanded and process_output_dir is not None:
        expanded["processed_dir"] = process_output_dir

    return expanded


def _merge_process_section(
    expanded: dict[str, Any],
    process_section: Mapping[str, Any],
    *,
    mode: Literal["process", "winrate"],
) -> None:
    resolved = None
    if "dir" in process_section:
        resolved = _resolve_process_dir_value(process_section["dir"], runs_dir=expanded.get("runs_dir"))
        if mode == "process" and "output_dir" not in expanded and resolved is not None:
            expanded["output_dir"] = resolved
        if mode == "winrate" and "processed_dir" not in expanded and resolved is not None:
            expanded["processed_dir"] = resolved
    if mode == "winrate" and "processed_dir" not in expanded and "output_dir" in process_section:
        expanded["processed_dir"] = process_section["output_dir"]
    key_map = {"runs_dir": "runs_dir"}
    if mode == "process":
        key_map.update(
            {
                "output_dir": "output_dir",
                "env_config_root": "env_config_root",
                "processed_at": "processed_at",
                "status": "status",
                "exclude_datasets": "exclude_datasets",
                "exclude_models": "exclude_models",
                "replace_models": "replace_models",
                "replace_envs": "replace_envs",
                "dry_run": "dry_run",
                "clean": "clean",
                "yes": "yes",
                "max_workers": "max_workers",
                "max_results_missing_pct": "max_results_missing_pct",
            }
        )
    for key, target in key_map.items():
        if key in process_section and target not in expanded:
            expanded[target] = process_section[key]


def _merge_winrate_section(
    expanded: dict[str, Any],
    winrate_section: Mapping[str, Any],
    *,
    process_output_dir: Path | None,
) -> None:
    if "dir" in winrate_section and "output_dir" not in expanded:
        resolved = _resolve_winrate_dir_value(winrate_section["dir"], process_output_dir=process_output_dir)
        if resolved is not None:
            expanded["output_dir"] = resolved
    key_map = {
        "processed_dir": "processed_dir",
        "output_dir": "output_dir",
        "output_name": "output_name",
        "processed_at": "processed_at",
        "missing_policy": "missing_policy",
        "epsilon": "epsilon",
        "min_common": "min_common",
        "weight_policy": "weight_policy",
        "weight_cap": "weight_cap",
        "dataset_coverage": "dataset_coverage",
        "include_model": "include_models",
        "include_models": "include_models",
        "exclude_model": "exclude_models",
        "exclude_models": "exclude_models",
        "exclude_dataset": "exclude_datasets",
        "exclude_datasets": "exclude_datasets",
        "partial_datasets": "partial_datasets",
        "hf_processed_pull": "hf_processed_pull",
        "hf_winrate_dir": "hf_winrate_dir",
    }
    for key, target in key_map.items():
        if key in winrate_section and target not in expanded:
            expanded[target] = winrate_section[key]


def _resolve_processed_dir_from_payload(
    payload: Mapping[str, Any], *, mode: Literal["process", "winrate"]
) -> Path | None:
    if "processed_dir" in payload and payload["processed_dir"] is not None:
        return Path(str(payload["processed_dir"]))
    if mode == "process" and "output_dir" in payload and payload["output_dir"] is not None:
        return Path(str(payload["output_dir"]))
    process_section = payload.get("process")
    if isinstance(process_section, Mapping) and "dir" in process_section:
        return _resolve_process_dir_value(process_section["dir"], runs_dir=payload.get("runs_dir"))
    return None


def _resolve_process_dir_value(value: Any, *, runs_dir: Any | None) -> Path | None:
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    runs_base = Path(str(runs_dir)).parent if runs_dir is not None else DEFAULT_RUNS_RAW_DIR.parent
    return runs_base / candidate


def _resolve_winrate_dir_value(value: Any, *, process_output_dir: Path | None) -> Path | None:
    raw = str(value).strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    base = process_output_dir if process_output_dir is not None else DEFAULT_PROCESSED_DIR
    return base / candidate


def _config_has_embedded_winrate(path: Path) -> bool:
    payload = dict(load_mapping_file(path, label="Process config"))
    winrate_payload = payload.get("winrate")
    if isinstance(winrate_payload, Mapping):
        return bool(winrate_payload.get("enabled", True))
    return bool(winrate_payload) if isinstance(winrate_payload, bool) else False


def _normalize_mode_payload(payload: dict[str, Any], *, mode: Literal["process", "winrate"]) -> None:
    if mode == "winrate":
        if "hf_processed_repo" in payload and "hf_repo" not in payload:
            payload["hf_repo"] = payload["hf_processed_repo"]
        if "hf_winrate_repo" in payload:
            raise ValueError("Winrate config field 'hf_winrate_repo' was removed; use 'hf.repo' and 'hf.winrate_dir'.")

    hf_payload = payload.get("hf")
    if isinstance(hf_payload, Mapping):
        for key, value in hf_payload.items():
            if mode == "winrate":
                if key == "repo":
                    payload.setdefault("hf_repo", value)
                    continue
                if key == "branch":
                    payload.setdefault("hf_branch", value)
                    continue
                if key == "token":
                    payload.setdefault("hf_token", value)
                    continue
                if key == "private":
                    payload.setdefault("hf_private", value)
                    continue
                if key == "winrate_repo":
                    raise ValueError(
                        "Winrate config field 'hf.winrate_repo' was removed; use 'hf.repo' and 'hf.winrate_dir'."
                    )
            payload.setdefault(f"hf_{key}", value)

    if "exclude_datasets" not in payload and "exclude_dataset" in payload:
        payload["exclude_datasets"] = payload["exclude_dataset"]
    if "exclude_models" not in payload and "exclude_model" in payload:
        payload["exclude_models"] = payload["exclude_model"]


def _load_and_apply_config(
    args: argparse.Namespace,
    path: Path,
    *,
    mode: Literal["process", "winrate"],
    parser: argparse.ArgumentParser,
) -> None:
    try:
        payload = _load_config_payload(path, mode=mode)
        _normalize_mode_payload(payload, mode=mode)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    path_fields = {
        "process": {
            "runs_dir": "runs_dir",
            "output_dir": "output_dir",
            "env_config_root": "env_config_root",
            "winrate": "winrate",
        },
        "winrate": {"processed_dir": "processed_dir", "output_dir": "output_dir", "output": "output"},
    }[mode]
    string_fields = {
        "process": {
            "processed_at": "processed_at",
            "hf_repo": "hf_repo",
            "hf_branch": "hf_branch",
            "hf_token": "hf_token",
            "hf_pull_policy": "hf_pull_policy",
        },
        "winrate": {
            "output_name": "output_name",
            "processed_at": "processed_at",
            "missing_policy": "missing_policy",
            "weight_policy": "weight_policy",
            "partial_datasets": "partial_datasets",
            "dataset_coverage": "dataset_coverage",
            "hf_repo": "hf_repo",
            "hf_winrate_dir": "hf_winrate_dir",
            "hf_branch": "hf_branch",
            "hf_token": "hf_token",
        },
    }[mode]
    boolean_fields = {
        "process": {
            "dry_run": "dry_run",
            "clean": "clean",
            "yes": "yes",
            "hf_private": "hf_private",
        },
        "winrate": {"hf_processed_pull": "hf_processed_pull", "hf_private": "hf_private"},
    }[mode]
    int_fields = {
        "process": {
            "max_workers": "max_workers",
            "hf_retries": "hf_retries",
            "hf_max_files_per_commit": "hf_max_files_per_commit",
        },
        "winrate": {"min_common": "min_common", "weight_cap": "weight_cap"},
    }[mode]
    float_fields = {
        "process": {
            "hf_request_timeout": "hf_request_timeout",
            "max_results_missing_pct": "max_results_missing_pct",
        },
        "winrate": {"epsilon": "epsilon"},
    }[mode]
    repeatable_fields = {
        "process": {
            "status": "status",
            "exclude_datasets": "exclude_dataset",
            "exclude_models": "exclude_model",
            "replace_models": "replace_model",
            "replace_envs": "replace_env",
        },
        "winrate": {
            "include_models": "include_model",
            "exclude_models": "exclude_model",
            "exclude_datasets": "exclude_dataset",
        },
    }[mode]

    for key, attr in path_fields.items():
        if key in payload and _is_unset(args, attr):
            _set_if_unset(args, attr, Path(str(payload[key])))
    for key, attr in string_fields.items():
        if key in payload and _is_unset(args, attr):
            try:
                resolved = _resolve_config_string_value(key, payload[key])
            except ValueError as exc:
                parser.error(str(exc))
            _set_if_unset(args, attr, resolved)
    for key, attr in boolean_fields.items():
        if key in payload and _is_unset(args, attr):
            _set_if_unset(args, attr, bool(payload[key]))
    for key, attr in int_fields.items():
        if key in payload and _is_unset(args, attr):
            parsed = _parse_config_numeric(payload[key], parser=parser, mode=mode, field=key, cast_type=int)
            _set_if_unset(args, attr, parsed)
    for key, attr in float_fields.items():
        if key in payload and _is_unset(args, attr):
            parsed = _parse_config_numeric(payload[key], parser=parser, mode=mode, field=key, cast_type=float)
            _set_if_unset(args, attr, parsed)
    for key, attr in repeatable_fields.items():
        if key in payload and _is_unset(args, attr):
            parsed = _parse_config_list(payload[key])
            if parsed is not None:
                _set_if_unset(args, attr, parsed)


def _build_winrate_args_from_config(path: Path, *, parser: argparse.ArgumentParser) -> argparse.Namespace:
    args = argparse.Namespace(
        processed_dir=None,
        output_dir=None,
        output=None,
        output_name=None,
        processed_at=None,
        missing_policy=None,
        epsilon=None,
        min_common=None,
        weight_policy=None,
        weight_cap=None,
        dataset_coverage=None,
        include_model=None,
        exclude_model=None,
        exclude_dataset=None,
        partial_datasets=None,
        hf_repo=None,
        hf_processed_pull=None,
        hf_winrate_dir=None,
        hf_branch=None,
        hf_token=None,
        hf_private=None,
    )
    _load_and_apply_config(args, path, mode="winrate", parser=parser)
    args._output_dir_explicit = args.output_dir is not None
    _finalize_config_args(args, mode="winrate")
    return args


def _finalize_config_args(args: argparse.Namespace, *, mode: Literal["process", "winrate"]) -> None:
    """Fill any unset process/winrate args with defaults after config + CLI merge."""
    defaults = {
        "process": {
            "runs_dir": DEFAULT_RUNS_RAW_DIR,
            "output_dir": DEFAULT_PROCESSED_DIR,
            "env_config_root": DEFAULT_ENV_CONFIG_ROOT,
            "max_workers": 4,
            "hf_private": False,
            "dry_run": False,
            "clean": False,
            "yes": False,
            "max_results_missing_pct": 2.5,
            "exclude_dataset": [],
            "exclude_model": [],
            "replace_model": [],
            "replace_env": [],
        },
        "winrate": {
            "processed_dir": DEFAULT_PROCESSED_DIR,
            "missing_policy": "neg-inf",
            "epsilon": 1e-9,
            "min_common": 0,
            "weight_policy": "ln",
            "weight_cap": 0,
            "dataset_coverage": "all-models",
            "include_model": [],
            "exclude_model": [],
            "exclude_dataset": [],
            "partial_datasets": "strict",
            "hf_processed_pull": False,
            "hf_winrate_dir": "winrate",
            "hf_private": False,
            "yes": False,
        },
    }[mode]
    for attr, default in defaults.items():
        if getattr(args, attr, None) is None:
            setattr(args, attr, default)
    if mode == "winrate" and getattr(args, "output_dir", None) is None:
        args.output_dir = _default_winrate_output_dir(Path(args.processed_dir))

    if hasattr(args, "exclude_dataset"):
        args.exclude_dataset = _parse_repeatable_csv(args.exclude_dataset)
    if mode == "process" and hasattr(args, "exclude_model"):
        args.exclude_model = _parse_repeatable_csv(args.exclude_model)
    if mode == "process" and hasattr(args, "replace_model"):
        args.replace_model = _parse_repeatable_csv(args.replace_model)
    if mode == "process" and hasattr(args, "replace_env"):
        args.replace_env = _parse_repeatable_csv(args.replace_env)


def _default_winrate_output_dir(processed_dir: Path) -> Path:
    return Path(processed_dir) / "winrate"


def _upload_winrate_outputs(
    *,
    output_dir: Path,
    output_paths: Sequence[Path],
    repo_id: str,
    token: str | None,
    branch: str | None,
    private: bool,
    winrate_dir: str | None,
    assume_yes: bool = False,
) -> None:
    if not output_paths:
        return
    raw_dir = "winrate" if winrate_dir is None else str(winrate_dir).strip()
    if not raw_dir:
        raw_dir = "winrate"
    if resolve_under(Path("."), raw_dir) is None:
        logger.error("Invalid winrate_dir '%s'; skipping upload.", winrate_dir)
        return
    output_dir = Path(output_dir)
    files: list[str] = []
    for path in output_paths:
        try:
            rel_path = path.relative_to(output_dir).as_posix()
        except ValueError:
            if len(output_paths) == 1:
                output_dir = path.parent
                files = [path.name]
                break
            logger.warning("Winrate output %s is outside output_dir %s; skipping upload.", path, output_dir)
            return
        files.append(rel_path)
    message = f"Update {len(files)} winrate file(s) from medarc-eval winrate"
    sync_files_to_hub(
        repo_id=repo_id,
        output_dir=output_dir,
        files=files,
        token=token,
        private=private,
        message=message,
        branch=branch,
        path_in_repo_prefix=raw_dir,
        is_tty=sys.stdin.isatty(),
        assume_yes=assume_yes,
        prompt_func=input,
    )


def _run_winrate_mode(argv: Sequence[str]) -> int:
    parser = build_winrate_parser()
    args = parser.parse_args(argv)

    if args.config:
        _load_and_apply_config(args, args.config, mode="winrate", parser=parser)
    args._output_dir_explicit = args.output_dir is not None
    _finalize_config_args(args, mode="winrate")

    hf_config = HFSyncConfig.from_cli(
        repo=args.hf_repo,
        branch=args.hf_branch,
        token=args.hf_token,
        private=bool(args.hf_private),
        dry_run=False,
    )

    if args.list_models:
        source_dir, datasets, source_desc = _resolve_source(
            args.processed_dir,
            hf_config=hf_config if args.hf_repo else None,
            hf_processed_pull=bool(args.hf_processed_pull),
        )
        if args.exclude_dataset:
            try:
                datasets = _filter_winrate_datasets(datasets, args.exclude_dataset)
            except ValueError as exc:
                parser.error(str(exc))
        if not datasets:
            logger.error("No datasets found from %s.", source_desc)
            return 1
        models = list_models(datasets)
        if models:
            print("\n".join(models))
        else:
            logger.info("No models found in datasets from %s.", source_desc)
        return 0

    cfg = WinrateConfig(
        missing_policy=args.missing_policy,
        epsilon=args.epsilon,
        min_common=args.min_common,
        weight_policy=args.weight_policy,
        weight_cap=args.weight_cap,
        dataset_coverage=args.dataset_coverage,
        include_models=tuple(args.include_model or ()),
        exclude_models=tuple(args.exclude_model or ()),
        exclude_datasets=tuple(args.exclude_dataset or ()),
        partial_datasets=args.partial_datasets,
    )

    try:
        winrate_result = run_winrate(
            processed_dir=args.processed_dir,
            output_dir=args.output_dir,
            output_path=args.output,
            output_name=args.output_name,
            config=cfg,
            processed_at=args.processed_at,
            hf_config=hf_config,
            hf_processed_pull=bool(args.hf_processed_pull),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Win rate computation failed: %s", exc)
        return 1

    logger.info("Computed win rates for %d dataset(s): %s", len(winrate_result.datasets), winrate_result.output_path)
    print_winrate_summary_markdown(winrate_result.result)
    if args.hf_repo:
        _upload_winrate_outputs(
            output_dir=args.output_dir,
            output_paths=winrate_result.output_paths,
            repo_id=args.hf_repo,
            token=args.hf_token,
            branch=args.hf_branch,
            private=bool(args.hf_private),
            winrate_dir=args.hf_winrate_dir,
            assume_yes=bool(args.yes),
        )
    return 0


def _validate_toml_selection_args(args: argparse.Namespace, *, parser: argparse.ArgumentParser) -> None:
    for attr, flag in (("eval_index", "--eval-index"), ("start_at", "--start-at"), ("stop_after", "--stop-after")):
        value = getattr(args, attr, None)
        if value is not None and value < 1:
            parser.error(f"{flag} must be a 1-based index.")
    if args.eval_index is not None and (args.start_at is not None or args.stop_after is not None):
        parser.error("--eval-index cannot be combined with --start-at or --stop-after.")
    if args.start_at is not None and args.stop_after is not None and args.stop_after < args.start_at:
        parser.error("--stop-after must be greater than or equal to --start-at.")


def _run_toml_bench(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser()
    raw_configs = _prepare_toml_raw_configs(load_toml_eval_configs(config_path), args)
    overrides = EvalConfigOverrides(
        model=args.model,
        provider=args.provider,
        api_base_url=args.api_base_url,
        api_key_var=args.api_key_var,
        endpoints_path=args.endpoints_path if getattr(args, "endpoints_path_explicit", False) else None,
        max_concurrent=args.max_concurrent,
        env_args=getattr(args, "cli_env_args", None),
        sampling_args=getattr(args, "cli_sampling_args", None),
    )
    eval_configs = [build_eval_config(raw, overrides=overrides) for raw in raw_configs]
    plan_inputs = [_eval_config_identity_payload(config) for config in eval_configs]
    output_root = _resolve_toml_output_root(eval_configs, args)
    path_plans = plan_eval_paths(plan_inputs, output_root=output_root)
    eval_configs, path_plans = _select_toml_plan(eval_configs, path_plans, args)

    _print_toml_bench_plan(eval_configs, path_plans, dry_run=bool(args.dry_run))
    if args.dry_run:
        return 0
    return _execute_toml_plan(eval_configs, path_plans, args)


def _prepare_toml_raw_configs(raw_configs: Sequence[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for raw in raw_configs:
        item = dict(raw)
        item.setdefault("save_results", True)
        if args.max_concurrent is None and "max_concurrent" not in item:
            item["max_concurrent"] = 1
        if args.timeout is not None:
            item["timeout"] = args.timeout
        if args.rollout_max_retries is not None:
            item["max_retries"] = args.rollout_max_retries
        prepared.append(item)
    return prepared


def _resolve_toml_output_root(eval_configs: Sequence[Any], args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir).expanduser()

    configured_roots = {str(config.output_dir) for config in eval_configs if config.output_dir}
    if len(configured_roots) > 1:
        raise ValueError(
            "TOML bench deterministic output supports one output_dir per run; use a single global output_dir."
        )
    if configured_roots:
        return Path(configured_roots.pop()).expanduser()
    return DEFAULT_EVALS_DIR


def _select_toml_plan(
    eval_configs: Sequence[Any],
    path_plans: Sequence[EvalPathPlan],
    args: argparse.Namespace,
) -> tuple[list[Any], list[EvalPathPlan]]:
    indexed = list(zip(eval_configs, path_plans))
    if args.eval_index is not None:
        start = args.eval_index - 1
        indexed = indexed[start : start + 1]
    else:
        if args.start_at is not None:
            indexed = indexed[args.start_at - 1 :]
        if args.stop_after is not None:
            indexed = indexed[: args.stop_after - (args.start_at or 1) + 1]
    if not indexed:
        raise ValueError("No TOML evals matched the requested selection.")
    selected_configs, selected_paths = zip(*indexed)
    return list(selected_configs), list(selected_paths)


def _execute_toml_plan(
    eval_configs: Sequence[Any], path_plans: Sequence[EvalPathPlan], args: argparse.Namespace
) -> int:
    failures = 0
    for index, (config, path_plan) in enumerate(zip(eval_configs, path_plans), start=1):
        metadata_fields = metadata_identity_fields(_eval_config_identity_payload(config), path_plan.identity)
        results_path = path_plan.results_path
        try:
            _prepare_toml_results_dir(results_path, metadata_fields, config, force=bool(args.force))
            run_config = config.model_copy(update={"resume_path": results_path, "save_results": True})
            logger.info("Running TOML eval %d/%d: %s on %s", index, len(eval_configs), config.env_id, config.model)
            asyncio.run(_run_one_toml_eval(run_config, results_path, metadata_fields))
            _merge_metadata_fields(results_path, metadata_fields)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            logger.exception("TOML eval %d failed: %s", index, exc)
            if not args.continue_on_error:
                return 1
        if args.sleep and index < len(eval_configs):
            import time

            time.sleep(float(args.sleep))
    return 1 if failures else 0


async def _run_one_toml_eval(config: Any, results_path: Path, metadata_fields: Mapping[str, Any]) -> Any:
    import verifiers.envs.environment as environment_module

    def add_medarc_metadata(_all_outputs: Any, _new_outputs: Any, metadata: MutableMapping[str, Any]) -> None:
        metadata.update(metadata_fields)

    original_save_metadata = environment_module.save_metadata

    def save_metadata_with_medarc_fields(metadata: MutableMapping[str, Any], result_path: Path) -> Any:
        if Path(result_path) == results_path:
            metadata.update(metadata_fields)
        return original_save_metadata(metadata, result_path)

    environment_module.save_metadata = save_metadata_with_medarc_fields
    try:
        return await run_evaluation(config, on_progress=add_medarc_metadata)
    finally:
        environment_module.save_metadata = original_save_metadata


def _prepare_toml_results_dir(
    results_path: Path,
    metadata_fields: Mapping[str, Any],
    config: Any,
    *,
    force: bool,
) -> None:
    if results_path.exists() and force:
        _archive_existing_path(results_path)

    metadata_path = results_path / "metadata.json"
    results_file = results_path / "results.jsonl"
    has_existing_state = metadata_path.exists() or results_file.exists()
    if has_existing_state:
        _validate_toml_resume_metadata(results_path, metadata_fields)

    results_path.mkdir(parents=True, exist_ok=True)
    results_file.touch(exist_ok=True)
    if has_existing_state:
        _merge_metadata_fields(results_path, metadata_fields)
        return

    metadata = _initial_toml_metadata(config)
    metadata.update(metadata_fields)
    _write_json(metadata_path, metadata)


def _validate_toml_resume_metadata(results_path: Path, metadata_fields: Mapping[str, Any]) -> None:
    metadata_path = results_path / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"Cannot resume {results_path}: metadata.json is missing.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot resume {results_path}: metadata.json is invalid JSON.") from exc
    expected = metadata_fields.get("medarc_config_fingerprint")
    current = metadata.get("medarc_config_fingerprint") if isinstance(metadata, Mapping) else None
    if current != expected:
        raise ValueError(
            f"Cannot resume {results_path}: MedARC config fingerprint mismatch "
            f"(saved={current!r}, current={expected!r}). Use --force to archive and rerun."
        )


def _initial_toml_metadata(config: Any) -> dict[str, Any]:
    return {
        "env_id": config.env_id,
        "env_args": dict(config.env_args or {}),
        "model": config.model,
        "base_url": config.client_config.api_base_url,
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "sampling_args": dict(config.sampling_args or {}),
        "avg_reward": None,
        "avg_metrics": {},
        "avg_error": None,
        "state_columns": list(config.state_columns or []),
    }


def _merge_metadata_fields(results_path: Path, metadata_fields: Mapping[str, Any]) -> None:
    metadata_path = results_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        metadata = {}
    metadata.update(metadata_fields)
    _write_json(metadata_path, metadata)


def _archive_existing_path(path: Path) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.name}__old_{timestamp}")
    suffix = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}__old_{timestamp}_{suffix}")
        suffix += 1
    shutil.move(str(path), str(candidate))
    return candidate


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, default=make_serializable, sort_keys=True), encoding="utf-8")


def _eval_config_identity_payload(config: Any) -> dict[str, Any]:
    return {
        "env_args": dict(config.env_args or {}),
        "env_id": config.env_id,
        "model": config.model,
        "num_examples": config.num_examples,
        "rollouts_per_example": config.rollouts_per_example,
        "sampling_args": dict(config.sampling_args or {}),
    }


def _print_toml_bench_plan(eval_configs: Sequence[Any], path_plans: Sequence[EvalPathPlan], *, dry_run: bool) -> None:
    console = Console(width=240)
    action = "dry-run" if dry_run else "run"
    table = Table(
        title="TOML Bench Dry Run" if dry_run else "TOML Bench Plan",
        caption=f"{len(eval_configs)} eval(s) to {action}",
        expand=True,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="magenta", overflow="fold")
    table.add_column("Environment", style="green", overflow="fold")
    table.add_column("Variant", style="cyan", overflow="fold")
    table.add_column("Examples", justify="right")
    table.add_column("Rollouts", justify="right")
    table.add_column("Max Concurrency", justify="right")
    table.add_column("Output Path", overflow="fold")

    for index, (config, path_plan) in enumerate(zip(eval_configs, path_plans), start=1):
        table.add_row(
            str(index),
            config.model,
            config.env_id,
            path_plan.identity.variant_id or "-",
            str(config.num_examples),
            str(config.rollouts_per_example),
            str(config.max_concurrent),
            str(path_plan.results_path),
        )

    console.print(table)


def _parse_repeatable_csv(values: Sequence[str] | None) -> list[str]:
    parsed: list[str] = []
    for chunk in values or ():
        if chunk is None:
            continue
        for item in str(chunk).split(","):
            value = item.strip()
            if value:
                parsed.append(value)
    return parsed


def _option_was_provided(argv: Sequence[str], long_flag: str) -> bool:
    for token in argv:
        if token == long_flag or token.startswith(f"{long_flag}="):
            return True
    return False


def _filter_winrate_datasets(
    datasets: Sequence[tuple[str, Sequence[Path]]],
    exclude_datasets: Sequence[str],
) -> list[tuple[str, Sequence[Path]]]:
    from medarc_verifiers.cli.process.rollout import derive_base_env_id

    exclude_set = normalize_dataset_ids(exclude_datasets, label="winrate exclude dataset")
    if not exclude_set:
        return list(datasets)
    filtered: list[tuple[str, Sequence[Path]]] = []
    for name, paths in datasets:
        dataset = str(name).strip()
        base, _ = derive_base_env_id(dataset)
        if dataset_is_excluded(dataset, exclude_set, base_dataset_id=base):
            continue
        filtered.append((name, paths))
    return filtered


def _print_general_help() -> None:
    message = dedent(
        f"""\
        Usage:
          {COMMAND} <ENV> [options]                 # Single run (ENV must be first; use ENV --help for details)
          {COMMAND} {BENCH_COMMAND} --config CONFIG.toml ...  # Sequential TOML bench
          {COMMAND} {PROCESS_COMMAND} [options]               # Export raw runs to parquet (see: {COMMAND} {PROCESS_COMMAND} --help)
          {COMMAND} {WINRATE_COMMAND} [options]               # Compute win rates from processed parquet outputs

        First argument must be the environment slug for single runs. Use '{COMMAND} {BENCH_COMMAND} --help' for TOML bench options."""
    )
    print(message)


def _log_process_result(result: ProcessResult) -> None:
    logger.info(
        "Processed %d record(s) into %d environment dataset(s) (%d rows).",
        result.records_processed,
        len(result.env_summaries),
        result.rows_processed,
    )
    for summary in result.env_summaries:
        path_display = summary.output_path if not summary.dry_run else f"(planned) {summary.output_path}"
        logger.info(
            "  %s -> %d rows @ %s (%s)",
            summary.env_id or summary.base_env_id,
            summary.row_count,
            path_display,
            summary.action,
        )
        if summary.job_run_ids_added or summary.job_run_ids_replaced:
            added = ", ".join(summary.job_run_ids_added)
            replaced = ", ".join(summary.job_run_ids_replaced)
            if added:
                logger.info("    added: %s", added)
            if replaced:
                logger.info("    replaced: %s", replaced)
    if result.hf_summary:
        logger.info(
            "HF sync: repo=%s strategy=%s rows=%d files=%d",
            result.hf_summary.repo_id,
            result.hf_summary.strategy,
            result.hf_summary.total_rows,
            result.hf_summary.total_files,
        )


def _load_env_export_map(root: Path | None) -> dict[str, EnvironmentExportConfig]:
    if root is None:
        return {}
    root = Path(root).expanduser()
    if not root.exists():
        logger.debug("Env config root %s does not exist; skipping export overrides.", root)
        return {}

    if root.is_file():
        files = [root]
    else:
        files = sorted(p for pattern in ("*.yaml", "*.yml") for p in root.rglob(pattern) if p.is_file())

    export_map: dict[str, EnvironmentExportConfig] = {}
    for path in files:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to read env config %s: %s", path, exc)
            continue

        if isinstance(payload, list):
            entries = [entry for entry in payload if isinstance(entry, Mapping)]
        elif isinstance(payload, Mapping):
            entries = [payload]
        else:
            continue

        for entry in entries:
            try:
                env_cfg = EnvironmentConfigSchema(**dict(entry))
            except ValidationError:
                continue
            if env_cfg.export is None:
                continue
            keys = {env_cfg.id, env_cfg.matrix_base_id}
            if env_cfg.module and env_cfg.env_args:
                keys.add(f"{env_cfg.module}::{generate_variant_id({'env_args': env_cfg.env_args})}")
            for key in filter(None, keys):
                export_map[key] = env_cfg.export

    return export_map


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
