"""Unified MedARC evaluation CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any, Mapping, Sequence

from rich.console import Console
from rich.table import Table

import yaml
from pydantic import ValidationError

from medarc_verifiers.cli._config_loader import ConfigFormatError, load_run_config
from medarc_verifiers.cli._job_builder import ResolvedJob, build_jobs
from medarc_verifiers.cli._job_executor import ExecutorSettings, JobExecutionResult, execute_jobs
from medarc_verifiers.cli._manifest import ManifestJobEntry, RunManifest, compute_snapshot_checksum
from medarc_verifiers.cli._manifest_planner import ManifestPlanner
from medarc_verifiers.cli._single_run import run_single_mode
from medarc_verifiers.cli.process import ProcessOptions, ProcessResult, run_process
from medarc_verifiers.cli.hf import HFSyncConfig, sync_files_to_hub
from medarc_verifiers.cli.winrate import WinrateConfig
from medarc_verifiers.cli.winrate import (
    _resolve_source,
    list_models,
    print_winrate_summary_markdown,
    run_winrate,
)
from medarc_verifiers.cli.utils.overrides import build_cli_override
from medarc_verifiers.cli._schemas import EnvironmentConfigSchema, EnvironmentExportConfig
from medarc_verifiers.cli._constants import (
    BENCH_COMMAND,
    COMMAND,
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_VAR,
    DEFAULT_ENV_CONFIG_ROOT,
    DEFAULT_ENV_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RUNS_RAW_DIR,
    DEFAULT_WINRATE_DIR,
    PROCESS_COMMAND,
    WINRATE_COMMAND,
)

logger = logging.getLogger(__name__)
HELP_FLAGS = {"-h", "--help"}


def build_batch_parser() -> argparse.ArgumentParser:
    """Construct the unified CLI parser."""
    parser = argparse.ArgumentParser(
        prog=COMMAND,
        description="Run MedARC evaluations using unified configuration files.",
    )
    parser.add_argument("-c", "--config", required=True, type=Path, help="Path to a run configuration YAML file.")
    parser.add_argument("--run-id", help="Override the generated run identifier.")
    parser.add_argument("--name", help="Override the human-friendly run name (defaults to the config name).")
    parser.add_argument(
        "--restart",
        help="Seed jobs from a previous run identifier (reuse completed jobs when configs match).",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically resume the newest matching run (default: enabled). "
            "Pass --no-auto-resume to force a fresh run."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Re-run every job regardless of manifest state.")
    parser.add_argument(
        "--forced",
        action="append",
        help="Re-run jobs for the specified environment(s); repeat or comma-separate values.",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the output directory from the configuration.")
    parser.add_argument(
        "--env-dir",
        type=Path,
        default=DEFAULT_ENV_DIR,
        help="Directory containing environments (default: %(default)s).",
    )
    parser.add_argument(
        "--env-config-root",
        type=Path,
        default=DEFAULT_ENV_CONFIG_ROOT,
        help="Directory containing environment YAMLs for auto-discovery (default: %(default)s).",
    )
    parser.add_argument("--endpoints-path", type=Path, help="Override the default endpoints registry path.")
    parser.add_argument(
        "--default-api-key-var",
        default=DEFAULT_API_KEY_VAR,
        help=f"Default API key environment variable (default: {DEFAULT_API_KEY_VAR}).",
    )
    parser.add_argument(
        "--default-api-base-url",
        default=DEFAULT_API_BASE_URL,
        help=f"Default API base URL (default: {DEFAULT_API_BASE_URL}).",
    )
    parser.add_argument(
        "--job-id", action="append", help="Run only the specified job identifier (repeat to select multiple)."
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
        "--save-results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Persist evaluation outputs (default: enabled).",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upload results to the Hugging Face Hub.",
    )
    parser.add_argument("--hf-hub-dataset-name", help="Custom dataset name when uploading to the Hub.")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Override env max_concurrent for all jobs (CLI > model > env > defaults).",
    )
    parser.add_argument("--max-concurrent-generation", type=int, help="Override generation concurrency for all jobs.")
    parser.add_argument("--max-concurrent-scoring", type=int, help="Override scoring concurrency for all jobs.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override request timeout in seconds for all jobs (CLI > model > default).",
    )
    parser.add_argument(
        "--sleep",
        "--sleep-seconds",
        dest="sleep",
        type=float,
        default=0.0,
        help="Sleep this many seconds after each job (overridden by per-job sleep).",
    )
    parser.add_argument(
        "--enable-additional-retries",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable per-call model retry wrapper (default: disabled).",
    )
    parser.add_argument(
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include usage reporting in API requests (extra_body.usage.include). "
            "Default: auto-detect (enabled for Prime Inference, disabled otherwise)."
        ),
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
        "--clean",
        action="store_true",
        default=None,
        help="Delete processed outputs in --output-dir and rebuild from --runs-dir.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=None,
        help="Skip confirmation prompts (use with --clean).",
    )
    parser.add_argument("--processed-at", default=None, help="Override processed_at timestamp (ISO8601).")
    parser.add_argument("--dry-run", action="store_true", default=None, help="Plan processing without writing outputs.")
    parser.add_argument(
        "--process-incomplete",
        dest="process_incomplete",
        action="store_true",
        default=None,
        help="Include runs where run_manifest.json summary has completed < total.",
    )
    parser.add_argument(
        "--winrate",
        type=Path,
        default=None,
        help="Run winrate after processing using the provided winrate config file.",
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
        choices=("prompt", "pull", "clean"),
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
        help=f"Directory to store winrate outputs (default: {DEFAULT_WINRATE_DIR}).",
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
        "--partial-datasets",
        choices=("strict", "include"),
        default=None,
        help=(
            "Dataset selection policy when --include-model is set: "
            "strict drops datasets missing any included models, "
            "include keeps them with missing models treated as all-missing."
        ),
    )
    parser.add_argument("--hf-processed-repo", help="Hugging Face repo id for processed dataset download.")
    parser.add_argument(
        "--hf-processed-pull",
        action="store_true",
        default=None,
        help="Pull missing processed files from HF even when --processed-dir is non-empty.",
    )
    parser.add_argument("--hf-branch", help="Target HF branch or revision for processed download.")
    parser.add_argument("--hf-token", help="Auth token for HF operations.")
    parser.add_argument("--hf-winrate-repo", help="Hugging Face repo id for winrate artifact upload.")
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
    except ValueError as exc:
        parser.error(str(exc))

    if args.restart:
        args.auto_resume = False
    # Restarting is an explicit workflow; disable auto-resume selection when --restart is set.
    # The planner may restart in-place when --restart points to an existing run directory.

    try:
        return _execute_batch(args)
    except KeyboardInterrupt:
        logger.warning("Batch run interrupted by user.")
        return 1
    except ConfigFormatError as exc:
        parser.error(str(exc))
    except SystemExit:  # pragma: no cover - argparse already handled messaging
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unhandled error: %s", exc)
        return 1


def _run_process_mode(argv: Sequence[str]) -> int:
    parser = build_process_parser()
    args = parser.parse_args(argv)

    if args.config:
        _apply_process_config(args, args.config)
    _finalize_process_args(args)
    if args.winrate:
        winrate_path = Path(args.winrate).expanduser()
        if not winrate_path.exists():
            parser.error(f"Winrate config not found: {winrate_path}")
        args.winrate = winrate_path

    try:
        env_export_map = _load_env_export_map(args.env_config_root)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load environment export configs: %s", exc)
        env_export_map = {}

    hf_config = HFSyncConfig.from_cli(
        repo=args.hf_repo,
        branch=args.hf_branch,
        token=args.hf_token,
        private=args.hf_private,
        dry_run=args.dry_run,
    )

    processed_with_args = {
        "status": args.status or [],
        "dry_run": bool(args.dry_run),
        "clean": bool(args.clean),
        "only_complete_runs": not bool(args.process_incomplete),
        "hf_repo": args.hf_repo,
        "hf_pull_policy": args.hf_pull_policy,
        "max_workers": args.max_workers,
    }

    options = ProcessOptions(
        runs_dir=args.runs_dir,
        output_dir=args.output_dir,
        processed_at=args.processed_at,
        processed_with_args=processed_with_args,
        status_filter=args.status or (),
        only_complete_runs=not bool(args.process_incomplete),
        dry_run=bool(args.dry_run),
        clean=bool(args.clean),
        assume_yes=bool(args.yes),
        hf_config=hf_config,
        hf_pull_policy=args.hf_pull_policy,
        max_workers=args.max_workers,
    )

    try:
        result = run_process(options, env_export_map=env_export_map)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Process pipeline failed: %s", exc)
        return 1

    _log_process_result(result)

    if args.winrate:
        if options.dry_run:
            logger.info("Skipping winrate post-step for dry-run process.")
            return 0
        winrate_args = _build_winrate_args_from_config(Path(args.winrate))
        winrate_args.processed_dir = options.output_dir
        winrate_args.hf_processed_repo = None
        winrate_args.hf_processed_pull = False
        winrate_cfg = WinrateConfig(
            missing_policy=winrate_args.missing_policy,
            epsilon=winrate_args.epsilon,
            min_common=winrate_args.min_common,
            weight_policy=winrate_args.weight_policy,
            weight_cap=winrate_args.weight_cap,
            include_models=tuple(winrate_args.include_model or ()),
            exclude_models=tuple(winrate_args.exclude_model or ()),
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
        logger.info(
            "Computed win rates for %d dataset(s): %s", len(winrate_result.datasets), winrate_result.output_path
        )
        print_winrate_summary_markdown(winrate_result.result)
        if winrate_args.hf_winrate_repo:
            _upload_winrate_outputs(
                output_dir=winrate_args.output_dir,
                output_paths=winrate_result.output_paths,
                repo_id=winrate_args.hf_winrate_repo,
                token=winrate_args.hf_token,
                private=bool(winrate_args.hf_private),
            )

    return 0


def _load_process_config(path: Path) -> Mapping[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Process config not found: {path}")
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    elif suffix == ".json":
        payload = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported process config format: {path} (expected .yaml/.yml/.json)")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Process config must be a mapping at top level: {path}")
    return payload


def _apply_process_config(args: argparse.Namespace, path: Path) -> None:
    """Apply config file defaults to process args (CLI flags win)."""
    payload = dict(_load_process_config(path))

    hf_payload = payload.get("hf")
    if isinstance(hf_payload, Mapping):
        for key, value in hf_payload.items():
            payload.setdefault(f"hf_{key}", value)

    def _set_if_unset(attr: str, value: Any) -> None:
        if not hasattr(args, attr):
            return
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    # Paths / simple defaults
    if "runs_dir" in payload:
        _set_if_unset("runs_dir", Path(str(payload["runs_dir"])))
    if "output_dir" in payload:
        _set_if_unset("output_dir", Path(str(payload["output_dir"])))
    if "env_config_root" in payload:
        _set_if_unset("env_config_root", Path(str(payload["env_config_root"])))
    if "max_workers" in payload:
        try:
            _set_if_unset("max_workers", int(payload["max_workers"]))
        except Exception:
            pass
    if "winrate" in payload:
        _set_if_unset("winrate", Path(str(payload["winrate"])))
    if "processed_at" in payload:
        _set_if_unset("processed_at", str(payload["processed_at"]))
    if "dry_run" in payload:
        _set_if_unset("dry_run", bool(payload["dry_run"]))
    if "clean" in payload:
        _set_if_unset("clean", bool(payload["clean"]))
    if "yes" in payload:
        _set_if_unset("yes", bool(payload["yes"]))
    if "process_incomplete" in payload:
        _set_if_unset("process_incomplete", bool(payload["process_incomplete"]))
    if "status" in payload and getattr(args, "status", None) is None:
        status_value = payload["status"]
        if isinstance(status_value, str) and status_value.strip():
            args.status = [status_value.strip()]
        elif isinstance(status_value, Sequence):
            args.status = [str(item).strip() for item in status_value if str(item).strip()]

    # HF settings (only apply when unset on CLI)
    if "hf_repo" in payload:
        _set_if_unset("hf_repo", str(payload["hf_repo"]))
    if "hf_branch" in payload:
        _set_if_unset("hf_branch", str(payload["hf_branch"]))
    if "hf_token" in payload:
        _set_if_unset("hf_token", str(payload["hf_token"]))
    if "hf_private" in payload:
        _set_if_unset("hf_private", bool(payload["hf_private"]))
    if "hf_pull_policy" in payload:
        _set_if_unset("hf_pull_policy", str(payload["hf_pull_policy"]))


def _load_winrate_config(path: Path) -> Mapping[str, Any]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Winrate config not found: {path}")
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw)
    elif suffix == ".json":
        payload = yaml.safe_load(raw)
    else:
        raise ValueError(f"Unsupported winrate config format: {path} (expected .yaml/.yml/.json)")
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Winrate config must be a mapping at top level: {path}")
    return payload


def _apply_winrate_config(args: argparse.Namespace, path: Path) -> None:
    """Apply config file defaults to winrate args (CLI flags win)."""
    payload = dict(_load_winrate_config(path))

    hf_payload = payload.get("hf")
    if isinstance(hf_payload, Mapping):
        for key, value in hf_payload.items():
            if key == "repo":
                payload.setdefault("hf_processed_repo", value)
            elif key == "branch":
                payload.setdefault("hf_branch", value)
            elif key == "token":
                payload.setdefault("hf_token", value)
            elif key == "private":
                payload.setdefault("hf_private", value)
            else:
                payload.setdefault(f"hf_{key}", value)

    if "exclude_models" not in payload and "exclude_model" in payload:
        payload["exclude_models"] = payload["exclude_model"]

    def _set_if_unset(attr: str, value: Any) -> None:
        if not hasattr(args, attr):
            return
        if getattr(args, attr) is None:
            setattr(args, attr, value)

    if "processed_dir" in payload:
        _set_if_unset("processed_dir", Path(str(payload["processed_dir"])))
    if "output_dir" in payload:
        _set_if_unset("output_dir", Path(str(payload["output_dir"])))
    if "output" in payload:
        _set_if_unset("output", Path(str(payload["output"])))
    if "output_name" in payload:
        _set_if_unset("output_name", str(payload["output_name"]))
    if "processed_at" in payload:
        _set_if_unset("processed_at", str(payload["processed_at"]))
    if "missing_policy" in payload:
        _set_if_unset("missing_policy", str(payload["missing_policy"]))
    if "epsilon" in payload:
        try:
            _set_if_unset("epsilon", float(payload["epsilon"]))
        except Exception:
            pass
    if "min_common" in payload:
        try:
            _set_if_unset("min_common", int(payload["min_common"]))
        except Exception:
            pass
    if "weight_policy" in payload:
        _set_if_unset("weight_policy", str(payload["weight_policy"]))
    if "weight_cap" in payload:
        try:
            _set_if_unset("weight_cap", int(payload["weight_cap"]))
        except Exception:
            pass
    if "partial_datasets" in payload:
        _set_if_unset("partial_datasets", str(payload["partial_datasets"]))
    if "include_models" in payload:
        include_value = payload["include_models"]
        if isinstance(include_value, str):
            _set_if_unset("include_model", [include_value])
        elif isinstance(include_value, Sequence):
            _set_if_unset("include_model", [str(item) for item in include_value if str(item).strip()])
    if "exclude_models" in payload:
        exclude_value = payload["exclude_models"]
        if isinstance(exclude_value, str):
            _set_if_unset("exclude_model", [exclude_value])
        elif isinstance(exclude_value, Sequence):
            _set_if_unset("exclude_model", [str(item) for item in exclude_value if str(item).strip()])
    if "hf_processed_repo" in payload:
        _set_if_unset("hf_processed_repo", str(payload["hf_processed_repo"]))
    if "hf_processed_pull" in payload:
        _set_if_unset("hf_processed_pull", bool(payload["hf_processed_pull"]))
    if "hf_winrate_repo" in payload:
        _set_if_unset("hf_winrate_repo", str(payload["hf_winrate_repo"]))
    if "hf_branch" in payload:
        _set_if_unset("hf_branch", str(payload["hf_branch"]))
    if "hf_token" in payload:
        _set_if_unset("hf_token", str(payload["hf_token"]))
    if "hf_private" in payload:
        _set_if_unset("hf_private", bool(payload["hf_private"]))


def _build_winrate_args_from_config(path: Path) -> argparse.Namespace:
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
        include_model=None,
        exclude_model=None,
        partial_datasets=None,
        hf_processed_repo=None,
        hf_processed_pull=None,
        hf_winrate_repo=None,
        hf_branch=None,
        hf_token=None,
        hf_private=None,
    )
    _apply_winrate_config(args, path)
    _finalize_winrate_args(args)
    return args


def _finalize_process_args(args: argparse.Namespace) -> None:
    """Fill any unset process args with their defaults (config overrides defaults)."""
    if getattr(args, "runs_dir", None) is None:
        args.runs_dir = DEFAULT_RUNS_RAW_DIR
    if getattr(args, "output_dir", None) is None:
        args.output_dir = DEFAULT_PROCESSED_DIR
    if getattr(args, "env_config_root", None) is None:
        args.env_config_root = DEFAULT_ENV_CONFIG_ROOT
    if getattr(args, "max_workers", None) is None:
        args.max_workers = 4
    if getattr(args, "hf_private", None) is None:
        args.hf_private = False
    if getattr(args, "dry_run", None) is None:
        args.dry_run = False
    if getattr(args, "clean", None) is None:
        args.clean = False
    if getattr(args, "yes", None) is None:
        args.yes = False
    if getattr(args, "process_incomplete", None) is None:
        args.process_incomplete = False


def _finalize_winrate_args(args: argparse.Namespace) -> None:
    """Fill any unset winrate args with their defaults (config overrides defaults)."""
    if getattr(args, "processed_dir", None) is None:
        args.processed_dir = DEFAULT_PROCESSED_DIR
    if getattr(args, "output_dir", None) is None:
        args.output_dir = DEFAULT_WINRATE_DIR
    if getattr(args, "missing_policy", None) is None:
        args.missing_policy = "neg-inf"
    if getattr(args, "epsilon", None) is None:
        args.epsilon = 1e-9
    if getattr(args, "min_common", None) is None:
        args.min_common = 0
    if getattr(args, "weight_policy", None) is None:
        args.weight_policy = "ln"
    if getattr(args, "weight_cap", None) is None:
        args.weight_cap = 0
    if getattr(args, "include_model", None) is None:
        args.include_model = []
    if getattr(args, "exclude_model", None) is None:
        args.exclude_model = []
    if getattr(args, "partial_datasets", None) is None:
        args.partial_datasets = "strict"
    if getattr(args, "hf_processed_pull", None) is None:
        args.hf_processed_pull = False
    if getattr(args, "hf_private", None) is None:
        args.hf_private = False


def _upload_winrate_outputs(
    *,
    output_dir: Path,
    output_paths: Sequence[Path],
    repo_id: str,
    token: str | None,
    private: bool,
) -> None:
    if not output_paths:
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
    )


def _run_winrate_mode(argv: Sequence[str]) -> int:
    parser = build_winrate_parser()
    args = parser.parse_args(argv)

    if args.config:
        _apply_winrate_config(args, args.config)
    _finalize_winrate_args(args)

    hf_config = HFSyncConfig.from_cli(
        repo=args.hf_processed_repo,
        branch=args.hf_branch,
        token=args.hf_token,
        private=False,
        dry_run=False,
    )

    if args.list_models:
        source_dir, datasets, source_desc = _resolve_source(
            args.processed_dir,
            hf_config=hf_config if args.hf_processed_repo else None,
            hf_processed_pull=bool(args.hf_processed_pull),
        )
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
        include_models=tuple(args.include_model or ()),
        exclude_models=tuple(args.exclude_model or ()),
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
    if args.hf_winrate_repo:
        _upload_winrate_outputs(
            output_dir=args.output_dir,
            output_paths=winrate_result.output_paths,
            repo_id=args.hf_winrate_repo,
            token=args.hf_token,
            private=bool(args.hf_private),
        )
    return 0


def _execute_batch(args: argparse.Namespace) -> int:
    # Set the include_usage environment variable if explicitly specified
    if getattr(args, "include_usage", None) is not None:
        import os

        os.environ["MEDARC_INCLUDE_USAGE"] = "true" if args.include_usage else "false"

    config_path = Path(args.config).expanduser()
    env_root_override = Path(args.env_config_root).expanduser().resolve() if args.env_config_root else None
    run_config = load_run_config(config_path, env_default_root=env_root_override)

    run_name = args.name or run_config.name
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else Path(run_config.output_dir).expanduser()
    output_dir = output_dir.resolve()
    run_id = args.run_id  # May be None when using --auto-resume discovery

    if args.enable_additional_retries:
        from medarc_verifiers.utils.retry import patch_verifiers_model_response_retry
        from datetime import datetime

        cwd = Path.cwd()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        retry_log_path = cwd / "logs" / f"medarc_model_retry_{ts}.log"
        patch_verifiers_model_response_retry(log_path=retry_log_path)

    jobs = build_jobs(run_config)
    if not jobs:
        logger.error("Configuration %s did not produce any jobs.", config_path)
        return 1

    selected_jobs = _filter_jobs(jobs, args.job_id)
    if not selected_jobs:
        logger.error("No jobs matched the provided filters.")
        return 1

    env_args_map, sampling_args_map = _build_effective_args(jobs)
    config_checksum = compute_snapshot_checksum(run_config.model_dump())
    forced_envs = _parse_forced_envs(args.forced)
    forced_envs.update(_collect_rerun_envs(run_config.envs))

    planner = ManifestPlanner(
        output_dir=output_dir,
        run_id=run_id,
        run_name=run_name,
        config_path=config_path,
        config_checksum=config_checksum,
        jobs=jobs,
        env_args_map=env_args_map,
        sampling_args_map=sampling_args_map,
        restart_source=args.restart,
        auto_resume=bool(args.auto_resume),
        persist=not bool(args.dry_run),
    )

    try:
        manifest_plan = planner.plan(force_all=bool(args.force), forced_envs=forced_envs)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    runnable_ids = manifest_plan.runnable_job_ids
    selected_ids = {job.job_id for job in selected_jobs}
    planned_jobs = [job for job in jobs if job.job_id in runnable_ids and job.job_id in selected_ids]

    _print_job_plan(
        selected_jobs,
        manifest=manifest_plan.manifest,
        runnable_job_ids=runnable_ids,
        discovered_total=len(jobs),
        dry_run=bool(args.dry_run),
    )

    if not planned_jobs:
        if manifest_plan.reused_job_ids:
            logger.info(
                "All jobs already completed (reused %d job(s) from prior manifests).",
                len(manifest_plan.reused_job_ids),
            )
        else:
            logger.info("No jobs were scheduled after applying filters and resume settings.")

        # Check if all selected jobs are completed (not just filtered out)
        all_completed = all(
            manifest_plan.manifest.job_entry(job.job_id)
            and manifest_plan.manifest.job_entry(job.job_id).status == "completed"
            for job in selected_jobs
        )

        if all_completed and selected_jobs and not args.dry_run and not args.force:
            # Prompt user for action
            choice = _prompt_completed_jobs_action()
            if choice == "new":
                logger.info("Creating a new run with all jobs...")
                # Create a fresh run by disabling auto-resume and forcing a new run_id
                # Recursively call with updated args to create new manifest
                new_args = argparse.Namespace(**vars(args))
                new_args.auto_resume = False
                new_args.run_id = None  # Force generation of new run_id
                new_args.restart = None
                return _execute_batch(new_args)
            elif choice == "rerun":
                logger.info("Rerunning all completed jobs...")
                # Set all selected jobs to runnable
                runnable_ids = {job.job_id for job in selected_jobs}
                planned_jobs = [job for job in jobs if job.job_id in runnable_ids and job.job_id in selected_ids]
                # Continue execution below
            elif choice == "exit":
                logger.info("Exiting without running jobs.")
                _log_summary([], manifest_plan.manifest)
                return 0
            else:  # continue/skip
                logger.info("Continuing without running jobs.")
                _log_summary([], manifest_plan.manifest)
                return 0
        else:
            _log_summary([], manifest_plan.manifest)
            return 0

    if not planned_jobs:
        # After prompting, still no planned jobs (shouldn't happen, but safety check)
        _log_summary([], manifest_plan.manifest)
        return 0

    settings = ExecutorSettings(
        run_id=manifest_plan.manifest.model.run_id or "",
        output_dir=output_dir,
        env_dir=Path(args.env_dir).expanduser(),
        endpoints_path=Path(args.endpoints_path).expanduser() if args.endpoints_path else None,
        default_api_key_var=args.default_api_key_var,
        default_api_base_url=args.default_api_base_url,
        log_level="DEBUG" if args.verbose else "INFO",
        verbose=args.verbose,
        save_results=args.save_results,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=_coerce_optional_str(args.hf_hub_dataset_name),
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        max_concurrent=args.max_concurrent,  # CLI override (None if not provided)
        timeout=args.timeout,
        sleep=args.sleep,
        dry_run=args.dry_run,
        cli_env_args=getattr(args, "cli_env_args", None),
        cli_sampling_args=getattr(args, "cli_sampling_args", None),
    )

    logger.info(
        "Loaded %d job(s); executing %d after filters (%d reusable).",
        len(jobs),
        len(planned_jobs),
        len(manifest_plan.reused_job_ids),
    )

    endpoints_cache: dict[str, Any] = {}
    env_metadata_cache: dict[str, Any] = {}

    results = execute_jobs(
        planned_jobs,
        settings,
        endpoints_cache=endpoints_cache,
        env_metadata_cache=env_metadata_cache,
        manifest=None if args.dry_run else manifest_plan.manifest,
    )

    _log_summary(results, manifest_plan.manifest)

    has_failures = any(result.status == "failed" for result in results if result.status != "skipped")
    return 1 if has_failures else 0


def _build_effective_args(
    jobs: Sequence[ResolvedJob],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    env_map: dict[str, dict[str, Any]] = {}
    sampling_map: dict[str, dict[str, Any]] = {}
    for job in jobs:
        env_map[job.job_id] = dict(job.env_args)
        sampling_map[job.job_id] = dict(job.sampling_args)
    return env_map, sampling_map


def _parse_forced_envs(values: Sequence[str] | None) -> set[str]:
    forced: set[str] = set()
    if not values:
        return forced
    for chunk in values:
        if not chunk:
            continue
        for item in chunk.split(","):
            value = item.strip()
            if value:
                forced.add(value.lower())
    return forced


def _collect_rerun_envs(envs: Mapping[str, EnvironmentConfigSchema]) -> set[str]:
    rerun: set[str] = set()
    for env in envs.values():
        if getattr(env, "rerun", False):
            for key in (env.id, env.module, env.matrix_base_id):
                if key:
                    rerun.add(str(key).lower())
    return rerun


def _filter_jobs(jobs: Sequence[ResolvedJob], job_filters: Sequence[str] | None) -> list[ResolvedJob]:
    if not job_filters:
        return list(jobs)
    filters = set(job_filters)
    selected = [job for job in jobs if job.job_id in filters]
    missing = filters - {job.job_id for job in selected}
    if missing:
        logger.warning("Unknown job ids requested: %s", ", ".join(sorted(missing)))
    return selected


def _coerce_optional_str(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    return value


def _prompt_completed_jobs_action() -> str:
    """Prompt user to choose what to do when all jobs are completed.

    Returns:
        "new", "rerun", "continue", or "exit"
    """
    console = Console()

    message = "\n[bold yellow]All jobs are already completed.[/bold yellow]\n"
    message += "What would you like to do?\n"
    message += "  [bold cyan]n[/bold cyan] - Create a new run\n"
    message += "  [bold cyan]r[/bold cyan] - Rerun all jobs (ignore completion status)\n"
    message += "  [bold cyan]c[/bold cyan] - Continue without running (default)\n"
    message += "  [bold cyan]e[/bold cyan] - Exit\n"

    console.print(message)

    try:
        response = input("Choose [n/r/c/e]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()  # New line after Ctrl+C
        return "exit"

    if response == "n" or response == "new":
        return "new"
    elif response == "r" or response == "rerun":
        return "rerun"
    elif response == "e" or response == "exit":
        return "exit"
    else:
        # Default to continue for any other input (including empty/enter)
        return "continue"


def _log_summary(results: Sequence[JobExecutionResult], manifest: RunManifest | None = None) -> None:
    if manifest is not None:
        summary = manifest.summary
        logger.info(
            "Run complete: %d completed, %d pending, %d failed, %d skipped (total %d).",
            summary.get("completed", 0),
            summary.get("pending", 0),
            summary.get("failed", 0),
            summary.get("skipped", 0),
            summary.get("total", 0),
        )
        return
    total = len(results)
    succeeded = sum(result.status == "succeeded" for result in results)
    skipped = sum(result.status == "skipped" for result in results)
    failed = sum(result.status == "failed" for result in results)
    logger.info("Run complete: %d succeeded, %d skipped, %d failed (total %d).", succeeded, skipped, failed, total)


def _print_general_help() -> None:
    message = dedent(
        f"""\
        Usage:
          {COMMAND} <ENV> [options]                 # Single run (ENV must be first; use ENV --help for details)
          {COMMAND} {BENCH_COMMAND} --config CONFIG.yaml ...  # Batch run (see: {COMMAND} {BENCH_COMMAND} --help)
          {COMMAND} {PROCESS_COMMAND} [options]               # Export raw runs to parquet (see: {COMMAND} {PROCESS_COMMAND} --help)
          {COMMAND} {WINRATE_COMMAND} [options]               # Compute win rates from processed parquet outputs

        First argument must be the environment slug for single runs. Use '{COMMAND} {BENCH_COMMAND} --help' for batch mode options."""
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
            for key in filter(None, keys):
                export_map[key] = env_cfg.export

    return export_map


def _print_job_plan(
    jobs: Sequence[ResolvedJob],
    *,
    manifest: RunManifest | None,
    runnable_job_ids: set[str],
    discovered_total: int,
    dry_run: bool,
) -> None:
    """Render a human-friendly summary of the jobs scheduled for execution."""
    listed_total = len(jobs)
    scheduled_total = sum(1 for job in jobs if job.job_id in runnable_job_ids)
    caption_parts: list[str] = [f"{listed_total} job(s) listed"]
    caption_parts.append(f"{scheduled_total} to {'dry-run' if dry_run else 'run'}")
    if discovered_total != listed_total:
        caption_parts.append(f"{discovered_total} discovered")
    caption = " | ".join(part for part in caption_parts if part)

    if not jobs:
        logger.info("No jobs to display (%s).", caption)
        return

    def _format_label(primary: str | None, secondary: str | None) -> str:
        if primary and secondary and primary != secondary:
            return f"{primary} ({secondary})"
        return primary or secondary or "-"

    def _resolve_status(job_id: str, entry: ManifestJobEntry | None) -> str:
        if job_id in runnable_job_ids:
            return "next"
        if entry and entry.status == "completed":
            return "completed"
        return "pending"

    entries = {}
    if manifest is not None:
        entries = {entry.job_id: entry for entry in manifest.jobs if entry.job_id}

    console = Console()
    table = Table(title="Planned Jobs", caption=caption, expand=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Job ID", style="bold cyan", overflow="fold")
    table.add_column("Status", style="yellow")
    table.add_column("Name", style="white", overflow="fold")
    table.add_column("Model", style="magenta", overflow="fold")
    table.add_column("Environment", style="green", overflow="fold")
    table.add_column("Examples", justify="right")
    table.add_column("Rollouts", justify="right")

    for index, job in enumerate(jobs, start=1):
        entry = entries.get(job.job_id)
        model_label = _format_label(job.model.id, job.model.model)
        env_label = _format_label(job.env.id, job.env.module)
        status = _resolve_status(job.job_id, entry)
        table.add_row(
            str(index),
            job.job_id,
            status,
            job.name or "-",
            model_label,
            env_label,
            str(job.env.num_examples),
            str(job.env.rollouts_per_example),
        )

    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
