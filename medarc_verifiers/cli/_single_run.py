"""Single-run CLI implementation with dynamic environment flags."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from verifiers.utils.eval_utils import run_evaluation

from medarc_verifiers.cli._constants import (
    BENCH_COMMAND,
    COMMAND,
    DEFAULT_API_BASE_URL,
    DEFAULT_API_KEY_VAR,
    DEFAULT_ENDPOINTS_PATH,
)
from medarc_verifiers.cli._eval_builder import build_client_config, build_eval_config
from medarc_verifiers.cli._schemas import ModelConfigSchema
from medarc_verifiers.cli.utils.env_args import EnvParam, MissingEnvParamError, gather_env_cli_metadata, merge_env_args
from medarc_verifiers.cli.utils.endpoint_utils import load_endpoint_registry
from medarc_verifiers.cli.utils.overrides import build_cli_override
from medarc_verifiers.cli.utils.resume import (
    format_resume_mismatch_lines,
    is_resume_metadata_mismatch_error,
    load_resume_metadata_values,
    resolve_resume_path,
)
from medarc_verifiers.cli.utils.shared import (
    HEADER_SEPARATOR,
    STATE_COLUMNS_SEPARATOR,
    DEFAULT_SINGLE_RUN_MAX_CONCURRENT,
    ensure_root_logging,
    flatten_state_columns,
    merge_sampling_args,
    normalize_headers,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnvOptionBinding:
    """Track how an environment parameter is bound to an argparse destination."""

    param: EnvParam
    dest: str
    default: Any


@dataclass
class _SingleRunEnvConfig:
    """Lightweight env config to reuse the shared EvalConfig builder."""

    id: str
    module: str | None = None
    matrix_base_id: str | None = None
    num_examples: int = 5
    rollouts_per_example: int = 1
    max_concurrent: int | None = None
    independent_scoring: bool = True
    state_columns: list[str] | None = None
    verbose: bool | None = False


def run_single_mode(argv: Sequence[str] | None = None) -> int:
    """Entry point for single-run (medarc-eval style) execution."""
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if not args_list:
        _print_env_first_error()
        return 2

    first_token = args_list[0]
    if first_token.startswith("-"):
        _print_env_first_error()
        return 2

    env_id = first_token
    remaining = args_list[1:]
    endpoints_path_explicit = _option_was_provided(remaining, "--endpoints-path", "-e")
    api_key_var_explicit = _option_was_provided(remaining, "--api-key-var", "-k")

    parser, env_group, reserved_dests = _build_base_parser_layout(require_env=True, add_help=True, env_id=env_id)
    try:
        metadata = gather_env_cli_metadata(env_id)
    except ImportError as exc:
        parser.error(str(exc))

    bindings = register_env_options(env_group, reserved_dests, env_id, metadata)

    try:
        args = parser.parse_args([env_id, *remaining])
    except SystemExit as exc:  # pragma: no cover - argparse already emitted error/help
        return int(exc.code)
    try:
        args.model_call_retries = _resolve_model_call_retries(
            args.model_call_retries,
            args.enable_additional_retries,
        )
        if args.http_max_retries is not None and args.http_max_retries < 0:
            raise ValueError("--http-max-retries must be >= 0.")
        if args.rollout_max_retries < 0:
            raise ValueError("--rollout-max-retries must be >= 0.")
    except ValueError as exc:
        parser.error(str(exc))

    try:
        env_override_mapping = build_cli_override(
            json_payload=args.env_args,
            pairs=args.env_arg,
            json_flag="--env-args",
            pair_flag="--env-arg",
        )
    except ValueError as exc:
        parser.error(str(exc))
    json_env_args: Mapping[str, Any] = env_override_mapping or {}
    explicit_cli_args = extract_env_cli_args(args, bindings)

    try:
        merged_env_args = merge_env_args(
            env_id,
            sources=[json_env_args, explicit_cli_args],
            metadata=metadata,
            allow_unknown=True,
            enforce_required=True,
            verbose=args.verbose,
        )
    except (MissingEnvParamError, ValueError) as exc:
        parser.error(str(exc))

    try:
        sampling_override_mapping = build_cli_override(
            json_payload=args.sampling_args,
            pairs=args.sampling_arg,
            json_flag="--sampling-args",
            pair_flag="--sampling-arg",
        )
    except ValueError as exc:
        parser.error(str(exc))
    merged_sampling_args = merge_sampling_args(
        sampling_override_mapping or {},
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        n=args.n,
    )

    try:
        headers = normalize_headers(args.header, header_file=args.header_file)
    except ValueError as exc:
        parser.error(str(exc))

    state_columns = flatten_state_columns(args.state_columns)

    ensure_root_logging("DEBUG" if args.verbose else "INFO")

    if args.model_call_retries > 0 and not args.dry_run:
        from datetime import datetime

        from medarc_verifiers.utils.retry import patch_verifiers_model_response_retry

        cwd = Path.cwd()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        retry_log_path = cwd / "logs" / f"medarc_model_retry_{ts}.log"
        patch_verifiers_model_response_retry(
            attempts=args.model_call_retries,
            log_path=retry_log_path,
        )

    endpoints_path = Path(args.endpoints_path).expanduser()
    default_endpoints_path = Path(DEFAULT_ENDPOINTS_PATH).expanduser()
    if not endpoints_path.exists():
        if endpoints_path_explicit:
            logger.error("Explicit endpoints registry path does not exist: %s", endpoints_path)
            return 2
        if _same_path(endpoints_path, default_endpoints_path):
            logger.warning(
                "Default endpoints registry '%s' not found; continuing without endpoint aliases.",
                endpoints_path,
            )
        endpoints = {}
    else:
        try:
            endpoints = load_endpoint_registry(endpoints_path)
        except Exception as exc:  # noqa: BLE001
            if endpoints_path_explicit:
                logger.error("Failed to load explicit endpoints registry '%s': %s", endpoints_path, exc)
                return 2
            logger.warning(
                "Failed to load default endpoints registry '%s'; continuing without endpoint aliases: %s",
                endpoints_path,
                exc,
            )
            endpoints = {}

    if endpoints_path_explicit and not endpoints:
        logger.error("Failed to load endpoint registry from explicit path: %s", endpoints_path)
        return 2

    model_cfg = ModelConfigSchema(model=args.model)
    resolved_model, client_config, prime_sampling_overrides = build_client_config(
        model_cfg,
        endpoints=endpoints,
        default_api_key_var=args.api_key_var,
        default_api_key_var_explicit=api_key_var_explicit,
        default_api_base_url=args.api_base_url,
        api_base_url_override=None,
        http_max_retries_override=args.http_max_retries,
        timeout_override=args.timeout,
        headers=headers,
    )

    # Merge Prime Inference overrides with user sampling args (user args take precedence)
    merged_sampling_args = {**prime_sampling_overrides, **merged_sampling_args}

    env_cfg = _SingleRunEnvConfig(
        id=args.env,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        independent_scoring=not args.group_scoring,
        state_columns=state_columns or None,
        verbose=args.verbose,
    )

    try:
        resume_path = resolve_resume_path(
            resume_arg=args.resume,
            env_id=args.env,
            model=resolved_model,
            num_examples=args.num_examples,
            rollouts_per_example=args.rollouts_per_example,
            env_dir_path=Path(args.env_dir_path).expanduser(),
        )
    except ValueError as exc:
        parser.error(str(exc))

    if isinstance(args.resume, str):
        logger.info("Resuming from explicit path: %s", resume_path)
    elif args.resume is True:
        if resume_path is not None:
            logger.info("Auto-resuming from: %s", resume_path)
        else:
            logger.info("No matching incomplete run found for --resume; starting a new run.")

    eval_config = build_eval_config(
        job_label=args.env,
        model_cfg=model_cfg,
        env_cfg=env_cfg,
        env_args=merged_env_args,
        sampling_args=merged_sampling_args,
        cli_env_args=None,
        cli_sampling_args=None,
        resolved_model=resolved_model,
        client_config=client_config,
        env_dir=Path(args.env_dir_path).expanduser(),
        max_concurrent_override=args.max_concurrent,
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        rollout_max_retries=args.rollout_max_retries,
        resume_path=resume_path,
        default_max_concurrent=DEFAULT_SINGLE_RUN_MAX_CONCURRENT,
        save_results=args.save_results,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name or None,
        verbose=args.verbose,
        env_metadata_cache=None,
        enforce_required_env_args=True,
    )

    if args.dry_run:
        print(eval_config.model_dump_json(indent=2))
        return 0

    if args.save_every != -1:
        logger.warning("Single-run option --save-every is deprecated and ignored.")

    # Set the include_usage environment variable if explicitly specified
    if args.include_usage is not None:
        os.environ["MEDARC_INCLUDE_USAGE"] = "true" if args.include_usage else "false"

    try:
        asyncio.run(run_evaluation(eval_config))
    except KeyboardInterrupt:
        logger.error("Evaluation interrupted by user.")
        return 1
    except Exception as exc:  # noqa: BLE001
        if resume_path is not None and is_resume_metadata_mismatch_error(exc):
            logger.error("Resume metadata mismatch for %s.", resume_path)
            saved_values = load_resume_metadata_values(resume_path)
            current_values = {
                "env_id": eval_config.env_id,
                "model": eval_config.model,
                "rollouts_per_example": eval_config.rollouts_per_example,
                "num_examples": eval_config.num_examples,
            }
            for line in format_resume_mismatch_lines(saved_values=saved_values, current_values=current_values):
                logger.error("  %s", line)
            logger.error("Use --resume PATH with matching settings, or start a new run.")
            return 1
        if args.verbose:
            logger.exception("Evaluation failed.")
        else:
            logger.error("Evaluation failed: %s", exc)
        return 1

    return 0


def _build_base_parser_layout(
    *,
    require_env: bool,
    add_help: bool,
    env_id: str | None = None,
) -> tuple[argparse.ArgumentParser, Any, set[str]]:
    parser = argparse.ArgumentParser(
        prog=COMMAND,
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run verifiers evaluations with dynamic environment parameters. "
            f"ENV must be provided first. Use '{COMMAND} <env> --help' for env options "
            f"or '{COMMAND} {BENCH_COMMAND} --help' for batch runs."
        ),
    )
    reserved_dests: set[str] = set()

    def _add_and_track(group, *args: str, **kwargs: Any) -> None:
        action = group.add_argument(*args, **kwargs)
        reserved_dests.add(action.dest)

    env_kwargs = {"metavar": "ENV", "help": "Environment slug or module path"}
    if require_env:
        _add_and_track(parser, "env", **env_kwargs)
    else:
        _add_and_track(parser, "env", nargs="?", **env_kwargs)

    if add_help:
        _add_and_track(
            parser, "-h", "--help", action="help", default=argparse.SUPPRESS, help="show this help message and exit"
        )

    env_group_title = f"{env_id} environment options" if env_id else "Environment options (ENV=<env>)"
    env_group = parser.add_argument_group(env_group_title)
    core_group = parser.add_argument_group(f"{COMMAND} options")

    _add_and_track(
        core_group,
        "--env-arg",
        action="append",
        help="Override an environment argument (KEY=VALUE). Repeat for multiple overrides.",
    )
    _add_and_track(core_group, "--env-args", help='Environment arguments as JSON object (e.g., \'{"key": "value"}\').')
    _add_and_track(core_group, "--env-dir-path", "-p", default="./environments", help="Path to environments directory.")
    _add_and_track(
        core_group,
        "--endpoints-path",
        "-e",
        default=str(DEFAULT_ENDPOINTS_PATH),
        help="Path to API endpoints registry (TOML preferred).",
    )
    _add_and_track(core_group, "--model", "-m", default="gpt-4.1-mini", help="Model identifier to evaluate.")
    _add_and_track(
        core_group,
        "--api-key-var",
        "-k",
        default=DEFAULT_API_KEY_VAR,
        help="Environment variable name for the API key.",
    )
    _add_and_track(
        core_group, "--api-base-url", "-b", default=DEFAULT_API_BASE_URL, help="Base URL for the inference API."
    )
    _add_and_track(
        core_group,
        "--header",
        action="append",
        help=f"Extra HTTP header to send ('Name{HEADER_SEPARATOR} Value'). Repeatable.",
    )
    _add_and_track(
        core_group,
        "--header-file",
        type=Path,
        help="File containing newline-delimited 'Name: Value' header entries. Overrides --header on conflicts.",
    )
    _add_and_track(core_group, "--num-examples", "-n", type=int, default=5, help="Number of examples to evaluate.")
    _add_and_track(
        core_group, "--rollouts-per-example", "-r", type=int, default=3, help="Number of rollouts per example."
    )
    _add_and_track(
        core_group,
        "--max-concurrent",
        "-c",
        type=int,
        default=DEFAULT_SINGLE_RUN_MAX_CONCURRENT,
        help="Maximum number of concurrent requests.",
    )
    _add_and_track(
        core_group,
        "--max-concurrent-generation",
        type=int,
        default=None,
        help="Deprecated: ignored.",
    )
    _add_and_track(
        core_group,
        "--max-concurrent-scoring",
        type=int,
        default=None,
        help="Deprecated: ignored.",
    )
    _add_and_track(
        core_group,
        "--timeout",
        type=float,
        default=None,
        help="Override request timeout in seconds (defaults to the verifier client default).",
    )
    _add_and_track(
        core_group,
        "--http-max-retries",
        type=int,
        default=None,
        help="HTTP/client-level retries for model calls (overrides model config max_retries).",
    )
    _add_and_track(
        core_group,
        "--rollout-max-retries",
        type=int,
        default=0,
        help="Retry full rollout/group on retryable infra/invalid-response errors.",
    )
    _add_and_track(
        core_group,
        "--model-call-retries",
        type=int,
        default=None,
        help="Per-model-call MedARC retry attempts (0 disables the monkeypatch).",
    )
    _add_and_track(
        core_group,
        "--enable-additional-retries",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated alias for --model-call-retries (true maps to 3 attempts).",
    )
    _add_and_track(
        core_group,
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum tokens to generate (unset to use model defaults).",
    )
    _add_and_track(core_group, "--temperature", "-T", type=float, default=None, help="Sampling temperature.")
    _add_and_track(core_group, "--top-p", type=float, default=None, help="Top-p nucleus sampling value.")
    _add_and_track(core_group, "--top-k", type=int, default=None, help="Top-k sampling value.")
    _add_and_track(
        core_group,
        "--n",
        type=int,
        default=None,
        help="Number of responses per prompt (passes through sampling_args.n).",
    )
    _add_and_track(
        core_group, "--sampling-arg", action="append", help="Override sampling args with KEY=VALUE (repeatable)."
    )
    _add_and_track(core_group, "--sampling-args", help="Sampling arguments as JSON object.")
    _add_and_track(core_group, "--verbose", "-v", action="store_true", help="Enable verbose logging.")
    _add_and_track(
        core_group,
        "--group-scoring",
        action="store_true",
        default=False,
        help="Score rollouts per-example group (default: independent scoring).",
    )
    _add_and_track(
        core_group,
        "--state-columns",
        action="append",
        type=parse_state_columns_arg,
        metavar="COLUMNS",
        help=(
            f"Comma-separated list of state columns to persist (use '{STATE_COLUMNS_SEPARATOR}' between values); repeatable."
        ),
    )
    _add_and_track(
        core_group,
        "--save-results",
        "--save-dataset",
        "-s",
        dest="save_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save evaluation results to disk (use --no-save-results to disable; accepts legacy --save-dataset alias).",
    )
    _add_and_track(
        core_group,
        "--resume",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Resume from PATH (must contain results.jsonl and metadata.json), or pass --resume with no PATH to "
            "auto-discover the latest matching incomplete run. Resume requires matching env/model/rollouts; "
            "num_examples may increase but cannot decrease."
        ),
    )
    _add_and_track(
        core_group,
        "--save-every",
        "-f",
        type=int,
        default=-1,
        help="Deprecated: ignored.",
    )
    _add_and_track(
        core_group,
        "--save-to-hf-hub",
        "-H",
        action="store_true",
        default=False,
        help="Push evaluation dataset to the Hugging Face Hub.",
    )
    _add_and_track(
        core_group, "--hf-hub-dataset-name", "-D", default="", help="Custom Hugging Face dataset name when saving."
    )
    _add_and_track(
        core_group, "--dry-run", action="store_true", help="Print the resolved EvalConfig and exit without running."
    )
    _add_and_track(
        core_group,
        "--include-usage",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Include usage reporting in API requests (extra_body.usage.include). "
            "Default: auto-detect (enabled for Prime Inference, disabled otherwise)."
        ),
    )
    return parser, env_group, reserved_dests


def build_base_parser(*, require_env: bool, add_help: bool) -> argparse.ArgumentParser:
    parser, _env_group, _reserved_dests = _build_base_parser_layout(require_env=require_env, add_help=add_help)
    return parser


def register_env_options(
    group: Any,
    reserved_dests: set[str],
    env_id: str,
    metadata: Sequence[EnvParam],
) -> dict[str, EnvOptionBinding]:
    bindings: dict[str, EnvOptionBinding] = {}

    for param in metadata:
        if not param.supports_cli:
            logger.debug(
                "Parameter '%s' in env '%s' requires --env-args (reason: %s).",
                param.name,
                env_id,
                param.unsupported_reason,
            )
            continue

        dest = param.name
        option = f"--{param.cli_name}"
        if dest in reserved_dests:
            dest = f"env_{dest}"
            option = f"--env-{param.cli_name}"
        reserved_dests.add(dest)

        kwargs: dict[str, Any] = {"dest": dest, "help": param.help}
        if param.choices:
            kwargs["choices"] = param.choices

        if param.action == "BooleanOptionalAction" or param.kind == "bool":
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs["default"] = param.default if param.default is not None else None
        elif param.is_list:
            kwargs["action"] = "append"
            kwargs["type"] = param.element_type
            kwargs["default"] = None
        else:
            if param.argparse_type is not None:
                kwargs["type"] = param.argparse_type
            kwargs["default"] = param.default

        action = group.add_argument(option, **kwargs)
        bindings[action.dest] = EnvOptionBinding(param=param, dest=action.dest, default=action.default)

    return bindings


def extract_env_cli_args(
    namespace: argparse.Namespace,
    bindings: Mapping[str, EnvOptionBinding],
) -> dict[str, Any]:
    explicit: dict[str, Any] = {}

    for binding in bindings.values():
        value = getattr(namespace, binding.dest)
        param = binding.param
        default = binding.default

        if param.is_list:
            if value is not None:
                explicit[param.name] = value
            continue

        if param.action == "BooleanOptionalAction" or param.kind == "bool":
            # BooleanOptionalAction defaults to None when the flag is not provided.
            # Treat unset flags as absent (do not inject `param: None` overrides).
            if value is None:
                continue
            if param.required or default is None or value != default:
                explicit[param.name] = value
            continue

        if value is None:
            continue

        if param.required or default is None or value != default:
            explicit[param.name] = value

    return explicit


def parse_state_columns_arg(value: str) -> list[str]:
    columns = [part.strip() for part in value.split(STATE_COLUMNS_SEPARATOR)]
    return [column for column in columns if column]


def _option_was_provided(argv: Sequence[str], long_flag: str, short_flag: str | None = None) -> bool:
    for token in argv:
        if token == long_flag or token.startswith(f"{long_flag}="):
            return True
        if short_flag and (token == short_flag or token.startswith(f"{short_flag}=")):
            return True
        if short_flag and token.startswith(short_flag) and not token.startswith("--") and len(token) > len(short_flag):
            return True
    return False


def _resolve_model_call_retries(model_call_retries: int | None, deprecated_toggle: bool | None) -> int:
    if model_call_retries is not None:
        if model_call_retries < 0:
            raise ValueError("--model-call-retries must be >= 0.")
        if deprecated_toggle is not None:
            logger.warning(
                "Ignoring deprecated --enable-additional-retries because --model-call-retries was explicitly set."
            )
        return model_call_retries

    if deprecated_toggle is None:
        return 0

    logger.warning("Flag --enable-additional-retries is deprecated; use --model-call-retries <attempts> instead.")
    return 3 if deprecated_toggle else 0


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve(strict=False) == right.resolve(strict=False)
    except OSError:
        return left == right


def _print_env_first_error() -> None:
    message = f"First argument must be ENV (e.g., medqa). For batch mode, run: {COMMAND} {BENCH_COMMAND} --help."
    print(message, file=sys.stderr)


__all__ = ["run_single_mode"]
