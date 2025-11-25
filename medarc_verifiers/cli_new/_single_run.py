"""Single-run CLI implementation with dynamic environment flags."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils.eval_utils import run_evaluation

from medarc_verifiers.cli_new.utils.env_args import (
    EnvParam,
    MissingEnvParamError,
    gather_env_cli_metadata,
    validate_env_arg_values,
)
from medarc_verifiers.cli_new.utils.endpoint_utils import load_endpoint_registry
from medarc_verifiers.cli_new.utils.overrides import build_cli_override
from medarc_verifiers.cli_new.utils.shared import (
    HEADER_SEPARATOR,
    STATE_COLUMNS_SEPARATOR,
    DEFAULT_SINGLE_RUN_MAX_CONCURRENT,
    ensure_required_params,
    ensure_root_logging,
    flatten_state_columns,
    merge_env_args,
    merge_sampling_args,
    normalize_headers,
    resolve_endpoint_selection,
)
from medarc_verifiers.utils import sanitize_sampling_args_for_openai

logger = logging.getLogger(__name__)

PROGRAM_NAME = "medarc-new"


@dataclass(frozen=True)
class EnvOptionBinding:
    """Track how an environment parameter is bound to an argparse destination."""

    param: EnvParam
    dest: str
    default: Any


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

    parser = build_base_parser(require_env=True, add_help=True)
    try:
        metadata = gather_env_cli_metadata(env_id)
    except ImportError as exc:
        parser.error(str(exc))

    bindings = register_env_options(parser, env_id, metadata)

    try:
        args = parser.parse_args([env_id, *remaining])
    except SystemExit as exc:  # pragma: no cover - argparse already emitted error/help
        return int(exc.code)

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
        ensure_required_params(metadata, explicit_cli_args, json_env_args)
    except MissingEnvParamError as exc:
        parser.error(str(exc))

    merged_env_args = merge_env_args(explicit_cli_args, json_env_args)
    try:
        validate_env_arg_values(env_id, merged_env_args, metadata)
    except ValueError as exc:
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

    endpoints_path = Path(args.endpoints_path).expanduser()
    try:
        endpoints = load_endpoint_registry(endpoints_path)
    except Exception as exc:  # noqa: BLE001
        parser.error(f"Failed to load endpoints registry: {exc}")

    resolved_model, api_key_var, api_base_url = resolve_endpoint_selection(
        args.model,
        endpoints,
        default_key_var=args.api_key_var,
        default_base_url=args.api_base_url,
    )

    client_kwargs: dict[str, Any] = {
        "api_key_var": api_key_var,
        "api_base_url": api_base_url,
        "extra_headers": headers or None,
    }
    if args.timeout is not None:
        client_kwargs["timeout"] = args.timeout
    client_config = ClientConfig(**client_kwargs)
    client_kwargs: dict[str, Any] = {
        "api_key_var": api_key_var,
        "api_base_url": api_base_url,
        "extra_headers": headers or None,
    }
    if args.timeout is not None:
        client_kwargs["timeout"] = args.timeout
    client_config = ClientConfig(**client_kwargs)

    sanitized_sampling_args = sanitize_sampling_args_for_openai(merged_sampling_args)

    eval_config = EvalConfig(
        env_id=args.env,
        env_args=merged_env_args,
        env_dir_path=str(Path(args.env_dir_path).expanduser()),
        model=resolved_model,
        client_config=client_config,
        sampling_args=sanitized_sampling_args,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        interleave_scoring=not args.no_interleave_scoring,
        print_results=True,
        verbose=args.verbose,
        state_columns=state_columns or None,
        save_results=args.save_results,
        save_every=args.save_every,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name or None,
    )

    if args.dry_run:
        print(eval_config.model_dump_json(indent=2))
        return 0

    try:
        asyncio.run(run_evaluation(eval_config))
    except KeyboardInterrupt:
        logger.error("Evaluation interrupted by user.")
        return 1
    except Exception as exc:  # noqa: BLE001
        if args.verbose:
            logger.exception("Evaluation failed.")
        else:
            logger.error("Evaluation failed: %s", exc)
        return 1

    return 0


def build_base_parser(*, require_env: bool, add_help: bool) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        add_help=add_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run verifiers evaluations with dynamic environment parameters. "
            "ENV must be provided first. Use 'medarc-new <env> --help' for env options "
            "or 'medarc-new bench --help' for batch runs."
        ),
    )
    for group in parser._action_groups:
        if group.title in {"optional arguments", "options"}:
            group.title = "medarc-new options"
            break

    env_kwargs = {"metavar": "ENV", "help": "Environment slug or module path"}
    if require_env:
        parser.add_argument("env", **env_kwargs)
    else:
        parser.add_argument("env", nargs="?", **env_kwargs)

    parser.add_argument(
        "--env-arg",
        action="append",
        help="Override an environment argument (KEY=VALUE). Repeat for multiple overrides.",
    )
    parser.add_argument("--env-args", help='Environment arguments as JSON object (e.g., \'{"key": "value"}\').')
    parser.add_argument("--env-dir-path", "-p", default="./environments", help="Path to environments directory.")
    parser.add_argument(
        "--endpoints-path", "-e", default="./configs/endpoints.py", help="Path to API endpoints registry."
    )
    parser.add_argument("--model", "-m", default="gpt-4.1-mini", help="Model identifier to evaluate.")
    parser.add_argument(
        "--api-key-var", "-k", default="OPENAI_API_KEY", help="Environment variable name for the API key."
    )
    parser.add_argument(
        "--api-base-url", "-b", default="https://api.openai.com/v1", help="Base URL for the inference API."
    )
    parser.add_argument(
        "--header",
        action="append",
        help=f"Extra HTTP header to send ('Name{HEADER_SEPARATOR} Value'). Repeatable.",
    )
    parser.add_argument(
        "--header-file",
        type=Path,
        help="File containing newline-delimited 'Name: Value' header entries. Overrides --header on conflicts.",
    )
    parser.add_argument("--num-examples", "-n", type=int, default=5, help="Number of examples to evaluate.")
    parser.add_argument("--rollouts-per-example", "-r", type=int, default=3, help="Number of rollouts per example.")
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=DEFAULT_SINGLE_RUN_MAX_CONCURRENT,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--max-concurrent-generation", type=int, default=None, help="Maximum number of concurrent generation requests."
    )
    parser.add_argument(
        "--max-concurrent-scoring", type=int, default=None, help="Maximum number of concurrent scoring requests."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override request timeout in seconds (defaults to the verifier client default).",
    )
    parser.add_argument(
        "--max-tokens", "-t", type=int, default=None, help="Maximum tokens to generate (unset to use model defaults)."
    )
    parser.add_argument("--temperature", "-T", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p nucleus sampling value.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling value.")
    parser.add_argument(
        "--n", type=int, default=None, help="Number of responses per prompt (passes through sampling_args.n)."
    )
    parser.add_argument("--sampling-arg", action="append", help="Override sampling args with KEY=VALUE (repeatable).")
    parser.add_argument("--sampling-args", help="Sampling arguments as JSON object.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    parser.add_argument(
        "--no-interleave-scoring", "-N", action="store_true", help="Disable interleaving of scoring requests."
    )
    parser.add_argument(
        "--state-columns",
        action="append",
        type=parse_state_columns_arg,
        metavar="COLUMNS",
        help=(
            f"Comma-separated list of state columns to persist (use '{STATE_COLUMNS_SEPARATOR}' between values); repeatable."
        ),
    )
    parser.add_argument(
        "--save-results",
        "--save-dataset",
        "-s",
        dest="save_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save evaluation results to disk (use --no-save-results to disable; accepts legacy --save-dataset alias).",
    )
    parser.add_argument(
        "--save-every",
        "-f",
        type=int,
        default=-1,
        help="Save results every N rollouts when --save-results is set (-1 disables periodic saves).",
    )
    parser.add_argument(
        "--save-to-hf-hub",
        "-H",
        action="store_true",
        default=False,
        help="Push evaluation dataset to the Hugging Face Hub.",
    )
    parser.add_argument("--hf-hub-dataset-name", "-D", default="", help="Custom Hugging Face dataset name when saving.")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved EvalConfig and exit without running."
    )
    return parser


def register_env_options(
    parser: argparse.ArgumentParser,
    env_id: str,
    metadata: Sequence[EnvParam],
) -> dict[str, EnvOptionBinding]:
    reserved_dests = {action.dest for action in parser._actions}
    group = parser.add_argument_group(f"Environment options (ENV={env_id})")
    parser._action_groups.remove(group)
    parser._action_groups.insert(1, group)

    bindings: dict[str, EnvOptionBinding] = {}
    env_actions: list[argparse.Action] = []

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
        env_actions.append(action)
        bindings[action.dest] = EnvOptionBinding(param=param, dest=action.dest, default=action.default)

    if env_actions:
        help_action_index = next(
            (index for index, action in enumerate(parser._actions) if action.dest == "help"),
            None,
        )
        insert_at = (help_action_index + 1) if help_action_index is not None else 0
        for action in reversed(env_actions):
            parser._actions.remove(action)
            parser._actions.insert(insert_at, action)

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


def _print_env_first_error() -> None:
    message = "First argument must be ENV (e.g., medqa). For batch mode, run: medarc-new bench --help."
    print(message, file=sys.stderr)


__all__ = ["run_single_mode"]
