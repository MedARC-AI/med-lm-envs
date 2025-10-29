import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Sequence

from verifiers import setup_logging
from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils.eval_utils import load_endpoints, run_evaluation

from medarc_verifiers.utils.cli_env_args import EnvParam, gather_env_cli_metadata

logger = logging.getLogger(__name__)

PROGRAM_NAME = "medarc-eval"
HELP_FLAGS = {"-h", "--help"}
HEADER_SEPARATOR = ":"
STATE_COLUMNS_SEPARATOR = ","


@dataclass(frozen=True)
class EnvOptionBinding:
    """Track how an environment parameter is bound to an argparse destination."""

    param: EnvParam
    dest: str
    default: Any


class MissingEnvParamError(Exception):
    """Raised when required environment parameters are missing."""


def parse_state_columns_arg(value: str) -> list[str]:
    """Split a comma-delimited state column string into individual entries."""
    columns = [part.strip() for part in value.split(STATE_COLUMNS_SEPARATOR)]
    return [column for column in columns if column]


def build_base_parser(require_env: bool, add_help: bool) -> argparse.ArgumentParser:
    """Create the base parser shared by both parsing passes."""
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        add_help=add_help,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run verifiers evaluations with dynamic environment parameters.\n"
            "After resolving ENV, medarc-eval inspects the environment's load_environment signature and adds matching CLI flags.\n"
            "Use `medarc-eval <env> --help` to list both global and environment-specific options."
        ),
    )
    for group in parser._action_groups:
        if group.title in {"optional arguments", "options"}:
            group.title = "medarc-eval options"
            break
    env_kwargs: Dict[str, Any] = {"metavar": "ENV", "help": "Environment module name"}
    if require_env:
        parser.add_argument("env", **env_kwargs)
    else:
        parser.add_argument("env", nargs="?", **env_kwargs)
    parser.add_argument(
        "--env-args",
        "-a",
        type=json.loads,
        default=None,
        help='Environment arguments as JSON object (e.g., \'{"key": "value", "num": 42}\').',
    )
    parser.add_argument(
        "--env-dir-path",
        "-p",
        type=str,
        default="./environments",
        help="Path to environments directory.",
    )
    parser.add_argument(
        "--endpoints-path",
        "-e",
        type=str,
        default="./configs/endpoints.py",
        help="Path to API endpoints registry.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="gpt-4.1-mini",
        help="Model identifier to evaluate.",
    )
    parser.add_argument(
        "--api-key-var",
        "-k",
        type=str,
        default="OPENAI_API_KEY",
        help="Environment variable name for the API key.",
    )
    parser.add_argument(
        "--api-base-url",
        "-b",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the inference API.",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=None,
        help="Extra HTTP header to send ('Name: Value'). Repeat to provide multiple.",
    )
    parser.add_argument(
        "--num-examples",
        "-n",
        type=int,
        default=5,
        help="Number of examples to evaluate.",
    )
    parser.add_argument(
        "--rollouts-per-example",
        "-r",
        type=int,
        default=3,
        help="Number of rollouts per example.",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=32,
        help="Maximum number of concurrent requests.",
    )
    parser.add_argument(
        "--max-concurrent-generation",
        type=int,
        default=None,
        help="Maximum number of concurrent generation requests.",
    )
    parser.add_argument(
        "--max-concurrent-scoring",
        type=int,
        default=None,
        help="Maximum number of concurrent scoring requests.",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=None,
        help="Maximum tokens to generate (unset to use model defaults).",
    )
    parser.add_argument(
        "--temperature",
        "-T",
        type=float,
        default=None,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sampling-args",
        "-S",
        type=json.loads,
        default=None,
        help=(
            "Sampling arguments as JSON object. Keys override --max-tokens/--temperature when provided. "
            'Example: \'{"enable_thinking": false, "max_tokens": 256}\''
        ),
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Enable verbose logging.",
    )
    parser.add_argument(
        "--no-interleave-scoring",
        "-N",
        default=False,
        action="store_true",
        help="Disable interleaving of scoring requests.",
    )
    parser.add_argument(
        "--state-columns",
        action="append",
        type=parse_state_columns_arg,
        default=None,
        metavar="COLUMNS",
        help="Comma-separated list of state columns to persist; repeatable.",
    )
    parser.add_argument(
        "--save-results",
        "--save-dataset",
        "-s",
        dest="save_results",
        default=False,
        action="store_true",
        help="Save evaluation results to disk (accepts legacy --save-dataset alias).",
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
        default=False,
        action="store_true",
        help="Push evaluation dataset to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-hub-dataset-name",
        "-D",
        type=str,
        default="",
        help="Custom Hugging Face dataset name when saving.",
    )
    parser.add_argument(
        "--print-env-schema",
        action="store_true",
        default=False,
        help="Print JSON metadata for the environment parameters and exit.",
    )
    return parser


def register_env_options(
    parser: argparse.ArgumentParser,
    env_id: str,
    metadata: Sequence[EnvParam],
) -> Dict[str, EnvOptionBinding]:
    """Register environment-specific arguments and return their bindings."""
    reserved_dests = {action.dest for action in parser._actions}
    group = parser.add_argument_group(f"{env_id} options")
    # argparse prints groups in insertion order; move env options ahead of globals.
    parser._action_groups.insert(1, parser._action_groups.pop())

    bindings: Dict[str, EnvOptionBinding] = {}
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
        # When the parameter collides with a global flag reuse the env- prefix.
        if dest in reserved_dests:
            dest = f"env_{dest}"
            option = f"--env-{param.cli_name}"
        kwargs: Dict[str, Any] = {
            "dest": dest,
            "help": param.help,
        }
        if param.choices:
            kwargs["choices"] = param.choices
        if param.kind == "bool":
            kwargs["action"] = argparse.BooleanOptionalAction
            kwargs["default"] = param.default if param.default is not None else None
        elif param.kind == "list":
            kwargs["action"] = "append"
            kwargs["type"] = param.element_type
            kwargs["default"] = None
        else:
            if param.argparse_type is not None:
                kwargs["type"] = param.argparse_type
            kwargs["default"] = param.default
        action = group.add_argument(option, **kwargs)
        env_actions.append(action)
        bindings[action.dest] = EnvOptionBinding(
            param=param,
            dest=action.dest,
            default=action.default,
        )

    if env_actions:
        # Reorder usage string so env-specific options precede globals.
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
) -> Dict[str, Any]:
    """Collect CLI-provided environment values, ignoring defaults."""
    explicit: Dict[str, Any] = {}

    for binding in bindings.values():
        value = getattr(namespace, binding.dest)
        param = binding.param
        default = binding.default

        if param.kind == "list":
            if value is not None:
                explicit[param.name] = value
            continue

        if param.kind == "bool":
            if param.required or default is None or value != default:
                explicit[param.name] = value
            continue

        if value is None:
            continue

        if param.required or default is None or value != default:
            explicit[param.name] = value

    return explicit


def merge_env_args(explicit: Mapping[str, Any], json_args: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge JSON arguments with CLI overrides (CLI wins on conflict)."""
    merged = dict(json_args)
    for key, value in explicit.items():
        if key in merged and merged[key] != value:
            logger.debug(
                "CLI option '%s' overriding JSON value '%s' with '%s'.",
                key,
                merged[key],
                value,
            )
        merged[key] = value
    return merged


def ensure_required_params(
    metadata: Sequence[EnvParam],
    explicit: Mapping[str, Any],
    json_args: Mapping[str, Any],
) -> None:
    """Ensure all required parameters are supplied via CLI or JSON."""
    missing = [
        param.name
        for param in metadata
        if param.required and param.name not in explicit and param.name not in json_args
    ]
    if missing:
        joined = ", ".join(missing)
        raise MissingEnvParamError(f"Missing required environment arguments: {joined}")


def _serialize_value(value: Any) -> Any:
    """Convert parameter defaults/choices into JSON-friendly values."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return repr(value)


def build_env_schema(env_id: str, metadata: Sequence[EnvParam]) -> Dict[str, Any]:
    """Build a JSON-serializable schema describing environment parameters."""
    params = []
    for param in metadata:
        param_entry = {
            "name": param.name,
            "cli": f"--{param.cli_name}",
            "kind": param.kind,
            "required": param.required,
            "supports_cli": param.supports_cli,
            "default": _serialize_value(param.default),
            "choices": _serialize_value(list(param.choices)) if param.choices else None,
            "help": param.help,
            "unsupported_reason": param.unsupported_reason,
        }
        params.append(param_entry)
    return {"env": env_id, "parameters": params}


def print_env_schema(env_id: str, metadata: Sequence[EnvParam]) -> None:
    """Print the environment schema to stdout."""
    schema = build_env_schema(env_id, metadata)
    json_output = json.dumps(schema, indent=2, sort_keys=True)
    print(json_output)


def build_headers(header_values: Iterable[str] | None) -> Dict[str, str]:
    """Convert repeated --header flags into a single mapping."""
    headers: Dict[str, str] = {}
    if not header_values:
        return headers
    for item in header_values:
        if HEADER_SEPARATOR not in item:
            raise ValueError(f"--header must be 'Name: Value', got: {item!r}")
        name, value = item.split(HEADER_SEPARATOR, 1)
        name, value = name.strip(), value.strip()
        if not name:
            raise ValueError("--header name cannot be empty.")
        headers[name] = value
    return headers


def _coerce_json_mapping(value: Any, flag: str) -> Dict[str, Any]:
    """Validate that a JSON argument decoded to an object."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = f"{flag} must be a JSON object."
        raise ValueError(msg)
    return value


def _merge_sampling_args(
    sampling_args: Mapping[str, Any],
    max_tokens: int | None,
    temperature: float | None,
) -> Dict[str, Any]:
    """Combine sampling args with scalar overrides."""
    merged = dict(sampling_args)
    if "max_tokens" not in merged:
        merged["max_tokens"] = max_tokens
    if temperature is not None and "temperature" not in merged:
        merged["temperature"] = temperature
    return merged


def _flatten_state_columns(values: Iterable[list[str]] | None) -> list[str]:
    """Flatten repeated --state-columns arguments into a single list."""
    if not values:
        return []
    flattened: list[str] = []
    for group in values:
        flattened.extend(group)
    return flattened


def _resolve_endpoint_selection(
    model: str,
    endpoints: Mapping[str, Mapping[str, str]],
    default_key_var: str,
    default_base_url: str,
) -> tuple[str, str, str]:
    """Resolve model alias and endpoint configuration."""
    if model in endpoints:
        entry = endpoints[model]
        resolved_model = entry.get("model", model)
        api_key_var = entry.get("key", default_key_var)
        api_base_url = entry.get("url", default_base_url)
        logger.debug(
            "Resolved model '%s' using endpoint registry entry '%s'.", model, resolved_model
        )
        return resolved_model, api_key_var, api_base_url

    logger.debug(
        "Model '%s' not found in endpoint registry; using CLI-specified API config.",
        model,
    )
    return model, default_key_var, default_base_url


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint implementing two-phase parsing and delegation."""
    args_list = list(argv) if argv is not None else sys.argv[1:]

    # Phase 1: parse known globals to discover which environment to inspect.
    base_parser = build_base_parser(require_env=False, add_help=False)
    base_args, _ = base_parser.parse_known_args(args_list)

    help_requested = any(token in HELP_FLAGS for token in args_list)

    if base_args.env is None:
        if help_requested:
            help_parser = build_base_parser(require_env=False, add_help=True)
            help_parser.print_help()
            return 0
        help_parser = build_base_parser(require_env=False, add_help=True)
        help_parser.error("Environment argument is required (e.g., 'medarc-eval medqa').")

    env_id = base_args.env
    try:
        metadata = gather_env_cli_metadata(env_id)
    except ImportError as exc:
        help_parser = build_base_parser(require_env=True, add_help=True)
        help_parser.error(str(exc))

    if base_args.print_env_schema and not help_requested:
        print_env_schema(env_id, metadata)
        return 0

    # Phase 2: build a parser that includes the env-specific flags.
    parser = build_base_parser(require_env=True, add_help=True)
    bindings = register_env_options(parser, env_id, metadata)

    if help_requested:
        try:
            parser.parse_args(args_list)
        except SystemExit as exc:  # pragma: no cover - argparse handles exit
            return int(exc.code)
        return 0

    try:
        args = parser.parse_args(args_list)
    except SystemExit as exc:
        return int(exc.code)

    try:
        json_env_args = _coerce_json_mapping(args.env_args, "--env-args")
    except ValueError as exc:
        parser.error(str(exc))

    explicit_cli_args = extract_env_cli_args(args, bindings)

    try:
        ensure_required_params(metadata, explicit_cli_args, json_env_args)
    except MissingEnvParamError as exc:
        parser.error(str(exc))

    merged_env_args = merge_env_args(explicit_cli_args, json_env_args)

    try:
        sampling_args = _coerce_json_mapping(args.sampling_args, "--sampling-args")
    except ValueError as exc:
        parser.error(str(exc))

    merged_sampling_args = _merge_sampling_args(
        sampling_args,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    try:
        headers = build_headers(args.header)
    except ValueError as exc:
        parser.error(str(exc))

    state_columns = _flatten_state_columns(args.state_columns)

    # Configure logging now that we know whether verbose output is requested.
    setup_logging("DEBUG" if args.verbose else "INFO")

    try:
        endpoints = load_endpoints(args.endpoints_path)
    except Exception as exc:  # pragma: no cover - load_endpoints already logs details
        logger.error("Failed to load endpoints registry: %s", exc)
        return 1

    resolved_model, api_key_var, api_base_url = _resolve_endpoint_selection(
        args.model,
        endpoints,
        args.api_key_var,
        args.api_base_url,
    )

    client_config = ClientConfig(
        api_key_var=api_key_var,
        api_base_url=api_base_url,
        extra_headers=headers,
    )

    eval_config = EvalConfig(
        # environment
        env_id=args.env,
        env_args=merged_env_args,
        env_dir_path=args.env_dir_path,
        # evaluation
        model=resolved_model,
        client_config=client_config,
        sampling_args=merged_sampling_args,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        max_concurrent=args.max_concurrent,
        max_concurrent_generation=args.max_concurrent_generation,
        max_concurrent_scoring=args.max_concurrent_scoring,
        interleave_scoring=not args.no_interleave_scoring,
        # logging
        print_results=True,
        verbose=args.verbose,
        # saving
        state_columns=state_columns or None,
        save_results=args.save_results,
        save_every=args.save_every,
        save_to_hf_hub=args.save_to_hf_hub,
        hf_hub_dataset_name=args.hf_hub_dataset_name or None,
    )

    logger.debug("Evaluation config: %s", eval_config.model_dump_json(indent=2))

    try:
        asyncio.run(run_evaluation(eval_config))
    except KeyboardInterrupt:
        logger.error("Evaluation interrupted by user.")
        return 1
    except Exception as exc:  # pragma: no cover - bubbled errors logged below
        if args.verbose:
            logger.exception("Evaluation failed.")
        else:
            logger.error("Evaluation failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
