"""Environment argument helpers for the unified CLI."""

from __future__ import annotations

import enum
import importlib
import inspect
import logging
import types
from dataclasses import dataclass
from typing import (
    Annotated,
    Any,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from docstring_parser import ParseError
from docstring_parser import parse as parse_docstring

logger = logging.getLogger(__name__)

HEADER_SEPARATOR = ":"
_EMPTY = inspect._empty
_NONE_TYPE = type(None)


class MissingEnvParamError(Exception):
    """Raised when required environment parameters are missing."""


@dataclass(slots=True, frozen=True)
class EnvParam:
    """Metadata describing a parameter accepted by an environment loader."""

    name: str
    cli_name: str
    kind: str
    default: Any
    required: bool
    help: str
    annotation: Any
    argparse_type: type | None
    choices: Tuple[Any, ...] | None
    action: str | None
    is_list: bool
    element_type: type | None
    unsupported_reason: str | None

    @property
    def supports_cli(self) -> bool:
        return self.unsupported_reason is None


@dataclass(slots=True, frozen=True)
class ArgSpec:
    """Intermediate specification inferred for a loader parameter."""

    kind: str
    argparse_type: type | None
    choices: Tuple[Any, ...] | None
    action: str | None
    is_list: bool
    element_type: type | None
    unsupported_reason: str | None


def build_headers(values: Iterable[str] | None) -> dict[str, str]:
    """Convert repeated header flags into a dictionary."""
    headers: dict[str, str] = {}
    if not values:
        return headers
    for item in values:
        if HEADER_SEPARATOR not in item:
            msg = f"--header must be 'Name: Value', got: {item!r}"
            raise ValueError(msg)
        name, value = item.split(HEADER_SEPARATOR, 1)
        name, value = name.strip(), value.strip()
        if not name:
            raise ValueError("--header name cannot be empty.")
        headers[name] = value
    return headers


def ensure_required_params(
    metadata: Sequence[EnvParam],
    explicit: Mapping[str, Any],
    json_args: Mapping[str, Any],
) -> None:
    """Ensure required parameters are provided by CLI overrides or merged args."""

    def _allows_none(annotation: Any) -> bool:
        """Return True if the type annotation allows None (Optional/Union[..., None])."""
        if annotation is None:
            return False
        if annotation is Any:
            return True
        if annotation is Optional:
            return True
        origin = get_origin(annotation)
        if origin is Annotated:
            args = get_args(annotation)
            return _allows_none(args[0] if args else None)
        if annotation is _NONE_TYPE:
            return True
        if origin in {types.UnionType, Union}:
            return any(arg is _NONE_TYPE for arg in get_args(annotation))
        return False

    missing: list[str] = []
    for param in metadata:
        if not param.required:
            continue

        allows_none = _allows_none(getattr(param, "annotation", None))
        provided = False
        for source in (explicit, json_args):
            if param.name not in source:
                continue
            value = source[param.name]
            if value is None and not allows_none:
                # Treat explicit `null`/`None` as "not provided" for required params unless the
                # annotation allows None. We don't error immediately because the other source
                # (explicit vs json_args) may still provide a valid non-None value; we'll raise
                # MissingEnvParamError after checking both sources if neither does.
                continue
            provided = True
            break

        if not provided:
            missing.append(param.name)
    if missing:
        joined = ", ".join(sorted(missing))
        raise MissingEnvParamError(f"Missing required environment arguments: {joined}")


def validate_env_arg_values(
    env_id: str,
    env_args: Mapping[str, Any],
    metadata: Sequence[EnvParam],
) -> None:
    """Validate env arg types and choices for the provided values."""
    if not env_args:
        return

    param_map = {param.name: param for param in metadata if getattr(param, "supports_cli", True)}
    for name, value in env_args.items():
        param = param_map.get(name)
        if param is None:
            continue
        _validate_env_arg_value(env_id, param, value)


def validate_env_args_or_raise(
    env_id: str,
    env_args: Mapping[str, Any],
    metadata: Sequence[EnvParam] | None = None,
    *,
    metadata_cache: dict | None = None,
    allow_unknown: bool = False,
    enforce_required: bool = False,
) -> None:
    """Validate env args using environment metadata.

    This is used in two phases with different strictness levels:

    Phase 1 (config load time, enforce_required=False):
        - Check for unknown parameters (typos in config)
        - Validate types and choices for provided values
        - Allow missing required parameters (configs might be partial)

    Phase 2 (execution time, enforce_required=True):
        - All Phase 1 checks PLUS
        - Enforce that required parameters are present
        - Happens after CLI overrides are merged

    Args:
        env_id: Environment identifier for error messages
        env_args: Arguments to validate
        metadata: Pre-loaded environment metadata (loads if None)
        metadata_cache: Cache for loaded metadata
        allow_unknown: If True, don't error on unknown parameter names
        enforce_required: If True, error on missing required parameters

    Raises:
        ValueError: For unknown params, type mismatches, or missing required params

    Note: Loads metadata when not provided (with optional cache), tolerating
    ImportError by logging a warning and returning early.
    """
    if metadata is None:
        try:
            from .endpoint_utils import load_env_metadata  # local import to avoid circular dependency

            metadata = load_env_metadata(env_id, cache=metadata_cache)  # type: ignore[arg-type]
        except ImportError as exc:
            logger.warning("Skipping env_args validation for '%s': %s", env_id, exc)
            return
    if not metadata:
        return

    param_names = {param.name for param in metadata if getattr(param, "supports_cli", True)}
    unknown = sorted(set(env_args) - param_names)
    if unknown and not allow_unknown:
        valid_params = ", ".join(sorted(param_names)) or "<none>"
        raise ValueError(
            f"Environment '{env_id}' env_args contain unknown parameters: {', '.join(unknown)}. Valid parameters: {valid_params}."
        )
    if enforce_required:
        ensure_required_params(metadata, {}, env_args)
    validate_env_arg_values(env_id, env_args, metadata)


def merge_env_args(
    env_id: str | None,
    *,
    sources: Sequence[Mapping[str, Any]],
    metadata: Sequence[EnvParam] | None = None,
    metadata_loader=None,
    metadata_cache: Mapping[str, Sequence[EnvParam]] | None = None,
    allow_unknown: bool = False,
    enforce_required: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Merge env args from multiple sources and optionally validate."""
    merged: dict[str, Any] = {}
    for source in sources:
        if not source:
            continue
        if verbose and env_id is not None:
            overridden = {k: f"{merged[k]} → {source[k]}" for k in merged.keys() & source.keys()}
            new_keys = sorted(set(source) - set(merged))
            if overridden:
                logger.debug("Env args overriding for %s: %s", env_id, overridden)
            if new_keys:
                logger.debug("Env args adding for %s: %s", env_id, new_keys)
        merged.update(source)

    if metadata is None and metadata_loader is not None:
        if env_id is None:
            raise ValueError("env_id is required to validate env args with a loader.")
        try:
            metadata = metadata_loader(env_id, cache=metadata_cache)
        except TypeError:
            metadata = metadata_loader(env_id)
        except ImportError as exc:
            logger.warning("Skipping env_args validation for '%s': %s", env_id, exc)
            return merged

    if metadata:
        if env_id is None:
            raise ValueError("env_id is required to validate env args.")
        validate_env_args_or_raise(
            env_id,
            merged,
            metadata,
            metadata_cache=metadata_cache,
            allow_unknown=allow_unknown,
            enforce_required=enforce_required,
        )

    return merged


def merge_env_args_with_validation(
    env_id: str,
    *,
    base_args: Mapping[str, Any],
    override_args: Mapping[str, Any] | None,
    metadata: Sequence[EnvParam] | None = None,
    metadata_loader=None,
    metadata_cache: Mapping[str, Sequence[EnvParam]] | None = None,
    allow_unknown: bool = False,
    enforce_required: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Backward-compatible wrapper for merge_env_args."""
    return merge_env_args(
        env_id,
        sources=[base_args, override_args or {}],
        metadata=metadata,
        metadata_loader=metadata_loader,
        metadata_cache=metadata_cache,
        allow_unknown=allow_unknown,
        enforce_required=enforce_required,
        verbose=verbose,
    )


def gather_env_cli_metadata(env_id: str) -> list[EnvParam]:
    """Collect parameter metadata for the given environment identifier."""
    load_fn = _resolve_load_environment(env_id)
    signature = inspect.signature(load_fn, follow_wrapped=True)
    type_hints = _safe_get_type_hints(load_fn)
    doc_map = _build_docstring_param_map(inspect.getdoc(load_fn))

    metadata: list[EnvParam] = []
    for param in signature.parameters.values():
        if param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}:
            logger.debug("Skipping variadic parameter '%s' for env '%s'.", param.name, env_id)
            continue

        annotation = type_hints.get(param.name, param.annotation)
        spec = _infer_argparse_spec(annotation, param.default)
        help_text = _select_help_text(param.name, doc_map.get(param.name), spec, param.default)
        metadata.append(
            EnvParam(
                name=param.name,
                cli_name=param.name.replace("_", "-"),
                kind=spec.kind,
                default=None if param.default is _EMPTY else param.default,
                required=param.default is _EMPTY and spec.unsupported_reason is None,
                help=help_text,
                annotation=annotation,
                argparse_type=spec.argparse_type,
                choices=spec.choices,
                action=spec.action,
                is_list=spec.is_list,
                element_type=spec.element_type,
                unsupported_reason=spec.unsupported_reason,
            )
        )

    return metadata


def _resolve_load_environment(env_id: str):
    """Resolve the load_environment callable for a given env id."""
    module_name = env_id.replace("-", "_")
    candidates = [
        module_name,
        f"{module_name}.{module_name}",
        f"environments.{module_name}",
        f"environments.{module_name}.{module_name}",
    ]

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            module = importlib.import_module(candidate)
            load_fn = getattr(module, "load_environment", None)
            if load_fn is None:
                raise AttributeError(f"Module '{candidate}' does not expose load_environment.")
            return load_fn
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    msg = f"Unable to locate load_environment for env '{env_id}'. Tried: {', '.join(candidates)}"
    if last_error:
        raise ImportError(msg) from last_error
    raise ImportError(msg)


def _safe_get_type_hints(load_fn) -> dict[str, Any]:
    """Resolve type hints while tolerating missing imports or forward refs."""
    try:
        return get_type_hints(load_fn, include_extras=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to resolve type hints via get_type_hints: %s", exc)
        return {}


def _build_docstring_param_map(docstring: str | None) -> dict[str, str]:
    """Parse the loader docstring and map parameter names to descriptions."""
    if not docstring:
        return {}

    try:
        parsed = parse_docstring(docstring)
    except (ParseError, ValueError) as exc:
        logger.debug("Failed to parse environment docstring: %s", exc)
        return {}

    result: dict[str, str] = {}
    for param in parsed.params:
        description = (param.description or "").strip()
        if description:
            result[param.arg_name] = description
    return result


def _select_help_text(
    name: str,
    doc_help: str | None,
    spec: ArgSpec,
    default: Any,
) -> str:
    if doc_help:
        return doc_help

    if spec.unsupported_reason:
        return f"{name} requires --env-args (reason: {spec.unsupported_reason})."

    if default is _EMPTY:
        return f"Required {spec.kind} parameter."

    return f"Defaults to {default!r} ({spec.kind})."


def _infer_argparse_spec(annotation: Any, default: Any) -> ArgSpec:
    """Infer argparse-related metadata for a parameter annotation."""
    normalized = _normalize_annotation(annotation)
    normalized, _ = _strip_optional(normalized)

    if normalized is None:
        inferred = _infer_from_default(default)
        if inferred is not None:
            return inferred
        reason = "no annotation or default"
        logger.debug("Falling back to --env-args for parameter without type info (%s).", reason)
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    if _is_literal(normalized):
        values = tuple(get_args(normalized))
        elem_type = type(values[0]) if values else str
        return ArgSpec("literal", elem_type, values, None, False, None, None)

    origin = get_origin(normalized)

    if origin is list:
        return _infer_list_spec(normalized)

    if _is_enum(normalized):
        choices = tuple(member.value for member in normalized)
        return ArgSpec("enum", None, choices, None, False, None, None)

    if normalized is bool:
        return ArgSpec("bool", None, None, "BooleanOptionalAction", False, None, None)

    if normalized in {int, float, str}:
        return ArgSpec(normalized.__name__, normalized, None, None, False, None, None)

    if normalized is Any:
        reason = "Any annotation unsupported"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    if origin in {dict, set, frozenset, tuple}:
        reason = f"{origin.__name__} unsupported"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    if _is_union(normalized):
        args = get_args(normalized)
        enum_args = [arg for arg in args if _is_enum(arg)]
        non_enum_args = [arg for arg in args if not _is_enum(arg)]
        if len(enum_args) == 1 and all(arg is str for arg in non_enum_args):
            enum_cls = enum_args[0]
            choices = tuple(member.value for member in enum_cls)
            return ArgSpec("enum", None, choices, None, False, None, None)
        reason = "non-optional union unsupported"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    reason = f"unsupported annotation {normalized!r}"
    return ArgSpec("unsupported", None, None, None, False, None, reason)


def _infer_list_spec(annotation: Any) -> ArgSpec:
    """Infer metadata for list-like annotations."""
    args = get_args(annotation)
    if len(args) != 1:
        reason = "list requires a single element annotation"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    elem_annotation = args[0]
    elem_spec = _infer_argparse_spec(elem_annotation, _EMPTY)
    if elem_spec.unsupported_reason:
        reason = f"list element unsupported ({elem_spec.unsupported_reason})"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    if elem_spec.is_list:
        reason = "nested list unsupported"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    if elem_spec.kind == "bool":
        reason = "list[bool] unsupported"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    return ArgSpec(
        "list",
        elem_spec.argparse_type,
        elem_spec.choices,
        "append",
        True,
        elem_spec.argparse_type,
        None,
    )


def _infer_from_default(default: Any) -> ArgSpec | None:
    """Fallback inference for parameters that only specify defaults."""
    if default is _EMPTY:
        return None

    if isinstance(default, bool):
        return ArgSpec("bool", None, None, "BooleanOptionalAction", False, None, None)

    if isinstance(default, int):
        return ArgSpec("int", int, None, None, False, None, None)

    if isinstance(default, float):
        return ArgSpec("float", float, None, None, False, None, None)

    if isinstance(default, str):
        return ArgSpec("str", str, None, None, False, None, None)

    if isinstance(default, list):
        if not default:
            reason = "list default requires annotation"
            return ArgSpec("unsupported", None, None, None, False, None, reason)
        elem_type = type(default[0])
        if elem_type not in {int, float, str}:
            reason = f"list default element type {elem_type.__name__} unsupported"
            return ArgSpec("unsupported", None, None, None, False, None, reason)
        if not all(isinstance(item, elem_type) for item in default):
            reason = "list default elements must share a type"
            return ArgSpec("unsupported", None, None, None, False, None, reason)
        return ArgSpec("list", elem_type, None, "append", True, elem_type, None)

    return None


def _normalize_annotation(annotation: Any) -> Any:
    """Normalize annotations so downstream helpers can reason about them."""
    if annotation is _EMPTY:
        return None

    origin = get_origin(annotation)
    if origin is types.GenericAlias and hasattr(annotation, "__args__"):
        return annotation

    if origin is Annotated:
        args = get_args(annotation)
        return args[0] if args else None

    return annotation


def _strip_optional(annotation: Any) -> tuple[Any, bool]:
    """Remove Optional/Union[None] wrappers, returning the core annotation."""
    if annotation is None:
        return (None, False)

    if annotation is _NONE_TYPE:
        return (None, True)

    origin = get_origin(annotation)
    if origin in {types.UnionType, Union}:
        args = [arg for arg in get_args(annotation) if arg is not _NONE_TYPE]
        if len(args) == 1:
            return (args[0], True)
        return (annotation, False)

    if annotation is Optional:
        return (None, True)

    return (annotation, False)


def _is_literal(annotation: Any) -> bool:
    """Return True when annotation is typing.Literal."""
    origin = get_origin(annotation)
    return origin is not None and getattr(origin, "__name__", "") == "Literal"


def _is_union(annotation: Any) -> bool:
    """Return True when annotation models a non-optional Union."""
    origin = get_origin(annotation)
    return origin in {types.UnionType, Union}


def _is_enum(annotation: Any) -> bool:
    """Return True when annotation is an Enum subclass."""
    return inspect.isclass(annotation) and issubclass(annotation, enum.Enum)


def _validate_env_arg_value(env_id: str, param: EnvParam, value: Any) -> None:
    if value is None:
        return

    if param.choices:
        if value not in param.choices:
            allowed = ", ".join(map(repr, param.choices))
            raise ValueError(f"Environment '{env_id}' env_args.{param.name} must be one of: {allowed}.")

    if param.is_list:
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                f"Environment '{env_id}' env_args.{param.name} must be a list, got {type(value).__name__}."
            )
        if param.element_type is not None:
            for item in value:
                if not isinstance(item, param.element_type):
                    raise ValueError(
                        f"Environment '{env_id}' env_args.{param.name} elements must be {param.element_type.__name__}."
                    )
        return

    if param.action == "BooleanOptionalAction" or param.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(
                f"Environment '{env_id}' env_args.{param.name} must be a boolean, got {type(value).__name__}."
            )
        return

    expected_type = param.argparse_type
    if expected_type and not isinstance(value, expected_type):
        raise ValueError(
            f"Environment '{env_id}' env_args.{param.name} must be {expected_type.__name__}, got {type(value).__name__}."
        )


__all__ = [
    "HEADER_SEPARATOR",
    "MissingEnvParamError",
    "EnvParam",
    "ArgSpec",
    "build_headers",
    "ensure_required_params",
    "gather_env_cli_metadata",
    "validate_env_arg_values",
    "validate_env_args_or_raise",
    "merge_env_args",
    "merge_env_args_with_validation",
]
