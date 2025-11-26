import enum
import importlib
import inspect
import logging
import types
from dataclasses import dataclass
from typing import Annotated, Any, Dict, Optional, Tuple, Union, get_args, get_origin, get_type_hints

from docstring_parser import ParseError
from docstring_parser import parse as parse_docstring

logger = logging.getLogger(__name__)

_EMPTY = inspect._empty
_NONE_TYPE = type(None)


@dataclass(slots=True, frozen=True)
class EnvParam:
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
    kind: str
    argparse_type: type | None
    choices: Tuple[Any, ...] | None
    action: str | None
    is_list: bool
    element_type: type | None
    unsupported_reason: str | None


def gather_env_cli_metadata(env_id: str) -> list[EnvParam]:
    """Collect parameter metadata for the given environment id."""
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
                annotation=param.annotation,
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
    """Resolve the load_environment callable for an environment id."""
    module_name = env_id.replace("-", "_")
    # Try common import patterns so environments can be addressed with or without package prefixes.
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
        except Exception as exc:
            last_error = exc

    msg = f"Unable to locate load_environment for env '{env_id}'. Tried: {', '.join(candidates)}"
    if last_error:
        raise ImportError(msg) from last_error
    raise ImportError(msg)


def _safe_get_type_hints(load_fn) -> Dict[str, Any]:
    """Resolve type hints, tolerating missing imports or forward references."""
    try:
        return get_type_hints(load_fn, include_extras=True)
    except Exception as exc:
        logger.debug("Failed to resolve type hints via get_type_hints: %s", exc)
        return {}


def _build_docstring_param_map(docstring: str | None) -> dict[str, str]:
    """Parse the loader docstring and map parameter names to their descriptions."""
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
    """Choose the help text shown for a parameter."""
    if doc_help:
        return doc_help

    if spec.unsupported_reason:
        return f"{name} requires --env-args (reason: {spec.unsupported_reason})."

    if default is _EMPTY:
        return f"Required {spec.kind} parameter."

    return f"Defaults to {default!r} ({spec.kind})."


def _infer_argparse_spec(annotation: Any, default: Any) -> ArgSpec:
    """Infer argparse configuration details from a parameter annotation."""
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
        union_args = get_args(normalized)
        enum_args = [arg for arg in union_args if _is_enum(arg)]
        non_enum_args = [arg for arg in union_args if not _is_enum(arg)]
        if len(enum_args) == 1 and all(arg is str for arg in non_enum_args):
            enum_cls = enum_args[0]
            choices = tuple(member.value for member in enum_cls)
            return ArgSpec("enum", None, choices, None, False, None, None)
        reason = "non-optional union unsupported"
        return ArgSpec("unsupported", None, None, None, False, None, reason)

    reason = f"unsupported annotation {normalized!r}"
    return ArgSpec("unsupported", None, None, None, False, None, reason)


def _infer_list_spec(annotation: Any) -> ArgSpec:
    """Infer argparse configuration for list-style parameters."""
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
