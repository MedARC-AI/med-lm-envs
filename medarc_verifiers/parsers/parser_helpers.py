from collections.abc import Mapping
from typing import Any


def get_parsed_field(result: object, field: str, default: Any = None) -> Any:
    """Extract a field from parsed output across dict/namespace types."""
    if result is None:
        return default
    if isinstance(result, Mapping):
        return result.get(field, default)
    if hasattr(result, field):
        return getattr(result, field)
    return default
