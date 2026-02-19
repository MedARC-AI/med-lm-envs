from __future__ import annotations

from typing import TypeAlias

from verifiers.types import Message as VerifiersMessage
from verifiers.types import Messages as VerifiersMessages


Message: TypeAlias = VerifiersMessage
# In 0.1.10, `Messages` already includes `str`; in 0.1.11 it does not.
# Keep a single compatibility alias that works for both versions.
Messages: TypeAlias = VerifiersMessages | str
