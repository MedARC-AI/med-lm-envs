# based on verifiers XMLParser: Copyright (c) 2025 William Brown - MIT License.

from typing import Any, Callable

from verifiers.parsers.parser import Parser
from verifiers.parsers.xml_parser import XMLParser as BaseXMLParser
from verifiers.types import Messages


class XMLParser(BaseXMLParser):
    """
    XMLParser that can parse either a raw string or a sequence of Messages.

    For Messages, it mirrors the logic of the base `parse_answer` by scanning
    assistant messages in reverse and returning the first parsed object that
    contains any of the configured XML fields.
    """

    def __init__(
        self,
        fields: list[str | tuple[str, ...]],
        answer_field: str = "answer",
        extract_fn: Callable[[str], str] = lambda x: x,
    ):
        # Re-implement BaseXMLParser.__init__ without the warning on "think".
        Parser.__init__(self, extract_fn=extract_fn)
        self._fields: list[tuple[str, list[str]]] = []

        self.answer_field = answer_field
        seen = set()
        for field in fields:
            if isinstance(field, str):
                canonical = field
                alternatives = [field]
            elif isinstance(field, tuple):
                if not field:
                    raise ValueError("Field tuple cannot be empty.")
                canonical = field[0]
                if not all(isinstance(alt, str) for alt in field):
                    raise TypeError("All alternatives in a tuple must be strings.")
                alternatives = list(field)
            else:
                raise TypeError("Each field must be a string or a tuple of strings.")
            if canonical in seen:
                raise ValueError(f"Duplicate field name: {canonical}")
            seen.add(canonical)
            self._fields.append((canonical, alternatives))

    def parse(self, completion: Messages | str, strip: bool = True, last: bool = False) -> Any:
        if isinstance(completion, str):
            return super().parse(completion, strip=strip, last=last)

        messages = self.get_assistant_messages(completion)
        for msg in reversed(messages):
            content = str(msg.get("content", ""))
            parsed = super().parse(content, strip=strip, last=last)
            if parsed is None:
                continue

            if self._has_any_field(parsed):
                return parsed
        return None

    def parse_answer(self, completion: Messages | str) -> str | None:
        """Extract the last answer from a completion.

        Overrides upstream to always use ``last=True`` when parsing chat messages.
        Qwen 3.5 reasoning models echo ``<answer>`` tags inside their thinking
        blocks, so we must take the **last** match to get the real answer.
        """
        if isinstance(completion, str):
            parsed = self.parse(completion, last=True)
            if parsed and hasattr(parsed, self.answer_field) and getattr(parsed, self.answer_field) is not None:
                return getattr(parsed, self.answer_field)
        else:
            for msg in reversed(self.get_assistant_messages(completion)):
                content = str(msg.get("content", ""))
                parsed = self.parse(content, last=True)
                if parsed and hasattr(parsed, self.answer_field) and getattr(parsed, self.answer_field) is not None:
                    return getattr(parsed, self.answer_field)
        return None

    def _has_any_field(self, parsed: Any) -> bool:
        for _, alternatives in self._fields:
            for alt in alternatives:
                if hasattr(parsed, alt) and getattr(parsed, alt) is not None:
                    return True
        return False
