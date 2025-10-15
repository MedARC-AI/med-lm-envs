# based on verifiers XMLParser: Copyright (c) 2025 William Brown - MIT License.

import json
from copy import deepcopy
from typing import Any, Callable, Iterator

from pydantic import BaseModel, ValidationError
from verifiers.parsers.parser import Parser
from verifiers.types import ChatMessage, Messages


class JSONParser(Parser):
    """Parser that extracts structured JSON payloads from model outputs."""

    def __init__(
        self,
        fields: list[str | tuple[str, ...]],
        answer_field: str = "answer",
        model: type[BaseModel] | None = None,
        extract_fn: Callable[[str], str] = lambda x: x,
    ) -> None:
        """
        Initialize the parser with field definitions.

        Each field may be:
          - a string (e.g. "reasoning"): the JSON key is fixed.
          - a tuple of alternatives (e.g. ("code", "answer")): the first element is
            the canonical name used for formatting, and all elements are allowed keys
            when parsing.

        The schema is assumed to have no duplicate names.
        """
        super().__init__(extract_fn=extract_fn)
        self._model: type[BaseModel] | None = model

        self.answer_field = answer_field
        self._fields: list[tuple[str, list[str]]] = []

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

    def _stringify(self, value: Any) -> str:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    def get_fields(self) -> list[str]:
        """Return canonical field names in order (parity with XMLParser)."""
        return [canonical for canonical, _ in self._fields]

    def parse(self, text: str, strip: bool = True) -> dict[str, Any] | BaseModel | None:
        """
        Parse the given JSON string and return a dictionary or Pydantic model.

        For each field defined:
          - If it is a simple field (e.g. 'reasoning'), the output dict will have
            a key 'reasoning' set to the value (or None if missing).
          - If it is defined with alternatives (e.g. ("code", "answer")), the output
            dict will have keys for *each* allowed field name. For example,
            if the schema is ['reasoning', ('code', 'answer')], then both
            `result['code']` and `result['answer']` are always accessible. If a key
            is not found in the JSON, its corresponding value is set to None.

        If a Pydantic model is provided, the parsed dict is validated against it.
        """
        processed_text = self.extract_fn(text)
        obj = self._extract_json_object(processed_text)
        if obj is None:
            return None

        obj_to_use = self._strip_strings(obj) if strip else obj

        if self._model is not None:
            try:
                return self._model.model_validate(obj_to_use)  # type: ignore[union-attr]
            except ValidationError:
                return None
        return obj_to_use

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        # heuristic: Use the final '}' and walk '{' positions backward.
        last_close = text.rfind("}")
        if last_close == -1:
            return None
        opening_indices = [idx for idx, ch in enumerate(text[: last_close + 1]) if ch == "{"]

        if not opening_indices:
            return None

        for open_idx in reversed(opening_indices):
            candidate = text[open_idx : last_close + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    def parse_answer(self, completion: Messages) -> str | None:
        """Extract the last answer from a completion."""
        if isinstance(completion, str):
            parsed = self.parse(completion, strip=True)
            return self._resolve_answer(parsed)

        for msg in reversed(self.get_assistant_messages(completion)):
            content = str(msg["content"])
            parsed = self.parse(content, strip=True)
            answer = self._resolve_answer(parsed)
            if answer is not None:
                return answer
        return None

    def _resolve_answer(self, parsed: dict[str, Any] | BaseModel | None) -> str | None:
        mapping = self._object_to_mapping(parsed)
        if mapping is None:
            return None
        projected = self._project_fields(mapping, strip=True)

        candidates: list[str] = []
        for canonical, alternatives in self._fields:
            if self.answer_field in alternatives:
                candidates = [canonical] + [a for a in alternatives if a != canonical]
                break
        if not candidates:
            candidates = [self.answer_field]

        for key in candidates:
            value = projected.get(key)
            if value is not None:
                return value
        return None

    def get_format_str(self) -> str:
        """
        Return a string that describes the format of the JSON.
        """
        lines: list[str] = ["{\n"]
        for index, (canonical, alternatives) in enumerate(self._fields):
            placeholder = alternatives[0] if len(alternatives) == 1 else " | ".join(alternatives)
            suffix = ",\n" if index < len(self._fields) - 1 else "\n"
            lines.append(f'  "{canonical}": "<{placeholder}>"{suffix}')

        lines.append("}")
        return "\n".join(lines)

    def format(self, **kwargs: Any) -> str:
        """
        Format the provided keyword arguments into a JSON string.

        For fields with alternatives (tuple), the canonical name (the first element)
        is used as the JSON key. The method looks for a provided value using any of the
        allowed names (preferring the canonical if present).

        Example usage:
            parser = JSONParser(['reasoning', ('code', 'answer')])
            formatted_str = parser.format(reasoning="...", code="...")
        """
        payload: dict[str, Any] = {}
        for canonical, alternatives in self._fields:
            key_to_use = None
            value: Any | None = None

            for alt in alternatives:
                if alt in kwargs:
                    key_to_use = canonical
                    value = kwargs[alt]
                    break

            if key_to_use is None:
                raise ValueError(f"Missing value for field set {alternatives}. Provide one of: {alternatives}")
            payload[key_to_use] = value

        return json.dumps(payload, ensure_ascii=False, indent=2)

    def get_format_reward_func(self) -> Callable[[list[ChatMessage]], float]:
        """
        Return a reward function that checks if messages follow the expected format.

        The function checks that:
        - The JSON can be successfully parsed from the message
        - Fields from the schema are present with valid content
        """

        def format_reward_func(completion: list[ChatMessage], **_: Any) -> float:
            messages = self.get_assistant_messages(completion)
            if not messages:
                return 0.0

            scores: list[float] = []
            for msg in messages:
                content = str(msg.get("content", ""))
                processed = self.extract_fn(content)

                obj = self._extract_json_object(processed)
                if obj is None:
                    scores.append(0.0)
                    continue

                score = 0.5
                total_field_sets = len(self._fields)
                coverage_ratio = 0.0

                obj_for_fields: dict[str, Any] | None = deepcopy(obj)
                if self._model is not None:
                    try:
                        inst = self._model.model_validate(obj)  # type: ignore[union-attr]
                        obj_for_fields = inst.model_dump()
                    except ValidationError:
                        obj_for_fields = None

                if obj_for_fields is not None and total_field_sets:
                    projected = self._project_fields(obj_for_fields, strip=True)
                    present_field_sets = 0
                    for _, alternatives in self._fields:
                        if any(projected.get(alt) is not None for alt in alternatives):
                            present_field_sets += 1
                    coverage_ratio = present_field_sets / total_field_sets

                score += 0.5 * coverage_ratio
                scores.append(score)

            return sum(scores) / len(scores)

        return format_reward_func

    def _iter_field_alternatives(self) -> Iterator[list[str]]:
        for _, alternatives in self._fields:
            yield alternatives

    def _strip_strings(self, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            return {key: self._strip_strings(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._strip_strings(item) for item in value]
        return value

    def _object_to_mapping(self, obj: dict[str, Any] | BaseModel | None) -> dict[str, Any] | None:
        if obj is None:
            return None
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if isinstance(obj, dict):
            return obj
        return None

    def _project_fields(self, mapping: dict[str, Any], *, strip: bool) -> dict[str, str | None]:
        projected: dict[str, str | None] = {}
        for _, alternatives in self._fields:
            for alt in alternatives:
                if alt in mapping and mapping[alt] is not None:
                    sval = self._stringify(mapping[alt])
                    projected[alt] = sval.strip() if strip and isinstance(sval, str) else sval
                else:
                    projected[alt] = None
        return projected
