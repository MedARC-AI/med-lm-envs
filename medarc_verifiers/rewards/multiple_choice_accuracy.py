"""MCQ raw-text grading with tail-authoritative long-response handling."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional


# Responses longer than this switch into tail long-mode behavior.
LONG_RESPONSE_THRESHOLD_CHARS = 4_000
# Long-mode explicit-answer and answer-text scans are limited to this terminal slice.
TERMINAL_WINDOW_CHARS = 4_000
# The looser last-token fallback only inspects this shorter tail inside the terminal slice.
STRONG_TAIL_WINDOW_CHARS = 2_000
# Local ambiguity checks can look this far backward from a candidate.
LOCAL_CONTEXT_BEFORE_CHARS = 160
# Local ambiguity checks can look this far forward from a candidate.
LOCAL_CONTEXT_AFTER_CHARS = 240
# Tail-choice fallback is only allowed when the trailing segment is this short or shorter.
TAIL_CHOICE_MAX_WORDS = 16

_UNICODE_PUNCT_TRANSLATIONS = str.maketrans(
    {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
_LIKELY_TEX_RE = re.compile(r"\\[A-Za-z]+|\\[$\\()\\[\\]{}]|[$]")
_THINK_OPEN_RE = re.compile(r"<\s*think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", re.IGNORECASE)
_ANSWER_TAG_RE = re.compile(r"</?\s*answer\s*>", re.IGNORECASE)

# Any standalone option-like token. This is intentionally broad and gets filtered by
# local ambiguity checks before it can count as a chosen answer.
_OPTION_TOKEN_RE = re.compile(r"(?<![\w+\-/])(?P<opt>[A-Za-z]|\d{1,2})(?![\w+\-/])", re.IGNORECASE)
# Anchored cues that usually indicate the model is committing to a final answer.
_ANCHOR_RE = re.compile(
    r"(?P<label>\bfinal\s+answer\b|\bthe\s+correct\s+answer\b|\bcorrect\s+answer\b|\bthe\s+answer\b|\banswer\b|\btherefore\b|\bi\s+choose\b)"
    r"\s*[:\-]?\s*(?:is\s+)?(?P<neg>not\s+|isn't\s+|isnt\s+)?(?:(?:option|choice)\s+)?"
    r"(?:[*_`~]+\s*)*(?:\\boxed\{\s*)?[\(\[\{<【]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*"
    r"[\)\]\}>】]*\s*(?:\}\s*)?(?:[*_`~]+\s*)?(?![\w+\-/])",
    re.IGNORECASE,
)
# Option-led lines like "B. Answer text" or "**(2)** Answer text".
_LEADING_OPTION_RE = re.compile(
    r"^\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?"
    r"(?:[*_`~]+\s*)?(?:\\boxed\{\s*)?[\(\[\{<【]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*"
    r"[\)\]\}>】]*\s*(?:\}\s*)?\s*(?:[).:\-])?\s*(?:[*_`~]+\s*)*\s+(?P<rest>.+?)\s*$",
    re.IGNORECASE,
)
_SENTENCE_OPTION_START_RE = re.compile(
    r"^\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?"
    r"(?:[*_`~]+\s*)*(?:\\boxed\{\s*)?[\(\[\{<【]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*"
    r"[\)\]\}>】]*\s*(?:\}\s*)?\s*(?:[).:\-])",
    re.IGNORECASE,
)
_EXACT_OPTION_RE = re.compile(r"^\s*(?:(?:option|choice)\s+)?(?P<opt>[A-Za-z]|\d{1,2})\s*[.!?]?\s*$", re.IGNORECASE)
_TERMINAL_OPTION_LINE_RE = re.compile(
    r"^\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?"
    r"(?:[*_`~]+\s*)?(?:(?:option|choice)\s+)?(?:\\boxed\{\s*)?[\(\[\{<【]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*"
    r"[\)\]\}>】]*\s*(?:\}\s*)?(?:[*_`~]+\s*)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
# Used by the tail-choice fallback after a short trailing segment has been extracted.
_TAIL_CHOICE_OPTION_RE = re.compile(
    r"(?<![\w+\-/])(?:(?:option|choice)\s+)?(?:\\boxed\{\s*)?[\(\[\{<【]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*"
    r"[\)\]\}>】]*\s*(?:\}\s*)?(?:[*_`~]+\s*)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)
_NEGATION_PREFIX_RE = re.compile(r"\b(?:not|isn't|isnt|wrong|incorrect|false)\b(?:\s+\w+){0,3}\s*$", re.IGNORECASE)
_BULLET_OR_LIST_LINE_RE = re.compile(r"^\s*(?:>\s*)?(?:[-*+]\s+|\d{1,3}[.)]\s+)")

_OUTER_WRAPPER_PAIRS = (
    ('"', '"'),
    ("'", "'"),
    ("\u201c", "\u201d"),
    ("\u2018", "\u2019"),
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("<", ">"),
    ("【", "】"),
)
_OUTER_MARKERS = ("**", "__", "*", "_", "`")
_AFTER_REJECTION_PREFIXES = (
    " is incorrect",
    " is wrong",
    " is false",
    " is not correct",
    " isn't correct",
    " isnt correct",
)
_CONTRAST_HINTS = (" but ", " however ", " instead ", " actually ", " rather ")
_COMPACT_OPTION_CONNECTORS = {"and", "or", "ou", "y", "e", "nor", "plus", "versus", "vs", "instead"}


@dataclass
class MCQAccuracyResult:
    """Detailed MCQ grading result."""

    is_correct: bool
    method: str
    matched_answer: Optional[str] = None
    correct_answer: Optional[str] = None


@dataclass
class _Candidate:
    """Normalized option candidate extracted from some region of the response."""

    token: str
    start: int
    end: int
    method: str


def normalize_for_structure(text: str) -> str:
    """Canonicalize structure while preserving line breaks and token boundaries."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_UNICODE_PUNCT_TRANSLATIONS)
    return text.casefold()


def normalize_for_match(text: str) -> str:
    """Canonicalize text for exact answer-text comparisons."""
    return _WHITESPACE_RE.sub(" ", normalize_for_structure(text)).strip()


def normalize_for_answer_text_match(text: str) -> str:
    """Canonicalize answer text under the explicit punctuation-normalization policy."""
    text = normalize_for_match(_strip_outer_wrappers(text))
    return text.rstrip(".,:;!?").strip()


def _answer_text_supports_fallback(answer_text: str) -> bool:
    """Reserve answer-text fallback for real text, not bare option labels like `A` or `2`."""
    return bool(answer_text) and _norm_option(answer_text) is None


@lru_cache(maxsize=1)
def _latex_to_text_converter():
    """Lazily construct the LaTeX-to-text converter used by `_strip_tex()`."""
    from pylatexenc.latex2text import LatexNodes2Text

    return LatexNodes2Text(math_mode="text")


def _strip_tex(text: str) -> str:
    """Best-effort LaTeX cleanup, leaving the original text on any failure."""
    if not text or not _LIKELY_TEX_RE.search(text):
        return text

    try:
        return _latex_to_text_converter().latex_to_text(text)
    except Exception:
        return text


def _norm_option(token: str) -> Optional[str]:
    """Normalize a predicted option to uppercase letter or digit string."""
    token = (token or "").strip()
    if not token:
        return None
    if token.isdigit():
        return token
    if token.isalpha() and len(token) == 1:
        return token.upper()
    return None


def _option_kind_matches(predicted: Optional[str], answer_letter: str) -> bool:
    """Require letter answers to match letters and numeric answers to match numbers."""
    if predicted is None:
        return False
    if answer_letter.isdigit():
        return predicted.isdigit()
    return predicted.isalpha()


def _result(
    is_correct: bool,
    method: str,
    predicted: Optional[str],
    actual: Optional[str],
    return_details: bool,
) -> bool | MCQAccuracyResult:
    """Return either a bare boolean or the structured grading result."""
    if not return_details:
        return is_correct
    return MCQAccuracyResult(
        is_correct=is_correct,
        method=method,
        matched_answer=predicted,
        correct_answer=actual,
    )


def _remove_think_tags(text: str) -> str:
    """Drop internal reasoning and keep only the answer region after the last `</think>` tag."""
    text = text or ""
    last_close_end: Optional[int] = None
    for match in _THINK_CLOSE_RE.finditer(text):
        last_close_end = match.end()
    if last_close_end is not None:
        return text[last_close_end:].lstrip()
    if _THINK_OPEN_RE.search(text):
        return ""
    return text


def _strip_outer_wrappers(text: str) -> str:
    """Peel simple answer wrappers like markdown, quotes, brackets, or `<answer>` tags."""
    text = (text or "").strip()
    changed = True
    while text and changed:
        changed = False
        lowered = text.lower()

        # Strip explicit answer wrappers before more generic marker peeling.
        if lowered[:8] == "<answer>" and lowered[-9:] == "</answer>":
            text = text[8:-9].strip()
            changed = True
            continue

        if lowered[:7] == "\\boxed{" and text.endswith("}"):
            text = text[7:-1].strip()
            changed = True
            continue

        for marker in _OUTER_MARKERS:
            if text.startswith(marker) and text.endswith(marker) and len(text) > len(marker) * 2:
                text = text[len(marker) : -len(marker)].strip()
                changed = True
                break
        if changed:
            continue

        for opener, closer in _OUTER_WRAPPER_PAIRS:
            if text.startswith(opener) and text.endswith(closer) and len(text) > len(opener) + len(closer):
                text = text[len(opener) : -len(closer)].strip()
                changed = True
                break

    return text


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Return the line boundaries that contain the span `[start, end)`."""
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return line_start, line_end


def _previous_nonempty_line_start(text: str, line_start: int) -> int:
    """Walk backward to the previous non-empty line start, if one exists."""
    cursor = line_start
    while cursor > 0:
        prev_end = cursor - 1
        prev_start = text.rfind("\n", 0, prev_end) + 1
        if text[prev_start:prev_end].strip():
            return prev_start
        cursor = prev_start
    return line_start


def _next_nonempty_line_end(text: str, line_end: int) -> int:
    """Walk forward to the next non-empty line end, if one exists."""
    cursor = line_end
    while cursor < len(text):
        next_start = cursor + 1 if cursor < len(text) and text[cursor] == "\n" else cursor
        next_end = text.find("\n", next_start)
        if next_end == -1:
            next_end = len(text)
        if text[next_start:next_end].strip():
            return next_end
        if next_end == len(text):
            break
        cursor = next_end
    return line_end


def _local_context(text: str, start: int, end: int) -> tuple[str, int, int]:
    """Return a bounded local region around a candidate plus its relative offsets."""
    line_start, line_end = _line_bounds(text, start, end)
    context_start = _previous_nonempty_line_start(text, line_start)
    context_end = _next_nonempty_line_end(text, line_end)
    # Prefer whole nearby lines, then cap to fixed windows so long CoTs stay cheap.
    context_start = max(context_start, start - LOCAL_CONTEXT_BEFORE_CHARS)
    context_end = min(context_end, end + LOCAL_CONTEXT_AFTER_CHARS)
    return text[context_start:context_end], start - context_start, end - context_start


def _candidate_is_negated(context: str, rel_start: int, rel_end: int) -> bool:
    """Detect local negation patterns that should invalidate a candidate option."""
    prefix = context[max(0, rel_start - 48) : rel_start]
    suffix = context[rel_end : min(len(context), rel_end + 40)]
    prefix = normalize_for_match(prefix).rstrip(" ([{<【")
    suffix = normalize_for_match(suffix)

    if _NEGATION_PREFIX_RE.search(prefix):
        return True
    if prefix.endswith("rather than") or prefix.endswith("except"):
        return True
    if "wrong diagnosis is" in prefix[-32:] or "incorrect diagnosis is" in prefix[-32:]:
        return True

    for prefix_text in _AFTER_REJECTION_PREFIXES:
        if suffix.startswith(prefix_text):
            return True

    return False


def _looks_like_option_connector(between_norm: str) -> bool:
    """Return True when the text between two options is just list/connector glue."""
    between_norm = between_norm.strip()
    if not between_norm:
        return True

    between_norm = re.sub(r"\b(?:option|choice)\b", " ", between_norm).strip()
    stripped = between_norm.strip(",;:./&+()[]{}<>-\\ ")
    if not stripped:
        return True

    return stripped in _COMPACT_OPTION_CONNECTORS


def _is_harmless_option_match(text: str, match: re.Match[str]) -> bool:
    """Ignore stray single-letter matches like pronoun `I` or apostrophe fragments."""
    token = match.group("opt").casefold()
    start = match.start("opt")
    end = match.end("opt")

    if token == "i":
        before = text[start - 1] if start > 0 else " "
        after = text[end] if end < len(text) else " "
        if before in {" ", "\n", "\t", ",", ";", ".", "(", "["} and after in {
            " ",
            "\n",
            "\t",
            ",",
            ";",
            ".",
            "!",
            "?",
            ")",
            "]",
        }:
            return True
    if token == "i" and start == 0:
        return True
    if start > 0 and text[start - 1] in {"'", "’"}:
        return True
    if end < len(text) and text[end] in {"'", "’"}:
        return True
    return False


def _candidate_has_local_competing_option(
    context: str, rel_start: int, rel_end: int, token: str, answer_letter: str
) -> bool:
    """Reject candidates that are locally entangled with another option token."""
    selected_span = (rel_start, rel_end)
    for match in _OPTION_TOKEN_RE.finditer(context):
        if _is_harmless_option_match(context, match):
            continue
        other = _norm_option(match.group("opt"))
        if other is None or not _option_kind_matches(other, answer_letter) or other == token:
            continue

        if match.end() <= selected_span[0]:
            between = context[match.end() : selected_span[0]]
        elif selected_span[1] <= match.start():
            between = context[selected_span[1] : match.start()]
        else:
            continue

        between_norm = normalize_for_match(between)
        if len(between_norm) > 24:
            continue
        # Treat only very short glue like commas, "and", or "or" as true ambiguity.
        if _looks_like_option_connector(between_norm):
            return True

    return False


def _candidate_is_contradicted(context: str, rel_end: int, token: str, answer_letter: str) -> bool:
    """Reject candidates that are immediately revised to a different option."""
    suffix = normalize_for_match(context[rel_end : min(len(context), rel_end + 80)])
    if not any(hint in suffix for hint in _CONTRAST_HINTS):
        return False

    for match in _OPTION_TOKEN_RE.finditer(suffix):
        other = _norm_option(match.group("opt"))
        if other is None or not _option_kind_matches(other, answer_letter):
            continue
        if other != token:
            return True
    return False


def _candidate_is_valid(text: str, candidate: _Candidate, answer_letter: str) -> bool:
    """Apply the local negation, ambiguity, and contradiction filters to a candidate."""
    context, rel_start, rel_end = _local_context(text, candidate.start, candidate.end)
    return not (
        _candidate_is_negated(context, rel_start, rel_end)
        or _candidate_has_local_competing_option(context, rel_start, rel_end, candidate.token, answer_letter)
        or _candidate_is_contradicted(context, rel_end, candidate.token, answer_letter)
    )


def _extract_exact_option(text: str, answer_letter: str) -> Optional[str]:
    """Accept responses that are exactly one standalone option token."""
    stripped = _strip_outer_wrappers(text)
    match = _EXACT_OPTION_RE.fullmatch(stripped)
    if not match:
        return None
    predicted = _norm_option(match.group("opt"))
    if predicted is None or not _option_kind_matches(predicted, answer_letter):
        return None
    return predicted


def _extract_exact_answer_text(text: str, answer_text: str) -> Optional[str]:
    """Accept responses that are exactly the answer text after wrapper normalization."""
    if not answer_text:
        return None
    stripped = _strip_outer_wrappers(text)
    if normalize_for_answer_text_match(stripped) != answer_text:
        return None
    return answer_text


def _extract_exact_option_plus_text(text: str, answer_letter: str, answer_text: str) -> Optional[str]:
    """Accept short option-led answers like `B. Correct answer text`."""
    stripped = _strip_outer_wrappers(text)
    match = _LEADING_OPTION_RE.fullmatch(stripped)
    if not match:
        return None
    predicted = _norm_option(match.group("opt"))
    if predicted is None or not _option_kind_matches(predicted, answer_letter):
        return None
    if normalize_for_answer_text_match(match.group("rest")) != answer_text:
        return None
    return predicted


@lru_cache(maxsize=64)
def _prefix_pattern(prefix_norm: str) -> re.Pattern[str]:
    """Compile the caller-provided anchor prefix into the same option-capture shape."""
    flexible_prefix = re.escape(prefix_norm).replace(r"\ ", r"\s+")
    return re.compile(
        rf"(?:^|(?<![a-z0-9])){flexible_prefix}\s*[:\-]?\s*(?:is\s+)?(?P<neg>not\s+|isn't\s+|isnt\s+)?"
        rf"(?:(?:option|choice)\s+)?"
        rf"(?:[*_`~]+\s*)*(?:\\boxed\{{\s*)?[\(\[\{{<【]*\s*(?P<opt>[A-Za-z]|\d{{1,2}})\s*"
        rf"[\)\]\}}>】]*\s*(?:\}}\s*)?(?:[*_`~]+\s*)?(?![\w+\-/])",
        re.IGNORECASE,
    )


def _latest_explicit_candidate(text: str, answer_letter: str, prefix: Optional[str]) -> Optional[_Candidate]:
    """Return the latest valid anchored candidate, preferring a caller-specified prefix."""
    if prefix:
        prefix_norm = normalize_for_match(prefix)
        if prefix_norm:
            saw_prefix_match = False
            latest_valid: Optional[_Candidate] = None
            for match in _prefix_pattern(prefix_norm).finditer(text):
                if not _prefix_match_has_standalone_start(text, match.start()):
                    continue
                saw_prefix_match = True
                if match.groupdict().get("neg"):
                    continue
                token = _norm_option(match.group("opt"))
                if token is None or not _option_kind_matches(token, answer_letter):
                    continue
                candidate = _Candidate(
                    token=token,
                    start=match.start("opt"),
                    end=match.end("opt"),
                    method="anchored_token",
                )
                if _candidate_is_valid(text, candidate, answer_letter):
                    latest_valid = candidate
            # If the caller supplied an explicit prefix, do not fall back to generic anchors
            # once that prefix appears at all.
            if saw_prefix_match:
                return latest_valid

    latest_valid = None
    for match in _ANCHOR_RE.finditer(text):
        if match.groupdict().get("neg"):
            continue
        token = _norm_option(match.group("opt"))
        if token is None or not _option_kind_matches(token, answer_letter):
            continue
        candidate = _Candidate(token=token, start=match.start("opt"), end=match.end("opt"), method="anchored_token")
        if _candidate_is_valid(text, candidate, answer_letter):
            latest_valid = candidate

    return latest_valid


def _prefix_match_has_standalone_start(text: str, start: int) -> bool:
    """Require prefix matches to start at a token boundary rather than inside a word."""
    cursor = start - 1
    while cursor >= 0 and text[cursor].isspace():
        cursor -= 1
    return cursor < 0 or not text[cursor].isalnum()


def _leading_option_candidate(text: str, answer_letter: str, answer_text: str) -> Optional[_Candidate]:
    """Parse a short option-led answer that starts with the selected option token."""
    source = text
    offset = 0
    if "\n" in text:
        # For multi-line responses, only trust the final non-empty line as a leading-option answer.
        source = _last_nonempty_line(text)
        if not source:
            return None
        offset = text.rfind(source)
        match = _LEADING_OPTION_RE.match(source)
    else:
        match = _LEADING_OPTION_RE.match(source)
        if not match:
            source = _last_nonempty_line(text)
            if not source:
                return None
            offset = text.rfind(source)
            match = _LEADING_OPTION_RE.match(source)
    if not match:
        return None

    token = _norm_option(match.group("opt"))
    if token is None or not _option_kind_matches(token, answer_letter):
        return None

    # Plain prose like "I think B works" should not be treated as an option-led format.
    separator = source[match.end("opt") : match.start("rest")]
    rest = match.group("rest").lstrip()
    if not any(char in separator for char in ")]}>】.:-*_`~\\") and not rest.startswith(
        ("(", "[", "{", "<", "【", '"', "'", "\\boxed{")
    ):
        return None

    # Reject enumerated multi-option payloads like "A. ...\nD. ...".
    if _contains_multiple_option_led_sentences(text, answer_letter):
        return None

    candidate = _Candidate(
        token=token,
        start=offset + match.start("opt"),
        end=offset + match.end("opt"),
        method="anchored_token",
    )
    if not _candidate_is_valid(text, candidate, answer_letter):
        return None
    return candidate


def _last_nonempty_line(text: str) -> str:
    """Return the final non-empty line from the response, if any."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _is_compact_multi_option_list(text: str, answer_letter: str) -> bool:
    """Detect short tails like `A, C` or `B and D` that should fail closed."""
    matches = [
        match
        for match in _OPTION_TOKEN_RE.finditer(text)
        if _option_kind_matches(_norm_option(match.group("opt")), answer_letter)
    ]
    if len(matches) < 2:
        return False

    if len(text.strip()) > 40:
        return False

    for idx in range(len(matches) - 1):
        between = normalize_for_match(text[matches[idx].end() : matches[idx + 1].start()])
        if not _looks_like_option_connector(between):
            return False

    return True


def _tail_choice_text(text: str) -> str:
    """Extract the short trailing segment that feeds the tail-choice fallback."""
    region = (text or "").strip()
    if not region:
        return ""

    parts = re.split(r"\n+|[.!?]\s+", region)
    tail_choice = parts[-1].strip() if parts else region
    if not tail_choice:
        tail_choice = _last_nonempty_line(region)
    # Long trailing prose is too ambiguous for the tail-choice heuristic.
    if len(tail_choice.split()) > TAIL_CHOICE_MAX_WORDS:
        return ""
    return tail_choice


def _contains_multiple_option_led_sentences(text: str, answer_letter: str) -> bool:
    """Detect multi-line or multi-sentence payloads that enumerate different option labels."""
    distinct: set[str] = set()
    # Newline-separated enumerations are common in model outputs, so keep lines intact in that case.
    chunks = (text or "").splitlines() if "\n" in (text or "") else re.split(r"[.!?]\s+", text or "")
    for chunk in chunks:
        match = _SENTENCE_OPTION_START_RE.match(chunk.strip())
        if not match:
            continue
        token = _norm_option(match.group("opt"))
        if token is None or not _option_kind_matches(token, answer_letter):
            continue
        distinct.add(token)
        if len(distinct) > 1:
            return True
    return False


def _tail_candidate(region: str, answer_letter: str) -> Optional[_Candidate]:
    """Extract a last-line or tail-choice option token from the terminal region."""
    line = _last_nonempty_line(region)
    # Prefer an exact last-line option like "(C)" before falling back to a looser tail-choice scan.
    if line and not _is_compact_multi_option_list(line, answer_letter):
        match = _TERMINAL_OPTION_LINE_RE.fullmatch(line)
        if match:
            token = _norm_option(match.group("opt"))
            if token is not None and _option_kind_matches(token, answer_letter):
                line_offset = region.rfind(line)
                start = line_offset + match.start("opt")
                end = line_offset + match.end("opt")
                candidate = _Candidate(token=token, start=start, end=end, method="last_token")
                if _candidate_is_valid(region, candidate, answer_letter):
                    return candidate

    tail_choice = _tail_choice_text(region)
    if not tail_choice or _is_compact_multi_option_list(tail_choice, answer_letter):
        return None

    match = _TAIL_CHOICE_OPTION_RE.search(tail_choice)
    if not match:
        return None

    token = _norm_option(match.group("opt"))
    if token is None or not _option_kind_matches(token, answer_letter):
        return None

    tail_choice_offset = region.rfind(tail_choice)
    candidate = _Candidate(
        token=token,
        start=tail_choice_offset + match.start("opt"),
        end=tail_choice_offset + match.end("opt"),
        method="last_token",
    )
    if not _candidate_is_valid(region, candidate, answer_letter):
        return None
    return candidate


def _answer_text_pattern(answer_text: str) -> re.Pattern[str]:
    """Compile a whitespace-tolerant exact-answer-text regex."""
    flexible_answer = re.escape(answer_text).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){flexible_answer}(?!\w)", re.IGNORECASE)


def _latest_answer_text_match(region: str, answer_text: str, answer_letter: str) -> Optional[str]:
    """Return the latest valid exact answer-text match inside a search region."""
    region_struct = normalize_for_structure(region)
    if not answer_text or not region_struct:
        return None

    latest_valid: Optional[str] = None
    for match in _answer_text_pattern(answer_text).finditer(region_struct):
        if _answer_text_match_is_valid(region_struct, match.start(), match.end(), answer_letter):
            latest_valid = answer_text

    return latest_valid


def _answer_text_match_is_valid(region_struct: str, start: int, end: int, answer_letter: str) -> bool:
    """Reject answer-text matches that sit inside obvious negation or option-list structure."""
    prefix = region_struct[max(0, start - 64) : start].rstrip()
    if _NEGATION_PREFIX_RE.search(prefix):
        return False
    if prefix.endswith("rather than") or prefix.endswith("except"):
        return False
    if "wrong diagnosis is" in prefix[-40:] or "incorrect diagnosis is" in prefix[-40:]:
        return False

    line_start, line_end = _line_bounds(region_struct, start, end)
    raw_line = region_struct[line_start:line_end]
    rel_start = start - line_start
    rel_end = end - line_start
    leading_match = _LEADING_OPTION_RE.match(raw_line.strip())
    if leading_match is not None:
        token = _norm_option(leading_match.group("opt"))
        if token is not None and _option_kind_matches(token, answer_letter):
            return False

    # Bulleted or numbered option-analysis lines often mention distractor answer text verbatim.
    if _BULLET_OR_LIST_LINE_RE.match(raw_line):
        before_match = raw_line[:rel_start]
        after_match = raw_line[rel_end:].lstrip(" *_`~)]}>】")
        if ":" in before_match or any(marker in before_match for marker in (" - ", " – ", " — ")):
            return False
        if after_match.startswith((":", "-", "–", "—")):
            return False

    return True


def _answer_text_regions(text: str, answer_text: str, is_long: bool) -> list[str]:
    """Choose the bounded regions where answer-text fallback is allowed to search."""
    if is_long:
        # In long mode, the tail is authoritative because earlier reasoning is frequently revised.
        return [text[-TERMINAL_WINDOW_CHARS:]]

    if len(text) <= 800:
        return [text]

    # For shorter responses, search bounded tail/head windows but align them to line
    # boundaries so local validation still sees bullet markers and nearby list structure.
    window = max(600, min(1_400, len(answer_text) + 400))
    line_slack = 200

    # Only stretch to the next line break when it is still close to the window edge.
    head_end = text.find("\n", window, min(len(text), window + line_slack + 1))
    if head_end == -1:
        head_end = min(len(text), window)
    head = text[:head_end]

    tail_start = max(0, len(text) - window)
    aligned_tail_start = text.rfind("\n", max(0, tail_start - line_slack), tail_start)
    if aligned_tail_start != -1:
        tail_start = aligned_tail_start + 1
    tail = text[tail_start:]

    if head == tail:
        return [head]
    return [tail, head]


def multiple_choice_accuracy(
    llm_answer: str,
    answer_letter: str,
    answer_text: str,
    prefix: Optional[str] = None,
    accept_answer_text: bool = True,
    strict: bool = False,
    strip_tex: bool = True,
    return_details: bool = False,
) -> bool | MCQAccuracyResult:
    """Grade an MCQ answer using exact matching or permissive MCQ extraction heuristics."""

    if not llm_answer:
        return _result(False, "none", None, None, return_details)

    # Strip reasoning wrappers and normalize before any extraction logic runs.
    processed_answer = _remove_think_tags(llm_answer)
    processed_answer = _ANSWER_TAG_RE.sub(" ", processed_answer)
    if strip_tex:
        processed_answer = _strip_tex(processed_answer)
        answer_text = _strip_tex(answer_text or "")

    structural_text = normalize_for_structure(processed_answer).strip()
    answer_letter = _norm_option(answer_letter)
    answer_text = normalize_for_answer_text_match(answer_text or "")
    exact_answer_text_allowed = accept_answer_text and bool(answer_text)
    answer_text_fallback_allowed = accept_answer_text and _answer_text_supports_fallback(answer_text)

    if answer_letter is None:
        raise ValueError(f"Invalid answer_letter '{answer_letter=}'. Must be a single letter or digit string.")

    if not structural_text:
        return _result(False, "none", None, None, return_details)

    # Strategy 1: exact standalone option, e.g. "C" or "(2)".
    direct_option = _extract_exact_option(structural_text, answer_letter)
    if direct_option == answer_letter:
        return _result(
            True,
            "direct_answer",
            direct_option.casefold(),
            answer_letter,
            return_details,
        )

    # Strategy 2: exact answer text after wrapper normalization. This remains allowed
    # even for numeric answer text, so parsed outputs like "\boxed{4}" can still match
    # the gold content answer text before a mismatched standalone numeral fails closed.
    if exact_answer_text_allowed:
        direct_text = _extract_exact_answer_text(structural_text, answer_text)
        if direct_text is not None:
            return _result(True, "answer_text", direct_text, answer_text, return_details)

    if direct_option is not None:
        return _result(
            False,
            "direct_answer",
            direct_option.casefold(),
            answer_letter,
            return_details,
        )

    # Strategy 3: short option-led answer that also includes the answer text.
    option_plus_text = _extract_exact_option_plus_text(structural_text, answer_letter, answer_text)
    if option_plus_text is not None:
        return _result(
            option_plus_text == answer_letter,
            "anchored_token",
            option_plus_text,
            answer_letter,
            return_details,
        )

    if strict:
        return _result(False, "none", None, None, return_details)

    is_long = len(structural_text) > LONG_RESPONSE_THRESHOLD_CHARS
    terminal_region = structural_text[-TERMINAL_WINDOW_CHARS:] if is_long else structural_text
    strong_tail_region = terminal_region[-STRONG_TAIL_WINDOW_CHARS:] if is_long else structural_text

    # Strategy 4: anchored commitments like "final answer: C".
    explicit_candidate = _latest_explicit_candidate(terminal_region, answer_letter, prefix)
    if explicit_candidate is not None:
        return _result(
            explicit_candidate.token == answer_letter,
            explicit_candidate.method,
            explicit_candidate.token,
            answer_letter,
            return_details,
        )

    # Strategy 5: leading-option forms are only trusted in short responses.
    if not is_long:
        leading_candidate = _leading_option_candidate(structural_text, answer_letter, answer_text)
        if leading_candidate is not None:
            return _result(
                leading_candidate.token == answer_letter,
                leading_candidate.method,
                leading_candidate.token,
                answer_letter,
                return_details,
            )

    # Strategy 6: tail-only token fallback from the last line or short tail choice text.
    tail_candidate = _tail_candidate(strong_tail_region, answer_letter)
    if tail_candidate is not None:
        return _result(
            tail_candidate.token == answer_letter,
            tail_candidate.method,
            tail_candidate.token,
            answer_letter,
            return_details,
        )

    # Strategy 7: exact answer-text fallback in bounded head/tail regions.
    if answer_text_fallback_allowed and answer_text:
        for region in _answer_text_regions(structural_text, answer_text, is_long):
            matched = _latest_answer_text_match(region, answer_text, answer_letter)
            if matched is not None:
                return _result(True, "answer_text", matched, answer_text, return_details)

    return _result(False, "none", None, None, return_details)
