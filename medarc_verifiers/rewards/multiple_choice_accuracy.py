"""
LLM multiple-choice question accuracy reward.

Main use case: Handle models that either return the letter/number (preferred)
or return the entire answer text verbatim (fallback).

Supports chain-of-thought by prioritizing anchored patterns like "answer is X"
before falling back to last token or text matching. Attempts to recognize
negations to avoid false positives (e.g., "the answer is not C").
"""

import re
import os
import sys
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from functools import wraps
from typing import Optional


_UNICODE_PUNCT_TRANSLATIONS = str.maketrans(
    {
        "\u00a0": " ",  # no-break space
        "\u2010": "-",  # hyphen
        "\u2011": "-",  # non-breaking hyphen
        "\u2012": "-",  # figure dash
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2015": "-",  # horizontal bar
        "\u2212": "-",  # minus sign
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
_INNER_PUNCT_SPACING_RE = re.compile(r"\s*([()\[\]{}.,;:])\s*")
_TRAILING_TERMINAL_PUNCT_RE = re.compile(r"(?<=\w)[.!?,;:]+$")
_LIKELY_TEX_RE = re.compile(r"\\[A-Za-z]+|\\[$\\()\\[\\]{}]|[$]")


def _mcq_perf_trace_enabled() -> bool:
    """Return whether lightweight MCQ performance tracing is enabled."""
    return os.getenv("MEDARC_MCQ_PERF_TRACE", "").strip().lower() in {"1", "true", "yes", "on"}


def _mcq_perf_trace_min_seconds() -> float:
    """Return the minimum elapsed time required before a helper emits a trace line."""
    raw = os.getenv("MEDARC_MCQ_PERF_TRACE_MIN_MS", "").strip()
    if not raw:
        return 0.0
    try:
        return max(float(raw) / 1000.0, 0.0)
    except ValueError:
        return 0.0


def _mcq_perf_trace_summary(args: tuple, kwargs: dict) -> str:
    """Build a compact summary string for performance trace logging."""
    parts: list[str] = []
    for idx, value in enumerate(args[:3]):
        if isinstance(value, str):
            parts.append(f"arg{idx}_len={len(value)}")
        elif isinstance(value, re.Match):
            try:
                start, end = value.span()
                parts.append(f"arg{idx}_span={start}:{end}")
            except Exception:
                parts.append(f"arg{idx}=match")
        elif isinstance(value, list):
            parts.append(f"arg{idx}_len={len(value)}")
    if "answer_letter" in kwargs and isinstance(kwargs["answer_letter"], str):
        parts.append(f"answer_letter={kwargs['answer_letter']!r}")
    return " ".join(parts)


def _trace_scan_perf(func):
    """Wrap a helper so it can emit elapsed-time traces when tracing is enabled."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Execute the wrapped helper and optionally log its runtime."""
        if not _mcq_perf_trace_enabled():
            return func(*args, **kwargs)

        started = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            if elapsed >= _mcq_perf_trace_min_seconds():
                summary = _mcq_perf_trace_summary(args, kwargs)
                print(
                    f"[mcq-perf] {func.__name__} elapsed_ms={elapsed * 1000:.3f}" + (f" {summary}" if summary else ""),
                    file=sys.stderr,
                    flush=True,
                )

    return wrapper


@dataclass
class MCQAccuracyResult:
    """Result of multiple-choice accuracy grading."""

    is_correct: bool
    """Whether the answer was graded as correct."""

    method: str
    """Method used for grading: 'direct_answer', 'anchored_token', 'last_token', 'answer_text', or 'none'."""

    matched_answer: Optional[str] = None
    """The extracted answer if found, otherwise None."""

    correct_answer: Optional[str] = None
    """The correct answer for reference, if available."""


@_trace_scan_perf
def normalize_for_structure(text: str) -> str:
    """Canonicalize text for structural matching without collapsing whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_UNICODE_PUNCT_TRANSLATIONS)
    return text.casefold()


@_trace_scan_perf
def normalize_for_match(text: str) -> str:
    """Canonicalize text for answer-text equivalence matching."""
    return _WHITESPACE_RE.sub(" ", normalize_for_structure(text)).strip()


@_trace_scan_perf
def normalize_for_answer_text_match(text: str) -> str:
    """Canonicalize text for answer-text matching while tolerating minor punctuation drift."""
    text = normalize_for_match(text)
    text = _INNER_PUNCT_SPACING_RE.sub(r"\1", text)
    return _TRAILING_TERMINAL_PUNCT_RE.sub("", text)


@_trace_scan_perf
@lru_cache(maxsize=1)
def _latex_to_text_converter():
    """Construct and cache the pylatexenc converter used for TeX stripping."""
    from pylatexenc.latex2text import LatexNodes2Text

    return LatexNodes2Text(math_mode="text")


@_trace_scan_perf
def _strip_tex(text: str) -> str:
    """Remove LaTeX formatting if pylatexenc is available."""
    if not text or not _LIKELY_TEX_RE.search(text):
        return text

    try:
        return _latex_to_text_converter().latex_to_text(text)
    except Exception:
        return text


@_trace_scan_perf
def _norm_letter(letter: str) -> Optional[str]:
    """Normalize a token to uppercase letter or digit string."""
    letter = (letter or "").strip()
    if not letter:
        return None
    if letter.isdigit():
        return letter
    if letter.isalpha() and len(letter) == 1:
        return letter.upper()
    return None


@_trace_scan_perf
def _token_kind_matches_answer_letter(predicted: Optional[str], answer_letter: str) -> bool:
    """Return True if predicted token type matches the task's option type.

    This prevents cases like '<answer>20' in a letter-based task (answer_letter='C')
    from being treated as an explicit option selection, which would incorrectly disable
    answer_text fallback.
    """
    if predicted is None:
        return False
    if answer_letter.isdigit():
        return predicted.isdigit()
    return predicted.isalpha()


_THINK_OPEN_RE = re.compile(r"<\s*think\b[^>]*>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</\s*think\s*>", re.IGNORECASE)


@_trace_scan_perf
def _remove_think_tags(completion_text: str) -> str:
    """Extract the answer section from completion text, handling think tags properly.

    Behavior is intentionally conservative:
    - If there is any explicit closing </think>, return everything after the last closing tag.
    - If there is an unclosed <think> with no closing tag, treat the output as missing
      a final-answer region.
    - Otherwise, return the full response.
    """
    text = completion_text or ""

    closes = list(_THINK_CLOSE_RE.finditer(text))
    if closes:
        return text[closes[-1].end() :].lstrip()

    if _THINK_OPEN_RE.search(text):
        return ""

    return text


# Anchored patterns like "final answer: C" or "the answer is D"
ANCHOR_PATTERN = re.compile(
    r"(?:\bfinal\s+answer\b|\banswer\b|\bans\b|\bchoice\b|\boption\b|\bselected\b|\bi\s+choose\b|\bi\s+pick\b|\btherefore\b|\bthus\b|\bso\b|\bconclusion\b|\bin\s+conclusion\b|\bmost\s+likely\b|\bbest[-\s]+supported\s+answer\b|<answer>)\s*"
    r"[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?"
    r"(?:[*_`~]+\s*)*"  # allow markdown wrappers before the option
    r"[\(\[\{<【]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*[\)\]\}>】]*"  # option token, possibly wrapped
    r"\s*[\)\.:]?\s*"  # optional delimiter (e.g., 'B.' or 'B)')
    r"(?:[*_`~]+\s*)*"  # allow markdown wrappers after the option
    r"(?![\w+\-/])",
    re.IGNORECASE,
)


# Any letter/number token that looks like an option
TOKEN_PATTERN = re.compile(
    r"(?<![\w+\-/])[\(\[\{<【]*\s*([A-Za-z]|\d{1,2})\s*[\)\]\}>】]*[\)\.:]?(?![\w+\-/])",
    re.IGNORECASE,
)

_LEADING_OPTION_PATTERN_BODY = (
    r"\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?\s*"
    r"(?:[*_`~]+)?\s*\(?\s*([A-Za-z]|\d{1,2})\s*"
    r"(?:[\)\.:]\s*\)?\s*(?:[*_`~]+)?\s*|(?=\s*(?:\(|[-–—])))(?!\w)"
)

# Leading option token like "B. Answer text" or "C) ..." at the start of the response
LEADING_OPTION_PATTERN = re.compile(rf"^{_LEADING_OPTION_PATTERN_BODY}", re.IGNORECASE)
# Same pattern without ^ so we can match from sentence offsets without slicing text.
SENTENCE_LEADING_OPTION_PATTERN = re.compile(_LEADING_OPTION_PATTERN_BODY, re.IGNORECASE)

# Standalone final-line option token like "C", "(C)", or "\boxed{C}".
TERMINAL_OPTION_LINE_PATTERN = re.compile(
    r"^\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?\s*"
    r"(?:\\boxed\{\s*)?(?:<answer>\s*)?"
    r"[\(\[\{<【]*\s*[*_`~]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*[*_`~]*[\)\]\}>】]*"
    r"\s*(?:</answer>\s*)?(?:\}\s*)?\s*[.!?]?\s*$",
    re.IGNORECASE,
)

FINAL_CLAUSE_TERMINAL_OPTION_RE = re.compile(
    r"(?<![\w+\-/])[\(\[\{<【]*\s*[*_`~]*\s*(?P<opt>[A-Za-z]|\d{1,2})\s*[*_`~]*[\)\]\}>】]*\s*[.!?]?\s*$",
    re.IGNORECASE,
)

# Negation/correction phrases that immediately precede an option or answer text
NEGATION_BEFORE_MATCH_PATTERN = re.compile(
    r"(?:\bnot\b|\bisn['’]t\b|\baren['’]t\b|\bwasn['’]t\b|\bweren['’]t\b|\bincorrect\b|\bwrong\b|\bfalse\b|\bexcept(?:\s+for)?\b|\brather\s+than\b)(?:\W+\w+){0,3}\W*$",
    re.IGNORECASE,
)

# Negative-context phrases that indicate an option mention is NOT a selected answer
NEGATIVE_AFTER_OPTION_PATTERN = re.compile(
    r"^\s*(?:is|are|was|were)\s+(?:incorrect|wrong|false|not\s+correct)\b|^\s*not\s+correct\b",
    re.IGNORECASE,
)

CONTRAST_PATTERN = re.compile(
    r"\b(?:but|however|instead(?!\s+of\b))\b"
    r".{0,40}?"
    r"(?<![\w+\-/])\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]?(?![\w+\-/])",
    re.IGNORECASE,
)

# Sentence boundary pattern - splits on period, exclamation, question mark, or newline
# Handles both single newlines (for line breaks in CoT) and double newlines (paragraphs)
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+|\n+")

_MULTI_OPTION_CONNECTOR_WORD_PATTERN = (
    r"\b(?:and|or|nor|to|through|then|plus|both|y|e|ou|und|et)\b"
    r"|\bas\s+well\s+as\b"
    r"|\btogether\s+with\b"
    r"|\bfollowed\s+by\b"
    r"|\bcorrect\ choices?\s+are\b"
    r"|\bchoices?\s+are\b"
)
_MULTI_OPTION_CONNECTOR_RE = re.compile(_MULTI_OPTION_CONNECTOR_WORD_PATTERN, re.IGNORECASE)

# Compact-list glue that should cause the last-token fallback to reject a tail as
# multi-answer rather than selecting the final option.
COMPACT_MULTI_OPTION_GLUE_PATTERN = re.compile(
    rf"(?:{_MULTI_OPTION_CONNECTOR_WORD_PATTERN}|[,:;/&+\-|])",
    re.IGNORECASE,
)

_MULTIPLE_OPTION_LED_SCAN_MAX_CHARS = 10_000


@_trace_scan_perf
def _get_sentence_containing_match(text: str, match: re.Match) -> tuple[int, int, int, int]:
    """Return (sentence_start, sentence_end, match_start, match_end) in the original text."""
    if getattr(match.re, "groupindex", None) and "opt" in match.re.groupindex:
        match_start, match_end = match.span("opt")
    else:
        try:
            match_start, match_end = match.span(1)
        except Exception:
            match_start, match_end = match.span()

    boundaries_before = [m.end() for m in SENTENCE_BOUNDARY.finditer(text[:match_start])]
    boundaries_after = [m.start() for m in SENTENCE_BOUNDARY.finditer(text[match_end:])]

    sentence_start = boundaries_before[-1] if boundaries_before else 0
    sentence_end = match_end + boundaries_after[0] if boundaries_after else len(text)
    return sentence_start, sentence_end, match_start, match_end


@dataclass
class _SentenceMatchContext:
    """Sentence-local context around a matched option or answer-text span."""

    prefix: str
    suffix: str
    token: Optional[str]


@_trace_scan_perf
def _match_token(match: re.Match) -> Optional[str]:
    """Extract and normalize the option token captured by a regex match."""
    if getattr(match.re, "groupindex", None) and "opt" in match.re.groupindex:
        return _norm_letter(match.group("opt"))
    try:
        return _norm_letter(match.group(1))
    except Exception:
        return None


@_trace_scan_perf
def _sentence_match_context(text: str, match: re.Match) -> _SentenceMatchContext:
    """Return the same-sentence prefix, suffix, and normalized token for a regex match."""
    sentence_start, sentence_end, match_start, match_end = _get_sentence_containing_match(text, match)
    return _SentenceMatchContext(
        prefix=text[sentence_start:match_start],
        suffix=text[match_end:sentence_end],
        token=_match_token(match),
    )


@_trace_scan_perf
def _match_is_negated(context: _SentenceMatchContext) -> bool:
    """Return True when a negation phrase appears before the match in the same sentence."""
    return bool(NEGATION_BEFORE_MATCH_PATTERN.search(context.prefix))


@_trace_scan_perf
def _match_has_negative_suffix(context: _SentenceMatchContext) -> bool:
    """Return True when the match is immediately followed by rejecting language."""
    return bool(NEGATIVE_AFTER_OPTION_PATTERN.search(context.suffix))


@_trace_scan_perf
def _match_is_contradicted(context: _SentenceMatchContext) -> bool:
    """Return True when a later contrast in the sentence points to a different option."""
    if context.token is None:
        return False
    later = CONTRAST_PATTERN.search(context.suffix)
    if not later:
        return False
    contrasted = _norm_letter(later.group(1))
    return contrasted is not None and contrasted != context.token


@_trace_scan_perf
def _tail_region(text: str, max_tokens: int = 64) -> str:
    """Return a short tail slice (last sentence/line) to reduce option-token noise."""
    boundaries = list(SENTENCE_BOUNDARY.finditer(text))
    tail = text[boundaries[-1].end() :] if boundaries else text
    tail = tail.strip()

    if not tail:
        for line in reversed(text.splitlines()):
            if line.strip():
                tail = line.strip()
                break

    tokens = tail.split()
    if len(tokens) > max_tokens:
        tail = " ".join(tokens[-max_tokens:])
    return tail


@_trace_scan_perf
def _last_nonempty_line(text: str) -> str:
    """Return the last non-empty line, if any."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


@_trace_scan_perf
def _option_candidate_invalid(text: str, match: re.Match) -> bool:
    """Return True if an option-like match is negated or contradicted in local context."""
    context = _sentence_match_context(text, match)
    return _match_is_negated(context) or _match_has_negative_suffix(context) or _match_is_contradicted(context)


@_trace_scan_perf
def _is_harmless_prefix_option_token(prefix: str, prior_match: re.Match) -> bool:
    """Ignore harmless single-letter artifacts before a terminal final-clause answer.

    This is limited to natural-language cases like:
    - leading pronoun "I"
    - article "a" before a normal word
    - trailing "'s" in contractions like "it's"
    """
    raw = prior_match.group(1).casefold()
    if raw == "i" and prior_match.start() == 0:
        return True
    if raw == "a" and re.match(r"\s+[a-z]{2,}\b", prefix[prior_match.end() :]):
        return True
    if raw == "s" and prior_match.start() > 0 and prefix[prior_match.start() - 1] in {"'", "’"}:
        return True
    return False


@_trace_scan_perf
def _has_connector_between(text: str, max_words: int = 5) -> bool:
    """Return True when a short span looks like connector text between option tokens."""
    text = text.strip()
    return bool(text) and len(text.split()) <= max_words and bool(_MULTI_OPTION_CONNECTOR_RE.search(text))


@_trace_scan_perf
def _normalized_option_matches(text: str) -> list[tuple[str, re.Match]]:
    """Return normalized option tokens paired with their regex matches in order."""
    matches: list[tuple[str, re.Match]] = []
    for match in TOKEN_PATTERN.finditer(text):
        token = _norm_letter(match.group(1))
        if token is not None:
            matches.append((token, match))
    return matches


@lru_cache(maxsize=64)
def _prefix_pattern(prefix_norm: str) -> re.Pattern:
    """Compile and cache the anchored prefix regex for a normalized answer prefix."""
    flexible_prefix = re.escape(prefix_norm).replace(r"\ ", r"\s+")
    return re.compile(
        rf"{flexible_prefix}\s*[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?\(?\s*(?P<opt>[A-Za-z]|\d{{1,2}})\s*[\)\.:]?(?![\w+\-/])",
        re.IGNORECASE,
    )


@_trace_scan_perf
def _extract_standalone_terminal_option(region: str) -> Optional[str]:
    """Extract a standalone terminal token like ``C`` or ``(C)`` from a region."""
    match = TERMINAL_OPTION_LINE_PATTERN.fullmatch(region)
    if not match:
        return None

    predicted = _norm_letter(match.group("opt"))
    if predicted is None:
        return None

    tokens = list(TOKEN_PATTERN.finditer(region))
    if len(tokens) != 1 or _option_candidate_invalid(region, tokens[0]):
        return None
    return predicted


@_trace_scan_perf
def _extract_leading_terminal_option(region: str) -> Optional[str]:
    """Extract a leading-option form like ``C. text`` from a region."""
    leading_match = LEADING_OPTION_PATTERN.match(region)
    if not leading_match:
        return None

    predicted = _norm_letter(leading_match.group(1))
    if predicted is None or _option_candidate_invalid(region, leading_match):
        return None
    return predicted


@_trace_scan_perf
def _extract_final_clause_terminal_option(region: str) -> Optional[str]:
    """Extract a final-clause token like ``I think it's C`` from a short region."""
    match = FINAL_CLAUSE_TERMINAL_OPTION_RE.search(region)
    if not match or _option_candidate_invalid(region, match):
        return None

    prefix = region[: match.start()]
    for _token, prior_match in _normalized_option_matches(prefix):
        if _is_harmless_prefix_option_token(prefix, prior_match):
            continue
        return None

    return _norm_letter(match.group("opt"))


@_trace_scan_perf
def _extract_terminal_option_line(line: str) -> Optional[str]:
    """Extract a standalone option token from the last line."""
    if not line or _is_compact_multi_option_list(line):
        return None
    predicted = _extract_standalone_terminal_option(line)
    if predicted is not None:
        return predicted
    return _extract_leading_terminal_option(line)


@_trace_scan_perf
def _extract_short_final_clause_option(text: str, max_words: int = 12) -> Optional[str]:
    """Extract a terminal option token from a short final clause like 'I think it's C'."""
    clause = _tail_region(text).strip()
    if not clause or len(clause.split()) > max_words or _is_compact_multi_option_list(clause):
        return None
    return _extract_final_clause_terminal_option(clause)


@_trace_scan_perf
def _anchored_match_in_multi_answer_phrase(text: str, matches: list[re.Match], idx: int) -> bool:
    """Return True if anchored match *idx* is part of a local multi-answer phrase."""
    match = matches[idx]
    current = _match_token(match)
    if current is None:
        return False

    if idx > 0:
        between = text[matches[idx - 1].end() : match.start()]
        if _has_connector_between(between):
            return True

    if idx < len(matches) - 1:
        between = text[match.end() : matches[idx + 1].start()]
        if _has_connector_between(between):
            return True

    sentence_start, sentence_end, match_start, match_end = _get_sentence_containing_match(text, match)
    sentence = text[sentence_start:sentence_end]
    local_match_start = match_start - sentence_start
    local_match_end = match_end - sentence_start

    for token, token_match in _normalized_option_matches(sentence):
        if token == current:
            continue
        between = ""
        if token_match.end() <= local_match_start:
            between = sentence[token_match.end() : local_match_start]
        elif local_match_end <= token_match.start():
            between = sentence[local_match_end : token_match.start()]
        if not between:
            continue
        if _has_connector_between(between):
            return True

    return False


@_trace_scan_perf
def _is_compact_multi_option_list(text: str) -> bool:
    """Return True for short multi-option tails like 'A, C' or '> **A** and C'."""
    text = (text or "").strip()
    if len(list(TOKEN_PATTERN.finditer(text))) < 2:
        return False

    residue = TOKEN_PATTERN.sub(" ", text)
    residue = COMPACT_MULTI_OPTION_GLUE_PATTERN.sub(" ", residue)
    residue = re.sub(r"[\s\[\]\(\)\{\}<>*_`~.!?]+", " ", residue)
    return residue.strip() == ""


@_trace_scan_perf
def _contains_multiple_option_led_sentences(text: str, answer_letter: str) -> bool:
    """Return True when different sentences/lines each start with different option labels.

    This catches payloads like "(A) ... . (D) ..." or "A. ...\\nD. ...", which should
    not be accepted for a single-answer MCQ unless a later anchored final answer overrides
    them.
    """

    text = text or ""
    distinct: set[str] = set()
    starts = [0]
    starts.extend(match.end() for match in SENTENCE_BOUNDARY.finditer(text))
    for start in starts:
        match = SENTENCE_LEADING_OPTION_PATTERN.match(text, pos=start)
        if not match:
            continue
        token = _norm_letter(match.group(1))
        if token is None or not _token_kind_matches_answer_letter(token, answer_letter):
            continue
        distinct.add(token)
        if len(distinct) > 1:
            return True
    return False


@_trace_scan_perf
def multiple_choice_accuracy(
    llm_answer: str,
    answer_letter: str,
    answer_text: str,
    prefix: Optional[str] = None,
    accept_answer_text: bool = True,
    strip_tex: bool = True,
    return_details: bool = False,
) -> bool | MCQAccuracyResult:
    """
    Grade a multiple-choice answer with layered strategies:

    1. Direct answer: Response is just the option letter/number
    2. Anchored token: Use the last occurrence of a provided prefix, otherwise general anchor phrases
    3. Last token: Parse a terminal option line or short final clause near the end
    4. Answer text: Match the full answer text (if long enough)

    Args:
        llm_answer: The model's response text
        answer_letter: The correct answer letter/number (e.g., "C" or "3")
        answer_text: The full correct answer text
        prefix: Optional prefix to strip (e.g., "The answer is: ")
        accept_answer_text: Whether to fall back to text matching
        strip_tex: Whether to strip LaTeX formatting
        return_details: If True, return MCQAccuracyResult dataclass instead of bool

    Returns:
        bool (if return_details=False) or MCQAccuracyResult (if return_details=True)
    """

    def _result(
        is_correct: bool, method: str, predicted: str | None, actual: str | None, return_details: bool
    ) -> bool | MCQAccuracyResult:
        """Helper to format return value."""
        if not return_details:
            return is_correct
        return MCQAccuracyResult(
            is_correct=is_correct,
            method=method,
            matched_answer=predicted,
            correct_answer=actual,
        )

    if not llm_answer:
        return _result(False, "none", None, None, return_details)

    # Normalize the response
    llm_answer = _remove_think_tags(llm_answer)

    if strip_tex:
        llm_answer = _strip_tex(llm_answer)
        answer_text = _strip_tex(answer_text)

    # Keep two views of the response:
    # - structural_text preserves original spacing for sentence/line-sensitive heuristics
    # - normalized_answer casefolds and normalizes punctuation for anchor/text matching
    structural_text = llm_answer
    normalized_answer = normalize_for_structure(llm_answer)

    answer_letter = _norm_letter(answer_letter)
    answer_text = normalize_for_match(answer_text or "")
    if answer_letter is None:
        raise ValueError(f"Invalid answer_letter '{answer_letter=}'. Must be a single letter or digit string.")

    # Once we see any explicit option selection of the right token kind, we stop lower-confidence
    # fallbacks from overriding it with a tail token or answer-text mention.
    explicit_choice_found = False

    # Strategy 1: Only answer letter anywhere (without anchoring)
    if answer_letter == _norm_letter(normalized_answer):
        return _result(True, "direct_answer", normalized_answer, answer_letter, return_details)

    # A response that begins like "B. ..." gets special handling: we may disable both the leading
    # shortcut and later tail/text fallbacks if it actually looks like multiple labeled options.
    leading_match = LEADING_OPTION_PATTERN.match(structural_text)
    multiple_option_led_sentences = False

    if leading_match:
        # Only pay for the additional answer scan when the payload actually starts with a leading
        # option pattern; otherwise we leave this guard disabled for the cheaper later paths.
        multiple_option_led_sentences = len(
            structural_text
        ) <= _MULTIPLE_OPTION_LED_SCAN_MAX_CHARS and _contains_multiple_option_led_sentences(
            structural_text, answer_letter
        )
        # If the response looks like multiple labeled answer statements, do not treat the first
        # label as the chosen answer.
        if multiple_option_led_sentences:
            leading_match = None

    # Strategy 2: Accept leading option token like "B. answer ..."
    if leading_match and answer_letter:
        predicted = _norm_letter(leading_match.group(1))
        if _token_kind_matches_answer_letter(predicted, answer_letter):
            explicit_choice_found = True
        if predicted == answer_letter:
            return _result(True, "anchored_token", predicted, answer_letter, return_details)

    # Strategy 3: Anchored token (prefix matches first, fallback to generic anchors)
    prefix_matches = []
    if prefix:
        prefix_norm = normalize_for_structure(prefix).strip()
        if prefix_norm:
            prefix_matches = list(_prefix_pattern(prefix_norm).finditer(normalized_answer))

    anchored_matches = prefix_matches if prefix_matches else list(ANCHOR_PATTERN.finditer(normalized_answer))
    if anchored_matches and answer_letter:
        # Walk anchored matches from the end so later corrections like "Answer: B ... final answer: C"
        # resolve to the last non-negated, non-multi-answer anchor.
        for idx in range(len(anchored_matches) - 1, -1, -1):
            match = anchored_matches[idx]
            predicted = _match_token(match)
            if predicted is None:
                continue
            if match.group("neg") is not None:
                continue
            if _match_is_contradicted(_sentence_match_context(normalized_answer, match)):
                continue
            if _anchored_match_in_multi_answer_phrase(normalized_answer, anchored_matches, idx):
                continue

            if _token_kind_matches_answer_letter(predicted, answer_letter):
                explicit_choice_found = True
            if predicted == answer_letter:
                return _result(True, "anchored_token", predicted, answer_letter, return_details)
            break

    # Strategy 4: Parse a terminal option line or short final clause near the end.
    # Tail parsing is lower confidence than explicit anchors, so it only runs when no explicit
    # option token has already been observed.
    if not explicit_choice_found and answer_letter and not multiple_option_led_sentences:
        predicted = _extract_terminal_option_line(_last_nonempty_line(normalized_answer))
        if predicted == answer_letter:
            return _result(True, "last_token", predicted, answer_letter, return_details)

        predicted = _extract_short_final_clause_option(normalized_answer)
        if predicted == answer_letter:
            return _result(True, "last_token", predicted, answer_letter, return_details)

    # Strategy 5: Exact answer text match if there's no explicit choice found
    # Only search at beginning and end to avoid matching reasoning in the middle
    if accept_answer_text and answer_text and not explicit_choice_found:
        # A multi-option-led payload is too ambiguous for answer-text fallback.
        if multiple_option_led_sentences:
            return _result(False, "none", None, None, return_details)

        # Calculate search regions based on token count
        answer_tokens = len(answer_text.split())
        buffer_tokens = answer_tokens + 15  # Extra tokens for preamble like "The answer is:"

        llm_tokens = normalized_answer.split()

        beginning_tokens = llm_tokens[:buffer_tokens]
        end_tokens = llm_tokens[-buffer_tokens:] if len(llm_tokens) > buffer_tokens else llm_tokens

        beginning_region = " ".join(beginning_tokens)
        end_region = " ".join(end_tokens)

        # First try the normalized answer text directly, then a slightly looser punctuation-tolerant
        # variant, but only in the beginning/end windows rather than the full reasoning trace.
        flexible_answer = re.escape(answer_text).replace(r"\ ", r"\s+")
        normed_answer_text = normalize_for_answer_text_match(answer_text)
        pattern = re.compile(rf"(?<!\w){flexible_answer}(?!\w)", re.IGNORECASE)
        loose_pattern = None
        if normed_answer_text:
            flexible_loose_answer = re.escape(normed_answer_text).replace(r"\ ", r"\s+")
            loose_pattern = re.compile(rf"(?<!\w){flexible_loose_answer}(?!\w)", re.IGNORECASE)

        # Check the beginning and end windows with the same two-stage matcher: exact normalized
        # answer text first, then the looser punctuation-tolerant variant.
        for region in (beginning_region, end_region):
            match = pattern.search(region)
            if match and not _match_is_negated(_sentence_match_context(region, match)):
                return _result(True, "answer_text", region, answer_text, return_details)

            if loose_pattern is None:
                continue

            loose_region = normalize_for_answer_text_match(region)
            match = loose_pattern.search(loose_region)
            if match and not _match_is_negated(_sentence_match_context(loose_region, match)):
                return _result(True, "answer_text", region, answer_text, return_details)

    return _result(False, "none", None, None, return_details)
