"""
Simplified multiple-choice question accuracy grader.

Main use case: Handle models that either return the letter/number (preferred)
or return the entire answer text verbatim (fallback).

Supports chain-of-thought by prioritizing anchored patterns like "answer is X"
before falling back to last token or text matching.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass
class MCQAccuracyResult:
    """Result of multiple-choice accuracy grading."""

    is_correct: bool
    """Whether the answer was graded as correct."""

    method: str
    """Method used for grading: 'anchored_token', 'last_token', 'answer_text', or 'none'."""

    predicted_letter: Optional[str] = None
    """The extracted letter/number if found, otherwise None."""


def _nfkc_casefold(s: str) -> str:
    """Unicode normalize + casefold for robust text comparison."""
    return unicodedata.normalize("NFKC", s or "").casefold()


def _normalize_spaces(s: str) -> str:
    """Collapse multiple whitespace to single space."""
    return re.sub(r"\s+", " ", s).strip()


def _strip_tex(s: str) -> str:
    """Remove LaTeX formatting if pylatexenc is available."""
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        return LatexNodes2Text(math_mode="text").latex_to_text(s)
    except Exception:
        return s


def _norm_letter(tok: str) -> Optional[str]:
    """Normalize a token to uppercase letter or digit string."""
    tok = (tok or "").strip()
    if not tok:
        return None
    if tok.isdigit():
        return tok
    if tok.isalpha() and len(tok) == 1:
        return tok.upper()
    return None


# Anchored patterns like "final answer: C" or "the answer is D"
ANCHOR_PATTERN = re.compile(
    r"(?:final\s+answer|answer|ans|choice|option|selected|i\s+choose|i\s+pick|therefore|thus|so)\s*"
    r"[:\-–—]?\s*(?:is\s*)?\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]?(?!\w)",
    re.IGNORECASE,
)

# Any letter/number token that looks like an option
TOKEN_PATTERN = re.compile(r"(?<!\w)\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]?(?!\w)", re.IGNORECASE)

# Negation words that invalidate nearby matches
NEGATION_PATTERN = re.compile(
    r"\b(?:not|isn['']t|no|incorrect|wrong|eliminat\w+|except|rather\s+than)\b", re.IGNORECASE
)

# Sentence boundary pattern - splits on period, exclamation, question mark, or newline
# Handles both single newlines (for line breaks in CoT) and double newlines (paragraphs)
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+|\n+")


def _get_sentence_containing_match(text: str, match: re.Match) -> str:
    """
    Extract the sentence/clause containing the match.
    This helps avoid false positives from negations in earlier sentences.
    """
    start, end = match.span()

    # Find sentence boundaries before and after the match
    boundaries_before = [m.end() for m in SENTENCE_BOUNDARY.finditer(text[:start])]
    boundaries_after = [m.start() for m in SENTENCE_BOUNDARY.finditer(text[end:])]

    # Get the start of current sentence (after last boundary before match, or beginning)
    sentence_start = boundaries_before[-1] if boundaries_before else 0

    # Get the end of current sentence (at first boundary after match, or end)
    sentence_end = end + boundaries_after[0] if boundaries_after else len(text)

    return text[sentence_start:sentence_end]


def _negated_near(text: str, match: re.Match) -> bool:
    """
    Check if a negation word appears in the same sentence as the match.
    This is more accurate than a fixed-window approach.
    """
    sentence = _get_sentence_containing_match(text, match)
    return bool(NEGATION_PATTERN.search(sentence))


def multiple_choice_accuracy(
    llm_answer: str,
    answer_letter: str,
    answer_text: str,
    prefix: Optional[str] = None,
    accept_answer_text: bool = True,
    strip_tex: bool = True,
    min_answer_len: int = 4,
    return_details: bool = False,
) -> bool | MCQAccuracyResult:
    """
    Grade a multiple-choice answer with three fallback strategies:

    1. Anchored token: Look for patterns like "answer is C"
    2. Last token: Take the last letter/number found anywhere
    3. Answer text: Match the full answer text (if long enough)

    Args:
        llm_answer: The model's response text
        answer_letter: The correct answer letter/number (e.g., "C" or "3")
        answer_text: The full correct answer text
        prefix: Optional prefix to strip (e.g., "The answer is: ")
        accept_answer_text: Whether to fall back to text matching
        strip_tex: Whether to strip LaTeX formatting
        min_answer_len: Minimum length for answer text matching (avoid false positives)
        return_details: If True, return MCQAccuracyResult dataclass instead of bool

    Returns:
        bool (if return_details=False) or MCQAccuracyResult (if return_details=True)
    """
    if not llm_answer:
        return _result(False, "none", None, return_details)

    # Normalize the response
    raw = llm_answer.strip()
    if strip_tex:
        raw = _strip_tex(raw)
        answer_text = _strip_tex(answer_text)

    # Strip optional prefix like "The answer is: ..."
    if prefix:
        raw = re.sub(rf"^\s*{re.escape(prefix)}\s*[:\-–—]*\s*", "", raw, flags=re.IGNORECASE)

    # Normalize: casefold only (preserve whitespace structure for sentence detection)
    normalized = _nfkc_casefold(raw)

    answer_letter_norm = _norm_letter(answer_letter)
    answer_text_norm = _nfkc_casefold(_normalize_spaces(answer_text or ""))

    # Disable text matching for very short answers (too risky)
    if accept_answer_text and len(answer_text_norm) < min_answer_len:
        accept_answer_text = False

    # Strategy 1: Anchored token (e.g., "final answer: C")
    anchored_matches = list(ANCHOR_PATTERN.finditer(normalized))
    if anchored_matches and answer_letter_norm:
        last_match = anchored_matches[-1]
        predicted = _norm_letter(last_match.group(1))
        if predicted == answer_letter_norm and not _negated_near(normalized, last_match):
            return _result(True, "anchored_token", predicted, return_details)

    # Strategy 2: Last token anywhere
    all_tokens = list(TOKEN_PATTERN.finditer(normalized))
    if all_tokens and answer_letter_norm:
        last_match = all_tokens[-1]
        predicted = _norm_letter(last_match.group(1))
        if predicted == answer_letter_norm and not _negated_near(normalized, last_match):
            return _result(True, "last_token", predicted, return_details)

    # Strategy 3: Exact answer text match
    if accept_answer_text and answer_text_norm:
        # Search in normalized text (preserves structure for negation checking)
        # Make answer_text_norm flexible for whitespace variations
        flexible_answer = re.escape(answer_text_norm).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){flexible_answer}(?!\w)", re.IGNORECASE)
        match = pattern.search(normalized)
        if match and not _negated_near(normalized, match):
            return _result(True, "answer_text", None, return_details)

    return _result(False, "none", None, return_details)


def _result(
    is_correct: bool,
    method: str,
    predicted_letter: Optional[str],
    return_details: bool,
) -> bool | MCQAccuracyResult:
    """Helper to format return value."""
    if not return_details:
        return is_correct
    return MCQAccuracyResult(
        is_correct=is_correct,
        method=method,
        predicted_letter=predicted_letter,
    )
