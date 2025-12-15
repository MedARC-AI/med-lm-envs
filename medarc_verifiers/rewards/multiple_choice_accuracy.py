"""
LLM multiple-choice question accuracy reward.

Main use case: Handle models that either return the letter/number (preferred)
or return the entire answer text verbatim (fallback).

Supports chain-of-thought by prioritizing anchored patterns like "answer is X"
before falling back to last token or text matching. Attempts to recognize
negations to avoid false positives (e.g., "the answer is not C").
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
    """Method used for grading: 'direct_answer', 'anchored_token', 'last_token', 'answer_text', or 'none'."""

    matched_answer: Optional[str] = None
    """The extracted answer if found, otherwise None."""

    correct_answer: Optional[str] = None
    """The correct answer for reference, if available."""


def _nfkc_casefold(text: str) -> str:
    """Unicode normalize + casefold for robust text comparison."""
    return unicodedata.normalize("NFKC", text or "").casefold()


def _normalize_spaces(text: str) -> str:
    """Collapse multiple whitespace to single space."""
    return re.sub(r"\s+", " ", text).strip()


def _strip_tex(text: str) -> str:
    """Remove LaTeX formatting if pylatexenc is available."""
    try:
        from pylatexenc.latex2text import LatexNodes2Text

        return LatexNodes2Text(math_mode="text").latex_to_text(text)
    except Exception:
        return text


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


# Anchored patterns like "final answer: C" or "the answer is D"
ANCHOR_PATTERN = re.compile(
    r"(?:final\s+answer|answer|ans|choice|option|selected|i\s+choose|i\s+pick|therefore|thus|so)\s*"
    r"[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?\(?\s*(?P<opt>[A-Za-z]|\d{1,2})\s*[\)\.:]?(?!\w)",
    re.IGNORECASE,
)

# Any letter/number token that looks like an option
TOKEN_PATTERN = re.compile(r"(?<!\w)\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]?(?!\w)", re.IGNORECASE)

# Leading option token like "B. Answer text" or "C) ..." at the start of the response
LEADING_OPTION_PATTERN = re.compile(
    r"^\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?\s*"  # blockquote / list prefixes
    r"(?:[*_`~]+)?\s*\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]\s*\)?\s*(?:[*_`~]+)?\s*(?!\w)",  # markdown wrappers
    re.IGNORECASE,
)

# Negation words that invalidate nearby matches
NEGATION_PATTERN = re.compile(r"\b(?:not|isn['’]t)\b", re.IGNORECASE)

# Sentence boundary pattern - splits on period, exclamation, question mark, or newline
# Handles both single newlines (for line breaks in CoT) and double newlines (paragraphs)
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+|\n+")


def _get_sentence_containing_match(text: str, match: re.Match) -> str:
    """
    Extract the sentence/clause containing the match.
    This helps avoid false positives from negations in earlier sentences.
    """
    # Use the span of the actual option group when available, so we don't start the match at leading whitespace.
    if getattr(match.re, "groupindex", None) and "opt" in match.re.groupindex:
        start, end = match.span("opt")
    else:
        try:
            start, end = match.span(1)
        except Exception:
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
    return_details: bool = False,
) -> bool | MCQAccuracyResult:
    """
    Grade a multiple-choice answer with layered strategies:

    1. Direct answer: Response is just the option letter/number
    2. Anchored token: Use the last occurrence of a provided prefix, otherwise general anchor phrases
    3. Last token: Take the last letter/number found anywhere
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
    llm_answer = llm_answer.strip()
    if strip_tex:
        llm_answer = _strip_tex(llm_answer)
        answer_text = _strip_tex(answer_text)

    llm_answer_original = llm_answer

    # Normalize: casefold only (preserve whitespace structure for sentence detection)
    llm_answer = _nfkc_casefold(llm_answer)

    answer_letter = _norm_letter(answer_letter)
    answer_text = _nfkc_casefold(_normalize_spaces(answer_text or ""))
    if answer_letter is None:
        raise ValueError(f"Invalid answer_letter '{answer_letter=}'. Must be a single letter or digit string.")

    explicit_choice_found = False

    # Strategy 1: Only answer letter anywhere (without anchoring)
    if answer_letter == _norm_letter(llm_answer):
        return _result(True, "direct_answer", llm_answer, answer_letter, return_details)

    # Strategy 2: Accept leading option token like "B. answer ..."
    leading_match = LEADING_OPTION_PATTERN.match(llm_answer_original)
    if leading_match and answer_letter:
        explicit_choice_found = True
        predicted = _norm_letter(leading_match.group(1))
        if predicted == answer_letter:
            return _result(True, "anchored_token", predicted, answer_letter, return_details)

    # Strategy 3: Anchored token (prefix matches first, fallback to generic anchors)
    prefix_matches = []
    if prefix:
        prefix_norm = _nfkc_casefold(prefix).strip()
        if prefix_norm:
            flexible_prefix = re.escape(prefix_norm).replace(r"\ ", r"\s+")
            prefix_pattern = re.compile(
                rf"{flexible_prefix}\s*[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?\(?\s*(?P<opt>[A-Za-z]|\d{{1,2}})\s*[\)\.:]?(?!\w)",
                re.IGNORECASE,
            )
            prefix_matches = list(prefix_pattern.finditer(llm_answer))

    anchored_matches = prefix_matches if prefix_matches else list(ANCHOR_PATTERN.finditer(llm_answer))
    if anchored_matches and answer_letter:
        last_match = anchored_matches[-1]
        predicted = _norm_letter(last_match.group("opt"))
        if last_match.group("neg") is None:
            explicit_choice_found = True
        if predicted == answer_letter and last_match.group("neg") is None:
            return _result(True, "anchored_token", predicted, answer_letter, return_details)

    # Strategy 4: Last token anywhere
    all_tokens = list(TOKEN_PATTERN.finditer(llm_answer))
    if all_tokens and answer_letter:
        last_match = all_tokens[-1]
        predicted = _norm_letter(last_match.group(1))
        if predicted == answer_letter and not _negated_near(llm_answer, last_match):
            return _result(True, "last_token", predicted, answer_letter, return_details)

    # Strategy 5: Exact answer text match if there's no explicit choice found
    if accept_answer_text and answer_text and not explicit_choice_found:
        # Search in normalized text (preserves structure for negation checking)
        # Make answer_text flexible for whitespace variations
        flexible_answer = re.escape(answer_text).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){flexible_answer}(?!\w)", re.IGNORECASE)
        match = pattern.search(llm_answer)
        if match:
            return _result(True, "answer_text", llm_answer, answer_text, return_details)

    return _result(False, "none", None, None, return_details)
