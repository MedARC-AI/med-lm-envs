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


def _remove_think_tags(completion_text: str) -> str:
    "Extract the answer section from completion text, handling think tags properly."
    # Check for proper think tags
    think_tag_pairs = re.findall(r"<think>.*?</think>", completion_text, re.DOTALL | re.IGNORECASE)

    has_exactly_one_proper_think_tag = len(think_tag_pairs) == 1

    # Check for malformed tags
    has_malformed_tags = (
        re.search(r"<think>(?:(?!</think>).)*$", completion_text, re.DOTALL | re.IGNORECASE) is not None
    )

    if has_exactly_one_proper_think_tag and not has_malformed_tags:
        # Extract everything after the properly closed </think> tag
        answer_section = re.sub(r".*?</think>", "", completion_text, flags=re.DOTALL | re.IGNORECASE).strip()
        return answer_section.strip()
    else:
        # If no proper think tags, return full response
        return completion_text.strip()


# Anchored patterns like "final answer: C" or "the answer is D"
ANCHOR_PATTERN = re.compile(
    r"(?:\bfinal\s+answer\b|\banswer\b|\bans\b|\bchoice\b|\boption\b|\bselected\b|\bi\s+choose\b|\bi\s+pick\b|\btherefore\b|\bthus\b|\bso\b)\s*"
    r"[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?"
    r"(?:[*_`~]+\s*)*"  # allow markdown wrappers before the option
    r"\(?\s*(?P<opt>[A-Za-z]|\d{1,2})\s*\)?"  # option token, possibly parenthesized
    r"\s*[\)\.:]?\s*"  # optional delimiter (e.g., 'B.' or 'B)')
    r"(?:[*_`~]+\s*)*"  # allow markdown wrappers after the option
    r"(?![\w+\-/])",
    re.IGNORECASE,
)

# Any letter/number token that looks like an option
TOKEN_PATTERN = re.compile(r"(?<![\w+\-/])\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]?(?![\w+\-/])", re.IGNORECASE)

# Leading option token like "B. Answer text" or "C) ..." at the start of the response
LEADING_OPTION_PATTERN = re.compile(
    r"^\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?\s*"  # blockquote / list prefixes
    r"(?:[*_`~]+)?\s*\(?\s*([A-Za-z]|\d{1,2})\s*[\)\.:]\s*\)?\s*(?:[*_`~]+)?\s*(?!\w)",  # markdown wrappers
    re.IGNORECASE,
)

# Negation words that invalidate nearby matches
NEGATION_PATTERN = re.compile(r"\b(?:not|isn['’]t)\b", re.IGNORECASE)

# Negative-context phrases that indicate an option mention is NOT a selected answer
NEGATIVE_AFTER_OPTION_PATTERN = re.compile(
    r"^\s*(?:is|are|was|were)\s+(?:incorrect|wrong|false|not\s+correct)\b|^\s*not\s+correct\b",
    re.IGNORECASE,
)

# Sentence boundary pattern - splits on period, exclamation, question mark, or newline
# Handles both single newlines (for line breaks in CoT) and double newlines (paragraphs)
SENTENCE_BOUNDARY = re.compile(r"[.!?]\s+|\n+")


def _get_sentence_containing_match(text: str, match: re.Match) -> str:
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


def _negated_near(text: str, match: re.Match) -> bool:
    """Check for negation that appears before the match within the same sentence.

    This is used for answer_text matching to avoid blocking answers that legitimately contain
    words like "not" (e.g., "do not resuscitate") while still blocking cases like
    "not <answer_text>".
    """
    sentence_start, sentence_end, match_start, _match_end = _get_sentence_containing_match(text, match)
    prefix = text[sentence_start:match_start]
    return bool(NEGATION_PATTERN.search(prefix))


def _negative_after_option(text: str, match: re.Match) -> bool:
    """Check if an option token is immediately followed by negative context like 'C is incorrect'."""
    _sentence_start, sentence_end, _match_start, match_end = _get_sentence_containing_match(text, match)
    suffix = text[match_end:sentence_end]
    return bool(NEGATIVE_AFTER_OPTION_PATTERN.search(suffix))


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
    llm_answer = _remove_think_tags(llm_answer)

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
                rf"{flexible_prefix}\s*[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?\(?\s*(?P<opt>[A-Za-z]|\d{{1,2}})\s*[\)\.:]?(?![\w+\-/])",
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

    # Strategy 4: Last token in the answer tail, ignore negative contexts like "C is incorrect",
    if not explicit_choice_found and answer_letter:
        tail = _tail_region(llm_answer)
        tail_tokens = list(TOKEN_PATTERN.finditer(tail))
        if tail_tokens:
            # Take the last non-negated, non-negative-context token.
            for token_match in reversed(tail_tokens):
                predicted = _norm_letter(token_match.group(1))
                if predicted is None:
                    continue
                if _negated_near(tail, token_match):
                    continue
                if _negative_after_option(tail, token_match):
                    continue
                if predicted == answer_letter:
                    return _result(True, "last_token", predicted, answer_letter, return_details)

    # Strategy 5: Exact answer text match if there's no explicit choice found
    # Only search at beginning and end to avoid matching reasoning in the middle
    if accept_answer_text and answer_text and not explicit_choice_found:
        # Calculate search regions based on token count
        answer_tokens = len(answer_text.split())
        buffer_tokens = answer_tokens + 8  # Extra tokens for preamble like "The answer is:"

        llm_tokens = llm_answer.split()

        beginning_tokens = llm_tokens[:buffer_tokens]
        end_tokens = llm_tokens[-buffer_tokens:] if len(llm_tokens) > buffer_tokens else llm_tokens

        beginning_region = " ".join(beginning_tokens)
        end_region = " ".join(end_tokens)

        # Make answer_text flexible for whitespace variations
        flexible_answer = re.escape(answer_text).replace(r"\ ", r"\s+")
        pattern = re.compile(rf"(?<!\w){flexible_answer}(?!\w)", re.IGNORECASE)

        # Check beginning first
        match = pattern.search(beginning_region)
        if match and not _negated_near(beginning_region, match):
            return _result(True, "answer_text", beginning_region, answer_text, return_details)

        # Then check end (after reasoning)
        match = pattern.search(end_region)
        if match and not _negated_near(end_region, match):
            return _result(True, "answer_text", end_region, answer_text, return_details)

    return _result(False, "none", None, None, return_details)
