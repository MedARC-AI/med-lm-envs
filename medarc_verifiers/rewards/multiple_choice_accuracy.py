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


def normalize_for_structure(text: str) -> str:
    """Canonicalize text for structural matching without collapsing whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_UNICODE_PUNCT_TRANSLATIONS)
    return text.casefold()


def normalize_for_match(text: str) -> str:
    """Canonicalize text for answer-text equivalence matching."""
    return _WHITESPACE_RE.sub(" ", normalize_for_structure(text)).strip()


def normalize_for_answer_text_match(text: str) -> str:
    """Canonicalize text for answer-text matching while tolerating minor punctuation drift."""
    text = normalize_for_match(text)
    text = _INNER_PUNCT_SPACING_RE.sub(r"\1", text)
    return _TRAILING_TERMINAL_PUNCT_RE.sub("", text)


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
    r"\s*(?:>\s*)?(?:(?:[-*+]\s+)|(?:\d{1,3}[.)]\s+))?\s*"  # blockquote / list prefixes
    r"(?:[*_`~]+)?\s*\(?\s*([A-Za-z]|\d{1,2})\s*"  # markdown wrappers before the option
    r"(?:"
    r"[\)\.:]\s*\)?\s*(?:[*_`~]+)?\s*"  # B. Answer text / C) ...
    r"|"
    r"(?=\s*(?:\(|[-–—]))"  # A (Answer text) / A - Answer text / A – Answer text
    r")"
    r"(?!\w)"
)

# Leading option token like "B. Answer text" or "C) ..." at the start of the response
LEADING_OPTION_PATTERN = re.compile(rf"^{_LEADING_OPTION_PATTERN_BODY}", re.IGNORECASE)
# Same pattern without ^ so we can match from sentence offsets without slicing `text[start:]`.
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

# Compact-list glue that should cause the last-token fallback to reject a tail as
# multi-answer rather than selecting the final option.
COMPACT_MULTI_OPTION_GLUE_PATTERN = re.compile(
    r"""
    \b(?:and|or|both|y|e|ou|und|et|plus)\b
    |
    \b(?:as\ well\ as|together\ with|followed\ by|correct\ choices?\s+are|choices?\s+are)\b
    |
    [,:;/&+\-|]
    """,
    re.IGNORECASE | re.VERBOSE,
)


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
    return bool(NEGATION_BEFORE_MATCH_PATTERN.search(prefix))


def _negative_after_option(text: str, match: re.Match) -> bool:
    """Check if an option token is immediately followed by negative context like 'C is incorrect'."""
    _sentence_start, sentence_end, _match_start, match_end = _get_sentence_containing_match(text, match)
    suffix = text[match_end:sentence_end]
    return bool(NEGATIVE_AFTER_OPTION_PATTERN.search(suffix))


def _contradicted_by_later_option(text: str, match: re.Match) -> bool:
    """Check for same-sentence corrections like 'C, but D is correct' or 'C rather than D'."""
    _sentence_start, sentence_end, _match_start, match_end = _get_sentence_containing_match(text, match)
    suffix = text[match_end:sentence_end]
    current = _norm_letter(
        match.group("opt") if getattr(match.re, "groupindex", None) and "opt" in match.re.groupindex else match.group(1)
    )
    later = CONTRAST_PATTERN.search(suffix)
    if not later:
        return False
    contrasted = _norm_letter(later.group(1))
    return contrasted is not None and contrasted != current


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


def _last_nonempty_line(text: str) -> str:
    """Return the last non-empty line, if any."""
    for line in reversed((text or "").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _option_candidate_invalid(text: str, match: re.Match) -> bool:
    """Return True if an option-like match is negated or contradicted in local context."""
    return _negated_near(text, match) or _negative_after_option(text, match) or _contradicted_by_later_option(text, match)


def _ignore_prior_option_like_token(prefix: str, prior_match: re.Match) -> bool:
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


def _extract_terminal_option_line(line: str) -> Optional[str]:
    """Extract a standalone option token from the last line."""
    if not line:
        return None

    match = TERMINAL_OPTION_LINE_PATTERN.fullmatch(line)
    if match:
        predicted = _norm_letter(match.group("opt"))
        if predicted is None:
            return None

        tokens = list(TOKEN_PATTERN.finditer(line))
        if len(tokens) != 1:
            return None

        token_match = tokens[0]
        if _option_candidate_invalid(line, token_match):
            return None

        return predicted

    leading_match = LEADING_OPTION_PATTERN.match(line)
    if not leading_match or _is_compact_multi_option_list(line):
        return None

    predicted = _norm_letter(leading_match.group(1))
    if predicted is None:
        return None

    if _option_candidate_invalid(line, leading_match):
        return None

    return predicted


def _extract_short_final_clause_option(text: str, max_words: int = 12) -> Optional[str]:
    """Extract a terminal option token from a short final clause like 'I think it's C'."""
    clause = _tail_region(text).strip()
    if not clause or len(clause.split()) > max_words:
        return None
    if _is_compact_multi_option_list(clause):
        return None

    match = FINAL_CLAUSE_TERMINAL_OPTION_RE.search(clause)
    if not match:
        return None

    token_match = match
    if _option_candidate_invalid(clause, token_match):
        return None

    # Reject short clauses that contain another meaningful option token before the final token.
    prefix = clause[:token_match.start()]
    for prior_match in TOKEN_PATTERN.finditer(prefix):
        token = _norm_letter(prior_match.group(1))
        if token is None:
            continue

        if _ignore_prior_option_like_token(prefix, prior_match):
            continue

        return None

    return _norm_letter(match.group("opt"))


# Connector words that join two option tokens into a multi-answer phrase.
# Catches "A and C", "A to C", "A through C", "neither A nor C", etc.
_MULTI_ANSWER_CONNECTOR_WORD_RE = re.compile(
    r"\b(?:and|or|nor|to|through|then|plus)\b"
    r"|\bas\s+well\s+as\b"
    r"|\btogether\s+with\b"
    r"|\bfollowed\s+by\b",
    re.IGNORECASE,
)

def _anchored_match_in_multi_answer_phrase(text: str, matches: list[re.Match], idx: int) -> bool:
    """Return True if anchored match *idx* is part of a local multi-answer phrase."""
    match = matches[idx]
    current = _norm_letter(match.group("opt"))
    if current is None:
        return False

    if idx > 0:
        between = text[matches[idx - 1].end() : match.start()]
        if len(between.split()) <= 5 and _MULTI_ANSWER_CONNECTOR_WORD_RE.search(between):
            return True

    if idx < len(matches) - 1:
        between = text[match.end() : matches[idx + 1].start()]
        if len(between.split()) <= 5 and _MULTI_ANSWER_CONNECTOR_WORD_RE.search(between):
            return True

    pre_text = text[max(0, match.start() - 20) : match.start()]
    if bool(
        re.search(
            r"(?<![\w+\-/])[\(\[\{<]*[*_`~]*\s*(?:[A-Za-z]|\d{1,2})\s*[*_`~]*[\)\]\}>]*"
            r"[\s,;]*"
            r"(?:and|or|nor|to|through|then|plus)\s*$",
            pre_text,
            re.IGNORECASE,
        )
    ):
        return True

    sentence_start, sentence_end, match_start, match_end = _get_sentence_containing_match(text, match)
    sentence = text[sentence_start:sentence_end]
    local_match_start = match_start - sentence_start
    local_match_end = match_end - sentence_start

    sentence_tokens = []
    for token_match in TOKEN_PATTERN.finditer(sentence):
        token = _norm_letter(token_match.group(1))
        if token is None:
            continue
        sentence_tokens.append((token, token_match))

    for token, token_match in sentence_tokens:
        if token == current:
            continue
        between = ""
        if token_match.end() <= local_match_start:
            between = sentence[token_match.end() : local_match_start]
        elif local_match_end <= token_match.start():
            between = sentence[local_match_end : token_match.start()]
        if not between:
            continue
        if len(between.split()) <= 5 and _MULTI_ANSWER_CONNECTOR_WORD_RE.search(between):
            return True

    return False


def _is_compact_multi_option_list(text: str) -> bool:
    """Return True for short multi-option tails like 'A, C' or '> **A** and C'."""
    text = (text or "").strip()
    matches = list(TOKEN_PATTERN.finditer(text))
    if len(matches) < 2:
        return False

    residue = TOKEN_PATTERN.sub(" ", text)
    residue = COMPACT_MULTI_OPTION_GLUE_PATTERN.sub(" ", residue)
    residue = re.sub(r"[\s\[\]\(\)\{\}<>*_`~.!?]+", " ", residue)
    return residue.strip() == ""


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

    llm_answer_original = llm_answer

    # Normalize: casefold only (preserve whitespace structure for sentence detection)
    llm_answer = normalize_for_structure(llm_answer)

    answer_letter = _norm_letter(answer_letter)
    answer_text = normalize_for_match(answer_text or "")
    if answer_letter is None:
        raise ValueError(f"Invalid answer_letter '{answer_letter=}'. Must be a single letter or digit string.")

    explicit_choice_found = False
    multiple_option_led_sentences = _contains_multiple_option_led_sentences(llm_answer_original, answer_letter)

    # Strategy 1: Only answer letter anywhere (without anchoring)
    if answer_letter == _norm_letter(llm_answer):
        return _result(True, "direct_answer", llm_answer, answer_letter, return_details)

    # Strategy 2: Accept leading option token like "B. answer ..."
    leading_match = None if multiple_option_led_sentences else LEADING_OPTION_PATTERN.match(llm_answer_original)
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
            flexible_prefix = re.escape(prefix_norm).replace(r"\ ", r"\s+")
            prefix_pattern = re.compile(
                rf"{flexible_prefix}\s*[:\-–—]?\s*(?:is\s*)?(?P<neg>not\s+|isn['’]t\s+)?\(?\s*(?P<opt>[A-Za-z]|\d{{1,2}})\s*[\)\.:]?(?![\w+\-/])",
                re.IGNORECASE,
            )
            prefix_matches = list(prefix_pattern.finditer(llm_answer))

    anchored_matches = prefix_matches if prefix_matches else list(ANCHOR_PATTERN.finditer(llm_answer))
    if anchored_matches and answer_letter:
        for idx in range(len(anchored_matches) - 1, -1, -1):
            match = anchored_matches[idx]
            predicted = _norm_letter(match.group("opt"))
            if predicted is None:
                continue
            if match.group("neg") is not None:
                continue
            if _contradicted_by_later_option(llm_answer, match):
                continue
            if _anchored_match_in_multi_answer_phrase(llm_answer, anchored_matches, idx):
                continue

            if _token_kind_matches_answer_letter(predicted, answer_letter):
                explicit_choice_found = True
            if predicted == answer_letter:
                return _result(True, "anchored_token", predicted, answer_letter, return_details)
            break

    # Strategy 4: Parse a terminal option line or short final clause near the end.
    if not explicit_choice_found and answer_letter and not multiple_option_led_sentences:
        predicted = _extract_terminal_option_line(_last_nonempty_line(llm_answer))
        if predicted == answer_letter:
            return _result(True, "last_token", predicted, answer_letter, return_details)

        predicted = _extract_short_final_clause_option(llm_answer)
        if predicted == answer_letter:
            return _result(True, "last_token", predicted, answer_letter, return_details)

    # Strategy 5: Exact answer text match if there's no explicit choice found
    # Only search at beginning and end to avoid matching reasoning in the middle
    if accept_answer_text and answer_text and not explicit_choice_found:
        if multiple_option_led_sentences:
            return _result(False, "none", None, None, return_details)

        # Calculate search regions based on token count
        answer_tokens = len(answer_text.split())
        buffer_tokens = answer_tokens + 15  # Extra tokens for preamble like "The answer is:"

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

        normed_answer_text = normalize_for_answer_text_match(answer_text)
        normed_beginning_region = normalize_for_answer_text_match(beginning_region)
        normed_end_region = normalize_for_answer_text_match(end_region)

        if normed_answer_text:
            flexible_loose_answer = re.escape(normed_answer_text).replace(r"\ ", r"\s+")
            loose_pattern = re.compile(rf"(?<!\w){flexible_loose_answer}(?!\w)", re.IGNORECASE)

            match = loose_pattern.search(normed_beginning_region)
            if match and not _negated_near(normed_beginning_region, match):
                return _result(True, "answer_text", beginning_region, answer_text, return_details)

            match = loose_pattern.search(normed_end_region)
            if match and not _negated_near(normed_end_region, match):
                return _result(True, "answer_text", end_region, answer_text, return_details)

    return _result(False, "none", None, None, return_details)
