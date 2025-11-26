import hashlib
import random
import re

# Matches phrases like "all/none of the above/following/these" and similar variants
ANCHOR = re.compile(
    r"""
    \b
    (?:all|none|some|both|neither)
    \s+
    (?:of\s+(?:the\s+)?)?
    (?:
        above
        |following
        |these
        |choices?
        |options?
        |answers?
        |statements?
        |responses?
        |listed
        |apply
        |applicable
        |them
    )
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Matches options that reference other option labels (e.g., "A or B", "Both A and C", "A, B, or C")
# This broader pattern detects label references embedded anywhere in the string, not only when the
# entire option equals a bare label list. It requires at least two label tokens connected by a
# conjunction or punctuation (and/or, and, or, nor, ",", "&", "/"). This helps avoid false
# positives like "Vitamin A deficiency" (no second label token), while catching phrases like
# "Both A and B are correct", "Choose A and/or B", or "Options A, B, and C".
_LABEL_TOKEN = r"(?:\((?-i:[A-Z0-9]+)\)|\[(?-i:[A-Z0-9]+)\]|\b(?-i:[A-Z])(?:[)\.:])?|\b\d+(?:[)\.:])?)"
LABEL_REF = re.compile(
    r"""
    (?:\b(?:both|either|neither|only)\b\s+)?   # optional leading qualifier
    {tok}                              # first label token
    (?:\s*[,&/]+\s*|\s+(?:and/or|and|or|nor)\s+)   # required separator
    {tok}                              # second label token
    (?:                                # optionally, additional tokens
        (?:\s*[,&/]+\s*|\s+(?:and/or|and|or|nor)\s+)
        {tok}
    )*
    """.format(tok=_LABEL_TOKEN),
    re.IGNORECASE | re.VERBOSE,
)


def _stable_options_hash(options: list[str] | dict[str, str] | object) -> int:
    """Build a stable hash from the option contents for deterministic seeding."""
    if isinstance(options, list):
        ser = "\x1e".join(["" if o is None else str(o) for o in options])
    elif isinstance(options, dict):
        ser = "\x1f".join([f"{k}\x1e{options[k]}" for k in sorted(options.keys())])
    else:
        ser = str(options)
    return int(hashlib.sha256(ser.encode()).hexdigest(), 16) & ((1 << 64) - 1)


def randomize_multiple_choice(
    options: list[str] | dict[str, str],
    answer_choice: str | int,
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
    return_mapping: bool = False,
) -> tuple[list[str] | dict[str, str], str, int] | tuple[list[str] | dict[str, str], str, int, list[int]]:
    """Randomize MCQ options while preserving anchor options in place.

    Anchors remain in their original positions; only non-anchor segments between/around anchors are shuffled.

    Anchor options (e.g., "All of the above", "None of the following") stay fixed.
    Only non-anchor options between anchors are shuffled within their blocks.

    Label Reference Detection: If any option references other option labels
    (e.g., "C) A or B", "D) Both A and B"), shuffling is SKIPPED entirely for that
    question to avoid breaking the references. This ensures correctness but means
    no randomization occurs when label references are detected.

    Args:
        options: List of option texts OR dict mapping labels to option texts.
        answer_choice: Original answer as 0-based index OR label string like "C", "(B)", "3.", etc.
        labels: Label strings for each option (e.g., ["A", "B", "C"]).
                For list inputs, labels are required and must match the options length.
        seed: Randomization policy:
            - None: No shuffling, return unchanged
            - -1: Non-deterministic random shuffle
            - int >= 0: Deterministic shuffle (combined with row_id if provided)
        row_id: Optional identifier mixed into deterministic seed for per-row variation.
        return_mapping: When True, also returns the permutation mapping list where
            mapping[new_position] = old_index.

    Returns:
        Tuple of (shuffled_options, new_answer_label, new_answer_index[, mapping])
        - shuffled_options: Same type as input (list or dict) with shuffled values
        - new_answer_label: Label string where the answer moved (e.g., "B", "(C)", "2.")
        - new_answer_index: 0-based index where the answer moved
        - mapping (optional): list[int] where mapping[new_position] = old_index

    Examples:
        >>> opts = ["Opt A", "Opt B", "All of the above"]
        >>> shuffled, label, idx = randomize_multiple_choice(opts, 0, labels=["A", "B", "C"], seed=42)
        >>> # First two options may shuffle, but "All of the above" stays at index 2

        >>> opts_dict = {"A": "Opt 1", "B": "Both of the above", "C": "Opt 2"}
        >>> shuffled, label, idx = randomize_multiple_choice(opts_dict, "A", seed=42, row_id="q1")
        >>> # "Both of the above" stays at position B, others may move
    """

    # normalize to parallel lists
    if isinstance(options, dict):
        labels = list(options.keys())
        texts = [options[k] for k in labels]
        dict_mode = True
    else:
        texts = list(options)
        if labels is None:
            raise ValueError("labels must be provided when options is a list")
        if len(labels) != len(texts):
            raise ValueError(
                f"labels length ({len(labels)}) must match number of options ({len(texts)}) for list inputs"
            )
        dict_mode = False

    # map answer_choice to index
    def norm_label(s):
        m = re.search(r"([A-Za-z]+|\d+)", str(s))
        return m.group(1).upper() if m else str(s).upper()

    if isinstance(answer_choice, int):
        answer_idx = answer_choice
        if not (0 <= answer_idx < len(texts)):
            raise ValueError(f"answer_choice={answer_choice!r} is out of range for {len(texts)} options")
    else:
        wanted = norm_label(answer_choice)
        # try alpha (A,B,...) then numeric (1,2,...)
        idx = None
        for i, lab in enumerate(labels):
            if norm_label(lab) == wanted:
                idx = i
                break
        if idx is None and wanted.isalpha():
            idx = ord(wanted) - ord("A")
        if idx is None and wanted.isdigit():
            idx = int(wanted) - 1  # "3" means the 3rd option
        if idx is None or not (0 <= idx < len(texts)):
            raise ValueError(f"answer_choice={answer_choice!r} not found or invalid among labels={labels}")
        answer_idx = idx

    if seed is None:
        rng = None  # no shuffle
    elif seed == -1:
        rng = random.Random()
    else:
        # choose RNG
        if row_id is None:
            row_id = _stable_options_hash(options)

        mix = f"{seed}::{row_id}" if row_id is not None else f"{seed}"
        rng = random.Random(int(hashlib.sha256(mix.encode()).hexdigest(), 16) & ((1 << 64) - 1))

    if rng is None:
        # return unchanged, but conform outputs
        if dict_mode:
            out_options = dict(zip(labels, texts))
            if return_mapping:
                return out_options, labels[answer_idx], answer_idx, list(range(len(texts)))
            return out_options, labels[answer_idx], answer_idx
        else:
            if return_mapping:
                return list(texts), labels[answer_idx], answer_idx, list(range(len(texts)))
            return list(texts), labels[answer_idx], answer_idx

    # check for label references - if found, skip randomization entirely
    # (shuffling would break references like "D) A or B")
    has_label_refs = any(LABEL_REF.search(t or "") for t in texts)
    if has_label_refs:
        if dict_mode:
            out_options = dict(zip(labels, texts))
            if return_mapping:
                return out_options, labels[answer_idx], answer_idx, list(range(len(texts)))
            return out_options, labels[answer_idx], answer_idx
        else:
            if return_mapping:
                return list(texts), labels[answer_idx], answer_idx, list(range(len(texts)))
            return list(texts), labels[answer_idx], answer_idx

    # find anchor positions and build shuffle blocks between them
    n = len(texts)
    anchors = [i for i, t in enumerate(texts) if ANCHOR.search(t or "")]
    blocks = []
    last = 0
    for a in anchors:
        if last < a:
            blocks.append((last, a))  # up to but not including anchor
        last = a + 1
    if last < n:
        blocks.append((last, n))

    # apply independent shuffles per block and track where items move
    index_map = list(range(n))  # new_position -> old_index
    for start, end in blocks:
        idxs = list(range(start, end))
        rng.shuffle(idxs)
        orig_txts = texts[start:end]
        orig_map = index_map[start:end]
        for off, dst in enumerate(idxs):
            texts[start + off] = orig_txts[dst - start]
            index_map[start + off] = orig_map[dst - start]

    new_answer_idx = index_map.index(answer_idx)

    # rebuild output in the same shape as input
    if dict_mode:
        out_options = dict(zip(labels, texts))  # same labels, texts moved
        if return_mapping:
            return out_options, labels[new_answer_idx], new_answer_idx, index_map
        return out_options, labels[new_answer_idx], new_answer_idx
    else:
        if return_mapping:
            return texts, labels[new_answer_idx], new_answer_idx, index_map
        return texts, labels[new_answer_idx], new_answer_idx


def randomize_multiple_choice_hf_map(
    example: dict,
    idx: int | None = None,
    *,
    seed: int = 1618,
    options_key: str = "options",
    answer_key: str = "answer",
    labels: list[str] | None = None,
    return_label: bool = True,
    answer_as_index: bool = True,
) -> dict:
    """HuggingFace datasets-friendly wrapper for randomizing MCQ options.

    Designed to work seamlessly with `dataset.map(randomize_multiple_choice_hf_map, with_indices=True)`.

    Args:
        example: Dataset row as dict (from HF datasets map).
        idx: Row index (provided by with_indices=True). Used as row_id for determinism.
        seed: Random seed for deterministic shuffling.
        options_key: Key in example dict containing the options list/dict.
        answer_key: Key in example dict containing the answer (index or label).
        labels: Optional custom labels. If None and options is a list, auto-generates ["A", "B", "C", ...].
        return_label: If True, adds 'answer_label' field with the letter label.
        answer_as_index: When True (default), stores the shuffled answer as an index.
            Set to False to keep the answer as its letter/label.

    Returns:
        Dict with updated 'options' and 'answer' (and optionally 'answer_label').
        Note: returned 'answer' defaults to an integer index unless `answer_as_index=False`.
        Determinism: when idx is None, the core helper hashes the option contents.
        Can be used directly: `return randomize_multiple_choice_hf_map(example, idx, seed=42)`

    Examples:
        >>> # Basic usage with dataset.map()
        >>> dataset = dataset.map(
        ...     randomize_multiple_choice_hf_map,
        ...     with_indices=True,
        ...     fn_kwargs={'seed': 42},
        ... )

        >>> # Custom field names
        >>> dataset = dataset.map(
        ...     randomize_multiple_choice_hf_map,
        ...     with_indices=True,
        ...     fn_kwargs={'seed': 42, 'options_key': 'choices', 'answer_key': 'correct'},
        ... )

    >>> # Without indices (uses options content hash as row_id)
        >>> def randomize_fn(example):
        ...     return randomize_multiple_choice_hf_map(example, seed=42)
        >>> dataset = dataset.map(randomize_fn)
    """
    options = example[options_key]
    answer = example[answer_key]

    # Use idx as row_id if provided
    row_id = idx

    # Auto-generate labels if not provided and options is a list
    if labels is None and isinstance(options, list):
        labels = [chr(ord("A") + i) for i in range(len(options))]

    shuffled_options, new_label, new_idx = randomize_multiple_choice(
        options=options,
        answer_choice=answer,
        labels=labels,
        seed=seed,
        row_id=row_id,
    )

    result = {options_key: shuffled_options}

    if answer_as_index:
        result[answer_key] = new_idx
    else:
        result[answer_key] = new_label

    if return_label:
        result["answer_label"] = new_label

    return result


def randomize_multiple_choice_row(
    row: dict,
    *,
    options_key: str = "options",
    answer_key: str = "answer",
    answer_text_key: str | None = "answer_text",
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
    return_mapping: bool = False,
) -> dict | tuple[dict, list[int]]:
    """Randomize a single MCQ row dict and return an updated copy.

    Args:
        row: Original row dict containing at least options and answer.
        options_key: Key in the row containing the options list/dict.
        answer_key: Key in the row containing the answer label/index.
        answer_text_key: Optional key to store the correct answer text after shuffling.
        labels: Optional labels for list options; auto-generated (A, B, …) when omitted.
        seed: Seed forwarded to `randomize_multiple_choice`.
        row_id: Optional identifier forwarded to `randomize_multiple_choice`.
        return_mapping: When True, also return the permutation mapping.

    Returns:
        Updated row dict (copy). When `return_mapping` is True, returns a tuple of
        (updated_row, mapping).
    """
    options = row[options_key]
    answer = row[answer_key]

    if labels is None and isinstance(options, list):
        labels = [chr(ord("A") + i) for i in range(len(options))]

    randomized = randomize_multiple_choice(
        options=options,
        answer_choice=answer,
        labels=labels,
        seed=seed,
        row_id=row_id,
        return_mapping=return_mapping,
    )

    if return_mapping:
        shuffled_options, new_label, new_idx, mapping = randomized
    else:
        shuffled_options, new_label, new_idx = randomized
        mapping = None

    updated = dict(row)
    updated[options_key] = shuffled_options
    updated[answer_key] = new_label

    if answer_text_key is not None:
        if isinstance(shuffled_options, dict):
            updated[answer_text_key] = shuffled_options.get(new_label)
        else:
            updated[answer_text_key] = shuffled_options[new_idx]

    if return_mapping:
        return updated, mapping
    return updated
