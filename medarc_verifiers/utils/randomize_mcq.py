import hashlib
import random
import re

ANCHOR = re.compile(r"\b(?:all|none|some|both|neither)\s+of\s+the\s+(?:above|following)\b", re.IGNORECASE)


def randomize_multiple_choice(
    options: list[str] | dict[str, str],
    answer_choice: str | int,
    labels: list[str] | None = None,
    seed: int | None = None,
    row_id: str | int | None = None,
) -> tuple[list[str] | dict[str, str], str, int]:
    """Randomize MCQ options while preserving anchor options in place.

    Anchor options (e.g., "All of the above", "None of the following") stay fixed.
    Only non-anchor options between anchors are shuffled within their blocks.

    Args:
        options: List of option texts OR dict mapping labels to option texts.
        answer_choice: Original answer as 0-based index OR label string like "C", "(B)", "3.", etc.
        labels: Required when options is a list. Label strings for each option (e.g., ["A", "B", "C"]).
        seed: Randomization policy:
            - None: No shuffling, return unchanged
            - -1: Non-deterministic random shuffle
            - int >= 0: Deterministic shuffle (combined with row_id if provided)
        row_id: Optional identifier mixed into deterministic seed for per-row variation.

    Returns:
        Tuple of (shuffled_options, new_answer_label, new_answer_index)
        - shuffled_options: Same type as input (list or dict) with shuffled values
        - new_answer_label: Label string where the answer moved (e.g., "B", "(C)", "2.")
        - new_answer_index: 0-based index where the answer moved

    Examples:
        >>> opts = ["Opt A", "Opt B", "All of the above"]
        >>> shuffled, label, idx = randomize_mcq(opts, 0, labels=["A", "B", "C"], seed=42)
        >>> # First two options may shuffle, but "All of the above" stays at index 2

        >>> opts_dict = {"A": "Opt 1", "B": "Both of the above", "C": "Opt 2"}
        >>> shuffled, label, idx = randomize_mcq(opts_dict, "A", seed=42, row_id="q1")
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

    # choose RNG
    if seed is None:
        rng = None  # no shuffle
    elif seed == -1:
        rng = random.Random()
    else:
        mix = f"{seed}::{row_id}" if row_id is not None else f"{seed}"
        h = int(hashlib.sha256(mix.encode()).hexdigest(), 16) & ((1 << 64) - 1)
        rng = random.Random(h)

    if rng is None:
        # return unchanged, but conform outputs
        if dict_mode:
            out_options = dict(zip(labels, texts))
            return out_options, labels[answer_idx], answer_idx
        else:
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
        return out_options, labels[new_answer_idx], new_answer_idx
    else:
        return texts, labels[new_answer_idx], new_answer_idx
