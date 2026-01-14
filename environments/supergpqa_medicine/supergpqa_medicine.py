import random
import re
from enum import Enum
from typing import Dict

import verifiers as vf
from datasets import Dataset, load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.rewards.multiple_choice_accuracy import multiple_choice_accuracy
from medarc_verifiers.utils.randomize_multiple_choice import randomize_multiple_choice
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, extract_boxed_answer

disable_progress_bar()  # suppress datasets mapping progress bar

ZERO_SHOT_PROMPT_TEMPLATE = """
Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: \\boxed{{$LETTER}}' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

{}
""".strip()

FIVE_SHOT_PROMPT_TEMPLATE = """
Answer the following multiple choice question. There is only one correct answer. The last line of your response should be in the format 'Answer: \\boxed{{$LETTER}}' (without quotes), where LETTER is one of A, B, C, D, E, F, G, H, I, or J.

Question:
A refracting telescope consists of two converging lenses separated by 100 cm. The eye-piece lens has a focal length of 20 cm. The angular magnification of the telescope is
A) 10
B) 40
C) 6
D) 25
E) 15
F) 50
G) 30
H) 4
I) 5
J) 20

Answer: Let's think step by step. In a refracting telescope, if both lenses are converging, the focus of both lenses must be between the two lenses, and thus the focal lengths of the two lenses must add up to their separation. Since the focal length of one lens is 20 cm, the focal length of the other must be 80 cm. The magnification is the ratio of these two focal lengths, or 4.
Answer: \\boxed{{H}}.

Question:
Say the pupil of your eye has a diameter of 5 mm and you have a telescope with an aperture of 50 cm. How much more light can the telescope gather than your eye?
A) 1000 times more
B) 50 times more
C) 5000 times more
D) 500 times more
E) 10000 times more
F) 20000 times more
G) 2000 times more
H) 100 times more
I) 10 times more
J) N/A

Answer: Let's think step by step. The amount of light a telescope can gather compared to the human eye is proportional to the area of its apertures. The area of a circle is given by the formula $A = \pi \left(\frac{{D}}{{2}}\right)^2$, where $D$ is the diameter. Therefore, the relative light-gathering power is calculated as:
\[
\frac{{\left(\frac{{50 \text{{ cm}}}}{{2}}\right)^2}}{{\left(\frac{{5 \text{{ mm}}}}{{2}}\right)^2}} = \frac{{\left(\frac{{50 \text{{ cm}}}}{{0.1 \text{{ cm}}}}\right)^2}}{{\left(\frac{{5 \text{{ mm}}}}{{0.1 \text{{ cm}}}}\right)^2}} = \frac{{500^2}}{{5^2}} = 10000.
\]
Answer: \\boxed{{E}}.

Question:
Where do most short-period comets come from and how do we know?
A) The Kuiper belt; short period comets tend to be in the plane of the solar system like the Kuiper belt.
B) The asteroid belt; short period comets tend to come from random directions indicating a spherical distribution of comets called the asteroid belt.
C) The asteroid belt; short period comets tend to be in the plane of the solar system just like the asteroid belt.
D) The Oort cloud; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the Oort cloud.
E) The Oort Cloud; short period comets tend to come from random directions indicating a spherical distribution of comets called the Oort Cloud.
F) The Oort cloud; short period comets tend to be in the plane of the solar system just like the Oort cloud.
G) The asteroid belt; short period comets have orbital periods similar to asteroids like Vesta and are found in the plane of the solar system just like the asteroid belt.

Answer: Let's think step by step. Most short-period comets originate from the Kuiper belt. This is deduced from the observation that these comets tend to follow orbits that lie in the plane of the solar system, similar to the distribution of objects in the Kuiper belt itself. Thus, the alignment of these cometary orbits with the ecliptic plane points to their Kuiper belt origin.
Answer: \\boxed{{A}}.

Question:
Colors in a soap bubble result from light
A) dispersion
B) deflection
C) refraction
D) reflection
E) interference
F) converted to a different frequency
G) polarization
H) absorption
I) diffraction
J) transmission

Answer: Let's think step by step. The colorful patterns observed in a soap bubble are caused by the phenomenon of light interference. This occurs when light waves bounce between the two surfaces of the soap film, combining constructively or destructively based on their phase differences and the varying thickness of the film. These interactions result in vibrant color patterns due to variations in the intensity of different wavelengths of light.
Answer: \\boxed{{E}}.

Question:
A microwave oven is connected to an outlet, 120 V, and draws a current of 2 amps. At what rate is energy being used by the microwave oven?
A) 240 W
B) 120 W
C) 10 W
D) 480 W
E) 360 W
F) 200 W
G) 30 W
H) 150 W
I) 60 W
J) 300 W

Answer: Let's think step by step. The rate of energy usage, known as power, in an electrical circuit is calculated by the product of voltage and current. For a microwave oven connected to a 120 V outlet and drawing a current of 2 amps, the power consumption can be calculated as follows:
\[
\text{{Power}} = \text{{Voltage}} \times \text{{Current}} = 120 \, \text{{V}} \times 2 \, \text{{A}} = 240 \, \text{{W}}.
\]
Therefore, the microwave oven uses energy at a rate of 240 watts.
Answer: \\boxed{{A}}.

Question:
{}

Answer: Let's think step by step.
""".strip()


class Field(str, Enum):
    ALL = "all"
    BASIC_MEDICINE = "basic_medicine"
    CLINICAL_MEDICINE = "clinical_medicine"
    PHARMACY = "pharmacy"
    PUBLIC_HEALTH_AND_PREVENTIVE_MEDICINE = "public_health_and_preventive_medicine"
    STOMATOLOGY = "stomatology"
    TRADITIONAL_CHINESE_MEDICINE = "traditional_chinese_medicine"


class Difficulty(str, Enum):
    ALL = "all"
    EASY = "easy"
    MIDDLE = "middle"
    HARD = "hard"


def _build_question(question: str, options: Dict[str, str]) -> str:
    opts = "\n".join(f"{k}) {v}" for k, v in options.items() if v not in [None, ""])
    return f"{question}\n{opts}"


def _process_age_strings(text: str, rng: random.Random) -> str:
    """Add a small decimal jitter (~±2 weeks) to ages from M-ARC."""
    age_pattern = re.compile(
        r"\b(\d+)"  # numeric age
        r"(?:\s*[-\s]?)"
        r"(year-old|year old|yo)\b",
        re.IGNORECASE,
    )

    def add_decimal(match: re.Match[str]) -> str:
        age = int(match.group(1))
        decimal = rng.uniform(0, 0.04)
        new_age = age + decimal
        return f"{new_age:.3f} year-old"

    return age_pattern.sub(add_decimal, text)


def _to_vf_format(
    ds: Dataset,
    few_shot: bool,
    shuffle_answers: bool,
    shuffle_seed: int | None,
    jitter_age: bool,
) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': string the env will turn into chat messages
      - 'answer':   top-level gold letter (A/B/C/D[/E])
      - 'info':     keep all original fields for bookkeeping

    Args:
      - ds: dataset to convert to vf format
      - num_few_shot: number of few-shot examples (0 or 5)
      - shuffle_answers: whether to shuffle the answer choices
      - shuffle_seed: deterministic seed for choice shuffling
      - jitter_age: whether to add a small decimal jitter (~±2 weeks) to ages
    """
    VALID = "ABCDEFGHIJ"
    prompt_template = FIVE_SHOT_PROMPT_TEMPLATE if few_shot else ZERO_SHOT_PROMPT_TEMPLATE

    def _format_row(row: dict, idx: int) -> dict:
        question = row.get("question", "") or ""  # question string
        opts = row.get("options", {}) or {}  # answer choices, map of letter to answer text
        opts = {k: v for k, v in opts.items() if v not in [None, ""]}

        # lift the answer to top-level, normalize to a single letter
        answer_letter = (row.get("answer_letter") or "").strip().upper()
        if answer_letter not in VALID:
            return None

        # shuffle answer choices if requested
        if shuffle_answers and answer_letter and answer_letter in opts:
            opts, answer_letter, _ = randomize_multiple_choice(
                options=opts,
                answer_choice=answer_letter,
                seed=shuffle_seed,
                row_id=idx,
            )

        # Per-row RNG keeps age jitter deterministic and process-safe for num_proc>1
        base_seed = shuffle_seed if (shuffle_answers and shuffle_seed is not None) else 2718
        question_text = _process_age_strings(question, random.Random(base_seed + idx)) if jitter_age else question
        question_prompt = _build_question(question_text, opts)
        prompt = prompt_template.format(question_prompt)

        # question and answer have been moved to top-level, so remove them here
        info = dict(row)

        # update shuffled answer choices in the info dict
        if shuffle_answers:
            info["answer"] = answer_letter
            info["options"] = opts
        info["answer_text"] = opts.get(answer_letter, None)

        return {"question": prompt, "answer": answer_letter, "info": info}

    return ds.map(_format_row, remove_columns=ds.column_names, load_from_cache_file=False, with_indices=True)


def load_environment(
    field: str | Field = Field.ALL,
    difficulty: str | Difficulty = Difficulty.ALL,
    few_shot: bool = False,
    shuffle_answers: bool = False,
    shuffle_seed: int | None = 1618,
    jitter_age: bool = False,
    **kwargs,
) -> vf.Environment:
    """
    Single-turn SuperGPQA Medicine environment using HuggingFace `m-a-p/SuperGPQA` dataset

    Args:
        field (str or Field): Filter by medical field (default: all).
        difficulty (str or Difficulty): Filter by question difficulty (default: all).
        few_shot (bool): Whether to include five-shot examples in prompts (`True` => 5-shot, `False` => 0-shot).
        shuffle_answers (bool): Whether to shuffle the answer choices.
        shuffle_seed (int or None): Deterministic seed for choice shuffling.
        jitter_age (bool): Whether to add a small decimal jitter (~±2 weeks) to ages from M-ARC.
    """

    # -------- load dataset --------
    test_raw = load_dataset("m-a-p/SuperGPQA", split="train").filter(lambda row: row["discipline"] == "Medicine")
    # normalize enum/string params
    field = Field(field) if isinstance(field, str) else field
    difficulty = Difficulty(difficulty) if isinstance(difficulty, str) else difficulty

    if field != Field.ALL:
        test_raw = test_raw.filter(lambda row: row["field"].lower() == field.value.replace("_", " "))
    if difficulty != Difficulty.ALL:
        test_raw = test_raw.filter(lambda row: row["difficulty"] == difficulty.value)

    # -------- convert options from list to dict --------
    # SuperGPQA options are stored as a list, convert to dict with letter keys
    def _convert_options(row: dict) -> dict:
        opts = row["options"]
        if isinstance(opts, list):
            row["options"] = {chr(ord("A") + i): v for i, v in enumerate(opts)}
        return row

    test_raw = test_raw.map(_convert_options, load_from_cache_file=False)

    # -------- convert rows to vf format and shuffle row order --------
    test_ds = _to_vf_format(
        test_raw,
        few_shot=few_shot,
        shuffle_answers=shuffle_answers,
        shuffle_seed=shuffle_seed,
        jitter_age=jitter_age,
    )
    del test_raw  # free memory

    # -------- construct prompts and questions --------
    parser = vf.Parser(extract_fn=extract_boxed_answer)

    # -------- rubric --------
    def accuracy(completion, answer: str, parser: vf.Parser, info: dict | None = None, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        answer_text = info.get("answer_text", None) if info else None
        is_correct = multiple_choice_accuracy(llm_answer=parsed, answer_letter=answer, answer_text=answer_text)
        return 1.0 if is_correct else 0.0

    rubric = vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        # dataset=..., # no train dataset.
        eval_dataset=test_ds,
        system_prompt=BOXED_SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
