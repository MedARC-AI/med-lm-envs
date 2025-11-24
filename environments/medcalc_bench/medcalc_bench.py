import math
import re
from datetime import datetime
from typing import Optional

import numpy as np
import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers import XMLParser

disable_progress_bar()  # suppress datasets progress indicators


def _build_prompt(patient_note, question) -> str:
    return f"""You are a helpful assistant for calculating a score for a given patient note.
Please think step-by-step to solve the question and then generate the required score.
\n\nPatient Note: {patient_note}
\n\nQuestion: {question}.


Please answer the question in the following format:
<think>
Your thought process here
</think>
<answer>
Your answer here without any units, just give the number.
</answer>
"""


def extract_answer(response, calid, parser: XMLParser):
    calid = int(calid)

    parsed = parser.parse(response, last=True)
    extracted_explanation = getattr(parsed, "think", None)
    extracted_explanation = extracted_explanation.strip() if extracted_explanation else "No Explanation"

    answer = getattr(parsed, "answer", None)
    answer = answer.strip() if isinstance(answer, str) else "Not Found"

    if (not isinstance(answer, str)) or (len(answer.strip()) == 0):
        extracted_answer = "Not Found"
    else:
        extracted_answer = answer.strip('"')

    if calid in [13, 68]:
        # Output Type: date
        match = re.search(r"^(0?[1-9]|1[0-2])\/(0?[1-9]|[12][0-9]|3[01])\/(\d{4})", extracted_answer)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = match.group(3)
            answer = f"{month:02}/{day:02}/{year}"
        else:
            answer = "N/A"

    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer
        )
        extracted_answer = extracted_answer.replace("[", "(").replace("]", ")").replace("'", "").replace('"', "")
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", extracted_answer
        )
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
        else:
            answer = "N/A"
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51, 69]:
        # Output Type: integer A
        match = re.search(r"(\d+) out of", extracted_answer)
        if match:  # cases like "3 out of 5"
            answer = match.group(1)
        else:
            match = re.search(r"-?\d+(, ?-?\d+)+", extracted_answer)
            if match:  # cases like "3, 4, 5"
                answer = str(len(match.group(0).split(",")))
            else:
                # match = re.findall(r"(?<!-)\d+", extracted_answer)
                match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                # match = re.findall(r"-?\d+", extracted_answer)
                if len(match) > 0:  # find the last integer
                    answer = match[-1][0]
                    # answer = match[-1].lstrip("0")
                else:
                    answer = "N/A"
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:  # fmt: skip
        # Output Type: decimal
        match = re.search(r"str\((.*)\)", extracted_answer)
        if match:  # cases like "str(round((140 * (3.15 - 136) / 1400) * 72.36)"
            expression = (
                match.group(1)
                .replace("^", "**")
                .replace("is odd", "% 2 == 1")
                .replace("is even", "% 2 == 0")
                .replace("sqrt", "math.sqrt")
                .replace(".math", "")
                .replace("weight", "")
                .replace("height", "")
                .replace("mg/dl", "")
                .replace("g/dl", "")
                .replace("mmol/L", "")
                .replace("kg", "")
                .replace("g", "")
                .replace("mEq/L", "")
            )
            expression = expression.split("#")[
                0
            ]  # cases like round(45.5 * 166 - 45.3 + 0.4 * (75 - (45.5 * 166 - 45.3))))) # Calculation: ...
            if expression.count("(") > expression.count(")"):  # add missing ')
                expression += ")" * (expression.count("(") - expression.count(")"))
            elif expression.count(")") > expression.count("("):  # add missing (
                expression = "(" * (expression.count(")") - expression.count("(")) + expression
            try:
                answer = eval(
                    expression,
                    {"__builtins__": None},
                    {
                        "min": min,
                        "pow": pow,
                        "round": round,
                        "abs": abs,
                        "int": int,
                        "float": float,
                        "math": math,
                        "np": np,
                        "numpy": np,
                    },
                )
            except Exception as e:
                print(f"Error in evaluating expression: {expression} - {e}")
                answer = "N/A"
        else:
            match = re.search(r"(-?\d+(\.\d+)?)\s*mL/min/1.73", extracted_answer)
            if match:  # cases like "8.1 mL/min/1.73 m\u00b2"
                answer = eval(match.group(1))
            else:
                match = re.findall(r"(-?\d+(\.\d+)?)\%", extracted_answer)
                if len(match) > 0:  # cases like "53.1%"
                    answer = eval(match[-1][0]) / 100
                else:
                    match = re.findall(r"(-?\d+(\.\d+)?)", extracted_answer)
                    if len(match) > 0:  # cases like "8.1 mL/min/1.73 m\u00b2" or "11.1"
                        answer = eval(match[-1][0])
                    else:
                        answer = "N/A"
        if answer != "N/A":
            answer = str(answer)

    return answer, extracted_explanation


def check_correctness(parser, completion, info, **kwargs):
    # Pull required fields from info

    ground_truth = info.get("ground_truth")
    calc_id = info.get("calc_id")
    upper_bound = info.get("upper_bound")
    lower_bound = info.get("lower_bound")

    calid = int(calc_id)

    # Coerce completion to final assistant text (string) before extraction
    if isinstance(completion, list) and completion:
        last = completion[-1]
        raw = str(last.get("content", "")) if isinstance(last, dict) else str(last)
    else:
        raw = str(completion)

    answer, _ = extract_answer(raw, calid, parser)

    if calid in [13, 68]:
        # Output Type: date

        try:
            if datetime.strptime(answer, "%m/%d/%Y").strftime("%-m/%-d/%Y") == datetime.strptime(
                ground_truth, "%m/%d/%Y"
            ).strftime("%-m/%-d/%Y"):
                correctness = 1
            else:
                correctness = 0
        except Exception:
            correctness = 0
    elif calid in [69]:
        # Output Type: integer (A, B)
        match = re.search(
            r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", str(ground_truth)
        )
        if match:
            ground_truth = f"({match.group(1)}, {match.group(3)})"
        match = re.search(r"\(?[\"\']?(\d+)\s*(weeks?)?[\"\']?,?\s*[\"\']?(\d+)\s*(days?)?[\"\']?\s*\)?", answer)
        if match:
            weeks = match.group(1)
            days = match.group(3)
            answer = f"({weeks}, {days})"
            if eval(answer) == eval(ground_truth):
                correctness = 1
            else:
                correctness = 0
        else:
            correctness = 0
    elif calid in [4, 15, 16, 17, 18, 20, 21, 25, 27, 28, 29, 32, 33, 36, 43, 45, 48, 51]:
        # Output Type: integer A
        try:
            numeric = eval(answer)
            correctness = 1 if numeric == eval(str(ground_truth)) else 0
        except Exception:
            correctness = 0
    elif calid in [2,  3,  5,  6,  7,  8,  9, 10, 11, 19, 22, 23, 24, 26, 30, 31, 38, 39, 40, 44, 46, 49, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]:  # fmt: skip
        # Output Type: decimal
        try:
            numeric = float(eval(answer))
            lo = float(eval(str(lower_bound)))
            hi = float(eval(str(upper_bound)))
            correctness = 1 if (lo <= numeric <= hi) else 0
        except Exception:
            correctness = 0
    else:
        raise ValueError(f"Unknown calculator ID: {calid}")
    return float(correctness)


def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
) -> vf.Environment:
    ds = load_dataset("ncbi/MedCalc-Bench-v1.2")

    def _map(ex):
        patient_note = ex["Patient Note"]
        q_text = ex["Question"]
        calc_id = ex["Calculator ID"]
        ground_truth = ex["Ground Truth Answer"]
        lower_bound = ex["Lower Limit"]
        upper_bound = ex["Upper Limit"]

        return {
            "question": _build_prompt(patient_note, q_text),
            "info": {
                "calc_id": calc_id,
                "ground_truth": ground_truth,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            },
        }

    train_mapped = ds["train"].map(_map, remove_columns=ds["train"].column_names)
    test_mapped = ds["test"].map(_map, remove_columns=ds["test"].column_names)

    # Use XMLParser to support <think> and <answer> formatting without strict enforcement
    parser = XMLParser(["think", "answer"], answer_field="answer")

    system_prompt = """You are a helpful assistant who will assist with calculating a score given a patient note and a question.
Please think step-by-step to solve the question and then generate the required score."""

    rubric = vf.Rubric(funcs=[check_correctness], weights=[1.0], parser=parser)

    return vf.SingleTurnEnv(
        dataset=train_mapped,
        eval_dataset=test_mapped,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
