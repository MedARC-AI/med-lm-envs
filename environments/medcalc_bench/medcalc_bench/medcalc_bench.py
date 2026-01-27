import math
import re
from datetime import datetime

import numpy as np
import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers.xml_parser import XMLParser
from medarc_verifiers.prompts import (
    BOXED_TOOL_SYSTEM_PROMPT,
    THINK_BOXED_TOOL_SYSTEM_PROMPT,
    THINK_XML_SYSTEM_PROMPT,
    THINK_XML_TOOL_SYSTEM_PROMPT,
    XML_SYSTEM_PROMPT,
    XML_TOOL_SYSTEM_PROMPT,
    AnswerFormat,
)
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer

from medcalc_bench.prompts import (
    get_tool_description,
    one_shot_prompt,
    tool_use_one_shot_prompt,
    tool_use_prompt,
    zero_shot_prompt,
)
from medcalc_bench.tools import SimpleToolEnv

disable_progress_bar()  # suppress datasets progress indicators


def extract_boxed_answer_strict(text: str) -> str:
    """Fail boxed parsing if box not found"""
    if r"\boxed{" not in text:
        return ""
    return extract_boxed_answer(text)


def _one_shot_response(
    answer: str, reasoning: str, use_think: bool = False, answer_format: AnswerFormat = AnswerFormat.XML
) -> str:
    """Format a one-shot response as XML or Boxed"""
    if use_think:
        reasoning = f"<think>{reasoning}</think>\n"
    if answer_format == AnswerFormat.XML:
        return f"{reasoning}\n<answer>{answer}</answer>"
    elif answer_format == AnswerFormat.BOXED:
        return f"{reasoning}\n\n\\boxed{{{answer}}}"
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")


def _build_prompt(
    row: dict,
    one_shot_examples: dict | None = None,
    add_python_tool: bool = False,
    add_calculator_tool: bool = False,
    one_shot: bool = False,
    use_think: bool = False,
    answer_format: AnswerFormat = AnswerFormat.XML,
) -> str:
    calc_id = int(row["Calculator ID"])
    tool_use = add_python_tool or add_calculator_tool
    tool_description = get_tool_description(add_python_tool, add_calculator_tool)

    if tool_use and one_shot:
        return tool_use_one_shot_prompt.format(
            tool_description=tool_description,
            example_note=one_shot_examples[calc_id]["Patient Note"],
            example_question=one_shot_examples[calc_id]["Question"],
            example_response=_one_shot_response(
                one_shot_examples[calc_id]["Ground Truth Answer"],
                one_shot_examples[calc_id]["Ground Truth Explanation"],
                use_think,
                answer_format,
            ),
            patient_note=row["Patient Note"],
            question=row["Question"],
        )
    elif tool_use:
        return tool_use_prompt.format(
            tool_description=tool_description,
            patient_note=row["Patient Note"],
            question=row["Question"],
        )
    elif one_shot:
        return one_shot_prompt.format(
            example_note=one_shot_examples[calc_id]["Patient Note"],
            example_question=one_shot_examples[calc_id]["Question"],
            example_response=_one_shot_response(
                one_shot_examples[calc_id]["Ground Truth Answer"],
                one_shot_examples[calc_id]["Ground Truth Explanation"],
                use_think,
                answer_format,
            ),
            patient_note=row["Patient Note"],
            question=row["Question"],
        )
    else:
        return zero_shot_prompt.format(
            patient_note=row["Patient Note"],
            question=row["Question"],
        )


def extract_answer(response, calid, parser):
    calid = int(calid)

    extracted_answer = parser.parse_answer(response) or ""

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

    return answer


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

    answer = extract_answer(raw, calid, parser)

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
    one_shot: bool = False,
    add_python_tool: bool = False,
    add_calculator_tool: bool = False,
    max_turns: int = 20,  # https://github.com/ncbi-nlp/MedCalc-Bench/blob/main/evaluation/generate_code_prompt.py#L145
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    use_think: bool = False,
    system_prompt: str | None = None,
    **kwargs,
) -> vf.Environment:
    # -------- normalize answer_format --------
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format

    if answer_format == AnswerFormat.XML:
        if system_prompt is None:
            if add_python_tool or add_calculator_tool:
                system_prompt = THINK_XML_TOOL_SYSTEM_PROMPT if use_think else XML_TOOL_SYSTEM_PROMPT
            else:
                system_prompt = THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT
        else:
            system_prompt = system_prompt
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = XMLParser(fields=parser_fields, answer_field="answer")  # medarc_verifiers' XMLParser
    elif answer_format == AnswerFormat.BOXED:
        if system_prompt is None:
            if add_python_tool or add_calculator_tool:
                system_prompt = THINK_BOXED_TOOL_SYSTEM_PROMPT if use_think else BOXED_TOOL_SYSTEM_PROMPT
            else:
                system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT
        else:
            system_prompt = system_prompt

    # -------- load dataset and convert to vf format --------
    ds = load_dataset("ncbi/MedCalc-Bench-v1.2")
    one_shot_examples = None
    if one_shot:
        # create mapping from calc id to one-shot example
        one_shot_ds = load_dataset("nsk7153/MedCalc-Bench-Verified", split="one_shot")
        one_shot_examples = {
            int(ex["Calculator ID"]): {
                "Patient Note": ex.get("Patient Note"),
                "Question": ex.get("Question"),
                "Ground Truth Answer": ex.get("Ground Truth Answer"),
                "Ground Truth Explanation": ex.get("Ground Truth Explanation"),
            }
            for ex in one_shot_ds
        }

    def _map(row: dict):
        return {
            "question": _build_prompt(
                row,
                one_shot_examples,
                add_python_tool=add_python_tool,
                add_calculator_tool=add_calculator_tool,
                one_shot=one_shot,
                use_think=use_think,
                answer_format=answer_format,
            ),
            "answer": row["Ground Truth Answer"],
            "task": "medcalc_bench",
            "info": {
                "calc_id": row["Calculator ID"],
                "ground_truth": row["Ground Truth Answer"],
                "lower_bound": row["Lower Limit"],
                "upper_bound": row["Upper Limit"],
            },
        }

    train_mapped = ds["train"].map(_map, remove_columns=ds["train"].column_names)
    test_mapped = ds["test"].map(_map, remove_columns=ds["test"].column_names)

    # -------- create rubric --------
    rubric = vf.Rubric(funcs=[check_correctness], weights=[1.0], parser=parser)

    # -------- create environment --------
    if add_python_tool or add_calculator_tool:
        env = SimpleToolEnv(
            dataset=train_mapped,
            eval_dataset=test_mapped,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            max_turns=max_turns,
            use_python=add_python_tool,
            use_calculator=add_calculator_tool,
            **kwargs,
        )
        # Add ToolRubric to track tool usage metrics
        tool_rubric = vf.ToolRubric(tools=env.tools)
        env.rubric = vf.RubricGroup(rubrics=[tool_rubric, env.rubric])
        return env
    else:
        return vf.SingleTurnEnv(
            dataset=train_mapped,
            eval_dataset=test_mapped,
            system_prompt=system_prompt,
            parser=parser,
            rubric=rubric,
            **kwargs,
        )
