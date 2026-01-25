# adapted from https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/main/evaluation/run.py#L16
zero_shot_prompt = """\
You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score.

When you are finished with all computations, please write your final answer value without any units, using the following formats inside the indicated answer tag:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Patient Note:

{patient_note}

Question:

{question}
""".strip()


# adapted from https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/main/evaluation/run.py#L26
one_shot_prompt = """\
You are a helpful assistant for calculating a score for a given patient note. Please think step-by-step to solve the question and then generate the required score.

When you are finished with all computations, please write your final answer value without any units, using the following formats:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Here is an example patient note and question:

Example Patient Note:

{example_note}

Example Question:

{example_question}

Example Response:

{example_response}

Here is the patient note and question you will be solving:

Patient Note:

{patient_note}

Question:

{question}
""".strip()


# Tool descriptions for prompts
PYTHON_TOOL_DESCRIPTION = "You have access to a python code execution tool that you should use to do any math based on values in the patient note. Make sure the code returns or prints all of its outputs. If there are any errors with compiling your or running script, you may need to re-write your code to obtain the output."

CALCULATOR_TOOL_DESCRIPTION = "You have access to a calculator tool that you should use to evaluate mathematical expressions based on values in the patient note."

BOTH_TOOLS_DESCRIPTION = "You have access to a calculator tool and a python code execution tool. Use them to do any math based on values in the patient note. If using the python tool, make sure the code returns or prints all of its outputs. If there are any errors with compiling or running your script, you may need to re-write your code to obtain the output."


def get_tool_description(allow_python: bool, allow_calculator: bool) -> str:
    """Get the appropriate tool description based on which tools are enabled."""
    if allow_python and allow_calculator:
        return BOTH_TOOLS_DESCRIPTION
    elif allow_python:
        return PYTHON_TOOL_DESCRIPTION
    elif allow_calculator:
        return CALCULATOR_TOOL_DESCRIPTION
    else:
        return ""


# adapted from https://github.com/nikhilk7153/MedCalc-Bench-Verified/blob/main/evaluation/generate_code_prompt.py#L52
# modified to remove instructions to only write python string and not perform calculations
tool_use_prompt = """\
You are a helpful assistant. Your task is to read a patient note and compute a medical value based on the following question.
If there are multiple values for a given measurement or attribute, then please use the value recorded based on when the patient note was written. You should not be using values that the patient had post-treatment or values from a patient's history in the past.
Additionally, if the problem doesn't directly imply or provide information regarding a particular patient attribute, assume the patient does not have it.
{tool_description}
Note that all of the necessary information is provided in the patient note and you should not need to prompt the user for any information.
When you are finished with all computations, please write your final answer value without any units, using the following formats inside the indicated answer tag:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Patient Note:

{patient_note}

Question:

{question}
""".strip()


tool_use_one_shot_prompt = """\
You are a helpful assistant. Your task is to read a patient note and compute a medical value based on the following question.
If there are multiple values for a given measurement or attribute, then please use the value recorded based on when the patient note was written. You should not be using values that the patient had post-treatment or values from a patient's history in the past.
Additionally, if the problem doesn't directly imply or provide information regarding a particular patient attribute, assume the patient does not have it.
{tool_description}
Note that all of the necessary information is provided in the patient note and you should not need to prompt the user for any information.
When you are finished with all computations, please write your final answer value without any units, using the following formats inside the indicated answer tag:

- Decimal Answer Format: 17.29
- Score-Based Answer Format: 5
- Estimated Date Answer Format: 5/21/2021
- Estimated Age Answer Format: (4 weeks, 3 days)

Here is an example patient note and question:

Example Patient Note:

{example_note}

Example Question:

{example_question}

Example Response:

{example_response}

Here is the patient note and question you will be solving:

Patient Note:

{patient_note}

Question:

{question}
""".strip()
