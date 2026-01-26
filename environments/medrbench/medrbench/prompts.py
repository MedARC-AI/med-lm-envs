# Original system prompt from MedRBench
# Source: oracle_diagnose.py and oracle_treatment_planning.py
DEFAULT_SYSTEM_PROMPT = "You are a professional doctor"

# Task prompts from the original MedRBench repository
# Source: https://github.com/MAGIC-AI4Med/MedRBench/tree/main/src/Inference/instructions

# Original prompt from oracle_diagnose.txt (typo "Resoning" fixed to "Reasoning")
DIAGNOSIS_TASK_PROMPT = """\
Please carefully study the following patient case summary, conduct a comprehensive and in-depth diagnostic analysis, and clearly provide the final diagnosis result.

{case}

---

**Format to Follow:**

```
### Reasoning:
[Please sort out your thinking process step by step, with each logical step in a separate paragraph.]
<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Answer:
[Just output the diagnostic result without any other explanation.]
```"""

# Original prompt from treatment_plan_prompt.txt
TREATMENT_TASK_PROMPT = """\
Please carefully study the following patient case summary, conduct a comprehensive and in-depth treatment planning analysis, and clearly provide the selected treatment for the patient.

{case}

---

**Format to Follow:**

```
### Chain of Thought:
[Please sort out your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]
<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Answer:
[Just output the selected treatment for the patient without any other explanation.]
```"""

# Original prompt from free_turn_first_turn_prompt.txt
MULTI_TURN_FIRST_TURN_PROMPT = """\
Please thoroughly examine the patient case summary presented below. Your objective is to perform a detailed diagnostic analysis utilizing all available information. Note that due to the potentially limited details, the preliminary diagnosis may encompass several possible conditions. Should you ascertain that the provided data is inadequate for a definitive conclusion, please enumerate any additional diagnostic tests or information that would be necessary. However, if you can deduce a conclusive diagnosis, please proceed to provide it. Too many requests for information are also inappropriate.

Patient Case Summary:
{case}

Guidelines:
Evaluate the patient's symptoms, medical history, and all pertinent details from the case summary.
Formulate differential diagnoses based on your analysis.
If the information is not sufficient for a conclusive diagnosis, specify the further tests or details required.

Always following the response format in the following dialogue, never change the section of ### format: 
```
### Chain of Thought:
[Please sort out your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]
<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Additional Information Required:
[Indicate if further information is needed by specifying the required tests or data. If a conclusive diagnosis has been made and no additional information is necessary, only output "Not required." directly without any other words in this section.]
For example:
Not required.

or

1. Laboratory tests: details
2. Imaging: details

### Conclusion:
[If do not require additional information, please provide a final conclusive diagnosis. Otherwise, summarize the current findings.]
..."""

# Original prompt from free_turn_following_turn_prompt.txt
MULTI_TURN_FOLLOWING_TURN_PROMPT = """\
Here is the additional information you required. Please proceed with the analysis.

Additional Information:
{additional_information}

Always following the response format in each turn of the dialogue, never change the section of ### format: 
```
### Chain of Thought:
[Please sort out your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]
<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Additional Information Required:
[Indicate if further information is needed by specifying the required tests or data. If a conclusive diagnosis has been made and no additional information is necessary, only output "Not required." directly without any other words in this section.]
For example:
Not required.

or

1. Laboratory tests: details
2. Imaging: details

### Conclusion:
[If do not require additional information, please provide a final conclusive diagnosis. Otherwise, summarize the current findings.]
..."""

# Original prompt from 1turn_prompt_examination_recommend.txt
SINGLE_TURN_FIRST_TURN_PROMPT = """\
Please thoroughly examine the patient case summary presented below. Your objective is to perform a detailed diagnostic analysis utilizing all available information. Note that due to the potentially limited details, the preliminary diagnosis may encompass several possible conditions. Should you ascertain that the provided data is inadequate for a definitive conclusion, please enumerate any additional diagnostic tests or information that would be necessary. However, if you can deduce a conclusive diagnosis, please proceed to provide it. Too many requests for information are also inappropriate.

Patient Case Summary:
{case}

Guidelines:
Evaluate the patient's symptoms, medical history, and all pertinent details from the case summary.
Formulate differential diagnoses based on your analysis.
If the information is not sufficient for a conclusive diagnosis, specify the further tests or details required.

Always following the response format in each turn of the dialogue, never change the section of ### format:
```
### Chain of Thought:
[Please sort out your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]
<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Conclusion:
[Give a preliminary conclusive if possible, or summarize the current findings.]

### Additional Information Required:
[Indicate if further information is needed by specifying the required tests or data. If a conclusive diagnosis has been made and no additional information is necessary, only output "Not required." directly without any other words in this section.]
For example:
Not required.

or

1. Laboratory tests: details
2. Imaging: details
...
```"""

# Original prompt from 1turn_prompt_make_diagnosis.txt
SINGLE_TURN_FINAL_TURN_PROMPT = """\
Please make a final diagnosis for the patient in light of the additional information provided below.

Additional Information:
{additional_information}

Guidelines:
- Evaluate the patient's symptoms, medical history, and all pertinent details from the case summary.
- Formulate differential diagnoses based on your analysis.

Always following the response format in each turn of the dialogue, never change the section of ### format:
```
### Chain of Thought:
[Please sort out your thinking process step by step, with each logical step in a separate paragraph, and use a format such as <step 1> to label each step.]
<step 1> Specific thinking content of this step
<step 2> Specific thinking content of this step
...
<step n> Specific thinking content of this step

### Conclusion:
[Directly output the diagnostic result without any other explanation.]
```"""

# Original prompt from patient_agent_prompt.txt
PATIENT_AGENT_PROMPT = """\
You are a medical expert providing guidance to a junior physician on a patient case. The junior physician will ask you for additional diagnostic information based on the patient's case details and any available ancillary test results. Your role is to provide accurate and relevant responses regarding the availability of specific diagnostic information.


Guidelines:
1. You will receive the patient's case information and any relevant ancillary test results.
2. The junior physician will ask questions about additional diagnostic information needed for the case.
3. If there is relevant ancillary test information available for the requested diagnostic area, provide the details 4. accurately.
4. If there is no relevant ancillary test information available for the requested diagnostic area, simply state: "There is no relevant ancillary test information available for this request."

Patient Case
{case}

Ancillary Test Results
{ancillary_test_results}

Example Interaction:
```
Junior Physician: "Does the patient have any imaging studies like an X-ray or CT scan?"
Your Response:
If there is relevant imaging information available:
"Based on the available ancillary test results, the patient has undergone a chest X-ray which shows [specific findings]."
If there is no relevant imaging information available:
"There is no relevant ancillary test information available for this request."
```
Note: Your responses should be factual and based solely on the provided patient case information and ancillary test results. Avoid speculation or hypotheticals unless explicitly requested.
"""

# Judge prompts from MedRBench evaluation metrics
# Source: external/MedRBench/src/Evaluation/metrics/instructions

# Original prompt from acc_diagnose.txt
DIAGNOSIS_JUDGE_PROMPT = """\
# Task Description
You are a professional medical diagnosis evaluation system. Now, you will receive two diagnosis results: one is the diagnosis predicted by the model ([pred_diagnose]), and the other is the verified correct diagnosis ([gt_diagnose]). Your task is to judge whether the model-predicted diagnosis([pred_diagnose]) is correct.

When evaluating, please consider the following factors:
1.The same disease may have multiple aliases, for example, "Heart disease" may also be called "Cardiac disease".
2.There may be diversity in language expression, for example, "heart attack" and "myocardial infarction" may refer to the same disease.
3.Only judge whether the diagnosis result is correct, information such as the cause of the disease, symptoms, and treatment recommendations are not included in the evaluation scope.
4.If the correct diagnosis [Ground-truth diagnosis] is included in the predicted diagnosis but some additional complications are mentioned, it is also considered correct

# Output Requirements
Only output your judgment result on the model-predicted [pred_diagnose] as "Correct|Wrong", do not output any other content.

# Format to Follow:
[Correct|Wrong]

Below is the diagnosis result predicted by the model and the correct diagnosis:
[Predicted Diagosis]
{pred_diagnose}

[Ground-truth Diagnosis]
{gt_diagnose}"""

# Original prompt from acc_treatment_plan.txt
# The upstream evaluation uses a system message ("You are a professional medical diagnosis evaluation system.")
# but verifiers' JudgeRubric does not support a system role; we prepended it to the user prompt for near parity.
TREATMENT_JUDGE_PROMPT = """\
You are a professional medical diagnosis evaluation system.

# Task Description
As a professional medical treatment planning evaluation system, you will now receive two treatment plan results for assessment: one is the treatment plan predicted by the model ([predicted treatment]), and the other is the verified correct treatment plan ([gt treatment]). Your task is to determine whether the model-predicted treatment ([predicted treatment]) is accurate.

When evaluating, please consider the following factors:
1. If predicted treatment and ground truth treatment have exactly the same meaning, then it is correct.
2. If the correct treatment plan [ground truth treatment] is included in the predicted treatment but some additional care are mentioned, it is also considered correct
3. Considering that even the same disease can sometimes be treated differently. If the model's predictions do not completely match ground truth treatment, you can refer to additional information to make a judgment.
4. If the predicted treatment and the ground-truth treatment ([ground truth treatment]) do not convey the same meaning, and there is no supporting evidence in the additional information to suggest that the predicted treatment is also applicable to the disease, it is considered wrong.

# Output Requirements
Only output your judgment result on the model-predicted [predicted treatment] as "Correct|Wrong", do not output any other content.

# Format to Follow:
[Correct|Wrong]

Below is the result predicted by the model and the correct Treatment plan:
[predicted treatment]
{pred_treatment}

[Ground-truth treatment]
{gt_treatment}
"""
