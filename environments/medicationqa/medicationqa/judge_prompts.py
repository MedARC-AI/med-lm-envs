# Judge template lightly adapted from HELM's MedDialog evaluation
# Judge template adapted from HELM-style prompts, rewritten for MedicationQA Q→A evaluation
JUDGE_TEMPLATE = """\
You are a medical expert tasked with evaluating how well a model answered a consumer medication question.

Your goal is to assess the quality of the model’s answer and how it compares to the reference answer.
You will rate the answer along three dimensions: accuracy, completeness, and clarity.

Definitions:
- Accuracy: Are the medical statements correct and free of misinformation?
- Completeness: Does the answer address all medically relevant aspects of the question?
- Clarity: Is the answer understandable, well-structured, and appropriate for a layperson?

The question will be provided in these tags:
<medication_question>
{question}
</medication_question>

The model's answer will be provided in these tags:
<model_answer>
{response}
</model_answer>

The reference answer will be provided in these tags:
<reference_answer>
{gold_response}
</reference_answer>

For each dimension:
1. Provide a brief explanation (1–3 sentences) describing why you assigned the score.
2. Then assign a score from 1 to 5:
   1 = very poor, 2 = poor, 3 = adequate, 4 = good, 5 = excellent.

Output Format:
{output_format}
"""

# JSON output schema
JUDGE_OUTPUT_JSON = """
Output your evaluation as a single valid JSON object matching the following structure:
{
  "accuracy": {
    "reason": "Brief explanation of why this score was given.",
    "score": 0
  },
  "completeness": {
    "reason": "Brief explanation of why this score was given.",
    "score": 0
  },
  "clarity": {
    "reason": "Brief explanation of why this score was given.",
    "score": 0
  }
}

Ensure the output is valid JSON:
- Use double quotes (") for all keys and string values.
- Escape any internal quotes inside the reason fields.
- Do not include any additional text outside the JSON object.
- Do not explain your reasoning outside the JSON object; all justification must appear only in the "reason" fields.
"""

# XML output schema
JUDGE_OUTPUT_XML = """
Output your evaluation as a single valid XML object matching the following structure:
<evaluation>
  <accuracy>
    <reason>Brief explanation of why this score was given.</reason>
    <score>0</score>
  </accuracy>
  <completeness>
    <reason>Brief explanation of why this score was given.</reason>
    <score>0</score>
  </completeness>
  <clarity>
    <reason>Brief explanation of why this score was given.</reason>
    <score>0</score>
  </clarity>
</evaluation>

Ensure the output is valid XML:
- Escape special characters in text nodes: & as &amp;, < as &lt;, > as &gt;, " as &quot;, ' as &apos;
- Do not include any additional text outside the XML object.
"""
