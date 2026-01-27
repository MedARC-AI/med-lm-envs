# Judge template adapted from HELM's MTSamples Procedures Annotator
# Evaluates clinical responses for treatment plan generation
JUDGE_TEMPLATE = """\
You are a medical expert responsible for evaluating a proposed treatment plan
based on provided patient information.

Your goal is to assess whether the treatment plan uses the patient's medical history,
medications, symptoms, and other relevant details appropriately, follows clinical best practices,
and aligns with the gold standard response in accuracy, completeness, and clarity.

The patient's information will be provided in these tags:
<patient_information>
{procedure_note}
</patient_information>

The proposed treatment plan will be provided in these tags:
<response>
{response}
</response>

The gold standard treatment plan will be provided in these tags:
<gold_plan>
{gold_plan}
</gold_plan>

Carefully review the <response> based on the <patient_information> and compare it to the <gold_response> when needed.

For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5)
- Does the generated response provide correct clinical advice that follows established clinical guidelines?

Completeness (1-5)
- Does the treatment plan include all important medical details?

Clarity (1-5)
- Is the treatment plan written clearly so clinicians can easily understand it?

Output Format:
{output_format}
"""

JUDGE_OUTPUT_JSON = """
Output your evaluation as a single valid JSON object matching the following structure:
{
    "accuracy": {
        "explanation": "Brief explanation of why this score was given.",
        "score": 0,
    },
    "completeness": {
        "explanation": "Brief explanation of why this score was given.",
        "score": 0,
    },
    "clarity": {
        "explanation": "Brief explanation of why this score was given.",
        "score": 0,
    }
}

Ensure the output is valid JSON:
- Use **double quotes** (") for all keys and string values.
- When quoting text or sections inside the explanations, use escaped double quotes (") to
  maintain valid JSON formatting.
- Do not include any additional information in the output.
"""

JUDGE_OUTPUT_XML = """
Output your evaluation as a single valid XML object matching the following structure:
<evaluation>
  <accuracy>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </accuracy>
  <completeness>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </completeness>
  <clarity>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </clarity>
</evaluation>

Ensure the output is valid XML:
- Escape special characters in text nodes: & as &amp;, < as &lt;, > as &gt;, " as &quot;, ' as &apos;.
  (Alternatively, wrap quoted passages inside <![CDATA[ ... ]]> blocks.)
- Do not include any additional information in the output.
"""
