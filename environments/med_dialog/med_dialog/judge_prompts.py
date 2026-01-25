# Judge template lightly adapted from HELM's MedDialog evaluation
JUDGE_TEMPLATE = """\
You are a medical expert tasked with evaluating the quality of medical dialogue summaries.

Your goal is to assess how well the summary captures the patient-doctor conversation,
and how it compares to the gold response in terms of accuracy, completeness, and clarity.

The patient-doctor conversation will be provided in these tags:
<conversation>
{conversation}
</conversation>

The response will be provided in these tags:
<response>
{response}
</response>

The reference response will be provided in these tags:
<gold_response>
{gold_response}
</gold_response>

Carefully review the <response> and compare it to the <gold_response> when needed.

Please evaluate the generated summary on a scale of 1-5 (1 = poor, 5 = excellent)
for each of these three key dimensions:

Evaluation Criteria:
Accuracy (1-5)
- Does the summary correctly capture the main medical issue and clinical details from the conversation?

Completeness (1-5)
- Does the summary include all important medical information from the conversation?

Clarity (1-5)
- Is the summary easy to understand for clinical use?

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
