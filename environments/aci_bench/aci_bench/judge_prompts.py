# Judge template lightly adapted from HELM's ACI-Bench evaluation prompt

JUDGE_DIMENSIONS = ("accuracy", "completeness", "clarity")

JUDGE_TEMPLATE = """\
You are a medical expert tasked with evaluating the quality of a generated clinical note based on a doctor-patient conversation.

For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

The patient-doctor conversation will be provided in these tags:
<conversation>
{conversation}
</conversation>

The generated note will be provided in these tags:
<response>
{response}
</response>

The reference response will be provided in these tags:
<gold_response>
{gold_response}
</gold_response>

Carefully review the <response> based on the <conversation> and compare it to the <gold_response> when needed.

For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria:
Accuracy (1-5)
- Does the note provide correct clinical information based on the conversation?

Completeness (1-5)
- Does the note include all important medical details from the conversation?

Clarity (1-5)
- Is the note written clearly and organized in a standard clinical format for clinicians?

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
