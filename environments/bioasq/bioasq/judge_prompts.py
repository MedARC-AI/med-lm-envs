# Judge template adapted from BioASQ manual assessment criteria
# Evaluates biomedical ideal answers for question answering
# Grounded in official Task 1b synthesis requirements

JUDGE_DIMENSIONS = ("precision", "recall", "repetition", "readability")

JUDGE_TEMPLATE = """\
You are a biomedical expert tasked with evaluating the quality of a generated answer to a biomedical question.

Your goal is to assess how well the generated answer addresses the question and how it compares to the reference answer in terms of precision, recall, repetition, and readability.

The biomedical question will be provided in these tags:
<question>
{question}
</question>

Supporting evidence (Context snippets used to derive the answer):
<context>
{context}
</context>

The generated response will be provided in these tags:
<response>
{response}
</response>

The reference answer will be provided in these tags:
<gold_answer>
{gold_answer}
</gold_answer>

Carefully review the <response> based on the <question> and the supporting <context>.

For each of the following criteria, rate the response on a scale of 1 to 5 (1 = very poor, 5 = excellent), and provide a short justification for your score.

Evaluation Criteria:
1. Precision (1-5)
- Does the generated response provide accurate biomedical information that is relevant to the question? Penalize information not supported by the context.

2. Recall (1-5)
- Does the response include all important biomedical concepts and facts mentioned in the reference answer?

3. Repetition (1-5)
- Does the response avoid unnecessary repetition? (1 = lots of repetition, 5 = no repetition)

4. Readability (1-5)
- Is the response written clearly and organized in a way that is easy to read for biomedical professionals?

Output Format:
{output_format}
"""

JUDGE_OUTPUT_JSON = """
Output your evaluation as a single valid JSON object matching the following structure:
{
  "precision": {
    "explanation": "Brief explanation of why this score was given.",
    "score": 0
  },
  "recall": {
    "explanation": "Brief explanation of why this score was given.",
    "score": 0
  },
  "repetition": {
    "explanation": "Brief explanation of why this score was given.",
    "score": 0
  },
  "readability": {
    "explanation": "Brief explanation of why this score was given.",
    "score": 0
  }
}

Ensure the output is valid JSON:
- Use **double quotes** (") for all keys and string values.
- When quoting text or sections inside the explanations, use escaped double quotes (\\") to maintain valid JSON formatting.
- Do not include any additional information in the output.
"""

JUDGE_OUTPUT_XML = """
Output your evaluation as a single valid XML object matching the following structure:
<evaluation>
  <precision>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </precision>
  <recall>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </recall>
  <repetition>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </repetition>
  <readability>
    <explanation>Brief explanation of why this score was given.</explanation>
    <score>0</score>
  </readability>
</evaluation>

Ensure the output is valid XML:
- Escape special characters in text nodes: & as &amp;, < as &lt;, > as &gt;, " as &quot;, ' as &apos;.
- Do not include any additional information in the output.
"""