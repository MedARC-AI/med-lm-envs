"""
Judge prompt template for evaluating ACI-BENCH clinical note generation.
Adapted from MedHELM's ACI Bench evaluation prompts.
"""

JUDGE_TEMPLATE = """\
You are an expert medical clinician tasked with evaluating the quality of an AI-generated clinical note.

Your goal is to assess how well the generated note summarizes the key medical information from a patient-doctor conversation. You will compare the AI's note against a reference "gold standard" note.

The patient-doctor conversation will be provided in these tags:
<conversation>
{conversation}
</conversation>

The AI-generated candidate note will be provided in these tags:
<candidate_note>
{candidate_note}
</candidate_note>

The reference "gold standard" note will be provided in these tags:
<reference_note>
{reference_note}
</reference_note>

Carefully review the <candidate_note> and compare it to both the original <conversation> and the <reference_note>.

Please evaluate the generated summary on a scale of 1-5 (1 = poor, 5 = excellent) for each of these three key dimensions:

Evaluation Criteria:
1.  **Accuracy (1-5):**
    - Does the candidate note correctly capture the main medical issues, symptoms, and clinical details from the conversation?
    - Are there any hallucinations or medically incorrect statements?

2.  **Completeness (1-5):**
    - Does the candidate note include all important medical information and key decisions from the conversation?
    - Are there any critical omissions compared to the reference note?

3.  **Clarity & Structure (1-5):**
    - Is the candidate note well-organized, easy to read, and suitable for clinical use?
    - Does it correctly follow the requested section structure?

Output Format:
{output_format}
"""

JUDGE_OUTPUT_JSON = """
Output your evaluation as a single valid JSON object matching the following structure:
{
    "accuracy": {
        "score": 0,
        "explanation": "Brief explanation of why this score was given for accuracy."
    },
    "completeness": {
        "score": 0,
        "explanation": "Brief explanation of why this score was given for completeness."
    },
    "clarity": {
        "score": 0,
        "explanation": "Brief explanation of why this score was given for clarity and structure."
    }
}

Ensure the output is valid JSON, using double quotes for all keys and string values.
"""