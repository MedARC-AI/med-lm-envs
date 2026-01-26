# System Prompt, create_prompt_no_knowledge, create_prompt_with_knowledge taken from MedHallu Github Repository with minor modifications( added \\boxed{} to the Final answer)

# source: https://github.com/MedHallu/MedHallu/blob/main/Detection/detection_vllm_notsurecase.py

# Paper: https://arxiv.org/abs/2502.14302


system_prompt = """
You are an AI assistant with extensive knowledge in the field of medicine. I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

Type: Misinterpretation of #Question#: These are hallucinated answers that misunderstands the question, leading to an off-topic or irrelevant response.
Type: Incomplete Information: These are hallucinated answers that Point out what is not true without providing correct information.
Type: Mechanism and Pathway Misattribution - These are hallucinated answer that falsely attribution of biological mechanisms, molecular pathways, or disease processes that contradicts established medical knowledge
Type: Methodological and Evidence Fabrication - Inventing false research methods, statistical data, or specific clinical outcomes


You are trying to determine if there is a factual contradiction between the answer and the world knowledge.
Some information in the answer might be fabricated. Answer \\boxed{{0}} if the answer is factual and \\boxed{{1}} if the answer is hallucinated and \\boxed{{2}} if you are unsure about the factuality and want to pass. If you are unsure about the answer choose \\boxed{{2}} instead of guessing and making a mistake.
"""


def create_prompt_no_knowledge(question, option1):
    prompt = f"""
Question: {question}
Answer: {option1}

Return just the answer, Answer \\boxed{{0}} if the answer is factual and \\boxed{{1}} if the answer is hallucinated and \\boxed{{2}} if you are unsure about the answer and want to pass.
If you are unsure about the answer choose \\boxed{{2}} instead of guessing and making a mistake.
Your Judgement:
"""
    return prompt


def create_prompt_with_knowledge(question, option1, knowledge):
    prompt = f"""
World Knowledge: {knowledge}
Question: {question}
Answer: {option1}

Return just the answer. Answer \\boxed{{0}} if the answer is factual, \\boxed{{1}} if the answer is hallucinated, and \\boxed{{2}} if you are unsure about the answer and want to pass.
If you are unsure about the answer, choose \\boxed{{2}} instead of guessing and making a mistake.
Your Judgement:
"""
    return prompt
