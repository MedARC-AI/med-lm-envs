QUESTION_SYSTEM_PROMPT = "You are an expert working for a Public Health agency."

# MCQ template
QUESTION_TEMPLATE_MCQ = """The following are multiple choice questions (with answers) about UK Government public health guidance.

Question: This question relates to UK Health Security Agency (UKHSA) guidance that could be found on the gov.uk website as of 08/01/2025.
{question}

Options:
{options}

{answer_instruction}

Answer:"""

ANSWER_INSTRUCTION_XML = 'Provide the letter (A, B, C, D, E, F, or G) of the correct answer. You should state "The answer is <answer>X</answer>", where the X contained in the tags is the correct letter choice, make sure you include the <answer></answer> tags around your final answer in your response. DO NOT provide any other information or text in your response.'

ANSWER_INSTRUCTION_BOXED = 'Provide the letter (A, B, C, D, E, F, or G) of the correct answer. You should state "The answer is \\boxed{{X}}", where the X contained in the box is the correct letter choice, make sure you include the \\boxed{{}} around your final answer in your response. DO NOT provide any other information or text in your response.'

# Freeform template
QUESTION_TEMPLATE_FREEFORM = """The following is a question about UK Government public health guidance.

Question: This question relates to UK Health Security Agency (UKHSA) guidance.
{question}

Please answer the question to the best of your knowledge.

Answer:"""

# Judge template for freeform evaluation (from paper, system + user prompt combined)
JUDGE_TEMPLATE = """You are an expert in UK public health. You are going to evaluate whether a given answer to a public health guidance question is correct.

You are tasked with evaluating whether a given answer is correct based on the ground truth answer and provided context. Carefully analyse the ground truth answer and context and determine whether the given answer correctly answers the question and aligns with the information given.

===========
Question:
===========
{question}

===========
Context:
===========
{context}

===========
Ground Truth Answer:
===========
{ground_truth_answer}

===========
Given Answer:
===========
{given_answer}

For the given answer to be correct it must align with the ground truth without omitting any key details **and** any additional detail in the given answer must be seen in the provided context. Determine, with reasoning, whether the given answer is correct based on the ground truth answer and context.

Give your response in the following json format:
{{"reasoning": "Why is the answer correct/incorrect", "predicted_correct": true or false}}"""
