from enum import Enum

THINK_XML_SYSTEM_PROMPT = "Think step-by-step inside <think>...</think> tags. Then, give your final answer inside <answer>...</answer> XML tags."

XML_SYSTEM_PROMPT = "Please reason step by step, then give your final answer within <answer>...</answer> XML tags."


class AnswerFormat(str, Enum):
    BOXED = "boxed"
    JSON = "json"
    XML = "xml"
