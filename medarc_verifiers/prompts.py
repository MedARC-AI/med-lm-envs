from enum import Enum

THINK_XML_SYSTEM_PROMPT = "Think step-by-step inside <think>...</think> tags. Then, give your final answer inside <answer>...</answer> XML tags."

THINK_XML_TOOL_SYSTEM_PROMPT = "Think step-by-step inside <think>...</think> tags and use tools as needed. Then, give your final answer inside <answer>...</answer> XML tags."

XML_SYSTEM_PROMPT = "Please reason step by step, then give your final answer within <answer>...</answer> XML tags."

XML_TOOL_SYSTEM_PROMPT = (
    "Please reason step by step, use tools as needed, then give your final answer within <answer>...</answer> XML tags."
)


THINK_BOXED_TOOL_SYSTEM_PROMPT = "Think step-by-step inside <think>...</think> tags and use tools as needed. Then, give your final answer inside \\boxed{}."

BOXED_TOOL_SYSTEM_PROMPT = (
    "Please reason step by step, use tools as needed, and put your final answer within \\boxed{}."
)


class AnswerFormat(str, Enum):
    BOXED = "boxed"
    JSON = "json"
    XML = "xml"
