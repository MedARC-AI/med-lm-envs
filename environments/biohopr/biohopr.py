from typing import Dict, Optional, List

import verifiers as vf
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT,AnswerFormat
from verifiers.utils.data_utils import (
    BOXED_SYSTEM_PROMPT,
    THINK_BOXED_SYSTEM_PROMPT,
    extract_boxed_answer,
)
from numpy.linalg import norm
from enum import Enum
from functools import partial
import os
from openai import AsyncOpenAI

# Cosine similarity threshold for embedded precision
TAU: float = 0.9 
TASKS: List[str] = ['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi']
# Mapping from task names to Huggingface dataset question keys
TASK_TO_QUESTION_KEY: Dict[str,str] = {
    'biohopr_hop1':'hop1_question',
    'biohopr_hop2':'hop2_question',
    'biohopr_hop1_multi':'hop1_question_multi',
    'biohopr_hop2_multi':'hop2_question_multi'
} 

# Judge prompt template for medical diagnosis evaluation
JUDGE_TEMPLATE = """\
Your job is to evaluate whether a medical diagnosis is equivalent to any of the ground truth diagnoses.

You will be given:
1. A medical case prompt (question)
2. The ground truth diagnoses (possible answers)  
3. A predicted diagnosis (response)

Your task is to determine if the predicted diagnosis is medically equivalent to the ground truths, even if worded differently.

Consider these guidelines:
- Medical terms that refer to the same condition should be considered equivalent
- Different levels of specificity may be acceptable (e.g., "pneumonia" vs "bacterial pneumonia")
- Spelling variations of medical terms should be considered equivalent
- The core medical meaning should match, even if additional details vary
- Consider both the primary diagnosis and any relevant differential diagnoses

Examples:
- "Acute myocardial infarction" and "heart attack" → EQUIVALENT
- "Type 2 diabetes mellitus" and "diabetes" → EQUIVALENT  
- "Upper respiratory infection" and "pneumonia" → NOT EQUIVALENT
- "Hypertension" and "high blood pressure" → EQUIVALENT

Question: {question}

Ground truth diagnoses: [{answer}]

Predicted diagnosis: {response}

Is the predicted diagnosis medically equivalent to any of the ground truth diagnoses?
Respond with either "EQUIVALENT" or "NOT_EQUIVALENT".
""".strip()

async def llm_as_a_judge(
    judge, 
    prompt: str,
    completion: str,
    answer: str,
    info: Dict,
    state: Dict,
    parser: vf.Parser,
    **kwargs
) -> float:
    """
    Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
    """
    parsed_completion = parser.parse_answer(completion)
    if(parsed_completion is None): return 0.0
    answers = info.get("answer", [])
    answer = ", ".join(answers)
    # Get judge response using the extracted answer
    judge_response = await judge(prompt, completion, answer, state, **kwargs)
    
    # Parse judge response
    judge_response_clean = judge_response.strip().upper()
    # Return 1.0 if equivalent, 0.0 otherwise
    if "EQUIVALENT" in judge_response_clean and "NOT_EQUIVALENT" not in judge_response_clean:
        return 1.0
    else:
        return 0.0

def _embedded_precision_f(model: SentenceTransformer):
    "Returns a function that calculates precision based on embedded cosine similarity."
    def embedded_precision(parser: vf.Parser, completion: str, info: Dict, **kwargs) -> float:
        answers = info.get("answer", [])
        parsed_completion = parser.parse_answer(completion)
        if(parsed_completion is None): return 0.0
        parsed_completion = parsed_completion.lower().strip()
        answer_embeds = model.encode(answers)
        completion_embed = model.encode(parsed_completion)
        similarities = (answer_embeds @ completion_embed) / ( 
            norm(answer_embeds,axis=1) * norm(completion_embed) )
        return 1.0 if(similarities.max() > TAU) else 0.0
    return embedded_precision

def question_to_prompt(question: str, task: str) -> str:
    """Wrap question into full prompt for BioHopR based on task.
    Args:
      - question: the question string from the dataset
      - task: which BioHopR task to evaluate against. Valid options are
        ['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi'].
    Returns:
        - full prompt string
    """
    # Prompt template matching huggingface.co/datasets/knowlab-research/BioHopR
    start = "You are an expert biomedical researcher.\n"
    end = "Answer:\n"
    single = ' Just give me the answer without any explanations.\n'
    multi = ' Just give me the answers without any explanations in a bullet-pointed list.\n'
    if('multi' in task):
        return start + question + multi + end
    else:
        return start + question + single + end

def _biohoper_format(exs, tasks:List[str]=TASKS) -> Dict:
    """Format BioHopR dataset examples into vf.SingleTurnEnv format.
    Each example may contain multiple tasks; this function expands them into separate entries.
    Args:
        exs: batch of examples from the BioHopR dataset
        tasks: list of BioHopR tasks to include
    Returns:
        Dict with keys 'question', 'info', and 'task' for vf.SingleTurnEnv
    """
    bs = len(exs['answer'])
    prompts = [ [question_to_prompt(q,task) for q in exs[TASK_TO_QUESTION_KEY[task]] ]
                    for task in tasks  ]
    prompts = [ o2 for o in zip(*prompts) for o2 in o]
    answers = [ {'answer': o} for o in exs['answer']]
    answers = [ o for o in answers for _ in range(len(tasks))]
    return { 'question':prompts,
            'info': answers,
            'task': bs*tasks}

def _prepare_parseing(answer_format,system_prompt,use_think=False): 
    """
    Prepares the parser, system prompt, and answer format based on input parameters.
    Args:
        answer_format: Desired answer format (AnswerFormat enum or str)
        system_prompt: Optional custom system prompt. If None, uses default based on answer_format and use_think.
        use_think: Whether to use a think-style parser/system prompt.
    Returns:
        Tuple of (answer_format, system_prompt, parser)
    """
    answer_format = AnswerFormat(answer_format) if isinstance(answer_format, str) else answer_format
    if answer_format == AnswerFormat.XML:
        system_prompt = system_prompt or (THINK_XML_SYSTEM_PROMPT if use_think else XML_SYSTEM_PROMPT)
        parser_fields = ["think", "answer"] if use_think else ["answer"]
        parser = vf.XMLParser(fields=parser_fields, answer_field="answer")
    elif answer_format == AnswerFormat.BOXED:
        parser = vf.ThinkParser(extract_boxed_answer) if use_think else vf.Parser(extract_boxed_answer)
        system_prompt = system_prompt or (THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT)
    else:
        raise ValueError(f"Unsupported answer format: {answer_format=}")
    return answer_format,system_prompt,parser

def _rubrics(eval_method:str,parser:vf.Parser,  judge_client, judge_model: str = "gpt-4o-mini"):
    """
    Creates and configures a list of Rubrics for BioHopR evaluation based on eval_method.
    Args:
        eval_method: "judge" (default), "metrics", or "judge-only"
        parser: Parser to extract answers from completions
        judge_client: AsyncOpenAI client instance for making judge API calls
        judge_model: Model name to use for judging (default: "gpt-4o-mini")
    Returns:
        List of Rubric instances for evaluation
    """
    if(eval_method not in ['judge','judge-only','metrics']): 
        raise ValueError(f"Unsupported eval_method: {eval_method=}")
    rubrics,weights = [],[]
    if(eval_method in ['judge','judge-only']):
        rubrics += [vf.JudgeRubric(
            funcs=[llm_as_a_judge],judge_client=judge_client, judge_model=judge_model, judge_prompt=JUDGE_TEMPLATE, parser=parser, weights = [1.0],
        )]
    if(eval_method in ['metrics', 'judge']):
        model = SentenceTransformer('FremyCompany/BioLORD-2023')
        weight = 1.0 if eval_method=='metrics' else 0.0
        rubrics += [vf.Rubric( funcs=[_embedded_precision_f(model)], weights=[weight], parser=parser)]
    return rubrics

def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    task: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    eval_method: str = "judge",  # "judge" (default), "metrics, or "judge-only"
) -> vf.Environment:
    """
    BioHopR multiple-hop biomedical question answering evaluation
    - Supports reasoning (use_think=True) or non-reasoning models
    - system_prompt: Optional custom system prompt. If None, uses default based on answer_format and use_think.
    - answer_format: Determines how to parse completion for answer. Also sets system prompt if system
    - task: which BioHopR task to evaluate against. Valid options are
      ['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi', 'all'].
      'all' evaluates on all tasks. Default is 'biohopr_hop2'.
    - eval_method: "judge" (default), "metrics", or "judge-only"
    - judge_model: Model name to use for judging (default: "gpt-4o-mini")
    - judge_base_url: Optional base URL for custom OpenAI-compatible API endpoint
    - judge_api_key: Optional API key for OpenAI-compatible API endpoint
    Returns:
        vf.Environment instance for BioHopR evaluation
    """
    if(task is None):
        tasks = ['biohopr_hop2']
    elif(task == 'all'):
        tasks = TASKS
    elif(task not in TASKS):
        raise ValueError(f"Unsupported task: {task=}")
    else:
        tasks = [task]

    ds = load_dataset("knowlab-research/BioHopR", split="train")
    ds = ds.map(partial(_biohoper_format,tasks=tasks), remove_columns=ds.column_names, batched=True)
    
    answer_format,system_prompt,parser = _prepare_parseing(answer_format,system_prompt,use_think=use_think)
    
    # Initialize OpenAI client for judge
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None

    rubrics = _rubrics(eval_method,parser, judge_client=judge_client, judge_model=judge_model)

    return vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=vf.RubricGroup(rubrics,parser=parser),
    )

