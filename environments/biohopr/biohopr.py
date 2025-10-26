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

TAU: float = 0.9
TASKS: List[str] = ['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi']
TASK_TO_QUESTION_KEY: Dict[str,str] = {
    'biohopr_hop1':'hop1_question',
    'biohopr_hop2':'hop2_question',
    'biohopr_hop1_multi':'hop1_question_multi',
    'biohopr_hop2_multi':'hop2_question_multi'
}

def _embedded_precision_f(model: SentenceTransformer):
    "Returns a function that calculates precision based on embedded cosine similarity."
    def embedded_precision(parser: vf.Parser, completion: str, info: Dict, **kwargs) -> float:
        answers = info.get("answer", [])
        completion = parser.parse_answer(completion)
        if(completion is None): return 0.0
        completion = completion.lower().strip()
        answer_embeds = model.encode(answers)
        completion_embed = model.encode(completion)
        similarities = (answer_embeds @ completion_embed) / ( 
            norm(answer_embeds,axis=1) * norm(completion_embed) )
        return 1.0 if(similarities.max() > TAU) else 0.0
    return embedded_precision

def question_to_prompt(question: str, task: str) -> str:
    single = ' Just give me the answer without any explanations.'
    multi = ' Just give me the answers without any explanations in a bullet-pointed list.'
    if('multi' in task):
        return question + multi
    else:
        return question + single

def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    task: Optional[str] = None,
) -> vf.Environment:
    '''
    BioHopR multiple-hop biomedical question answering evaluation
    - Supports reasoning (use_think=True) or non-reasoning models
    '''
    if(task is None):
        tasks = ['biohopr_hop2']
    elif(task == 'all'):
        tasks = TASKS
    elif(task not in TASKS):
        raise ValueError(f"Unsupported task: {task=}")
    else:
        tasks = [task]

    def _map(exs):
        bs = len(exs['answer'])
        prompts = [ [question_to_prompt(q,task) for q in exs[TASK_TO_QUESTION_KEY[task]] ]
                     for task in tasks  ]
        prompts = [ o2 for o in zip(*prompts) for o2 in o]
        answers = [ {'answer': o} for o in exs['answer']]
        answers = [ o for o in answers for _ in range(len(tasks))]
        return { 'question':prompts,
                'info': answers,
                'task': bs*tasks}
    ds = load_dataset("knowlab-research/BioHopR", split="train")
    ds = ds.map(_map, remove_columns=ds.column_names, batched=True)
    model = SentenceTransformer('FremyCompany/BioLORD-2023')
    precision = _embedded_precision_f(model)

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

    rubric = vf.Rubric(funcs=[precision], weights=[1.0], parser=parser)
    return vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )

