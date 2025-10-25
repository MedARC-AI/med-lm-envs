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

TAU: float = 0.9
TASKS: List[str] = ['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi']

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
            (answer_embeds**2).sum(axis=1)**0.5 )
        return 1.0 if(similarities.max() > TAU) else 0.0
    return embedded_precision

def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
) -> vf.Environment:
    '''
    BioHopR multiple-hop biomedical question answering evaluation
    - Supports reasoning (use_think=True) or non-reasoning models
    '''
    def _map(exs):
        bs = len(exs['answer'])
        single = ' Just give me the answer without any explanations.'
        hop1 = [ o+single for o in exs['hop1_question']]
        hop2 = [ o+single for o in exs['hop2_question']]
        multi = ' Just give me the answers without any explanations in a bullet-pointed list.'
        hop1m = [ o+multi for o in exs['hop1_question_multi']]
        hop2m = [ o+multi for o in exs['hop2_question_multi']]
        questions = sum([list(tasks) for tasks in zip(hop1,hop2,hop1m,hop2m)],[])
        tasks = bs*TASKS
        answers = [ {'answer': o} for o in exs['answer']]
        answers = [ o for o in answers for _ in range(len(TASKS))]
        return { 'question':questions,
                'info': answers,
                'task': tasks}
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

