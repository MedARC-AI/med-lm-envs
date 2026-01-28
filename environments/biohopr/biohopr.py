import asyncio
from typing import Dict, Optional, List, Tuple, NamedTuple

import verifiers as vf
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
import torch
from torch import nn
import tqdm
from collections import namedtuple
import numpy as np
from abc import ABC, abstractmethod
from typing import Coroutine, Any
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
from transformers.utils import logging

# Cosine similarity threshold for embedded precision
TAU: float = 0.9
TASKS: List[str] = ['biohopr_hop1', 'biohopr_hop2', 'biohopr_hop1_multi', 'biohopr_hop2_multi']
# Mapping from task names to Huggingface dataset question keys
TASK_TO_QUESTION_KEY: Dict[str,str] = {
    'biohopr_hop1':'hop1_question',
    'biohopr_hop2':'hop2_question',
    'biohopr_hop1_multi':'hop1_question_multi',
    'biohopr_hop2_multi':'hop2_question_multi'
} 

def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return F.normalize(token_embeddings, p=2, dim=1)

"""Credits: Code mostly from https://huggingface.co/FremyCompany/BioLORD-2023"""
class EncodingModel:

    pooling = {
        'FremyCompany/BioLORD-2023': mean_pooling,
        'Simonlee711/Clinical_ModernBERT': lambda output,mask: output[0][:,0],  # CLS pooling
    }

    def __init__(self, model_name: str):
        super(EncodingModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Set to ERROR to hide warnings/info
        verbosity = logging.get_verbosity()
        logging.set_verbosity_error() 
        self.model = AutoModel.from_pretrained(model_name)
        logging.set_verbosity(verbosity)
        self.use_cuda = False
        self.pooling_f = self.pooling.get(model_name, mean_pooling)

    def encode(self, input_texts: List[str]) -> torch.Tensor:
        # Tokenize sentences
        encoded_input = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors='pt')
        if self.use_cuda:
            encoded_input = {key: val.cuda() for key, val in encoded_input.items()}
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embeddings = self.pooling_f(model_output, encoded_input['attention_mask'])
        return embeddings
    
    def cuda(self):
        self.model = self.model.cuda()
        self.use_cuda = True
        return self

class AsyncEncoder(ABC):
    @abstractmethod
    def encode(self, texts: List[str]) -> Coroutine[Any, Any, torch.Tensor]:
        pass

class AsyncBufferedEncoder(AsyncEncoder):
    """
    Asynchronous buffered encoder for EncodingModel models.
    Batches encoding requests to improve efficiency.
    """
    def __init__(self, model: EncodingModel, batch_size: int = 32):
        self.model = model
        self.batch_size = batch_size
        self.buffer = []
        self.futures = []
        self.lock = asyncio.Lock()

    async def encode(self, texts: List[str]) -> torch.Tensor:
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        self.buffer.append(texts)
        self.futures.append(future)

        # Run flush asyncio task to avoid blocking
        
        asyncio.create_task(self._flush())
        return await future

    async def _flush(self):
        if not self.buffer:
            return
        async with self.lock:
            if not self.buffer:
                return
            await asyncio.sleep(0)  # slight delay to allow batching
            texts = self.buffer[:self.batch_size]
            futures = self.futures[:len(texts)]
            self.buffer = self.buffer[len(texts):]
            self.futures = self.futures[len(texts):]
            # indexs of texts in the batch
            idxs = [len(sublist) for sublist in texts]
            #flatten list of lists
            texts = [item for sublist in texts for item in sublist]
            # Compute set of embeddings in seperate thread to avoid blocking event loop, lock if necessary

            embeddings = await asyncio.to_thread(self.model.encode, texts)
        embeddings = embeddings if isinstance(embeddings, torch.Tensor) else torch.tensor(embeddings)

        # Reshape embeddings to match the original input structure
        split_embeddings = torch.split(embeddings, idxs)

        for future, embedding in zip(futures, split_embeddings):
            future.set_result(embedding)

class AsyncEmbeddingClient(AsyncEncoder):
    """
    Asynchronous embedding client for OpenAI-compatible APIs.
    """
    def __init__(self, client, model: str = "clinicalmodernbert"):
        self.client = client
        self.model = model

    async def encode(self, texts: List[str]) -> torch.Tensor:
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        embeddings = [torch.tensor(data.embedding) for data in response.data]
        return torch.stack(embeddings)

class model_config(NamedTuple):
    judge_model: str
    judge_api_key: Optional[str]
    judge_base_url: Optional[str]
    judge_client: Optional[AsyncOpenAI]
    judge_answer_num: int
    embeddings_model: EncodingModel
    encoder: AsyncEncoder
    tau: float = TAU

# Judge prompt template for medical diagnosis evaluation
JUDGE_TEMPLATE = """\
Your job is to evaluate whether a medical diagnosis is equivalent to any of the ground truth diagnoses.

You will be given:
1. A medical case prompt (question)
2. The ground truth diagnoses (possible answers)  
3. A predicted diagnosis (response)

Your task is to determine if the predicted diagnosis is medically equivalent to any of the ground truths, even if worded differently.

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



def parse_answers_completions(parser:vf.Parser, completion: str, info: Dict) -> Tuple[Optional[str], List[str]]:
    """
    Parses the completion to extract the answer using the provided parser.
    Args:
        parser: Parser instance to extract the answer
        answer: The ground truth answer string
        completion: The model's completion text
        info: Dictionary containing additional information, including ground truth answers
    Returns:
        Extracted answer string or None if parsing fails
    """
    parsed_completion = parser.parse_answer(completion)
    answers = info.get("answer", [])
    return (parsed_completion, answers)

async def answer_completion_similarity(encoder: AsyncEncoder, answer: List[str], completion: str) -> torch.Tensor:
    """
    Computes cosine similarity between the embeddings of the answer and completion.
    Args:
        answer: The ground truth answer string
        completion: The model's completion text
    Returns:
        Cosine similarity score between answer and completion embeddings
    """

    answer_completions = answer+[completion.strip().lower()]

    answer_comp_embeds = await encoder.encode(answer_completions)
    answer_comp_embeds = answer_comp_embeds if isinstance(answer_comp_embeds, torch.Tensor) else torch.tensor(answer_comp_embeds)
    similarities = (answer_comp_embeds[:len(answer)] @ answer_comp_embeds[len(answer):].T) / (
    torch.sqrt(torch.sum(answer_comp_embeds[:len(answer)]**2, keepdim=True, dim=1)) * torch.sqrt(torch.sum(answer_comp_embeds[len(answer):]**2))
    )
    return similarities.flatten().cpu()

async def llm_as_a_judge_impl(
    config: model_config,
    judge, 
    prompt: str,
    completion: str,
    answer: str,
    info: Dict,
    state: Dict,
    parser: vf.Parser,
    **kwargs
) -> float:
    """EncodingModel
    Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
    """
    parsed_completion,answers = parse_answers_completions(parser, completion, info)
    if(parsed_completion is None): return 0.0
    similarities = await answer_completion_similarity(config.encoder, answers, parsed_completion)

    # Select top-K most similar answers for judging
    top_n_indices = torch.topk(similarities, k=min(config.judge_answer_num, len(answers))).indices.tolist()
    answers = [answers[i] for i in top_n_indices]

    answer = ", ".join(answers)

    # Get judge response using the extracted answer
    judge_response = await judge(prompt, completion, answer, state)
    
    # Parse judge response
    judge_response_clean = judge_response.strip().upper()

    # Return 1.0 if equivalent, 0.0 otherwise
    if "EQUIVALENT" in judge_response_clean and "NOT_EQUIVALENT" not in judge_response_clean:
        return 1.0
    else:
        return 0.0

async def embedded_precision_impl(config: model_config, parser: vf.Parser, completion: str, info: Dict, **kwargs) -> float:
    """Calculates embedded precision based on cosine similarity between
    completion and ground truth answers.
    """
    parsed_completion,answers = parse_answers_completions(parser, completion, info)
    if(parsed_completion is None): return 0.0
    similarities = await answer_completion_similarity(config.encoder, answers, parsed_completion)
    return 1.0 if(similarities.max() > config.tau) else 0.0

def _rubric_f(config: model_config, use_judge=True):
    "Returns a function that calculates precision based on embedded cosine similarity."
    async def embedded_precision(parser: vf.Parser, completion: str, info: Dict) -> float:
        return await embedded_precision_impl(config, parser, completion, info)
    async def llm_as_a_judge(judge, prompt: str, completion: str, answer: str, info: Dict, state: Dict, parser: vf.Parser, **kwargs) -> float:
        return await llm_as_a_judge_impl(config, judge, prompt, completion, answer, info, state, parser, **kwargs)
    return llm_as_a_judge if(use_judge) else embedded_precision

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

def _rubrics(eval_method:str,parser:vf.Parser, config: model_config) -> List[vf.Rubric]:
    """
    Creates and configures a list of Rubrics for BioHopR evaluation based on eval_method.
    Args:
        eval_method: "judge" (default), "metrics", or "judge-only"
        parser: Parser to extract answers from completions
        config: model_config instance with judge and embedding model settings
    Returns:
        List of Rubric instances for evaluation
    """
    if(eval_method not in ['judge','judge-only','metrics']): 
        raise ValueError(f"Unsupported eval_method: {eval_method=}")
    rubrics,weights = [],[]
    if(eval_method in ['judge','judge-only']):
        rubrics += [vf.JudgeRubric(
            judge_client=config.judge_client, judge_model=config.judge_model, judge_prompt=JUDGE_TEMPLATE, parser=parser,
        )]
        rubrics[-1].add_reward_func(_rubric_f(config),weight = 1.0)
    if(eval_method in ['metrics', 'judge']):
        weight = 1.0 if eval_method=='metrics' else 0.0
        rubrics += [vf.Rubric( funcs=[_rubric_f(config,use_judge=False)], weights=[weight], parser=parser)]
    return rubrics

def create_model(model_name: str, use_cuda: bool = False) -> EncodingModel:
    """
    Creates an EncodingModel instance based on the provided model name.
    Args:
        model_name: Name of the EncodingModel to load
        use_cuda: Whether to use CUDA for the model
    Returns:
        Loaded EncodingModel instance
    """
    if model_name=='FremyCompany/BioLORD-2023':
        model= EncodingModel(model_name)
    elif model_name=='Simonlee711/Clinical_ModernBERT':
        model = EncodingModel(model_name)
    else:
        raise ValueError(f"Unsupported model name: {model_name=}")
    if(use_cuda):
        model = model.cuda()
    return model

def load_environment(
    use_think: bool = False,
    system_prompt: Optional[str] = None,
    answer_format: AnswerFormat | str = AnswerFormat.XML,
    task: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    eval_method: str = "metrics",  # "judge", "metrics (default), or "judge-only",
    judge_answer_num: int = 5,
    embeddings_model: str = 'FremyCompany/BioLORD-2023',
    tau: float = TAU,
    use_cuda: bool = torch.cuda.is_available(),
    embedding_batch_size: int = 32,
    embedding_model_url: Optional[str] = None,
    embedding_api_key: Optional[str] = None,
) -> vf.Environment:
    """
    BioHopR multiple-hop biomedical question answering evaluation
    - Supports reasoning (use_think=True) or non-reasoning models
    - system_prompt: Optional custom system prompt. If None, uses default based on answer_format and use_think.
    - answer_format: Determines how to parse completion for answer. Also sets system prompt if system
    - task: which BioHopR task to evaluate against. Valid options are
      ['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi', 'all'].
      'all' evaluates on all tasks. Default is 'biohopr_hop2'.
    - eval_method: "judge", "metrics" (default), or "judge-only"
    - judge_model: Model name to use for judging (default: "gpt-4o-mini")
    - judge_base_url: Optional base URL for custom OpenAI-compatible API endpoint
    - judge_api_key: Optional API key for OpenAI-compatible API endpoint
    - judge_answer_num: Number of top similar ground truth answers to consider for judging
    - embeddings_model: EncodingModel name for computing answer-completion similarity. Valid options: ['FremyCompany/BioLORD-2023', 'Simonlee711/Clinical_ModernBERT']
    - tau: Cosine similarity threshold for embedded precision
    - use_cuda: Whether to use CUDA for embedding model
    - embedding_batch_size: Maximum batch size for embedding model encoding, use in case of out of memory error
    - embedding_model_url: Optional URL for OpenAI-compatible embedding model API
    - embedding_api_key: Optional API key for OpenAI-compatible embedding model API

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
    embedding_client = None
    if (embedding_model_url is not None):
        embedding_client = AsyncEmbeddingClient(
            client=AsyncOpenAI(
                base_url=embedding_model_url,
                api_key=embedding_api_key if embedding_api_key else os.getenv("OPENAI_API_KEY"),
            ),
            model=embeddings_model
        )

    # Initialize OpenAI client for judge
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
    model = create_model(embeddings_model, use_cuda=use_cuda) if (not embedding_client) else None
    encoder = embedding_client if (embedding_client) else AsyncBufferedEncoder(model, batch_size=embedding_batch_size)

    model_c = model_config(
        judge_model=judge_model,
        judge_api_key=judge_api_key,
        judge_base_url=judge_base_url,
        judge_client=judge_client,
        judge_answer_num=judge_answer_num,
        embeddings_model=model,
        encoder=encoder,
        tau=tau,
    )
    rubrics = _rubrics(eval_method,parser, model_c)
    
    return vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=vf.RubricGroup(rubrics,parser=parser),
    )