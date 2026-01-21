from biohopr import _prepare_parseing,create_model, embedded_precision_impl,model_config,_rubrics,_biohoper_format, AsyncBufferedEncoder, _rubric_f,AsyncEmbeddingClient
from datasets import load_dataset
from medarc_verifiers.prompts import THINK_XML_SYSTEM_PROMPT, XML_SYSTEM_PROMPT,AnswerFormat
import os
import argparse
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai import AsyncOpenAI,OpenAI
import verifiers as vf
import asyncio
from functools import partial
import json
from dataclasses import dataclass
import torch


UNRELATED_SYSTEM = "You are a unhelpful biomedical reasoning model that gives unrelated medical advice to any queries. " + XML_SYSTEM_PROMPT

@dataclass
class RemoteModelConfig:
    api_key: str | None
    base_url: str | None
    judge_model: str
    completion_model: str | None = 'openai/gpt-oss-20b'


def test_judge(ds,config:RemoteModelConfig,parser,related=True, num_examples=32):
    """Test to compare LLM-as-a-Judge vs embedded precision metrics on BioHopR."""
    # Initialize OpenAI client for judge
    judge_client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=60*60) if config.api_key else None
    model = create_model('Simonlee711/Clinical_ModernBERT').cuda()
    model_c = model_config(
        judge_model=config.judge_model,
        judge_api_key=config.api_key,
        judge_base_url=config.base_url,
        judge_client=judge_client,
        judge_answer_num=7,
        embeddings_model=model,
        encoder=AsyncBufferedEncoder(model)
    )
    rubrics = _rubrics('judge-only',parser, model_c)

    env = vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=XML_SYSTEM_PROMPT if related else UNRELATED_SYSTEM,
        parser=parser,
        rubric=vf.RubricGroup(rubrics,parser=parser),
    )
    
    client = judge_client
    system_prompt = XML_SYSTEM_PROMPT
    results = env.evaluate(client, model='openai/gpt-oss-20b', num_examples=num_examples)
    results = asyncio.run(results)

    return results

def test_metrics(ds,config,parser,related=True, num_examples=32):
    """Test to compute embedded precision metrics on BioHopR."""
    model = create_model('FremyCompany/BioLORD-2023').cuda()
    model_c = model_config(
        judge_model=config.judge_model,
        judge_api_key=None,
        judge_base_url=None,
        judge_client=None,
        judge_answer_num=5,
        embeddings_model=model,
        encoder=AsyncBufferedEncoder(model, ds_len = len(ds))
    )
    rubrics = _rubrics('metrics',parser, model_c)

    env = vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=XML_SYSTEM_PROMPT if related else UNRELATED_SYSTEM,
        parser=parser,
        rubric=vf.RubricGroup(rubrics,parser=parser),
    )
    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=60*60)
    results = env.evaluate(client, model='openai/gpt-oss-20b', num_examples=num_examples)
    results = asyncio.run(results)
    return results

def test_metrics_vs_judge(ds, config, data_parser, num_examples=32):
    results = test_judge(ds, config, data_parser, num_examples=num_examples, related=True)

    logs = [
        {   
            "name": "biohopr_judge_related",
            "results":[{
                
                "example": i,
                "reward": results.reward[i],
                "prompt": results.prompt[i],
                "completion": results.completion[i],
                "answers": results.info[i]['answer'],
            }
            for i in range(len(results.reward))
            ]
        }
    ]

    # Calculate and display summary stats
    rewards = results.reward
    avg_reward = sum(rewards) / len(rewards)
    print(f"Related Judge Summary: {len(rewards)} examples, avg reward: {avg_reward:.3f}")

    results = test_judge(ds,config,data_parser,related=False, num_examples=num_examples)

    logs += [
        {   
            "name": "biohopr_judge_unrelated",
            "results":[{
                
                "example": i,
                "reward": results.reward[i],
                "prompt": results.prompt[i],
                "completion": results.completion[i],
                "answers": results.info[i]['answer'],
            }
            for i in range(len(results.reward))
            ]
        }
    ]

    # Calculate and display summary stats
    rewards = results.reward
    avg_reward = sum(rewards) / len(rewards)
    print(f"Unrelated Judge Summary: {len(rewards)} examples, avg reward: {avg_reward:.3f}")

    metrics_results = test_metrics(ds,config,data_parser, related=True, num_examples=num_examples)
    embedded_precisions = metrics_results.metrics['embedded_precision']
    avg_embedded_precision = sum(embedded_precisions) / len(embedded_precisions)
    print(f"Related Metrics Summary: {len(embedded_precisions)} examples, avg embedded precision: {avg_embedded_precision:.3f}")

    logs += [
        {
            "name": "biohopr_metrics_related",
            "results":[{
                "example": i,
                "embedded_precision": embedded_precisions[i],
                "prompt": metrics_results.prompt[i],
                "completion": metrics_results.completion[i],
                "answers": metrics_results.info[i]['answer'],
            }
            for i in range(len(embedded_precisions))
            ]
        }
    ]

    metrics_results = test_metrics(ds,config,data_parser,related=False, num_examples=num_examples)
    embedded_precisions = metrics_results.metrics['embedded_precision']
    avg_embedded_precision = sum(embedded_precisions) / len(embedded_precisions)
    print(f"Unrelated Metrics Summary: {len(embedded_precisions)} examples, avg embedded precision: {avg_embedded_precision:.3f}")

    logs += [
        {
            "name": "biohopr_metrics_unrelated",
            "results":[{
                "example": i,
                "embedded_precision": embedded_precisions[i],
                "prompt": metrics_results.prompt[i],
                "completion": metrics_results.completion[i],
                "answers": metrics_results.info[i]['answer'],
            }
            for i in range(len(embedded_precisions))
            ]
        }
    ]

    # Save detailed logs to file
    os.makedirs('outputs', exist_ok=True)
    output_file = f'outputs/biohopr_detailed_logs_metrics_vs_judge.jsonl'
    with open(output_file, 'w') as f:
        for log_entry in logs:
            f.write(json.dumps(log_entry) + '\n')

def test_biolord_vs_bert(ds, config, data_parser, num_examples=32):
    """Test to compare BioLORD vs Clinical ModernBERT embeddings on BioHopR."""
    model_biolord = create_model('FremyCompany/BioLORD-2023').cuda()
    model_bert = create_model('Simonlee711/Clinical_ModernBERT').cuda()

    model_c_biolord = model_config(
        judge_model=config.judge_model,
        judge_api_key=None,
        judge_base_url=None,
        judge_client=None,
        judge_answer_num=5,
        embeddings_model=model_biolord,
        encoder=AsyncBufferedEncoder(model_biolord,ds_len = num_examples),
    )
    model_c_bert = model_config(
        judge_model=config.judge_model,
        judge_api_key=None,
        judge_base_url=None,
        judge_client=None,
        judge_answer_num=5,
        embeddings_model=model_bert,
        encoder=AsyncBufferedEncoder(model_bert,ds_len = num_examples),
        tau=0.97,
    )

    rubrics_biolord, rubrics_bert = _rubric_f(model_c_biolord,use_judge=False),_rubric_f(model_c_bert,use_judge=False)

    async def embedded_precision_biolord(parser: vf.Parser, completion: str, info) -> float:
        return await rubrics_biolord(parser, completion, info)
    async def embedded_precision_bert(parser: vf.Parser, completion: str, info) -> float:
        return await rubrics_bert(parser, completion, info)

    rubrics = [vf.Rubric( funcs=[embedded_precision_biolord, embedded_precision_bert], weights=[1.0,1.0], parser=data_parser)]
    env = vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=XML_SYSTEM_PROMPT,
        parser=data_parser,
        rubric=vf.RubricGroup(rubrics,parser=data_parser),
    )

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=60*60)
    results = env.evaluate(client, model='openai/gpt-oss-20b', num_examples=num_examples, max_concurrent=128)
    results = asyncio.run(results)

    embedded_precisions_biolord = results.metrics['embedded_precision_biolord']
    embedded_precisions_bert = results.metrics['embedded_precision_bert']

    avg_embedded_precision_biolord = sum(embedded_precisions_biolord) / len(embedded_precisions_biolord)
    avg_embedded_precision_bert = sum(embedded_precisions_bert) / len(embedded_precisions_bert)

    print(f"BioLORD Metrics Summary: {len(embedded_precisions_biolord)} examples, avg embedded precision: {avg_embedded_precision_biolord:.3f}")
    print(f"Clinical ModernBERT Metrics Summary: {len(embedded_precisions_bert)} examples, avg embedded precision: {avg_embedded_precision_bert:.3f}")
    #Save detailed logs to file
    os.makedirs('outputs', exist_ok=True)
    output_file = f'outputs/biohopr_detailed_logs_biolord_vs_bert.jsonl'
    with open(output_file, 'w') as f:
        log_entry = {
            "name": "biohopr_biolord_vs_bert",
            "results":[{
                "example": i,
                "embedded_precision_biolord": embedded_precisions_biolord[i],
                "embedded_precision_bert": embedded_precisions_bert[i],
                "prompt": results.prompt[i],
                "completion": results.completion[i],
                "answers": results.info[i]['answer'],
            }
            for i in range(len(embedded_precisions_biolord)) if embedded_precisions_biolord[i] != embedded_precisions_bert[i]
            ]
        }
        f.write(json.dumps(log_entry) + '\n')

def test_server_vs_local_embeddings(ds, config:RemoteModelConfig, data_parser:vf.Parser, num_examples=32):
    client = AsyncOpenAI(base_url="http://localhost:8001/v1/", api_key=config.api_key, timeout=60*60)

    # Query clinicalmodernbert model for embeddings
    response = asyncio.run(client.embeddings.create(
        model="clinicalmodernbert",
        input=["Test sentence for embeddings."],
    ))
    print("Embeddings response from clinicalmodernbert model:", len(response.data[0].embedding), "dimensions.")

    client_encoder = AsyncEmbeddingClient(
            client=client,
            model="clinicalmodernbert",
        )

    print("Client encoder len:", asyncio.run(client_encoder.encode(["Test"])).shape)

    model_bert = create_model('Simonlee711/Clinical_ModernBERT').cuda()
    local_encoder = AsyncBufferedEncoder(model_bert, batch_size=32)

    print(model_bert[0])

    # Local vs Server embedding test
    local_tensor = asyncio.run(local_encoder.encode(["Test sentence for embeddings."]))
    server_tensor = asyncio.run(client_encoder.encode(["Test sentence for embeddings."]))


    print("Local encoder embedding:", local_tensor)
    print("Server encoder embedding:", server_tensor)

    # if server_tensor in local_tensor:
    for o in local_tensor:
        if torch.allclose(o.cpu(), server_tensor.cpu(), atol=1e-4):
            print("Local and server embeddings match closely.")
            break

    exit()

    model_c_local = model_config(
        judge_model=config.judge_model,
        judge_api_key=None,
        judge_base_url=None,
        judge_client=None,
        judge_answer_num=5,
        embeddings_model=model_bert,
        encoder=local_encoder,
    )

    model_c_server = model_config(
        judge_model=config.judge_model,
        judge_api_key=None,
        judge_base_url=None,
        judge_client=None,
        judge_answer_num=5,
        embeddings_model=model_bert,
        encoder=client_encoder,
    )

    rubric_local, rubric_server = _rubric_f(model_c_local,use_judge=False),_rubric_f(model_c_server,use_judge=False)
    async def embedded_precision_local(parser: vf.Parser, completion: str, info) -> float:
        return await rubric_local(parser, completion, info)
    async def embedded_precision_server(parser: vf.Parser, completion: str, info) -> float:
        return await rubric_server(parser, completion, info)
    
    rubrics = [vf.Rubric( funcs=[embedded_precision_server, embedded_precision_local], weights=[1.0,1.0], parser=None)]
    env = vf.SingleTurnEnv(
        eval_dataset=ds,
        system_prompt=XML_SYSTEM_PROMPT,
        parser=data_parser,
        rubric=vf.RubricGroup(rubrics,parser=data_parser),
    )

    client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key, timeout=60*60)
    results = env.evaluate(client, model='openai/gpt-oss-20b', num_examples=num_examples, max_concurrent=128)
    results = asyncio.run(results)

    embedded_precisions_local = results.metrics['embedded_precision_local']
    embedded_precisions_server = results.metrics['embedded_precision_server']

    avg_embedded_precision_local = sum(embedded_precisions_local) / len(embedded_precisions_local)
    avg_embedded_precision_server = sum(embedded_precisions_server) / len(embedded_precisions_server)

    print(f"Average embedded precision (local): {avg_embedded_precision_local}")
    print(f"Average embedded precision (server): {avg_embedded_precision_server}")

    os.makedirs('outputs', exist_ok=True)
    output_file = f'outputs/biohopr_detailed_logs_server_vs_local_embeddings.jsonl'
    with open(output_file, 'w') as f:
        log_entry = {
            "name": "biohopr_server_vs_local_embeddings",
            "results":[{
                "example": i,
                "embedded_precision_local": embedded_precisions_local[i],
                "embedded_precision_server": embedded_precisions_server[i],
                "prompt": results.prompt[i],
                "completion": results.completion[i],
                "answers": results.info[i]['answer'],
            }
            for i in range(len(embedded_precisions_local)) if embedded_precisions_local[i] != embedded_precisions_server[i]
            ]
        }
        f.write(json.dumps(log_entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Test BioHopR Judge vs Metrics')
    parser.add_argument('--api_key', type=str, default=None, help='OpenAI API key')
    parser.add_argument('--base_url', type=str, default=None, help='Base URL for custom OpenAI-compatible API endpoint')
    parser.add_argument('--judge_model', type=str, default='openai/gpt-oss-20b', help='Model name to use for judging')
    parser.add_argument('--num_examples', type=int, default=32, help='Number of examples to use for evaluation')
    parser.add_argument('--completion_model', type=str, default='openai/gpt-oss-20b', help='Model name to use for generating completions')

    args, _ = parser.parse_known_args()
    api_key = args.api_key if args.api_key else os.getenv("OPENAI_API_KEY")

    config = RemoteModelConfig(
        api_key=api_key,
        base_url=args.base_url,
        judge_model=args.judge_model,
        completion_model=args.completion_model,
    )

    # Test Data from every 1k examples in BioHopR
    ds = load_dataset("knowlab-research/BioHopR", split="train")
    ds = ds.map(partial(_biohoper_format,tasks=['biohopr_hop2']), remove_columns=ds.column_names, batched=True)

    answer_format = AnswerFormat.XML
    answer_format,system_prompt,data_parser = _prepare_parseing(answer_format,None)

    #test_metrics_vs_judge(ds,config,data_parser, num_examples=args.num_examples)

    #test_biolord_vs_bert(ds, config, data_parser, num_examples=args.num_examples)

    test_server_vs_local_embeddings(ds, config, data_parser, num_examples=args.num_examples)

if __name__ == "__main__":
    main()