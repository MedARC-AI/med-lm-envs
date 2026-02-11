#!/usr/bin/env python3
"""
Evaluate trained RL models on test sets using the verifiers library.

Usage:
    # Start vLLM server first (in another terminal):
    cd ~/med-lm-envs/prime-rl
    CUDA_VISIBLE_DEVICES=0,1 uv run vllm serve <model_path> --tensor-parallel-size 2 --port 8000
    
    # Then run evaluation:
    cd ~/med-lm-envs
    uv run python scripts/evaluate_models.py --api-base http://localhost:8000/v1 --dataset med_mcqa
    
    # For MedCaseReasoning (needs judge):
    export OPENAI_API_KEY=<your-key>
    uv run python scripts/evaluate_models.py --api-base http://localhost:8000/v1 --dataset medcasereasoning
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add environments to path
sys.path.insert(0, "/admin/home/nikhil/med-lm-envs/environments/medcalc_bench")
sys.path.insert(0, "/admin/home/nikhil/med-lm-envs/environments/med_mcqa")
sys.path.insert(0, "/admin/home/nikhil/med-lm-envs/environments/medcasereasoning")

import verifiers as vf
from openai import AsyncOpenAI

# Model paths
MODELS = {
    "base": "Qwen/Qwen3-4B-Instruct-2507",
    "medcalc_trained": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_verified_short/weights/step_300",
    "medmcqa_trained": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_medmcqa/weights/step_300",
    "medcase_trained": "/admin/home/nikhil/med-lm-envs/prime-rl/outputs_medcasereasoning/weights/step_300",
}

OUTPUT_DIR = Path("/admin/home/nikhil/med-lm-envs/eval_results")


def load_medcalc_env():
    """Load MedCalc-Bench-Verified environment."""
    from medcalc_bench.medcalc_bench import load_environment
    
    env = load_environment(
        use_think=True,
        use_verified_dataset=True,
        answer_format="xml",
    )
    return env


def load_medmcqa_env():
    """Load MedMCQA environment."""
    from med_mcqa import load_environment
    
    env = load_environment(
        use_think=True,
        answer_format="xml",
    )
    return env


def load_medcase_env(judge_api_key: str = None):
    """Load MedCaseReasoning environment."""
    import sys
    sys.path.insert(0, "/admin/home/nikhil/med-lm-envs/environments/medcasereasoning")
    from medcasereasoning import load_environment
    
    env = load_environment(
        judge_model="gpt-5-nano",
        reasoning_effort="low",
        judge_api_key=judge_api_key,
    )
    return env


async def evaluate_on_env(
    env: vf.SingleTurnEnv,
    client: AsyncOpenAI,
    model_name: str,
    n_samples: int = 100,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_concurrent: int = 10,
) -> dict:
    """
    Evaluate a model on a verifiers environment.
    
    Uses the environment's built-in evaluate() method which handles:
    - Prompt formatting with system prompt
    - Response parsing
    - Reward calculation via rubric
    """
    # Configure the environment
    env.set_kwargs(
        client=client,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    
    # Get eval dataset
    eval_dataset = env.get_eval_dataset()
    if eval_dataset is None:
        raise ValueError("Environment has no eval dataset")
    
    # Limit samples if needed
    if n_samples > 0 and n_samples < len(eval_dataset):
        eval_dataset = eval_dataset.select(range(n_samples))
    
    print(f"  Evaluating on {len(eval_dataset)} samples...")
    
    # Run evaluation using the environment's evaluate method
    results = await env.evaluate(
        eval_dataset,
        max_concurrent=max_concurrent,
        rollouts_per_example=1,
    )
    
    # Calculate accuracy from results
    rewards = [r.reward for r in results if r.reward is not None]
    accuracy = sum(1 for r in rewards if r > 0.5) / len(rewards) if rewards else 0
    mean_reward = sum(rewards) / len(rewards) if rewards else 0
    
    return {
        "n_samples": len(eval_dataset),
        "n_completed": len(rewards),
        "accuracy": accuracy,
        "mean_reward": mean_reward,
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models using verifiers")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                       help="API base URL for vLLM server")
    parser.add_argument("--api-key", type=str, default="EMPTY",
                       help="API key (use EMPTY for local vLLM)")
    parser.add_argument("--model-name", type=str, default="model",
                       help="Model name to use in API calls")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["medcalc_bench", "med_mcqa", "medcasereasoning", "all"],
                       help="Dataset to evaluate on")
    parser.add_argument("--n-samples", type=int, default=100,
                       help="Number of samples to evaluate")
    parser.add_argument("--max-concurrent", type=int, default=10,
                       help="Maximum concurrent API requests")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens to generate")
    parser.add_argument("--judge-api-key", type=str, default=None,
                       help="OpenAI API key for MedCaseReasoning judge")
    parser.add_argument("--output-name", type=str, default=None,
                       help="Name for output file (default: dataset name)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create client
    client = AsyncOpenAI(base_url=args.api_base, api_key=args.api_key)
    
    # Define dataset loaders
    loaders = {
        "medcalc_bench": load_medcalc_env,
        "med_mcqa": load_medmcqa_env,
        "medcasereasoning": lambda: load_medcase_env(args.judge_api_key),
    }
    
    if args.dataset == "all":
        datasets_to_eval = list(loaders.keys())
    else:
        datasets_to_eval = [args.dataset]
    
    results = {}
    
    for dataset_name in datasets_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'='*60}")
        
        try:
            env = loaders[dataset_name]()
            result = await evaluate_on_env(
                env=env,
                client=client,
                model_name=args.model_name,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_concurrent=args.max_concurrent,
            )
            results[dataset_name] = result
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  Mean Reward: {result['mean_reward']:.3f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = {"error": str(e)}
    
    # Save results
    output_name = args.output_name or args.dataset
    output_file = OUTPUT_DIR / f"{output_name}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for dataset, result in results.items():
        if "error" in result:
            print(f"  {dataset}: ERROR - {result['error']}")
        else:
            print(f"  {dataset}: {result['accuracy']:.1%} accuracy ({result['n_completed']}/{result['n_samples']} samples)")


if __name__ == "__main__":
    asyncio.run(main())
