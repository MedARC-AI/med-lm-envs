"""Validation script for MedHALT environment"""

import argparse
from openai import AsyncOpenAI
import verifiers as vf

async def main():
    parser = argparse.ArgumentParser(description='Validate MedHALT environment')
    parser.add_argument('--config', default='reasoning_FCT', choices=['reasoning_FCT', 'reasoning_nota'])
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to test')
    parser.add_argument('--model', default='qwen2.5:3b', help='Model to use')
    
    args = parser.parse_args()
    
    print(f"Testing MedHALT environment: {args.config}")
    
    # Load environment
    env = vf.load_environment('medhalt', config_name=args.config, num_examples=args.num_examples)
    print(f"✓ Loaded {len(env.dataset)} examples")
    
    # Test with Ollama
    client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    results = await env.evaluate(client, args.model, num_examples=args.num_examples)
    
    # Calculate metrics
    correct = sum(results.reward)
    total = len(results.reward)
    accuracy = (correct / total * 100)
    
    print(f"\nResults:")
    print(f"  Total:    {total}")
    print(f"  Correct:  {int(correct)}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"\n✅ Validation complete")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())