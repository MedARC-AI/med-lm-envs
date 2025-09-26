import argparse
import os
import json
from pubmedqa import load_environment
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
import random

class MockClient:
    def __init__(self):
        self.chat = MockChat()

class MockChat:
    def __init__(self):
        self.completions = MockCompletions()


class MockCompletions:
    def __init__(self):
        self.contents = [
            'ANSWER: A (using {model})',
            'ANSWER: AF',
            'ANSWER: yes',
            r'<think> dadada </think> my answer is \boxed{A}',
            r'<think> dadada </think> my answer is \boxed{yes}'
        ]
        self.index = 0;


    async def create(self, **kwargs):
        model = kwargs.get('model', 'unknown')

        content = self.contents[self.index % len(self.contents)]
        self.index += 1

        message = ChatCompletionMessage(
            role='assistant',
            content=content
        )
        choice = Choice(
            index=0,
            message=message,
            finish_reason='stop'
        )
        return ChatCompletion(
            id='mock-completion',
            object='chat.completion',
            created=1234567890,
            model=model,
            choices=[choice]
        )

'''
def submit_results(env_name, model_name, results):
    """callable via: #submit_results("pubmedqa", model_name, results)"""
    from prime_intellect import PrimeIntellectClient

    # Initialize the client with your API key
    client = PrimeIntellectClient(api_key="your_api_key")    

    rewards = results.reward
    average_reward = sum(rewards) / len(rewards)
    success_rate = sum(1 for r in rewards if r == 1.0) / len(rewards)

    # Define your evaluation results
    results = {
        "num_examples": len(rewards),
        "average_reward": average_reward,
        "success_rate": success_rate,
    }

    # Submit the results
    client.submit_evaluation_results(env_name=env_name, model_name=model_name, results=results)
'''


def main():
    parser = argparse.ArgumentParser(description='Run PubMedQA evaluation')
    parser.add_argument('--no-mock', action='store_true', help='Use real Mistral API instead of mock client')
    parser.add_argument('--think', action='store_true', help='Use thinking')
    parser.add_argument('--model', default='mistral-medium', help='Model name to use')
    parser.add_argument('--num_examples', type=int, default=5, help='Number of examples to evaluate (-1 for full benchmark)')
    
    args, _ = parser.parse_known_args()

    if args.no_mock:
        # Use real Mistral client
        from openai import AsyncOpenAI
        
        try:
            import api_tokens  # Load API keys
        except ImportError:
            pass  # API keys not available, will use environment variables

        client = AsyncOpenAI(
            api_key=os.getenv("MISTRAL_API_KEY"),
            base_url="https://api.mistral.ai/v1"
        )
        print(f"Using real Mistral API with model: {args.model}")
    else:
        # Use mock client
        client = MockClient()
        print(f"Using mock client with model: {args.model}")
    
    # conduct evaluation
    env = load_environment(use_think=args.think)
    #print(len(env.eval_dataset), env.eval_dataset[0])
    
    # Handle full benchmark case
    results = env.evaluate(client, model=args.model, num_examples=args.num_examples)
    
    # Create outputs directory and file
    os.makedirs('outputs', exist_ok=True)
    if args.num_examples == -1:
        output_file = f'outputs/{args.model}_full.jsonl'
    else:
        output_file = f'outputs/{args.model}_nsamples_{args.num_examples}.jsonl'
    
    # Write results to JSONL file
    with open(output_file, 'w') as f:
        #print(dir(results))
        for i in range(len(results.answer)):
            result_item = {}
            result_item["question"] = env.eval_dataset[i]['question']
            result_item["ground_truth"] = results.answer[i]
            result_item["long_answer"] = env.eval_dataset[i]['long_answer']
            result_item["reward"] = results.reward[i]
            result_item["completion"] = results.completion[i][0]['content']
            result_item["full_prompt"] = env.eval_dataset[i]['prompt']
            f.write(json.dumps(result_item) + '\n')
    
    # Calculate and display summary stats
    rewards = results.reward
    avg_reward = sum(rewards) / len(rewards)
    success_rate = sum(1 for r in rewards if r == 1.0) / len(rewards)
    
    print(f"Results saved to: {output_file}")
    print(f"Summary: {len(rewards)} examples, avg reward: {avg_reward:.3f}, success rate: {success_rate:.3f}")

if __name__ == "__main__":
    main()
