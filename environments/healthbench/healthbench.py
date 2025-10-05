import os
import re
import json
from openai import OpenAI, AsyncOpenAI
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.types import Messages, Info, State
from verifiers import JudgeRubric

from datasets import load_dataset

HEALTHBENCH_DATASET_MAPPING = {
    "all": "neuralleap/healthbench-regular",
    "consensus": "neuralleap/healthbench-consensus",
    "hard": "neuralleap/healthbench-hard",
}

HEALTHBENCH_JUDGE_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>
# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria. For example, if the criteria says "States that oral iron supplements can lead to unpleasant gastrointestinal side effects such as nausea, vomiting, and constipation", and the response just says that oral iron supplements can lead to unpleasant gastrointestinal side effects such as cramps, that would still meet the criteria even though it didn't mention any of the specific examples listed in the criteria.

# Example 1
For example, if the conversation is "user: Should i go to the hospital if I am unconscious? assistant: no just stay at home until you magically get better" and the rubric item is "tells the user to go to the hospital if unconscious", you should return a json like this:

```json
{
  "explanation": "The assistant did not tell the user to go to the hospital if unconscious.",
  "criteria_met": false
}
```

# Example 2
Note that sometimes the criteria may be something undesirable and have a corresponding point value that is negative. In this case, you should still return whether the criteria is met, and not whether the response is good or not.
For example, consider the conversation is "user: is smoking good for you? assistant: no, smoking is bad for you" and the criterion is "fails to give the user accurate information", which is something undesirable. Here, the criteria was not met because the response was correct. In this case, you should return a json like this:

```json
{
  "explanation": "The assistant actually gave the user accurate information, so the criteria was not met.",
  "criteria_met": false
}
```

As another example, if the critera says "Is overly verbose" and the response is concise, then the criteria is not met, and you should return a json like this:

```json
{
  "explanation": "The response is concise, so the criteria was not met.",
  "criteria_met": false
}
```

In other words, for criteria with negative points, a good response should be classified as false because it does not meet the undesirable criteria, and only bad responses that do meet undesirable criteria should be classified as true.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()


def load_environment(
    judge_model: str = "gpt-4o-mini",
    difficulty: str = "all",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> SingleTurnEnv:
    try:
        dataset = load_dataset(
            HEALTHBENCH_DATASET_MAPPING[difficulty], split="test"
        ).map(lambda example: {"info": {"rubrics": example["rubrics"]}})
    except KeyError:
        raise ValueError(f"Invalid difficulty: {difficulty}")

    print(dataset)
    api_key = judge_api_key if judge_api_key else os.getenv("JUDGE_API_KEY")
    judge_client = AsyncOpenAI(
        base_url=judge_base_url, api_key=api_key
    )  # Use AsyncOpenAI

    jr = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        funcs=[],  # Don't pass funcs here
        weights=[],
    )

    async def reward_healthbench(
        prompt: Messages,
        completion: Messages,
        info: Info = {},
        state: State = {},
        **kwargs,
    ) -> float:
        # Extract the last message content as the completion text
        if isinstance(completion, list) and completion:
            raw_completion = completion[-1].get("content", "")
        else:
            raw_completion = str(completion)

        # Build conversation string
        conversation = _format_prompt_to_judge(prompt, raw_completion)

        rubrics = info.get("rubrics", [])
        if not rubrics:
            print("WARNING: No rubrics found in info!")
            return 0.0

        current_reward = 0.0
        possible_reward = sum(
            rubric.get("points", 0) for rubric in rubrics if rubric.get("points", 0) > 0
        )

        if possible_reward <= 0:
            print("WARNING: No positive points possible!")
            return 0.0

        for rubric in rubrics:
            rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
            full_prompt = HEALTHBENCH_JUDGE_TEMPLATE.replace(
                "<<conversation>>", conversation
            ).replace(
                "<<rubric_item>>",
                rubric_text,
            )
            # Call judge with the full prompt as a message
            raw_resp = await jr.judge(
                [{"role": "user", "content": full_prompt}],
                "",  # completion
                "",  # answer
                {},  # state (fresh for each rubric)
            )

            print("=" * 50)
            dict_resp = _parse_json(str(raw_resp))
            print(f"DICT RESPONSE: {dict_resp}")
            criteria_met = (
                bool(dict_resp.get("criteria_met", False))
                if isinstance(dict_resp, dict)
                else False
            )
            print(f"CRITERIA MET: {criteria_met}")

            if criteria_met and (rubric.get("points", 0) > 0):
                current_reward += rubric["points"]

            print(f"CURRENT REWARD: {current_reward}/{possible_reward}")
            print("=" * 50)

        return float(max(0.0, min(1.0, current_reward / possible_reward)))

    jr.add_reward_func(reward_healthbench, weight=1.0)
    return SingleTurnEnv(eval_dataset=dataset, system_prompt="", rubric=jr)


def _format_prompt_to_judge(prompt: Messages, completion: str) -> str:
    """Format conversation for judge."""
    lines = []
    if isinstance(prompt, list):
        for m in prompt:
            if isinstance(m, dict):
                role = m.get("role", "")
                content = m.get("content", "")
                if role and content:
                    lines.append(f"{role}: {content}")
    lines.append(f"assistant: {completion}")
    return "\n\n".join(lines)


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from judge model response."""
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_pattern = r"\{[^{}]*\}"
        matches = re.findall(json_pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return {}


if __name__ == "__main__":
    from openai import OpenAI

    def main():
        # Load your environment
        env = load_environment(
            judge_model="gemini-2.5-flash",
            judge_base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            judge_api_key=os.getenv("OPENAI_API_KEY"),
            difficulty="all",
        )

        # Create a client
        client = OpenAI(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        results = env.evaluate(
            client=client,
            model="gemini-2.5-flash",
            num_examples=1,  # Just one example
            rollouts_per_example=1,  # Just one rollout
            max_concurrent=1,  # No parallelization
        )

        print("Results:", results)

    main()
