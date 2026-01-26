import asyncio
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers import JudgeRubric
from verifiers.envs.singleturn_env import SingleTurnEnv
from verifiers.types import Info, Messages, State

HEALTHBENCH_DATASET_MAPPING = {
    "all": "neuralleap/healthbench-regular",
    "consensus": "neuralleap/healthbench-consensus",
    "hard": "neuralleap/healthbench-hard",
}

# See pgs 33-36 (Appendix I) of the HealthBench paper for a complete listing
# of all consensus criteria organized by themes and outlined with separate
# consensus categories
# https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf

with open(Path(__file__).resolve().parent / "hb_consensus_criteria.json", "r") as fp:
    HEALTHBENCH_CONSENSUS_CRITERIA_LOOKUP = json.load(fp)

disable_progress_bar()  # suppress datasets progress indicators


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
    make_dataset: bool = False,
    max_parallel_judges: int = 5,
    **kwargs,
) -> SingleTurnEnv:
    try:
        dataset = load_dataset(
            HEALTHBENCH_DATASET_MAPPING[difficulty], split="test" if difficulty == "all" else "train"
        ).map(lambda example: {"info": _process_healthbench_dataset(example)})
    except KeyError:
        raise ValueError(f"Invalid difficulty: {difficulty}")

    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)

    jr = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    async def reward_healthbench(prompt: Messages, completion: Messages, info: Info, state: State) -> float:
        """
        Embedded reward function that asynchronously calls `judge` for every
        criterion for this rollout.
        NB: `make_dataset` and `max_parallel_judges` taken from outer scope
        `load_environment` function.
        """
        # Extract the last message content as the completion text
        if isinstance(completion, list) and completion:
            raw_completion = completion[-1].get("content", "")
        else:
            raw_completion = str(completion)

        # Build conversation string
        conversation = _format_prompt_to_judge(prompt, raw_completion)

        criteria = info.get("criteria", [])
        points_list = info.get("points_list", [])
        if not points_list:
            return 0.0

        total_reward = sum([pt for pt in points_list if pt > 0])
        current_reward = 0.0

        # Limit concurrent judge calls PER rollout using a shared semaphore
        semaphore = asyncio.Semaphore(max_parallel_judges)

        tasks = [
            _judge_single_criterion(
                idx=idx,
                criterion=criterion,
                points_possible=points_possible,
                conversation=conversation,
                judge_rubric=jr,
                semaphore=semaphore,
                state=state,
            )
            for idx, (criterion, points_possible) in enumerate(zip(criteria, points_list))
        ]

        judgments = await asyncio.gather(*tasks)
        current_reward += sum(judgment["points_possible"] if judgment["criteria_met"] else 0 for judgment in judgments)

        ## Update state to record performance by rubric
        if make_dataset:
            judgments_sorted = sorted(judgments, key=lambda x: x["idx"])
            for judg in judgments_sorted:
                judg.pop("idx", None)  # metadata do not report
                judg.pop("points_possible", None)  # already contained in `info`

            if state.get("performance_by_rubric", None) is None:
                state["performance_by_rubric"] = []

            state["performance_by_rubric"].append(judgments_sorted)

        return float(max(0.0, min(1.0, current_reward / total_reward)))

    jr.add_reward_func(reward_healthbench, weight=1.0)
    return SingleTurnEnv(eval_dataset=dataset, system_prompt="", rubric=jr)


async def _judge_single_criterion(
    idx: int,
    criterion: str,
    points_possible: int,
    conversation: str,
    judge_rubric: JudgeRubric,
    semaphore: asyncio.Semaphore,
    state: dict,
) -> dict[str, str | int | bool]:
    # Use the shared semaphore to bound concurrency across criteria for this rollout
    async with semaphore:
        rubric_text = f"[{points_possible}] {criterion}"
        full_prompt = HEALTHBENCH_JUDGE_TEMPLATE.replace("<<conversation>>", conversation).replace("<<rubric_item>>", rubric_text)  # fmt: skip
        raw_resp = await judge_rubric.judge(
            [{"role": "user", "content": full_prompt}],
            "",  # completion
            "",  # answer
            state,  # pass real state for token tracking
        )

        dict_resp = _parse_json(str(raw_resp))
        criteria_met = bool(dict_resp.get("criteria_met", False)) if isinstance(dict_resp, dict) else False

        return {
            "idx": idx,
            "points_possible": points_possible,
            "criteria_met": criteria_met,
            "judge_explanation": dict_resp.get("explanation", None),
        }


def _process_healthbench_dataset(example: dict) -> dict:
    """
    Massaging the Healthbench dataset to make it more amenable for analytics
    by theme and axis. Dataset is structured as follows (one example below):
    {
        example_tags: [
            "theme:some-theme",
            "physician_agreed_category:"some-consensus-criterion" (not always present)
        ],
        ideal_completions_data, prompt, prompt_id: self-explanatory,
        rubrics: [
            {
                criterion: "some criterion text"
                points: int,
                tags: [
                    "level:example" OR "level:cluster" if this is one of 34 consensus criteria

                    axis: one of the 5 specified axes (completeness, accuracy,
                    context awareness, communication quality, instruction following)

                    IF the criterion is a consensus criterion, then also will
                    contain the below item:

                    cluster:<theme repeated again>_<consensus criterion>_<behavior category>
                ]
            },
            ... more criteria ...
        ]
    }
    Ideally we would like it so that the `info` column for each rollout would be:
    info: {
        prompt_id: extracted from hb dataset
        theme: extracted from hb dataset
        criterion_ids: [<hash of criterion 1 text>, <hash of criterion 2 text>, ...]
        criteria: [<criterion 1 text>, <criterion 2 text>, ...]
        axes: [<axis of criterion 1>, ...]
        consensus_criteria: [
            null if not a consensus criterion

            If consensus criterion, then:
            {
                criterion: <ex: "emergent">,
                behavior_category: <ex: "emergency behavior">
            }
        ]
        points_list: [<list of ints>]
    }
    """

    def _gen_hash(criterion_text: str) -> str:
        data_bytes = criterion_text.encode("utf-8")
        hash_object = hashlib.blake2b(data_bytes, digest_size=8)
        return hash_object.hexdigest()

    prompt_id = example["prompt_id"]
    theme = [e for e in example["example_tags"] if e.startswith("theme")][0].split(":")[1]
    rubrics = example["rubrics"]
    info_data = defaultdict(list)
    for rubric in rubrics:
        info_data["criterion_ids"].append(_gen_hash(rubric["criterion"]))
        info_data["points_list"].append(rubric["points"])
        info_data["criteria"].append(rubric["criterion"])

        tags = {}
        for t in rubric.get("tags", []):
            try:
                key, value = t.split(":", 1)
                tags[key] = value
            except ValueError:
                continue

        info_data["axes"].append(tags["axis"])

        cluster_tag = tags.get("cluster")
        if cluster_tag:
            consensus_criterion = HEALTHBENCH_CONSENSUS_CRITERIA_LOOKUP[cluster_tag]
        else:
            consensus_criterion = None

        info_data["consensus_criteria"].append(consensus_criterion)

    final_info = dict(info_data)
    final_info["prompt_id"] = prompt_id
    final_info["theme"] = theme
    return final_info


# Function code directly copied from openai/simple-evals/healthbench_eval.py
# Credit to Rahul Arora; MIT licensed
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


# Function code directly copied from groq/openbench/utils/text.py:parse_json_from_response
# Credit to Aarush Sah; MIT licensed
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
