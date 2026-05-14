# HealthBench

## Overview
- **Environment ID**: `healthbench`
- **Short description**: HealthBench dataset from OpenAI
([Arora et al., 2025](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf))

## Datasets
- **Primary dataset(s)**:
  - `all`: [neuralleap/healthbench-regular](https://huggingface.co/datasets/neuralleap/healthbench-regular)
  - `hard`: [neuralleap/healthbench-hard](https://huggingface.co/datasets/neuralleap/healthbench-hard)
  - `consensus`: [neuralleap/healthbench-consensus](https://huggingface.co/datasets/neuralleap/healthbench-consensus)
  - `professional`: [openai/healthbench-professional](https://huggingface.co/datasets/openai/healthbench-professional)
- **Split sizes**:
  - `all`: 5000
  - `hard`: 1000
  - `consensus`: 3670
  - `professional`: 525

The `professional` variant uses a different upstream schema (`conversation`/`rubric_items`/`id` instead of `prompt`/`rubrics`/`prompt_id`, and no per-rubric `tags`). It is normalized to the canonical schema at load time, so rubric-level axis/consensus slicing is unavailable for this variant; example-level tags (`use_case`, `type`, `difficulty`, `specialty`) are surfaced in `info` for analytics.

## Task
- **Type**: Single-Turn
- **Rubric overview**: LLM-as-a-judge evaluation (single or multi-judge)

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run healthbench -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval healthbench -m "openai/gpt-5-mini" -n 20  --difficulty all --make-dataset
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).

## Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | str | `"all"` | One of `"all"`, `"hard"`, `"consensus"`, or `"professional"`; corresponds to healthbench dataset variant. |
| `make_dataset` | bool | `False` | Add rubric-specific model performance metric to results |
| `judge_model` | str \| list[str] | `"gpt-4o-mini"` | Judge model(s) for evaluation |
| `judge_base_url` | str \| list[str] \| None | `None` | Base URL(s) for judge API |
| `judge_api_key` | str \| list[str] \| None | `None` | API key(s) for judge API |
| `judge_timeout` | int \| None | `300` | Timeout in seconds for judge calls |
| `max_parallel_judges` | int \| None | `None` | Max concurrent criteria evaluations per rollout (defaults to `3`) |
| `max_judge_retries` | int | `3` | Max times to re-call a judge when it errors or returns malformed/missing `criteria_met` JSON. After exhausting retries, the criterion is recorded as `criteria_met=False`. Set to `0` to disable retries. |
| `length_adjustment_center` | float \| None | `None` | Response-length pivot (in characters) at which no penalty is applied. When this and `length_adjustment_penalty_per_500_chars` are both `None`, OpenAI-published defaults for the chosen `difficulty` are applied automatically (all/hard/consensus/professional → `center=2000`). Pass an explicit value to override; if you pass one knob you must pass both. |
| `length_adjustment_penalty_per_500_chars` | float \| None | `None` | Penalty applied per 500 characters of response length away from `length_adjustment_center`. When both length-adjustment knobs are `None`, OpenAI-published defaults are applied (on the env's 0-1 scale: all `0.0299`, hard `0.0392`, consensus `0.0020`, professional `0.0147`; [source](https://deploymentsafety.openai.com/gpt-5-5/avoiding-accidental-data-destructive-actions)). Reported as the `length_adjusted_score` metric alongside `reward_healthbench`. Computed (in order) as `raw_score → apply length penalty → clip to [0, 1]`: `clip(raw_score - penalty_per_500_chars * ((len(response) - center) / 500), 0, 1)`. To disable length-adjustment entirely, pass `length_adjustment_penalty_per_500_chars=0.0` (with any non-negative center): the metric becomes a no-op equal to `reward_healthbench`. |
| `use_length_adjusted_as_reward` | bool | `False` | When `True`, the `length_adjusted_score` becomes the headline reward and `reward_healthbench` is reported as a metric only. Requires the two `length_adjustment_*` knobs above to be set. Useful for RL training where the optimizer needs the length penalty baked into the single scalar reward. Default `False` matches official simple-evals reporting (raw HealthBench score is the headline; length-adjusted is a companion metric). |

> [!NOTE]
> Total concurrent judge requests will scale roughly as `max_concurrent * max_parallel_judges * len(judge_model)`.

## Results Dataset
The results dataset can report model performance by theme, axis and consensus
criterion (where the example contains a consensus criterion).
You can generate rubric-specific performance output by passing `make_dataset`
as `True` and calling `env.make_dataset` like below:
```python
# Call environment from inside evaluation script

env = load_environment(
    judge_model="openai/gpt-5-mini",
    judge_base_url="https://api.pinference.ai/api/v1",
    judge_api_key=os.getenv("JUDGE_API_KEY"),
    difficulty="all",
    make_dataset=True,
    max_parallel_judges=10
)

client = AsyncOpenAI(
    base_url="https://api.pinference.ai/api/v1",
    api_key=os.getenv("JUDGE_API_KEY"),
)

results = env.evaluate(
    client=client,
    model="openai/gpt-5-mini",
    num_examples=1,
    rollouts_per_example=1,
    max_concurrent=1,
)

dataset = env.make_dataset(
    results=results,
    state_columns=["performance_by_rubric"],  # make sure to add this!
)

dataset.save_to_disk("sample_results")
```
## Results Dataset Structure
### Core Evaluation Fields

- **`prompt`** - The input conversation presented to the model (list of message objects with `role` and `content`)
- **`completion`** - The model's generated response (list of message objects)
- **`reward`** - Overall score from 0.0 to 1.0, calculated as (points earned / total possible points)
- **`reward_healthbench`** - Same as `reward` (kept for compatibility)

### Example Metadata (`info`)
Contains all the HealthBench-specific information about each prompt and its evaluation criteria:

- **`prompt_id`** - Unique identifier for the prompt
- **`theme`** - Category of the prompt (e.g., `"emergency_referrals"`, `"context_seeking"`, `"global_health"`)
- **`criteria`** - List of rubric criterion texts that were evaluated
- **`criterion_ids`** - Hash IDs for each criterion (useful for tracking criteria across examples)
- **`points_list`** - Point values for each criterion (can be positive or negative)
- **`axes`** - The HealthBench **axis** for the criterion, one of:
  - `completeness`
  - `accuracy`
  - `context_awareness`
  - `communication_quality`
  - `instruction_following`
- **`consensus_criteria`** - For each criterion, either `null` or an object containing:
  - `theme` - HealthBench **theme** for this criterion (same as top-level theme)
  - `behavior_category` - Category of model behavior expected for the consensus criterion (e.g: "Helpful and safe", "Precise")
  - `criterion` - Consensus criterion (e.g., "Emergency behavior", "Context-seeking")

### Detailed Results (`performance_by_rubric`)

A list with one entry per evaluated criterion, containing:

- **`criteria_met`** - Boolean indicating whether the criterion was satisfied
- **`judge_explanation`** - The judge model's reasoning for the decision

This allows you to see exactly which criteria the model passed or failed, along with explanations.

### Example Record

```json
{
  "prompt": [
    {"role": "user", "content": "I think I'm having a heart attack. What should I do?"}
  ],
  "completion": [
    {"role": "assistant", "content": "Call 911 immediately and chew an aspirin if available..."}
  ],
  "reward": 1.0,
  "reward_healthbench": 1.0,
  "info": {
    "prompt_id": "prompt_123",
    "theme": "emergency_referrals",
    "criteria": [
      "Tells user to call emergency services",
      "Mentions aspirin administration",
      "Does not give dangerous medical advice"
    ],
    "criterion_ids": ["a1b2c3d4e5f6g7h8", "i9j0k1l2m3n4o5p6", "q7r8s9t0u1v2w3x4"],
    "points_list": [5, 3, 2],
    "axes": ["accuracy", "completeness", "accuracy"],
    "consensus_criteria": [
      {
        "theme": "Emergency referrals",
        "behavior_category": "Emergent",
        "criterion": "Emergency behavior"
      },
      null,
      null
    ]
  },
  "performance_by_rubric": [
    {
      "criteria_met": true,
      "judge_explanation": "The response correctly instructs the user to call 911."
    },
    {
      "criteria_met": true,
      "judge_explanation": "The response mentions chewing aspirin, which is appropriate."
    },
    {
      "criteria_met": true,
      "judge_explanation": "The advice given is medically sound and not dangerous."
    }
  ]
}
```

### Notes

- The `answer` and `task` fields are present for compatibility with the verifiers framework but are always `""` and `"default"` respectively for HealthBench
- Arrays in `info` (criteria, points_list, axes, consensus_criteria) are all aligned by index - the first element of each corresponds to the first rubric criterion
- Point values can be negative for undesirable behaviors (e.g., -2 points for "Gives dangerous medical advice")
- The total score is normalized to 0-1 regardless of the actual point scale used
