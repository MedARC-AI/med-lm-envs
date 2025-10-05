# HealthBench

### Overview
- **Environment ID**: `healthbench`
- **Short description**: HealthBench dataset from OpenAI
([Arora et al., 2025](https://cdn.openai.com/pdf/bd7a39d5-9e9f-47b3-903c-8b847ca650c7/healthbench_paper.pdf))

### Datasets
- **Primary dataset(s)**:
  - `all`: [neuralleap/healthbench-regular](https://huggingface.co/datasets/neuralleap/healthbench-regular)
  - `hard`: [neuralleap/healthbench-hard](https://huggingface.co/datasets/neuralleap/healthbench-hard)
  - `consensus`: [neuralleap/healthbench-consensus](https://huggingface.co/datasets/neuralleap/healthbench-consensus)
- **Split sizes**:
  - `all`: 5000
  - `hard`: 1000,
  - `consensus`: 3670

### Task
- **Type**: Single-Turn
- **Rubric overview**: JudgeRubric

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval healthbench
```

Configure model and sampling:

```bash
uv run vf-eval healthbench -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -a '{"difficulty": "all", "make_dataset", "true"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `difficulty` | str | `"all"` | One of 'all', 'hard', or 'consensus'; corresponds to healthbench dataset variant|
| `make_dataset` | bool | `False` | Add rubric-specific model performance metric to results |

### Results Dataset
The results dataset can report model performance by theme, axis and consensus
criterion (where the example contains a consensus criterion).
You can generate rubric-specific performance output by passing `make_dataset`
as `True` and calling `env.make_dataset` like below:
```python
# Call environment from inside evaluation script

env = load_environment(
    judge_model="gpt-4.1-mini",
    judge_base_url="https://api.openai.com/v1/chat/completions",
    judge_api_key=os.getenv("OPENAI_API_KEY"),
    difficulty="all",
    make_dataset=True,
)

client = AsyncOpenAI(
    base_url="https://api.openai.com/v1/chat/completions",
    api_key=os.getenv("OPENAI_API_KEY"),
)

results = env.evaluate(
    client=client,
    model="gpt-4.1-mini",
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
### Results Dataset Structure
#### Core Evaluation Fields

- **`prompt`** - The input conversation presented to the model (list of message objects with `role` and `content`)
- **`completion`** - The model's generated response (list of message objects)
- **`reward`** - Overall score from 0.0 to 1.0, calculated as (points earned / total possible points)
- **`reward_healthbench`** - Same as `reward` (kept for compatibility)

#### Example Metadata (`info`)
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

#### Detailed Results (`performance_by_rubric`)

A list with one entry per evaluated criterion, containing:

- **`criteria_met`** - Boolean indicating whether the criterion was satisfied
- **`judge_explanation`** - The judge model's reasoning for the decision

This allows you to see exactly which criteria the model passed or failed, along with explanations.

#### Example Record

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

#### Notes

- The `answer` and `task` fields are present for compatibility with the verifiers framework but are always `""` and `"default"` respectively for HealthBench
- Arrays in `info` (criteria, points_list, axes, consensus_criteria) are all aligned by index - the first element of each corresponds to the first rubric criterion
- Point values can be negative for undesirable behaviors (e.g., -2 points for "Gives dangerous medical advice")
- The total score is normalized to 0-1 regardless of the actual point scale used

### Dataset Schema
The output dataset has the following JSON schema:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "HealthBench Evaluation Dataset",
  "description": "Schema for HealthBench evaluation results including model completions, rubric-based scoring, and detailed per-criterion performance tracking",
  "type": "object",
  "required": ["prompt", "completion", "answer", "task", "reward", "reward_healthbench", "info"],
  "properties": {
    "prompt": {
      "type": "array",
      "description": "The input conversation messages presented to the model",
      "items": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {
            "type": "string",
            "enum": ["system", "user", "assistant"],
            "description": "The role of the message sender"
          },
          "content": {
            "type": "string",
            "description": "The message content"
          }
        }
      }
    },
    "completion": {
      "type": "array",
      "description": "The model's generated response messages",
      "items": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {
            "type": "string",
            "enum": ["assistant"],
            "description": "The role of the message sender (always 'assistant' for completions)"
          },
          "content": {
            "type": "string",
            "description": "The generated message content"
          }
        }
      }
    },
    "answer": {
      "type": "string",
      "description": "Ground truth answer (empty string for HealthBench as it uses rubric-based evaluation)",
      "const": ""
    },
    "task": {
      "type": "string",
      "description": "Task identifier",
      "const": "default"
    },
    "reward": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Overall normalized reward score (0-1) calculated as the sum of points earned divided by total possible points across all rubric criteria"
    },
    "reward_healthbench": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "HealthBench-specific reward metric (identical to 'reward' field)"
    },
    "info": {
      "type": "object",
      "description": "Metadata about the HealthBench example and its evaluation rubrics",
      "required": ["prompt_id", "theme", "criterion_ids", "criteria", "axes", "consensus_criteria", "points_list"],
      "properties": {
        "prompt_id": {
          "type": "string",
          "description": "Unique identifier for the HealthBench prompt"
        },
        "theme": {
          "type": "string",
          "description": "The thematic category of the prompt",
          "examples": ["emergency_referrals", "context_seeking", "global_health", "communication", "complex_responses", "health_data_tasks", "hedging"]
        },
        "criterion_ids": {
          "type": "array",
          "description": "BLAKE2b hash identifiers (8-byte digest) for each evaluation criterion",
          "items": {
            "type": "string",
            "pattern": "^[0-9a-f]{16}$"
          }
        },
        "criteria": {
          "type": "array",
          "description": "Full text descriptions of each evaluation criterion",
          "items": {
            "type": "string"
          }
        },
        "axes": {
          "type": "array",
          "description": "The evaluation axis for each criterion",
          "items": {
            "type": "string",
            "enum": ["completeness", "accuracy", "context_awareness", "communication_quality", "instruction_following"]
          }
        },
        "consensus_criteria": {
          "type": "array",
          "description": "Consensus criterion metadata for each rubric item (null if not a consensus criterion)",
          "items": {
            "oneOf": [
              {
                "type": "null"
              },
              {
                "type": "object",
                "required": ["theme", "behavior_category", "criterion"],
                "properties": {
                  "theme": {
                    "type": "string",
                    "description": "The consensus theme category",
                    "examples": ["Emergency referrals", "Context-seeking", "Global health", "Expertise tailored communication", "Response depth", "Health data tasks", "Responding under uncertainty"]
                  },
                  "behavior_category": {
                    "type": "string",
                    "description": "The behavioral subcategory within the theme",
                    "examples": ["Emergent", "Conditionally emergent", "Non-emergent", "Enough context", "Not enough context"]
                  },
                  "criterion": {
                    "type": "string",
                    "description": "The specific consensus criterion being evaluated",
                    "examples": ["Emergency behavior", "Context-seeking", "Helpful and safe", "Accuracy and completeness", "Tailored communication"]
                  }
                }
              }
            ]
          }
        },
        "points_list": {
          "type": "array",
          "description": "Point values for each criterion (positive for desirable behaviors, negative for undesirable ones)",
          "items": {
            "type": "integer"
          }
        }
      }
    },
    "performance_by_rubric": {
      "type": "array",
      "description": "Detailed per-criterion evaluation results from the judge model",
      "items": {
        "type": "object",
        "required": ["criteria_met", "judge_explanation"],
        "properties": {
          "criteria_met": {
            "type": "boolean",
            "description": "Whether the criterion was met according to the judge model"
          },
          "judge_explanation": {
            "type": ["string", "null"],
            "description": "The judge model's explanation for why the criterion was or was not met"
          }
        }
      }
    }
  },
  "additionalProperties": false
}
```
