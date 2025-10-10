# AgentClinic Environment

Multi-agent medical diagnosis environment for evaluating LLMs on clinical diagnosis through interactive conversations.

## Quick Start

### 1. Install

```bash
uv pip install -e .
```

### 2. Set API Keys

```bash
# Required for helper agents (patient, measurement, moderator)
export OPENAI_API_KEY="your-key"

# Optional: For using other models
export ANTHROPIC_API_KEY="your-key"
export MISTRAL_API_KEY="your-key"

```

---

## Running Evaluations

### Basic Command Structure

```bash
uv run --active -m verifiers.scripts.eval \
  -m MODEL_NAME \                    # Doctor model (evaluated)
  -b API_BASE_URL \                  # Doctor API endpoint
  -k API_KEY_VAR \                   # Doctor API key variable name
  agentclinic \
  -n NUM_CASES \                     # Number of cases to evaluate
  --rollouts-per-example 3 \         # Rollouts 
  --max-concurrent 2 \               # Parallel requests
  -T 0.0 \                          # Temperature
  -s \                              # Save results
  --env-args '{
    "dataset_path": "DATASET.jsonl",
    "patient_model": "MODEL",
    "measurement_model": "MODEL",
    "moderator_model": "MODEL"
  }'
```

---

## Examples

### MedQA with GPT-4o-mini (all agents)

```bash
export OPENAI_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m gpt-4o-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  agentclinic \
  -n 50 \
  --rollouts-per-example 3 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_medqa_extended.jsonl"}'
```

### NEJM with GPT-4o-mini (all agents)

```bash
export OPENAI_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m gpt-4o-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  agentclinic \
  -n 50 \
  --rollouts-per-example 3 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_nejm_extended.jsonl"}'
```

### Mistral Large (all agents)

```bash
export MISTRAL_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m mistral-large-latest \
  -b https://api.mistral.ai/v1 \
  -k MISTRAL_API_KEY \
  agentclinic \
  -n 50 \
  --rollouts-per-example 3 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{
    "dataset_path": "agentclinic_nejm_extended.jsonl",
    "patient_model": "mistral-large-latest",
    "patient_backend": "mistral",
    "measurement_model": "mistral-large-latest",
    "measurement_backend": "mistral",
    "moderator_model": "mistral-large-latest",
    "moderator_backend": "mistral"
  }'
```

### Claude 3.5 Sonnet (doctor) + GPT-4o-mini (helpers)

```bash
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m claude-3-5-sonnet-20241022 \
  -b https://api.anthropic.com/v1 \
  -k ANTHROPIC_API_KEY \
  agentclinic \
  -n 50 \
  --rollouts-per-example 3 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_medqa_extended.jsonl"}'
```



---

## Configuration Options

### Agent Configuration

Configure each agent separately via `--env-args`:

```json
{
  "dataset_path": "agentclinic_medqa_extended.jsonl",

  // Patient Agent
  "patient_model": "gpt-4o-mini",
  "patient_backend": "auto",
  "patient_api_key": null,
  "patient_api_base": null,

  // Measurement Agent
  "measurement_model": "gpt-4o-mini",
  "measurement_backend": "auto",
  "measurement_api_key": null,
  "measurement_api_base": null,

  // Moderator/Judge Agent
  "moderator_model": "gpt-4o-mini",
  "moderator_backend": "auto",
  "moderator_api_key": null,
  "moderator_api_base": null,

  // Other options
  "max_turns": 20,
  "use_think": false
}
```

### Supported Backends

- `openai` - OpenAI models (gpt-4, gpt-4o-mini, etc.)
- `anthropic` - Anthropic models (claude-3-5-sonnet, etc.)
- `mistral` - Mistral models (mistral-large-latest, etc.)
- `gemini` - Google Gemini models
- `vllm` - Local vLLM server
- `auto` - Auto-detect from model name (default)

### Datasets

- **MedQA Extended** (214 cases): `agentclinic_medqa_extended.jsonl`
- **NEJM Extended** (120 cases): `agentclinic_nejm_extended.jsonl`

---

## Agent Roles

- **Doctor** (evaluated model): Asks questions, orders tests, makes diagnosis
- **Patient** (helper): Simulates patient responses based on case data
- **Measurement** (helper): Returns test results from case data
- **Moderator** (helper): Judges if diagnosis matches ground truth

