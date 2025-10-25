# AgentClinic Environment

Multi-agent medical diagnosis environment for evaluating LLMs on clinical diagnosis through interactive conversations.

## Quick Start

```bash
# Install
uv pip install -e .

# Set API key
export OPENAI_API_KEY="your-key"
```

## Usage

### Basic Command

```bash
uv run --active -m verifiers.scripts.eval \
  -m MODEL_NAME \
  -b API_BASE_URL \
  -k API_KEY_VAR \
  agentclinic \
  -n NUM_CASES \
  --rollouts-per-example 3 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{
    "dataset_path": "DATASET.jsonl",
    "patient_model": "MODEL",
    "patient_base_url": "URL",
    "patient_api_key": "KEY",
    "measurement_model": "MODEL",
    "measurement_base_url": "URL",
    "moderator_model": "MODEL",
    "moderator_base_url": "URL"
  }'
```

### OpenAI Example (MedQA)

```bash
export OPENAI_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m gpt-4o-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  agentclinic \
  -n 10 \
  --rollouts-per-example 2 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_medqa_extended.jsonl"}'
```

### OpenAI Example (NEJM)

```bash
export OPENAI_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m gpt-4o-mini \
  -b https://api.openai.com/v1 \
  -k OPENAI_API_KEY \
  agentclinic \
  -n 10 \
  --rollouts-per-example 2 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{"dataset_path": "agentclinic_nejm_extended.jsonl"}'
```

### Mixed Providers (Mistral Doctor + OpenAI Helpers)

```bash
export MISTRAL_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

uv run --active -m verifiers.scripts.eval \
  -m mistral-large-latest \
  -b https://api.mistral.ai/v1 \
  -k MISTRAL_API_KEY \
  agentclinic \
  -n 10 \
  --rollouts-per-example 2 \
  --max-concurrent 2 \
  -T 0.0 \
  -s \
  --env-args '{
    "dataset_path": "agentclinic_medqa_extended.jsonl",
    "patient_model": "gpt-4o-mini",
    "patient_base_url": "https://api.openai.com/v1",
    "patient_api_key": "'$OPENAI_API_KEY'",
    "measurement_model": "gpt-4o-mini",
    "measurement_base_url": "https://api.openai.com/v1",
    "measurement_api_key": "'$OPENAI_API_KEY'",
    "moderator_model": "gpt-4o-mini",
    "moderator_base_url": "https://api.openai.com/v1",
    "moderator_api_key": "'$OPENAI_API_KEY'"
  }'
```

## Configuration

### Agent Parameters

Each agent (patient, measurement) can be configured via `--env-args`:

### Datasets

- **MedQA Extended** (214 cases): `agentclinic_medqa_extended.jsonl`
- **NEJM Extended** (120 cases): `agentclinic_nejm_extended.jsonl`

### Other Options

- `max_turns`: Maximum conversation turns (default: 20)
- `use_think`: Enable chain-of-thought prompting (default: false)


### Agent Roles

- **Doctor** (evaluated model): Asks questions, requests tests (e.g., "REQUEST TEST: MRI_Brain_Spine"), makes diagnosis
- **Patient** (auxiliary LLM): Simulates realistic patient responses based on case symptoms
- **Measurement** (auxiliary LLM): Returns test results from scenario data when requested
- **Moderator** (auxiliary LLM): Evaluates diagnosis accuracy using JudgeRubric

