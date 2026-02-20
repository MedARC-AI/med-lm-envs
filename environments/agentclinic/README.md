# AgentClinic Environment

Multi-agent medical diagnosis environment for evaluating LLMs on clinical diagnosis through interactive conversations.

## Quickstart

Run a quick evaluation with `prime eval`:

```bash
prime eval run agentclinic -m "openai/gpt-5-mini" -n 5 -s
```

Or use `medarc-eval` with a single judge:

```bash
medarc-eval agentclinic -m "openai/gpt-5-mini" -n 5 -s --judge-model "openai/gpt-5-mini"
```

This environment supports multi-judge via the `medarc-verifiers` `MultiJudge` rubric. When multiple judge models are specified, each scores independently and the final reward is their mean. `judge_model`, `judge_base_url`, and `judge_api_key` are all list-valued and correspond positionally; if only one `judge_base_url` or `judge_api_key` is provided, it is used for all judge models.

```bash
medarc-eval agentclinic \
  -m "openai/gpt-5-mini" \
  -n 10 \
  -s \
  --patient-model "openai/gpt-5-mini" \
  --measurement-model "openai/gpt-5-mini" \
  --judge-model "openai/gpt-5-mini" \
  --judge-model "google/gemini-3-flash-preview" \
```

## Usage

Filter to a specific dataset:

```bash
medarc-eval agentclinic \
  -m "openai/gpt-5-mini" \
  -n 10 \
  -s \
  --dataset-path "agentclinic_medqa_extended.jsonl" \
  --judge-model "openai/gpt-5-mini"
```

## Configuration

## Datasets

- **MedQA Extended** (214 cases): `agentclinic_medqa_extended.jsonl`
- **NEJM Extended** (120 cases): `agentclinic_nejm_extended.jsonl`
  - Text-only in this environment; `image_url` is passed as plain text.

## Other Options

- `dataset_type`: `medqa` or `nejm` (auto-detect if omitted)
- `max_turns`: Maximum conversation turns (default: 20)
- `use_think`: Enable chain-of-thought prompting (default: false)
- `patient_temperature` / `measurement_temperature`
- `aux_max_tokens`: Max tokens for patient/measurement agents
- `doctor_bias` / `patient_bias`: Cognitive bias injection (validated)


## Agent Roles

- **Doctor** (evaluated model): Asks questions, requests tests (e.g., "REQUEST TEST: MRI_Brain_Spine"), makes diagnosis
- **Patient** (auxiliary LLM): Simulates realistic patient responses based on case symptoms
- **Measurement** (auxiliary LLM): Returns test results from scenario data when requested
- **Judge** (auxiliary LLM): Evaluates diagnosis accuracy using the canonical AgentClinic moderator prompt
