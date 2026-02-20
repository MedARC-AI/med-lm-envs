# medcasereasoning

## Overview
- **Environment ID**: `medcasereasoning`
- **Short description**: MedCaseReasoning dataset from Stanford (Wu et al. 2025) — open-ended clinical case diagnosis evaluated by LLM judge
- **Tags**: medical, clinical, single-turn, diagnosis, llm-judge, train, eval

## Datasets
- **Primary dataset(s)**: MedCaseReasoning — clinical case presentations with ground truth final diagnoses
- **Source links**: [zou-lab/MedCaseReasoning](https://huggingface.co/datasets/zou-lab/MedCaseReasoning)
- **Split sizes**: 13.1k (train) / 500 (val) / 897 (test)

## Task
- **Type**: single-turn
- **Rubric overview**: LLM-as-a-Judge — judge is asked "Is our predicted diagnosis correct (yes/no)?"; returns 1.0 if the judge answers "yes", else 0.0

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medcasereasoning -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medcasereasoning -m "openai/gpt-5-mini" -n 20 -s
```

Judge example:

```bash
medarc-eval medcasereasoning -m "openai/gpt-5-mini" -n 20 -s --judge-model "openai/gpt-5-nano"
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str | `"openai/gpt-5-nano"` | Model to use for LLM-as-a-Judge evaluation |
| `judge_base_url` | str \| None | `None` | Optional base URL for judge model API |
| `judge_api_key` | str \| None | `None` | Optional API key for judge model (defaults to `OPENAI_API_KEY`) |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `medical_diagnosis_reward_func` | (weight 1.0): 1.0 if judge confirms the diagnosis is correct, else 0.0 |
