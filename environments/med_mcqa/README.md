# MEDMCQA

Evaluation environment for the MEDMCQA dataset.

### Overview
- **Environment ID**: `med_mcqa`
- **Short description**: Single-turn medical multiple-choice QA on MedMCQA 
- **Tags**: medical, single-turn, multiple-choice, train, eval

### Datasets
- **Primary dataset(s)**: MedMCQA (HF datasets)
- **Source links**: [lighteval/med_mcqa](https://huggingface.co/datasets/lighteval/med_mcqa)
- **Split sizes**: Uses provided train, validation splits. 

### Task
- **Type**: Single-Turn
- **Parser**: `Parser` (for standard MCQA) or `ThinkParser` (if using reasoning mode)

    Add Python import snippet
- **Rubric overview**: Binary scoring based on correct letter choice.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval med_mcqa
```

### Usage
# To run an evaluation using vf-eval with OpenAI-compatible endpoint and key already exported (OPENAI_API_KEY), use:

```bash
export OPENAI_API_KEY=sk-...
uv run vf-eval \
  -m gpt-4.1-mini \
  -n 5 \
  -s \
  med_mcqa
```
