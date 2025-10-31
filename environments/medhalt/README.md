# MedHALT

Evaluation environment for the MedHALT (Medical Domain Hallucination Test) dataset.

## Overview

MedHALT tests medical language models on multiple-choice questions from various medical exams. This environment implements evaluation for two configurations:

- `reasoning_FCT`: Functional correct thinking questions
- `reasoning_nota`: None-of-the-above questions

## Dataset

- **Source**: [openlifescienceai/Med-HALT](https://huggingface.co/datasets/openlifescienceai/Med-HALT)
- **Paper**: https://arxiv.org/abs/2307.15343
- **GitHub**: https://github.com/medhalt/medhalt
- **Size**: ~18,866 examples per configuration

## Installation
```bash
vf-install medhalt
```

## Quickstart

Run an evaluation with default settings:
```bash
uv run vf-eval medhalt
```

## Usage

To run an evaluation using `vf-eval` with the OpenAI API:
```bash
export OPENAI_API_KEY=sk-...
uv run vf-eval \
  -m gpt-4o-mini \
  -n 100 \
  -s \
  medhalt
```

Replace `OPENAI_API_KEY` with your actual API key.

### With Ollama
```bash
uv run vf-eval \
  -m qwen2.5:3b \
  -n 100 \
  --api-base-url http://localhost:11434/v1 \
  --api-key ollama \
  medhalt
```

### Specify configuration
```bash
# Test on None-of-the-Above questions
uv run vf-eval medhalt --config-name reasoning_nota

# Enable answer shuffling
uv run vf-eval medhalt --shuffle-answers --shuffle-seed 42

## Parameters

- `config_name`: One of `["reasoning_FCT", "reasoning_nota"]` (default: `"reasoning_FCT"`)
- `split`: Dataset split (default: `"train"`)
- `num_examples`: Limit number of examples (default: `None` for all)
- `shuffle_answers`: Randomize answer order (default: `False`)
- `shuffle_seed`: Seed for shuffling (default: `42`)

## Testing

A test script is provided to verify the environment works:
```bash
cd environments/medhalt
python test_medhalt.py --config reasoning_FCT --num-examples 10
```

## Example Results

Tested on qwen2.5:3b with 1000 examples:

| Config | Accuracy | Notes |
|--------|----------|-------|
| reasoning_FCT | 50.7% | Functional reasoning questions |
| reasoning_nota | 33.1% | None-of-the-above questions |

Results vary by model size and capability. Random guessing baseline is 25% for 4-option questions.