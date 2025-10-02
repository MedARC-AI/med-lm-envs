# med-lm-eval
Automated LLM evaluation suite for medical tasks
# MedQA Eval

This repository provides an evaluation environment for the [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

## Usage

To run an evaluation using [vf-eval](https://github.com/PrimeIntellect-ai/verifiers) with the OpenAI API, use:

```sh
export OPENAI_API_KEY=sk-...
vf-eval medqa -m gpt-4.1-mini -n 5 -s
```

Replace `OPENAI_API_KEY` with your actual API key.

## Environment

The evaluation environment is defined in `medqa.py` and uses the HuggingFace `GBaker/MedQA-USMLE-4-options` dataset.