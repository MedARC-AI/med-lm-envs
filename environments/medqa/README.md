# med-lm-eval
Automated LLM evaluation suite for medical tasks
# MedQA Eval

This repository provides an evaluation environment for the [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

## Usage

To run an evaluation using [vf-eval](https://github.com/PrimeIntellect-ai/verifiers) with the Mistral API, use:

```sh
uv run vf-eval \
	-m mistral-small-latest \
	-b https://api.mistral.ai/v1 \
	-k MISTRAL_API_KEY \
	--env-args '{"split":"test"}' \
	--num-examples 200 \
	-s \
	metamedqa
```

Replace `MISTRAL_API_KEY` with your actual API key.

## Environment

The evaluation environment is defined in `medqa.py` and uses the HuggingFace `GBaker/MedQA-USMLE-4-options` dataset.