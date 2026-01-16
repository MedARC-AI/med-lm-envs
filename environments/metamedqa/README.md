# med-lm-eval
Automated LLM evaluation suite for medical tasks
# MetaMedQA Eval

This repository provides an evaluation environment for the [MetaMedQA](https://huggingface.co/datasets/maximegmd/MetaMedQA).

## Usage

To run an evaluation using [vf-eval](https://github.com/EleutherAI/vf-eval) with the Mistral API, use:

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

The evaluation environment is defined in `metamedqa.py` and uses the HuggingFace `maximegmd/MetaMedQA` dataset.

## Authors
This environment has been put together by:

Aymane Ouraq - ([@aymaneo](https://github.com/aymaneo))
