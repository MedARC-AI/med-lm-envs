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


### References

```bibtex
@article{griot_large_2025,
    title = {Large {Language} {Models} lack essential metacognition for reliable medical reasoning},
    volume = {16},
    issn = {2041-1723},
    url = {https://doi.org/10.1038/s41467-024-55628-6},
    doi = {10.1038/s41467-024-55628-6},
    number = {1},
    journal = {Nature Communications},
    author = {Griot, Maxime and Hemptinne, Coralie and Vanderdonckt, Jean and Yuksel, Demet},
    month = jan,
    year = {2025},
    pages = {642},
}

```