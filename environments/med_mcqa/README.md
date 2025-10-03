# Med MCQA Eval

Evaluation environment for the [lighteval/med_mcqa](https://huggingface.co/datasets/lighteval/med_mcqa) multiple-choice dataset.

## Usage

After setting up the repo tooling (see root README), install this environment locally:

```bash
vf-install med_mcqa
```

Run an evaluation (defaults to the test split):

```bash
# Example with OpenAI-compatible endpoint and key already exported (OPENAI_API_KEY)
export OPENAI_API_KEY= 
uv run vf-eval \
  -m gpt-4.1-mini \
  --num-examples 5 \
  -s \
  med_mcqa
```