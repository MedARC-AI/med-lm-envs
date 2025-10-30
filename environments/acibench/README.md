# ACI Bench Overview

### Overview
- **Environment ID**: `acibench`
- **Short description**: A benchmark for generating structured clinical notes from doctor-patient dialogues, using dynamic prompts based on the section headers of the ground-truth note.
- **Tags**: `medical`, `clinical`, `summarization`, `note-generation`, `single-turn`, `evaluation`

### Datasets
- **Primary dataset(s)**: `harsh-c137/aci-bench-medarc-eval` on Hugging Face. This is the final, cleaned, and de-duplicated version of the ACI-BENCH corpus, containing 225 unique encounters.
- **Source links**: [Paper](https://arxiv.org/abs/2306.02022), [HF Dataset](https://huggingface.co/datasets/harsh-c137/aci-bench-medarc-eval)
- **Split sizes**: The script loads the entire dataset and performs a reproducible 80/20 split in-memory, resulting in:
    - **Train**: 180
    - **Validation**: 45

### Task
- **Type**: `single-turn`
- **Parser**: `vf.Parser` using a custom function to extract the text content from the model's completion object.
- **Dynamic Prompting**: Before each evaluation, the script inspects the ground-truth `note`. It extracts all lines written in ALL CAPS (e.g., "HISTORY OF PRESENT ILLNESS", "ASSESSMENT AND PLAN") and dynamically injects them into the prompt. This provides the model with the exact structure and order of section headers it is expected to generate, ensuring a fair evaluation.
- **Rubric overview**: The rubric calculates three key metrics to evaluate the quality of the generated clinical note: ROUGE, BERTScore, and BLEURT.

### Quickstart
Run a default zero-shot evaluation using an OpenAI model:

```bash
export OPENAI_API_KEY="your-openai-api-key"
uv run vf-eval acibench -m gpt-4-turbo -n 5 -s
```

Run a 1-shot evaluation using a Mistral model:

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
uv run vf-eval acibench \
  -m mistral-small-latest \
  -b https://api.mistral.ai/v1 \
  -k MISTRAL_API_KEY \
  --env-args '{"num_few_shot": 1}' \
  -n 5 -s
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg             | Type         | Default | Description                                                                 |
| --------------- | ------------ | ------- | --------------------------------------------------------------------------- |
| `system_prompt` | `str | None` | `None`  | An optional custom system prompt to override the default.                   |
| `device`        | `str | None` | `None`  | Device for metric computation (e.g., `"cuda:0"`). Defaults to CPU.          |
| `num_few_shot`  | `int`        | `0`     | The number of in-context examples to provide in the prompt (0 for zero-shot). |

### Metrics
The final reward is the sum of the following three metrics, each providing a different perspective on the quality of the generated note.

| Metric      | Meaning                                                                                       |
| ----------- | --------------------------------------------------------------------------------------------- |
| `rouge`     | Measures the lexical overlap (exact words and sequences) between the generated and reference notes. |
| `bertscore` | Measures the semantic similarity, understanding paraphrasing and context. A key metric for meaning. |
| `bleurt`    | A learned metric trained to predict human judgments of quality and coherence.                 |


### Author
This environment was developed by **[Harsh Deshpande](https://www.linkedin.com/in/harsh-deshpande-v1/)**. Contributions include the dataset curation and cleaning (`dataset_curation_notes.md`) for Medarc's evaluation task, and creation of the verifiers environment script (`acibench.py`).