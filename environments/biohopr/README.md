# BioHopR

### Overview
- **Environment ID**: `biohopr`
- **Short description**: BioHopR multiple-hop biomedical question answering evaluation
- **Tags**: biohopr, multi-hop, biomedical, question-answering, retrieval-augmented

### Datasets
- **Primary dataset(s)**: All datasets are based on the knowledge graph. [PrimeKG](https://zitniklab.hms.harvard.edu/projects/PrimeKG/). The BioHopR paper introduced 4 evaluation sets based on PrimeKG. Examples provided here: 
  - **1-hop**: Name a disease that is related to effect/phenotype Pain. 
  - **2-hop**(default eval): Name a disease that is related to a effect/phenotype that is associated with drug Benzyl benzoate. 
  - **1-hop multi**: Name all diseases that are related to effect/phenotype Pain.
  - **2-hop multi**: Name all diseases that are related to a effect/phenotype that is associated with drug Benzyl benzoate.

- **Source links**: [Paper](https://arxiv.org/abs/2505.22240) | 
[Dataset](https://huggingface.co/datasets/knowlab-research/BioHopR)
- **Split sizes**: Eval 7,633 (for 2-hop)

### Task
- **Type**: single-turn
- **Parser**: XMLParser(default)/ThinkParser
- **Rubric overview**: LLM as a Judge or Precision based on embedded cosine similarity of list of possible answers

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval biohopr
```

Configure model and sampling:

```bash
vf-eval biohopr -n 32 --api-base-url http://localhost:5000/v1/ -a '{"eval_method":"metrics","judge_base_url":"http://localhost:5000/v1/","judge_model":"openai/gpt-oss-20b","embedding_model_url":"http://localhost:8001/"}' -m 'openai/gpt-oss-20b'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `use_think` | bool | `False` | Use reasoning when set to true. |
| `system_prompt` | str | `None` | Optional custom system prompt. |
| `answer_format` | str | `AnswerFormat.XML` | Determines how to parse completion for answer. Also sets system prompt if `system_prompt` not set. |
| `task` | str | `biohopr_hop2` | The task to evaluate against. Determines which prompts are used from the BioHopR paper. Valid options are `['biohopr_hop1','biohopr_hop2','biohopr_hop1_multi','biohopr_hop2_multi', 'all']`. `task` also set for verifiers dataset for use with EnvGroup and RubricGroup |
| `eval_method` | str | `metrics` | Whether to use metrics from paper, or llm as a judge. Values: "judge" , "metrics", or "judge-only"
| `judge_model` | str | `gpt-4o-mini` | Model name to use for judging
| `judge_base_url` | str | `None` | Optional base URL for custom OpenAI-compatible API endpoint
| `judge_api_key` | str | `None` | Optional API key for OpenAI-compatible API endpoint
| `embeddings_model` | str | `FremyCompany/BioLORD-2023` | SentenceTransformer model name for computing answer-completion similarity. Valid options: `['FremyCompany/BioLORD-2023', 'Simonlee711/Clinical_ModernBERT']`
| `tau` | float | 0.9 | Cosine similarity threshold for embedded precision
| `use_cuda` | bool | "if available" | Whether to use CUDA for embedding model, uses cuda if available, set to false to disable. 
| `embedding_batch_size` | int | 32 |  Maximum batch size for embedding model encoding, use in case of out of memory error
| `embedding_model_url` |  str | None | Optional URL for OpenAI-compatible embedding model API
| `embedding_api_key` | str | None |  Optional API key for OpenAI-compatible embedding model API

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria). Is the same as `embedded_precision` |
| `llm_as_a_judge` | Reward function that uses LLM judge to evaluate medical diagnosis equivalence. |
| `embedded_precision` | Precision based on embedded cosine similarity. A completion is embedded, that compared to a list of possible answer embeddings. If cosine similarity is >tau(0.9) then it is considered a true positive. Final score is true_positives/predicted_responses. |

### Embedding Server Note 
We recommend using infinity embed server for serving embeddings. This was the only server we found compatible with BioLORD embeddings. You may find them here: https://github.com/michaelfeil/infinity
At the time of writing their server errors at the time of setup. Further instructions can be found here: https://github.com/michaelfeil/infinity/issues/649


### Citation
Dataset:
```bibtex
@misc{kim2025biohoprbenchmarkmultihopmultianswer,
      title={BioHopR: A Benchmark for Multi-Hop, Multi-Answer Reasoning in Biomedical Domain}, 
      author={Yunsoo Kim and Yusuf Abdulle and Honghan Wu},
      year={2025},
      eprint={2505.22240},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22240}, 
}
```