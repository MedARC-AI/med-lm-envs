# biohopr

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
- **Rubric overview**: Precision based on embedded cosine similarity of list of possible answers

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval biohopr
```

Configure model and sampling:

```bash
uv run vf-eval biohopr   -m gpt-4.1-mini   -n 20   -a '{"key": "value"}'  # env-specific args as JSON
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


### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria). Is the same as `embedded_precision` |
| `embedded_precision` | Precision based on embedded cosine similarity. A completion is embedded, that compared to a list of possible answer embeddings. If cosine similarity is >tau(0.9) then it is considered a true positive. Final score is true_positives/predicted_responses. |

