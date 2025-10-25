# biohopr

### Overview
- **Environment ID**: `biohopr`
- **Short description**: BioHopR multiple-hop biomedical question answering evaluation
- **Tags**: biohopr, multi-hop, biomedical, question-answering, retrieval-augmented

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: Eval 7,633

### Task
- **Type**: single-turn
- **Parser**: <e.g., ThinkParser, XMLParser, custom>
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

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria). Is the same as `embedded_precision` |
| `embedded_precision` | Precision based on embedded cosine similarity. A completion is embedded, that compared to a list of possible answer embeddings. If cosine similarity is >tau(0.9) then it is considered a true positive. Final score is true_positives/predicted_responses. |

