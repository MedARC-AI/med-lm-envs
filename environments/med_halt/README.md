# Med-HALT

### Overview
- **Environment ID**: `med_halt`
- **Short description**: Med-HALT reasoning hallucination tests for medical LLMs
- **Tags**: medical, hallucination, single-turn, multiple-choice, reasoning, test

### Datasets
- **Primary dataset(s)**: `Med-HALT` (reasoning subset)
- **Source links**: [Paper](https://arxiv.org/abs/2307.15343), [HF Dataset](https://huggingface.co/datasets/openlifescienceai/Med-HALT)
- **Split sizes**:

    | Test Type       | Examples  | Description |
    | --------------- | --------- | ----------- |
    | `reasoning_FCT` | **18,866** | False Confidence Test - evaluates if models can correctly assess proposed answers |
    | `reasoning_nota` | **18,866** | None of the Above Test - tests if models can identify when no option is correct |
    | `reasoning_fake` | **1,858** | Fake Questions Test - assesses if models can recognize nonsensical questions (excluded by default) |

### Task
- **Type**: single-turn
- **Parser**: `XMLParser` (default) or `Parser`/`ThinkParser` with `extract_fn=extract_boxed_answer` for \boxed{} format
- **Rubric overview**: Binary scoring based on correct letter choice (A, B, C, D, etc.)

### Quickstart
Run an evaluation with default settings (FCT and Nota tests):

```bash
uv run vf-eval med_halt
```

Configure model and sampling:

```bash
uv run vf-eval med_halt \
    -m gpt-4o-mini   \
    -n 100 -r 1 -t 1024 -T 0.7  \
    -a '{"use_think": false, "test_types": ["reasoning_FCT", "reasoning_nota"], "shuffle_answers": false}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The default includes both FCT and Nota test types. The Fake test type requires special evaluation logic and is excluded by default.

### Environment Arguments

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `use_think`          | bool | `False` | Whether to check for `<think>...</think>` formatting with `ThinkParser` or `XMLParser` |
| `system_prompt`      | str \| None | `None` | Custom system prompt (defaults to standard XML/BOXED prompt based on `answer_format`) |
| `shuffle_answers`    | bool | `False` | Whether to shuffle answer choices |
| `shuffle_seed`       | int \| None | `1618` | Random seed for reproducible answer shuffling |
| `answer_format`      | str  | `"xml"` | Answer format: `"xml"` (default) or `"boxed"` |
| `test_types`         | list[str] \| None | `["reasoning_FCT", "reasoning_nota"]` | Test types to include (options: `"reasoning_FCT"`, `"reasoning_nota"`, `"reasoning_fake"`) |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0 |

### Credits

Dataset:

```bibtex
@article{jeblick2023medhalt,
  title={Med-HALT: Medical Domain Hallucination Test for Large Language Models},
  author={Jeblick, Konstantin and Schachtner, Balthasar and Dexl, Jakob and 
          Mittermeier, Andreas and St{\"u}ber, Anna Theresa and Topalis, Johanna and
          Weber, Tobias and Wesp, Philipp and Sabel, Bastian Oliver and 
          Ricke, Jens and Ingrisch, Michael},
  journal={arXiv preprint arXiv:2307.15343},
  year={2023}
}
```
