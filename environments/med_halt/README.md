# Med-HALT

## Overview
- **Environment ID**: `med_halt`
- **Short description**: Evaluate clinical reasoning hallucinations using Med-HALT’s FCT (False Confidence Test) and NOTA (None of the Above) tasks.
- **Tags**: medical, hallucination, single-turn, multiple-choice, clinical reasoning, evaluation

## Datasets
- **Primary dataset(s)**: `Med-HALT` (reasoning subset)
- **Source links**: [Paper](https://arxiv.org/abs/2307.15343), [HF Dataset](https://huggingface.co/datasets/openlifescienceai/Med-HALT)
The upstream dataset ships only a **single split**, but internally includes a `split_type` field (`train` / `val` / `test`).  
This environment **filters to the `val` subset**, consistent with MedARC evaluation standards.

- **Split sizes**:

    | Test Type       | Examples  | Description |
    | --------------- | --------- | ----------- |
    | `reasoning_FCT` | **5,154** | False Confidence Test - evaluates if models can correctly assess proposed answers |
    | `reasoning_nota` | **5,154** | None of the Above Test - tests if models can identify when no option is correct |
    | `reasoning_fake` | _Not used_ | Fake Questions Test - assesses if models can recognize nonsensical questions (excluded by default) |

Notes:
- FCT’s validation subset is **purposely imbalanced**: almost all examples contain *incorrect* student answers.  
- This is expected and documented behavior of the source dataset.


## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring based on correct letter choice (A, B, C, D, etc.)

## Quickstart
Run an evaluation with default settings (FCT split):

```bash
prime eval run med_halt -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval med_halt -m "openai/gpt-5-mini" -n 100 -s --question-type reasoning_nota --num-few-shot 3 --no-shuffle-answers
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).

## Environment Arguments

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `system_prompt`      | str \| None | `None` | Custom system prompt (defaults to standard XML/BOXED prompt based on `answer_format`) |
| `shuffle_answers`    | bool | `False` | Whether to shuffle answer choices |
| `shuffle_seed`       | int \| None | `1618` | Random seed for reproducible answer shuffling |
| `question_type`      | str  | `"reasoning_fct"` | Question family to evaluate: `"reasoning_fct"` or `"reasoning_nota"` |
| `num_few_shot`       | int  | `2` | Number of few-shot examples to include in prompts |

## Metrics

| Metric | Meaning |
| ------ | ------- |
| `accuracy` | 1.0 if parsed letter matches the gold letter, else 0.0 |

## Credits

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
