# MedCalc-Bench

## Overview
- **Environment ID**: `medcalc-bench`
- **Short description**: Evaluate clinical calculator reasoning and numeric/date outputs. Optionally equips the model with a Python execution tool or a calculator tool.
- **Tags**: medical, clinical, single-turn, numeric, date, evaluation

## Dataset

Two dataset variants are available:

- `verified` (default): `nsk7153/MedCalc-Bench-Verified`
- `v1.2`: `ncbi/MedCalc-Bench-v1.2`

| Split | `v1.2` | `verified` |
| ----- | ------ | ---------- |
| train | 10,543 | 10,538 |
| test  | 1,100  | 1,100  |

Each example includes a `Patient Note`, `Question`, `Calculator ID`, `Ground Truth`, `Lower Bound`, and `Upper Bound`.

## Task
- **Type**: single-turn, multi-turn with tool use
- **Prompt**: `_build_prompt(patient_note, question)` instructs `<think>...</think>` and `<answer>...</answer>`.
- **Rubric**: `check_correctness` validates by calculator type:
  - IDs 13, 68: date equality (MM/DD/YYYY)
  - ID 69: tuple `(weeks, days)` equality
  - Integer IDs: integer equality (with rounding as needed)
  - Decimal IDs: numeric value within `[lower_bound, upper_bound]`

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medcalc-bench -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medcalc-bench -m "openai/gpt-5-mini" -n 5 -s --one-shot --add-python-tool
```

Notes:
- Use direct environment flags with `medarc-eval` (for example, `--split validation` or `--judge-model gpt-5-mini`).
- Setting `use_think` to `True` works best with `one_shot` set to `True`, so that the LLM can learn exactly how it should format its answer.
- The packaged `medarc_verifiers` XMLParser suppresses the upstream warning about `<think>` and still parses `<answer>` even if `<think>` is malformed.
- **Tool safety**: The Python tool uses [`RestrictedPython`](https://restrictedpython.readthedocs.io/) for sandboxed execution with limited builtins (only `math`, `numpy`, `scipy` imports allowed). The calculator tool uses [`simpleeval`](https://github.com/danthedeckie/simpleeval) with only safe math operations.

## Environment Arguments

| Arg         | Type | Default | Description |
| ----------- | ---- | ------- | ----------- |
| `one_shot` | bool | `False` | Whether to use the one-shot prompt |
| `add_python_tool` | bool | `False` | Add the Python code execution tool (uses restricted Python with limited builtins) |
| `add_calculator_tool` | bool | `False` | Add the calculator tool (uses simple eval with safe math operations) |
| `max_turns` | int | `20` | Maximum number of turns in tool use environment |
| `version` | str | `"verified"` | Dataset variant: `"verified"` (default) or `"1.2"` |
| `answer_format` | str | `"xml"` | Answer format: `"xml"` (default) or `"boxed"` |
| `use_think` | bool | `False` | Whether to instruct `<think>...</think>` formatting |
| `system_prompt` | str | `None` | Custom system prompt (defaults to standard XML/BOXED prompt based on `answer_format`) |


## Metrics

| Metric | Meaning |
| ------ | ------- |
| `check_correctness` | (weight 1.0): validates numeric/date/tuple answers per calc ID |

## Adjustments

Adjusted the prompt to output the step-by-step thinking and final answer with the <think> and <answer> tags instead of responding with a JSON.


## References

```bibtex
@misc{khandekar2024medcalcbench,
      title={MedCalc-Bench: Evaluating Large Language Models for Medical Calculations},
      author={Nikhil Khandekar and Qiao Jin and Guangzhi Xiong and Soren Dunn and Serina S Applebaum and Zain Anwar and Maame Sarfo-Gyamfi and Conrad W Safranek and Abid A Anwar and Andrew Zhang and Aidan Gilson and Maxwell B Singer and Amisha Dave and Andrew Taylor and Aidong Zhang and Qingyu Chen and Zhiyong Lu},
      year={2024},
      eprint={2406.12036},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.12036},
}
```
