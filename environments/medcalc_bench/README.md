# MedCalc-Bench

### Overview
- **Environment ID**: `medcalc-bench`
- **Short description**: Evaluate clinical calculator reasoning and numeric/date outputs.
- **Tags**: medical, clinical, single-turn, numeric, date, evaluation

### Dataset
- **Primary dataset**: `ncbi/MedCalc-Bench-v1.2`
- Each example includes `Patient Note`, `Question`, `Calculator ID`, `Ground Truth`, `Lower Bound`, `Upper Bound`.
- Mapped fields in env: `question` (formatted prompt), `calc_id`, `ground_truth`, `lower_bound`, `upper_bound`.

    | Split | Count |
    | ----- | ----- |
    | train | 10543 |
    | test  | 1100  |

### Task
- **Type**: single-turn
- **Prompt**: `_build_prompt(patient_note, question)` instructs `<think>...</think>` and `<answer>...</answer>`.
- **Parser**: medarc_verifiers' `XMLParser` reads both `<think>` and `<answer>` tags.
- **Rubric**: `check_correctness` validates by calculator type:
  - IDs 13, 68: date equality (MM/DD/YYYY)
  - ID 69: tuple `(weeks, days)` equality
  - Integer IDs: integer equality (with rounding as needed)
  - Decimal IDs: numeric value within `[lower_bound, upper_bound]`

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medcalc-bench
```

Configure model and sampling:

```bash
uv run vf-eval medcalc-bench \
    -m gpt-4.1-mini \
    -n 5 -r 3 -t 1024 -T 0.7 \
    -a '{"use_think": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The packaged `medarc_verifiers` XMLParser suppresses the upstream warning about `<think>` and still parses `<answer>` even if `<think>` is malformed.


### Environment Arguments

| Arg         | Type | Default | Description |
| ----------- | ---- | ------- | ----------- |
| `use_think` | bool | `False` | Whether to instruct `<think>...</think>` formatting |


### Metrics

| Metric | Meaning |
| ------ | ------- |
| `check_correctness` | (weight 1.0): validates numeric/date/tuple answers per calc ID |

### Adjustments

Adjusted the prompt to output the step-by-step thinking and final answer with the <think> and <answer> tags instead of responding with a JSON.


### References

```bibtex
@misc{khandekar2024medcalcbench,
      title={MedCalc-Bench: Evaluating Large Language Models for Medical Calculations}, 
      author={Nikhil Khandekar and Qiao Jin and Guangzhi Xiong and Soren Dunn and Serina S Applebaum and Zain Anwar and Maame Sarfo-Gyamfi and Conrad W Safranek and Abid A Anwar and Andrew Zhang and Aidan Gilson and Maxwell B Singer and Amisha Dave and Andrew Taylor and Aidong Zhang and Qingyu Chen and Zhiyong Lu},
      year={2024},
      eprint={2406.12036},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```