# medbullets

### Overview
- **Environment ID**: `medbullets`
- **Short description**: USMLE-style multiple-choice questions from Medbullets.
- **Tags**: medical, clinical, single-turn, multiple-choice, USMLE, train, evaluation

### Datasets
- **Primary dataset(s)**: `Medbullets-4` and `Medbullets-5`
- **Source links**: [Paper](https://arxiv.org/pdf/2402.18060), [Github](https://github.com/HanjieChen/ChallengeClinicalQA), [HF Dataset](https://huggingface.co/datasets/mkieffer/Medbullets)
- **Split sizes**:

    | Split       | Choices         | Count   |
    | ----------- | --------------- | ------- |
    | `op4_test` | {A, B, C, D}    | **308** |
    | `op5_test` | {A, B, C, D, E} | **308** |

    `op5_test` contains the same content as `op4_test`, but with one additional answer choice to increase difficulty. Note that while the content is the same, the letter choice corresponding to the correct answer is sometimes different between these splits.


### Task
- **Type**: single-turn
- **Parser**: `Parser` or `ThinkParser`, with `extract_fn=extract_boxed_answer` for strict letter-in-\boxed{}-format parsing
- **Rubric overview**: Binary scoring based on correctly boxed letter choice and optional think tag formatting

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval medbullets
```

Configure model and sampling:

```bash
uv run vf-eval medbullets \
    -m gpt-4.1-mini   \
    -n -1 -r 3 -t 1024 -T 0.7  \
    -a '{"use_think": false, "num_options": 4, "num_test_examples": -1, "shuffle": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg                  | Type | Default | Description                                                                                                                                                                          |
| -------------------- | ---- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `num_test_examples` | int  | `-1`    | Limit the number of test examples (`-1` for all)                                                                                                                            |
| `num_options`        | int  | `4`     | Number of options: `4` → {A, B, C, D}; `5` → {A, B, C, D, E}                                                |
| `use_think`          | bool | `False` | Whether to check for `<think>...</think>` formatting with `ThinkParser`|
| `shuffle`            | bool | `False` | Whether to shuffle answer choices |


### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `correct_answer_reward_func` | (weight 1.0): 1.0 if parsed letter is correct, else 0.0|
| `parser.get_format_reward_func()` | (weight 0.0): optional format adherence (not counted) |

### References

```bibtex
@inproceedings{chen-etal-2025-benchmarking,
    title = "Benchmarking Large Language Models on Answering and Explaining Challenging Medical Questions",
    author = "Chen, Hanjie  and
      Fang, Zhouxiang  and
      Singla, Yash  and
      Dredze, Mark",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.182/",
    doi = "10.18653/v1/2025.naacl-long.182",
    pages = "3563--3599",
    ISBN = "979-8-89176-189-6",
}
```