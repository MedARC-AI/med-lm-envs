# HEAD-QA

Evaluation environment for the HEAD-QA dataset.

### Overview
- **Environment ID**: `head-qa`
- **Short description**: Single-turn medical multiple-choice QA 
- **Tags**: medical, single-turn, multiple-choice, train, eval

### Datasets
- **Primary dataset(s)**: HEAD-QA (HF datasets)
- **Source links**: [EleutherAI/headqa](https://huggingface.co/datasets/EleutherAI/headqa) 
- **Split sizes**: Uses provided train and validation splits

### Task
- **Type**: Single-turn
- **Parser**: `XMLParser` or `Parser` (BOXED format) depending on `answer_format`, uses `ThinkParser` when `use_think=True`
- **Rubric overview**: Binary scoring (1.0 / 0.0), based on correct answer.
- **Reward function:** `accuracy` — returns 1.0 if the predicted answer matches, else 0.0.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval head_qa
```

### Usage
To run an evaluation using `vf-eval` with the OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
uv run vf-eval \
  -m gpt-4.1-mini \
  -n 5 \
  -s \
  head_qa
```
Replace `OPENAI_API_KEY` with your actual API key.

### Authors
This environment has been put together by:

Ratna Sagari Grandhi - ([@sagarigrandhi](https://github.com/sagarigrandhi))

### Credits 
Dataset:
```bibtex
@inproceedings{vilares-gomez-rodriguez-2019-head,
    title = "{HEAD}-{QA}: A Healthcare Dataset for Complex Reasoning",
    author = "Vilares, David  and
      G{\'o}mez-Rodr{\'i}guez, Carlos",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1092",
    doi = "10.18653/v1/P19-1092",
    pages = "960--966",
    abstract = "We present HEAD-QA, a multi-choice question answering testbed to encourage research on complex reasoning. The questions come from exams to access a specialized position in the Spanish healthcare system, and are challenging even for highly specialized humans. We then consider monolingual (Spanish) and cross-lingual (to English) experiments with information retrieval and neural techniques. We show that: (i) HEAD-QA challenges current methods, and (ii) the results lag well behind human performance, demonstrating its usefulness as a benchmark for future work.",
}
```

