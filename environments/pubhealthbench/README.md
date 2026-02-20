# PubHealthBench

Evaluation environment for the [Joshua-Harris/PubHealthBench](https://huggingface.co/datasets/Joshua-Harris/PubHealthBench) dataset.

## Overview
- **Environment ID**: `pubhealthbench`
- **Short description**: Public health MCQ and free-form evaluation derived from UK Health Security Agency (UKHSA) guidance documents
- **Tags**: medical, public-health, single-turn, multiple-choice, llm-judge, eval

## Datasets

PubHealthBench contains public health questions derived from UK Health Security Agency (UKHSA) guidance documents. Questions cover topics including:
- Gastro/food safety
- Chemicals/toxicology
- Vaccine-preventable diseases and immunisation
- And more

## Splits

| Split | Type | Questions | Description |
|-------|------|-----------|-------------|
| `full` | MCQ | 7,929 | Full test set |
| `validation` | MCQ | 161 | Validation set |
| `reviewed` | MCQ | 760 | Human-reviewed questions (default) |
| `freeform` | LLM-as-judge | 760 | Reviewed set with open-ended evaluation |
| `freeform_valid` | LLM-as-judge | 161 | Validation set with open-ended evaluation |

## Quickstart

Install:

```bash
vf-install pubhealthbench
```

Run MCQ evaluation (default: reviewed split):

```bash
prime eval run pubhealthbench -m "openai/gpt-5-mini" -n 5 -s
```

Use full test split:

```bash
medarc-eval pubhealthbench --split full -m "openai/gpt-5-mini" -n 10
```

With answer shuffling:

```bash
medarc-eval pubhealthbench --shuffle-answers -m "openai/gpt-5-mini" -n 10
```

Freeform (LLM-as-judge) single-judge evaluation:

```bash
medarc-eval pubhealthbench --split freeform -m "openai/gpt-5-mini" -n 10 --judge-model "openai/gpt-5-mini"
```

This environment supports multi-judge via the `medarc-verifiers` `MultiJudge` rubric. When multiple judge models are specified, each scores independently and the final reward is their mean. `judge_model`, `judge_base_url`, and `judge_api_key` are all list-valued and correspond positionally; if only one `judge_base_url` or `judge_api_key` is provided, it is used for all judge models.


```bash
medarc-eval pubhealthbench --split freeform -m "openai/gpt-5-mini" -n 10 --judge-model "openai/gpt-5-mini" --judge-model "google/gemini-3-flash-preview"
```

## Environment Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | "reviewed" | Dataset split (see table above) |
| `shuffle_answers` | bool | False | Randomize answer option order (MCQ only) |
| `shuffle_seed` | int | 1618 | Seed for deterministic shuffling (MCQ only) |
| `answer_format` | str | "xml" | Answer format: "xml" or "boxed" (MCQ only) |
| `judge_model` | str \| list[str] | "gpt-4o-mini" | Judge model(s) for freeform evaluation |
| `judge_base_url` | str \| list[str] | None | Base URL(s) for judge API |
| `judge_api_key` | str \| list[str] | None | API key(s) for judge |

## Authors
This environment has been put together by:

Benjamin Warner - ([@warner-benjamin](https://github.com/warner-benjamin))

## Citation
Dataset:
```bibtex
@misc{harris2025healthyllmsbenchmarkingllm,
      title={Healthy LLMs? Benchmarking LLM Knowledge of UK Government Public Health Information},
      author={Joshua Harris and Fan Grayson and Felix Feldman and Timothy Laurence and Toby Nonnenmacher and Oliver Higgins and Leo Loman and Selina Patel and Thomas Finnie and Samuel Collins and Michael Borowitz},
      year={2025},
      eprint={2505.06046},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.06046},
}
```
