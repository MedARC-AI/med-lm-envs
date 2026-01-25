# PubHealthBench

Evaluation environment for the [Joshua-Harris/PubHealthBench](https://huggingface.co/datasets/Joshua-Harris/PubHealthBench) dataset.

## Dataset

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

## Usage

```bash
# Install
vf-install pubhealthbench

# Run MCQ evaluation (default: reviewed split)
vf-eval pubhealthbench -m gpt-5-mini -n 10

# Use full test split
uv run vf-eval pubhealthbench --split full -m gpt-5-mini -n 10

# With answer shuffling
uv run vf-eval pubhealthbench --shuffle-answers -m gpt-5-mini -n 10

# Freeform (LLM-as-judge) evaluation
uv run vf-eval pubhealthbench --split freeform -m gpt-5-mini -n 10
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | "reviewed" | Dataset split (see table above) |
| `shuffle_answers` | bool | False | Randomize answer option order (MCQ only) |
| `shuffle_seed` | int | 1618 | Seed for deterministic shuffling (MCQ only) |
| `answer_format` | str | "xml" | Answer format: "xml" or "boxed" (MCQ only) |
| `judge_model` | str | "gpt-4o-mini" | Judge model for freeform evaluation |
| `judge_base_url` | str | None | Base URL for judge API |
| `judge_api_key` | str | None | API key for judge |

### Authors
This environment has been put together by:

Benjamin Warner - ([@warner-benjamin](https://github.com/warner-benjamin))

### Citation
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