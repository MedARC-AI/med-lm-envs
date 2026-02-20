# MedQA

Evaluation environment for the MedQA (USMLE) dataset.

## Overview
- **Environment ID**: `medqa`
- **Short description**: Single-turn USMLE-style medical multiple-choice QA
- **Tags**: medical, single-turn, multiple-choice, usmle, train, eval

## Datasets
- **Primary dataset(s)**: MedQA USMLE 4-options
- **Source links**: [GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- **Split sizes**: Uses provided train and test splits

## Task
- **Type**: single-turn
- **Rubric overview**: Binary scoring (1.0 / 0.0) based on correct letter match

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run medqa -m "openai/gpt-5-mini" -n 5 -s
```

Configure model and sampling:

```bash
medarc-eval medqa -m "openai/gpt-5-mini" -n 5 -s --answer-format boxed
```

## Authors
This environment has been put together by:

Ahmed Essouaied - ([@ahmedessouaied](https://github.com/ahmedessouaied))

## Credits
Dataset:

```bibtex
@misc{jin2020diseasedoespatienthave,
      title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
      author={Di Jin and Eileen Pan and Nassim Oufattole and Wei-Hung Weng and Hanyi Fang and Peter Szolovits},
      year={2020},
      eprint={2009.13081},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2009.13081},
}
```
