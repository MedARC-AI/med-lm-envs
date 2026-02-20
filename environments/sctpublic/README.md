# sctpublic

Evaluation environment for SCT-Bench public dataset.

## Overview
- **Environment ID**: `sctpublic`
- **Short description**: Single-turn SCT dataset environment
- **Tags**: medical, clinical, single-turn, eval

## Datasets
- **Primary dataset(s)**: SCT-Bench public 
- **Source links**: https://github.com/SCT-Bench/sctpublic
- **Split sizes**: Evaluation only

## Task
- **Type**: Single-turn clinical reasoning evaluation
- **Rubric overview**: Custom `sct_rubric` that normalizes the answer distribution so that the greatest score is always 1

## Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `reason` | bool | `False` | If True, prompts include an explanation requirement |
| `few_shot` | bool | `False` | If True, includes 5 example ratings in the prompt |

## Quickstart
Run an evaluation with default settings:

```bash
prime eval run sctpublic -m "openai/gpt-5-mini" -n 5 -s
```

## Usage
To run an evaluation using `medarc-eval` with few-shot prompting and reasoning enabled:

```bash
medarc-eval sctpublic -m "openai/gpt-5-mini" -n 5 -s --reason --few-shot
```

## Authors
This environment has been put together by:

Ratna Sagari Grandhi - ([@sagarigrandhi](https://github.com/sagarigrandhi))

## Credits 
Dataset:
```bibtex
@article{mccoy2025assessment,
  title={Assessment of large language models in clinical reasoning: a novel benchmarking study},
  author={McCoy, Liam G and Swamy, Rajiv and Sagar, Nidhish and Wang, Minjia and Bacchi, Stephen and Fong, Jie Ming Nigel and Tan, Nigel CK and Tan, Kevin and Buckley, Thomas A and Brodeur, Peter and others},
  journal={NEJM AI},
  volume={2},
  number={10},
  pages={AIdbp2500120},
  year={2025},
  publisher={Massachusetts Medical Society}
}
```
