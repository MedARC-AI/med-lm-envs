# med-lm-eval
Automated LLM evaluation suite for medical tasks
# MedQA Eval

This repository provides an evaluation environment for the [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options).

## Usage

To run an evaluation using [vf-eval](https://github.com/PrimeIntellect-ai/verifiers) with the OpenAI API, use:

```sh
export OPENAI_API_KEY=sk-...
vf-eval medqa -m gpt-4.1-mini -n 5 -s
```

Replace `OPENAI_API_KEY` with your actual API key.

## Environment

The evaluation environment is defined in `medqa.py` and uses the HuggingFace `GBaker/MedQA-USMLE-4-options` dataset.

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

### Authors
This environment has been put together by:

Ahmed Essouaied - ([@ahmedessouaied](https://github.com/ahmedessouaied))