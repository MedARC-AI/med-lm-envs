# CaseReportBench

CaseReportBench is a benchmark designed for dense information extraction from clinical case reports.

### Overview
- **Environment ID**: `casereportbench`
- **Short description**: Dense clinical information extraction from case reports.
- **Tags**: medical, information-extraction, single-turn, eval

### Datasets
- **Primary dataset(s)**: [cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark](https://huggingface.co/datasets/cxyzhang/caseReportBench_ClinicalDenseExtraction_Benchmark)
- **Source links**: [Original Repository](https://github.com/cindyzhangxy/CaseReportBench)
- **Split sizes**: 138 expert-annotated case reports 

### Task
- **Type**: Single-turn information extraction.
- **Parser**: `JSONParser` (expects JSON with keys like `extractions`, `findings`, or `output`).
- **Input Format**: Case report text followed by category-specific extraction instructions.

### Metrics
This environment replicates the paper's metrics. When running `vf-eval`, the `reward` column corresponds to the **Token Set Ratio (TSR)**.

| Metric | Meaning |
| ------ | ------- |
| `reward` (TSR) | **Primary**. Token Set Ratio normalized by token length (0.0 to 1.0). |
| `bleu1` | 1-gram precision of extracted findings. |
| `bleu4` | 4-gram precision of extracted findings. |
| `rougeL` | Longest Common Subsequence overlap. |
| `omission` | 1.0 if model extracted info when expert did; 0.0 if failure to extract. |
| `hallucination` | 1.0 if model stayed silent when expert was; 0.0 if invention. |

### Quickstart
Run an evaluation with default settings (all categories, first 5 examples):

```bash
# Install the environment
vf-install casereportbench

# Run evaluation
vf-eval casereportbench -m gpt-4o-mini -n 5
```

### Usage
To run an evaluation using `vf-eval` with the OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
vf-eval \
  -m gpt-4o-mini \
  -n 10 \
  -s \
  casereportbench
```

To evaluate a specific clinical category:
```bash
vf-eval casereportbench -m gpt-4o-mini -a '{"task": "Neuro"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `task` | str | `"all"` | Which category to evaluate: `"Neuro"`, `"CVS"`, `"RESP"`, etc. Use `"all"` for all 13. |
| `max_examples` | int | `-1` | Limit number of examples (-1 for all) |

### Authors
This environment has been put together by:

Shamus Sim Zi Yang - ([@ss8319](https://github.com/ss8319))

### Credits 
Dataset:

```bibtex
@inproceedings{zhang2025casereportbench,
title={CaseReportBench: An LLM Benchmark Dataset for Dense Information Extraction in Clinical Case Reports},
author={Zhang, Xiao Yu Cindy and Ferreira, Carlos R. and Rossignol, Francis and Ng, Raymond T. and Wasserman, Wyeth and Zhu, Jian},
booktitle={Proceedings of the Sixth Conference on Health, Inference, and Learning},
series={Proceedings of Machine Learning Research},
volume={287},
pages={527--542},
year={2025},
publisher={PMLR}
}
```
