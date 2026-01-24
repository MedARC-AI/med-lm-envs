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
- **Methodology**: Supports the paper’s **UCP** and **UGP** settings with **FS**/**ZS** prompting.

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

```bash
# Install the environment
vf-install casereportbench

# Run evaluation (default: UCP + Few-Shot, all 13 categories)
export OPENAI_API_KEY=sk-...
vf-eval casereportbench -m gpt-4o-mini -n 5 -s
```

**Note**: With default UCP mode, this evaluates all 13 categories separately (13 prompts per case report).

### Paper Configurations

```bash
# UCP (Uniform Category-Specific) + Few-Shot [default]
vf-eval casereportbench -m gpt-4o-mini -a '{"method":"UCP","prompting":"FS"}'

# UCP + Zero-Shot
vf-eval casereportbench -m gpt-4o-mini -a '{"method":"UCP","prompting":"ZS"}'

# UGP (Unified Global Prompting) + Few-Shot
vf-eval casereportbench -m gpt-4o-mini -a '{"method":"UGP","prompting":"FS"}'

# UGP + Zero-Shot
vf-eval casereportbench -m gpt-4o-mini -a '{"method":"UGP","prompting":"ZS"}'
```

**Note**: FCSP (Filtered Category-Specific Prompting) is not implemented due to missing subheading metadata in the HuggingFace dataset.

### Evaluate Specific Categories

```bash
# Single category
vf-eval casereportbench -m gpt-4o-mini -a '{"task":"Neuro"}'

# Limit examples
vf-eval casereportbench -m gpt-4o-mini -a '{"max_examples":10}'
```

### Metrics

| Metric | Description |
|--------|-------------|
| `reward` (TSR) | **Primary metric**. Token Set Ratio normalized by token length (0–1) |
| `bleu1` | 1-gram precision |
| `bleu4` | 4-gram precision |
| `rougeL` | Longest Common Subsequence overlap |
| `omission` | Penalty for missing expert-labeled extractions |
| `hallucination` | Penalty for extracting when expert found nothing |

### Environment Arguments

| Argument | Type | Default | Options |
|----------|------|---------|---------|
| `task` | str | `"all"` | `"Neuro"`, `"CVS"`, `"RESP"`, `"GI"`, `"GU"`, `"MSK"`, `"DERM"`, `"EENT"`, `"LYMPH"`, `"ENDO"`, `"History"`, `"Pregnancy"`, `"Vitals_Hema"`, or `"all"` |
| `method` | str | `"UCP"` | `"UCP"` (per-category prompt), `"UGP"` (unified prompt) |
| `prompting` | str | `"FS"` | `"FS"` (few-shot), `"ZS"` (zero-shot) |
| `max_examples` | int | `-1` | Number of examples (-1 for all 138) |

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
