# MTSamples Medical Specialty Classification

A medical specialty classification benchmark using the MTSamples dataset.

## Dataset

- **Source**: [MTSamples](https://mtsamples.com)
- **HuggingFace**: [NickyNicky/medical_mtsamples](https://huggingface.co/datasets/NickyNicky/medical_mtsamples)
- **Size**: ~5,000 medical transcription samples
- **License**: CC0 (Public Domain)
- **Specialties**: 40 medical specialties

## Task

**Medical Specialty Classification (5-way MCQ)**

Given a clinical transcription, identify the correct medical specialty from 5 options (1 correct + 4 distractors).

### Specialties Covered

The dataset covers 40 medical specialties including:
- Allergy / Immunology
- Cardiovascular / Pulmonary
- Dermatology
- Emergency Room Reports
- Gastroenterology
- General Medicine
- Neurology
- Obstetrics / Gynecology
- Orthopedic
- Psychiatry / Psychology
- Radiology
- Surgery
- And 28 more...

## Usage

```bash
# Basic evaluation
uv run medarc-eval mtsamples -m gpt-4.1-mini -n 50 -s

# With shuffled answers
uv run medarc-eval mtsamples -m gpt-4.1-mini --shuffle-answers -n 50 -s

# Use descriptions instead of full transcriptions
uv run medarc-eval mtsamples -m gpt-4.1-mini --use-description -n 50 -s

# Custom number of options
uv run medarc-eval mtsamples -m gpt-4.1-mini --num-options 4 -n 50 -s
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | str | "test" | Dataset split ("train" or "test") |
| `num_options` | int | 5 | Number of MCQ options |
| `shuffle_answers` | bool | True | Whether to shuffle answer options |
| `shuffle_seed` | int | 1618 | Random seed for shuffling |
| `use_description` | bool | False | Use brief description instead of full transcription |
| `max_transcription_length` | int | 4000 | Maximum transcription length |
| `system_prompt` | str | None | Optional system prompt override |

## Evaluation

- **Metric**: Multiple-choice accuracy
- **Parser**: JSON parser expecting `{"answer": "X"}` format
- **Grading**: Uses `multiple_choice_accuracy` from medarc_verifiers

## Example Prompt

```
You are a medical expert. Based on the following clinical transcription, 
identify the most appropriate medical specialty.

Transcription:
SUBJECTIVE: This 23-year-old white female presents with complaint of allergies...

Options:
A. Neurology
B. Allergy / Immunology
C. Cardiology
D. Dermatology
E. Gastroenterology

Provide your answer in JSON format: {"answer": "X"} where X is the letter.
```

## Citation

```bibtex
@misc{mtsamples,
  title={MTSamples - Medical Transcription Samples},
  url={https://mtsamples.com},
  note={Public domain medical transcription dataset}
}
```

## License

- **Dataset**: CC0 (Public Domain)
- **Code**: MIT License
