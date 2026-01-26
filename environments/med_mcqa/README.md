# MEDMCQA

Evaluation environment for the MEDMCQA dataset.

### Overview
- **Environment ID:** `med_mcqa`
- **Short description:** Single-turn medical multiple-choice QA
- **Tags:** medical, single-turn, multiple-choice, train, eval

### Datasets
- **Primary dataset(s):** MedMCQA (HF datasets)
- **Source links:** [lighteval/med_mcqa](https://huggingface.co/datasets/lighteval/med_mcqa)
- **Split sizes:** Uses provided train and validation splits

### Task
- **Type:** Single-turn
- **Parser:** `Parser` (standard) or `ThinkParser` (if using reasoning mode) depending on `use_think`
- **Rubric overview:** Binary scoring (1.0 / 0.0), based on correct letter or answer text match.  
- **Reward function:** `accuracy` — returns 1.0 if the predicted answer matches, else 0.0.

### Model Input Format
Each example is formatted as a single-turn user message: 

```
Give a letter answer among A, B, C or D.
Question: {question}
A. {opa}
B. {opb}
C. {opc}
D. {opd}
Answer:
```

The model should respond with a letter choice (A–D).

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval med_mcqa
```

### Usage
To run an evaluation using `vf-eval` with the OpenAI API:

```bash
export OPENAI_API_KEY=sk-...
uv run vf-eval \
  -m gpt-4.1-mini \
  -n 5 \
  -s \
  med_mcqa
```
Replace `OPENAI_API_KEY` with your actual API key.

### Authors
This environment has been put together by:

Ratna Sagari Grandhi - ([@sagarigrandhi](https://github.com/sagarigrandhi))

### Credits 
Dataset:

```bibtex
@InProceedings{pmlr-v174-pal22a,
  title = 	 {MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering},
  author =       {Pal, Ankit and Umapathi, Logesh Kumar and Sankarasubbu, Malaikannan},
  booktitle = 	 {Proceedings of the Conference on Health, Inference, and Learning},
  pages = 	 {248--260},
  year = 	 {2022},
  editor = 	 {Flores, Gerardo and Chen, George H and Pollard, Tom and Ho, Joyce C and Naumann, Tristan},
  volume = 	 {174},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {07--08 Apr},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v174/pal22a/pal22a.pdf},
  url = 	 {https://proceedings.mlr.press/v174/pal22a.html},
  abstract = 	 {This paper introduces MedMCQA, a new large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions. More than 194k high-quality AIIMS & NEET PG entrance exam MCQs covering 2.4k healthcare topics and 21 medical subjects are collected with an average token length of 12.77 and high topical diversity. Each sample contains a question, correct answer(s), and other options which requires a deeper language understanding as it tests the 10+ reasoning abilities of a model across a wide range of medical subjects & topics. A detailed explanation of the solution, along with the above information, is provided in this study.}
}
```
