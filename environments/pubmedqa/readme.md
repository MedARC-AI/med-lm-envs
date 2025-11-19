# pubmedqa environment

### Overview
- **Environment ID**: `pubmedqa`
- **Short description**: pubmedqa test set from Jin et al. 2019

### Dataset

- **Primary dataset(s)**: PubMedQA 1k expert-annotated QA instances (500 instances test subset)

The 1k instances are available directly on the [pubmedqa githib repoisitory](https://github.com/pubmedqa/pubmedqa/blob/master/data/ori_pqal.json), and is available on the huggingface hub as ['qiaojin/PubMedQA'](https://huggingface.co/datasets/qiaojin/PubMedQA), which is also the dataset used by InspectEval.

The IDs of the 500 test instances are mapped in [test_ground_truth.json](https://github.com/pubmedqa/pubmedqa/blob/master/data/test_ground_truth.json).

This should correspond to the 'pqal_test_set.json' (from the bigbio/pubmed_qa [pqal.zip](https://huggingface.co/datasets/bigbio/pubmed_qa/blob/main/pqal.zip) file), resulting after [splitting](https://github.com/pubmedqa/pubmedqa/blob/master/preprocess/split_dataset.py) the 1k set.


### Task
- **Type**: single-turn
- **Parser**: InspectEval-style MCQ parser
- **Rubric**: Classification-based rubric (1 point for correct answer, 0 points for incorrect answer). Three choices: (yes/no/maybe)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval pubmedqa

uv run test_pubmedqa.py --num_examples 4 --model mock-model

# MISTRAL_API_KEY needs to be set 
uv run test_pubmedqa.py --num_examples 4 --model mistral-small --no-mock

```

### Model Input Format

By default, no system prompt is given. Each item is formatted as a single-turn prompt, formatted as a user chat message (by default). 

The message contents look like:
```
Answer the following multiple choice question about medical knowledge given the context.
The entire content of your response should be of the following format: 'ANSWER: $LETTER'
(without quotes) where LETTER is one of A, B, C.

Context: 
(BACKGROUND) Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.
(RESULTS) The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.

Question: Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?

A) yes
B) no
C) maybe
```

Choices to letter assignments are fixed and always presented in identical order.



### Credits

For the original publication, cite:
```bibtex
@article{jin2019pubmedqa,
  title={Pubmedqa: A dataset for biomedical research question answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William W and Lu, Xinghua},
  journal={arXiv preprint arXiv:1909.06146},
  year={2019}
}
```

The evaluation code draws strongly on https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/pubmedqa for consistency across implementations. 

### Authors
This environment has been put together by:

Robert Scholz - ([@rscgh](https://github.com/rscgh))