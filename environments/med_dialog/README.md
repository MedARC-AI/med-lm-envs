# MedDialog (English)

### Overview
- **Environment ID**: `med_dialog`
- **Short description**: MedDialog is a benchmark of real-world doctor-patient conversations focused on health-related concerns and advice. Each dialogue is paired with a one-sentence summary that reflects the core patient question or exchange. The benchmark evaluates a model's ability to condense medical dialogue into concise, informative summaries.

### Dataset
- **Split sizes**:
  - Train: 205,973
  - Valid: 25,746
  - Test: 25,750
- **Source**:
  - [MedDialog: a large-scale medical dialogue dataset](https://arxiv.org/abs/2004.03329) (Chen et al., 2020)
  - Preprocessing by MedHELM following [BioBART](https://arxiv.org/abs/2204.03905) (Yuan et al., 2022)
  - Original dataset: [Medical-Dialogue-System](https://github.com/UCSD-AI4H/Medical-Dialogue-System) (Chen et al., 2020)

### Task
- **Type**: Single-Turn
- **Rubric overview**: JudgeRubric (LLM-as-a-Judge evaluation using prompts adapted from MedHELM)
- **Evaluation dimensions**:
  - **Accuracy** (1-5): Does the summary correctly capture the main medical issue and clinical details?
  - **Completeness** (1-5): Does the summary include all important medical information?
  - **Clarity** (1-5): Is the summary easy to understand for clinical use?

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval med_dialog
```

Use a custom judge model:

```bash
uv run vf-eval med_dialog -m gpt-4.1-mini --env-args '{"judge_model": "gpt-5-mini"}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The environment defaults to using `gpt-4o-mini` as the judge model.

### Environment Arguments
Document any supported environment arguments and their meaning:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cache_dir` | str \| Path \| None | `~/.cache/meddialog` | Local directory to cache downloaded datasets. Can also be set via `MEDDIALOG_CACHE_DIR` environment variable. |
| `judge_model` | str | `"gpt-4o-mini"` | Model identifier for the LLM judge evaluating summaries |
| `judge_base_url` | str \| None | `None` | Custom API base URL for judge model (defaults to OpenAI API) |
| `judge_api_key` | str \| None | `None` | API key for judge model. Falls back to `JUDGE_API_KEY` environment variable if not provided |

### Results Dataset Structure
#### Core Evaluation Fields

- **`prompt`** - The input conversation presented to the model (list of message objects with `role` and `content`)
- **`completion`** - The model's generated summary (list of message objects)
- **`reward`** - Overall score from 0.0 to 1.0, calculated as the average of normalized dimension scores: `(accuracy/5 + completeness/5 + clarity/5) / 3`

#### Example Metadata (`info`)
Contains all the MedDialog-specific information about each dialogue:

- **`id`** - Unique identifier for the dialogue
- **`conversation`** - The full patient-doctor conversation text
- **`reference_response`** - Gold standard one-sentence summary
- **`subset`** - Either `"healthcaremagic"` or `"icliniq"`
- **`index`** - Original index in the source dataset

#### Notes

- The `question` field in the dataset maps to the full conversation text
- The `answer` field contains the gold standard summary (also available as `reference_response` in `info`)
- Scores are normalized to 0-1 by dividing each dimension score (1-5) by 5 and averaging across dimensions
- If judge response parsing fails, dimension scores default to `None` and do not contribute to the final reward

### Dataset Examples

```
Patient: I get cramps on top of my left forearm and hand and it causes my hand and
fingers to draw up and it hurts. It mainly does this when I bend my arm. I ve been
told that I have a slight pinch in a nerve in my neck. Could this be a cause? I don t
think so.

Doctor: Hi there. It may sound difficult to believe it, but the nerves which supply
your forearms and hand, start at the level of spinal cord and on their way towards the
forearm and hand regions which they supply, the course of these nerves pass through
difference fascial and muscular planes that can make them susceptible to entrapment
neuropathies...

Summary: Could painful forearms be related to pinched nerve in neck?
```

```
Patient: Hello doctor, We are looking for a second opinion on my friend's MRI scan of
both the knee joints as he is experiencing excruciating pain just above the patella.
He has a sudden onset of severe pain on both the knee joints about two weeks ago...

Doctor: Hi. I viewed the right and left knee MRI images. Left knee: The MRI, left knee
joint shows a complex tear in the posterior horn of the medial meniscus area and mild
left knee joint effusion...

Summary: My friend has excruciating knee pain. Please interpret his MRI report
```

### References

**MedDialog Dataset**
```bibtex
@misc{chen2020meddiag,
  title={MedDialog: a large-scale medical dialogue dataset},
  author={Chen, Shu and Ju, Zeqian and Dong, Xiangyu and Fang, Hongchao and Wang, Sicheng and Yang, Yue and Zeng, Jiaqi and Zhang, Ruisi and Zhang, Ruoyu and Zhou, Meng and Zhu, Penghui and Xie, Pengtao},
  publisher = {arXiv},
  year={2020},
  url = {https://arxiv.org/abs/2004.03329},
}
```