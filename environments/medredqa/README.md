# medredqa

### Overview
- **Environment ID**: `medredqa`
- **Short description**: MedRedQA dataset from https://data.csiro.au/collection/csiro:62454
- **Tags**: 

### Datasets
- **Primary dataset(s)**: MedRedQA - large non-factoid English consumer Question Answering (QA) dataset containing 51,000 pairs of consumer questions and their corresponding expert answers. This dataset is useful for bench-marking or training systems on more difficult real-world questions and responses which may contain spelling or formatting errors, or lexical gaps between consumer and expert vocabularies.
- **Source links**: https://huggingface.co/datasets/bagga005/medredqa
- **Split sizes**: 40.7k (train) / 5.1k (val) / 5k (test)

### Task
- **Type**: single-turn
- **Parser**: JudgeRubric
- **Rubric overview and Example**: 
Evaluates model opinion and recommendation against ground truth (provided by a medical professional) opinion and recommendation. Judge breaks ground truth into atomic facts (inspired by FactSore) and assigns a score based on occurance of each atomic fact in model output.

#### Sample Evaluation
##### User Query
28M. Small “grain of rice” under the skin on the shaft of my penis
Age - 28
Sex - M
Height - 6’2”
Weight - 230lbs 
Race - White
Duration of complaint - 3-4 months
28M. I’ve already scheduled an appointment with my PCP, but the appointment is not for a couple of weeks. Was hoping to get an opinion or thought about what this may before then. 
I have a small, grain of rice sized lump on the shaft of my penis near the dorsal vein. It doesn’t hurt, it popped up around 3-4 months ago as far as I remember, and it does not seem to be getting larger or smaller. I can’t see it while flaccid, so it is definitely under the skin. 
I have no other relevant conditions that I’m aware of. Currently taking medications that include levothyroxine, loratadine, and Baclofen. 
Here is a picture for reference:
##### Professional Opinion and Recommendation:
The small spots may be Fordyce spots but the  lump in the circle is not. If it is firm and non compressible then it is probably an epidermoid cyst. If it can be squashed flat it may be a blood vessel. It doesn’t look worrying and if it has not changed then there is a low likelihood of being serious.

##### Model Opinion and Recommendation:
Based on the information you provided, the small, painless, stable lump under the skin of your penile shaft is most likely a benign lesion such as an epidermoid cyst or a normal vein irregularity. Since it has been present for several months without change or symptoms, this is reassuring. 

Your plan to see your primary care provider is appropriate. In the meantime, avoid manipulating or irritating the area, monitor for any changes (growth, pain, ulceration, discharge), and seek urgent care if these occur.

At your appointment, your provider can examine the lump and, if needed, refer you to a urologist or dermatologist for further evaluation or possible removal.

If you develop new symptoms such as pain, rapid growth, redness, or ulceration before your appointment, seek medical evaluation sooner.

Overall, keep track of any changes and follow up as scheduled.

##### Judge reasoning and Score:
Atomic Facts and matching 
Fact: The small spots may be Fordyce spots., Score: False
Fact: A firm and non-compressible lump is probably an epidermoid cyst., Score: True
Fact: A firm, noncompressible lump is likely an epidermoid cyst., Match: True
Fact: If the lump doesn’t look worrying and has not changed, there is a low likelihood of it being serious., Score: True
Total Matches: 2, Total Facts: 4, Score: **0.5**


### Environment Arguments
Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str | `"gpt-4o-mini"` | Model to use for LLM-as-a-Judge evaluation |
| `judge_base_url` | str | `None` | Optional base URL for judge model API |
| `judge_api_key` | str | `None` | Optional API key for judge model (defaults to OPENAI_API_KEY env var) |
| `use_think` | bool | `False` | Whether to use ThinkParser with `<think>...</think>` tags and `\boxed{}` format for reasoning models |

### Usage Examples

Standard evaluation (free-form responses):

```bash
uv run vf-eval medredqa -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Verbose with saved output:

```bash
uv run vf-eval medredqa -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -s -v
```

Evaluation with reasoning model (with think tags and boxed answers):

```bash
uv run vf-eval medredqa \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"use_think": true}' \
  -s
```

Custom judge model:

```bash
uv run vf-eval medredqa \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"judge_model": "gpt-4o", "use_think": true}' \
  -s
```

### Authors
This environment has been put together by:

Kunal Bagga - ([@bagga005](https://github.com/bagga005))