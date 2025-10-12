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
Thank you for sharing detailed information about your symptom. Based on your description of a small, painless, stable-sized lump the size of a grain of rice under the skin on the shaft of your penis near the dorsal vein, here are some possible considerations:

1. **Pearly Penile Papules (PPP)**
These are small, dome-shaped or filiform papules that typically appear around the corona (the rim of the glans) but can sometimes be near the shaft. They are benign, normal anatomical variants and are usually asymptomatic. However, they tend to be multiple rather than a single lump.

2. **Epidermoid cyst (sebaceous cyst)**
A small, benign cyst that can arise under the skin. Usually painless and slow-growing. They feel like a small, firm lump under the skin. Sometimes they can have a central punctum.

3. **Peyronie's disease nodule**
This is a fibrous plaque or nodule under the skin of the penile shaft. Usually associated with penile curvature or pain during erection, but early plaques can be painless and stable. Usually feels firm.

4. **Lymphocele or enlarged lymph node**
Less likely on the shaft itself; usually lymph nodes are in the groin area.

5. **Other benign lesions**
Such as angiomas, fibromas, or lipomas (fatty lumps).

6. **Infectious causes**
Small bumps from infections (warts, molluscum contagiosum) usually have a characteristic appearance and often more than one lesion.

7. **Malignant lesions**
Very rare in this context, especially if stable, painless, and small, but any new persistent growth should be evaluated.
Given that your lump is stable in size, painless, and you have no other symptoms such as redness, discharge, pain, or rapid growth, it is likely benign.

**Recommendations:**
Since you already have an appointment with your PCP, it is appropriate to wait for a professional evaluation.
Monitor for any changes in size, color, pain, or new symptoms like discharge. If such changes occur, seek medical attention sooner.
Avoid manipulating or squeezing the lump as it may cause irritation or infection.
Your PCP may examine the lesion, and if necessary, order an ultrasound or refer you to a dermatologist or urologist for further evaluation.
If the lump is bothersome or cosmetically concerning, options such as excision or biopsy can be discussed with your doctor.
If you develop any urgent symptoms such as rapid enlargement, severe pain, ulceration, bleeding, or signs of infection (redness, warmth, fever), you should seek medical care promptly.
In summary, the lump you describe sounds like it could be a benign cyst or other harmless lesion, but confirmation by your PCP is important. Your planned appointment is a good step, and you should keep an eye on any changes in the lesion until then.

##### Judge reasoning and Score:
Atomic Facts and matching 
Fact: Small spots may be Fordyce spots., Match: False
Fact: The circled lump is unlikely to be a Fordyce spot., Match: True
Fact: A firm, noncompressible lump is likely an epidermoid cyst., Match: True
Fact: A lump that can be flattened may be a blood vessel., Match: False
Fact: The lesion’s appearance is not concerning., Match: True
Fact: If the lesion has not changed over time, it is less likely to be serious., Match: True
Total Matches: 4, Total Facts: 6, Score: **0.66**


### Using
Run evaluation with configured model and sampling:

```bash
uv run vf-eval medredqa  -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Verbose with saved output

```bash
uv run vf-eval medredqa  -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -s -v
```
