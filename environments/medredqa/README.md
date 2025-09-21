# medcasereasoning

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
- **Rubric overview**: 
Evaluates model opinion and recommendation against ground truth (provided by a medical professional) opinion and recommendation. Judge determines number of points made in ground truth and percentage of that covered in models opinion and recommendation.

Here is a sample
Professional Opinion and Recommendation:
`
This topic can be a little sensitive.  Without knowing more of your background, I'd advise going to the doctor to differentiate a UTI from an STI or some other infection or other cutaneous condition like bacterial vaginosis.  STIs in particular can progress to pelvic inflammatory disease which has the potential to cause you significant long term issues as well as discomfort.\n\nUltimately, if it is a UTI, it might go away on its own or it might not.  The risk is yours to take.  Is there any particular reason you don't want to go to a doctor?  The basic tests that they might consider ordering (e.g. swabs, urine sample) can often be self-collected.  They may recommend a visual inspection +\/- examination of the area.  You can very reasonably ask for a female doctor or specialised nurse at any point.
`
Model Opinion and Recommendation:
`
Thank you for sharing your symptoms and concerns. Let's go through your situation step-by-step.\n\n1. *Symptoms*: You mentioned having orange-ish urine, blood when wiping, and a sensation of discomfort when urinating. These symptoms are commonly associated with a urinary tract infection (UTI). The presence of blood (hematuria) is particularly concerning and should not be overlooked.\n\n2. *Potential Risks*: While some mild UTIs can resolve on their own, there is a risk that if left untreated, the infection could progress to involve the kidneys or lead to more severe complications. This is especially true if you are sexually active, as UTIs can occur more frequently in sexually active individuals.\n\n3. *Diagnosis*: A healthcare provider can perform a urine test to confirm if you have a UTI and determine the appropriate treatment. Self-diagnosing and treating UTIs without professional guidance can be problematic. \n\n4. *Antibiotics*: If it is indeed a UTI, antibiotics are typically required to effectively treat the infection. Without treatment, symptoms may worsen or lead to complications. \n\n5. *Recommendations*: Given your symptoms, especially the presence of blood and the discomfort while urinating, I strongly recommend that you see a doctor as soon as possible. They can provide a proper diagnosis and prescribe the necessary treatment. It\u2019s important to address these symptoms promptly to prevent any potential complications.\n\nIn conclusion, it is advisable not to wait and see if the symptoms go away on their own. Seeking medical attention is important for your health.
`
Judge reasoning and Score:
`
To evaluate the similarity between the certified medical professional's opinion and recommendations and the predicted opinion and recommendations, I'll break down the unique points made in the ground truth and compare them with those in the response.\n\n### Ground Truth Opinion and Recommendations Points:\n1. *Urge to see a doctor*: Advises going to the doctor to differentiate between a UTI and other potential conditions (like STIs).\n2. *Potential risks of untreated UTI*: Provides information about the risks of an untreated UTI, mentioning that it might resolve on its own, but the risk is significant.\n3. *Questioning reasons for avoiding doctor visit*: Asks if there\u2019s a particular reason to avoid seeing a doctor.\n4. *Testing and examination*: Mentions the basic tests that may be ordered, including urine samples and physical inspection.\n5. *Comfort in seeking care*: Encourages the patient to request a female doctor or specialized nurse, emphasizing patient comfort during the exam.\n\n### Predicted Opinion and Recommendations Points:\n1. *Symptom acknowledgment*: Acknowledges the symptoms shared.\n2. *Potential risks of untreated UTI*: Discusses risks of untreated infections and the possibility of complications, especially since the person is sexually active.\n3. *Recommendation to see a doctor*: Strongly recommends seeing a doctor for diagnosis and treatment.\n4. *Importance of not self-diagnosing*: Highlights the issue with self-diagnosing and the necessity for urine tests.\n5. *Antibiotic necessity if UTI confirmed*: Mentions that if it is a UTI, antibiotics are needed.\n\n### Comparison and Similarity Score Calculation:\n- Both pieces emphasize the *importance of seeing a doctor*, but the ground truth suggests asking why the individual might not want to go, which is not covered in the predicted response.\n- Both mention the *risks associated with untreated UTIs*. The predicted response elaborates on complications tied to sexual activity.\n- The predicted response underscores the necessity for *professional diagnosis and treatment*, directly linking this to potential need for antibiotics if a UTI is confirmed.\n- The predicted response does not cover the specific point of *testing and examination* procedures as described in the ground truth.\n- Neither response emphasizes the user\u2019s comfort when asking for medical help in terms of seeking a female doctor or specialized nurse.\n\n### Unique Points Matching Analysis:\n- Ground Truth: 5 unique points\n- Predicted Response: 4 points match.\n\nSince 4 out of 5 unique points from the ground truth are addressed in the predicted opinion and recommendations, the similarity percentage is 80%.\n\nFinal Score:\n*SCORE:0.8*
`
### Quickstart
Run evaluation with configured model and sampling:

```bash
uv run vf-eval medredqa  -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7
```

Verbose with saved output

```bash
uv run vf-eval medredqa  -m gpt-4.1-mini -n 20 -r 3 -t 1024 -T 0.7 -s -v
```