##################

# ALL PROMPTS ARE FROM THE PAPER (Source: https://arxiv.org/abs/2401.14493)

##################


def _prompt_for_has_contradiction(question: str, llm_answer: str, answer: str) -> str:
    return f"""
# OVERALL INSTRUCTIONS
- You have a deep understanding of logical relationships, such as entailment and contradiction, to evaluate given triplets of (question, premise, hypothesis).

# TASK INSTRUCTIONS
Your goal is to determine whether the Premise effectively contradicts the corresponding Hypothesis. Carefully analyze each triplet, focusing on details.
- If the premise and the hypothesis are unrelated or lack sufficient evidence to ascertain their truthfulness, label your answer as False.
- be vigilant in identifying cases where the premise doesn't rule out the possibility of an entity (e.g., vaccine, symptom) appearing in the hypothesis. In such cases, classify the answer as False.
- Approach each question methodically, considering the step-by-step process outlined below.
- Put your Final Answer in the format: Final Answer: \\boxed{{True}} or Final Answer: \\boxed{{False}}

# INPUT DATA
Question: What does trich test for? Let's think step by step.
Premise: The term "trich test" can refer to two different medical tests, depending on the context. Here are the two possibilities:
Trichomoniasis Test: Trichomoniasis is a sexually transmitted infection (STI) caused by the parasite Trichomonas vaginalis.
The trichomoniasis test, also known as a trich test or trichomonas test, is used to detect the presence of this parasite in the body. The test is typically performed on a sample of vaginal discharge in women or urine in men.
Trichogram: A trichogram is a diagnostic test used to evaluate hair loss and assess the health and condition of hair follicles. It involves plucking a small number of hairs from the scalp and examining them under a microscope.
It's important to note that without additional context, it's difficult to determine which specific test you are referring to.
Hypothesis: Trichamoniasis- a parasitic infection that can cause your symptoms.
Answer: According to the premise "trich test" refer to two different medical tests. A Trichamoniasis test is one of them, which is used to detect this parasite's presence.
As stated in the hypothesis, the trich test is used to diagnose parasitic infections. Ths premise entails the hypothesis. The answer is False.
Final Answer: \\boxed{{False}}
###
Question: Can diabetics eat sweets? Let's think step by step.
Premise: Individuals with diabetes are recommended to limit their consumption of sweets to one or two times per week.
It is also suggested being selective with desserts and to focus on foods with a low glycemic index, such as high fiber foods like whole grains and legumes, as well as certain lower sugar fruits like berries, melons, and apples.
Hypothesis: It is recommended that diabetics avoid sweets.
Answer: The premise suggests that diabetics can eat sweets but limit their consumption. According to the hypothesis diabetics should avoid sweets.
Diabetics are allowed to consume sweets according to the premise, but they are prohibited according to the hypothesis. There is a contradiction between the premise and the hypothesis. The answer is True.
Final Answer: \\boxed{{True}}
###
Question: 25 yo female with right lower abdominal pain, what might be causing it? Let's think step by step.
Premise: Right lower abdominal pain in a 25-year-old female could be caused by a variety of medical conditions.
Some potential causes include: Ovarian cyst: a fluid-filled sac on the ovary - Ectopic pregnancy: a pregnancy that occurs outside the uterus.
Hypothesis: possible cause for right lower abdominal pain in a young female can be Appendicitis.
Answer: The premise lists several potential causes of right lower abdominal pain in a 25-year-old female, not including appendicitis. The hypothesis states that Appendicitis could be a cause of right lower abdominal pain in a young female.
There is no direct contradiction between the premise and the hypothesis, as the premise does not exclude the possibility of appendicitis as the cause of the pain. The answer is False.
Final Answer: \\boxed{{False}}
###
Question: Can a headache last longer than a few days? Let's think step by step.
Premise: Yes, it is possible. If you are experiencing a headache that lasts longer than a few days, it is important to see a doctor to get the appropriate treatment.
This will help to relieve the pain and prevent any further complications.
Hypothesis: It is not a cause for concern if a headache lasts longer than a few days.
Answer: This premise acknowledges that a headache can last for several days, but emphasizes that seeing a doctor to prevent further complications is important. According to this hypothesis, headaches lasting longer than a few days are not cause of concern.
There is a contradiction between the premise and hypothesis due to the premise emphasizing the importance of seeking medical consultation, while the hypothesis posits that there is no cause for concern. The answer is True.
Final Answer: \\boxed{{True}}
###
Question: {question} Let's think step by step.
Premise: {llm_answer}
Hypothesis: {answer}
Answer:
Final Answer: \\boxed{{True|False}}
"""


def _prompt_for_is_entails(question: str, llm_answer: str, answer: str) -> str:
    return f"""


# OVERALL INSTRUCTIONS
- You have a deep understanding of logical relationships, such as entailment and contradiction, to evaluate given triplets of (question, premise, hypothesis).

# TASK INSTRUCTIONS
Your goal is to determine whether the Premise effectively entails the corresponding Hypothesis. Carefully analyze each triplet, focusing on details.
- If the premise disagrees with, is unrelated to, or does not support the hypothesis, there is not enough evidence to determine whether it is true, and so you answer should be False.
- Approach each question methodically, considering the step-by-step process outlined below.
- Put your Final Answer in the format: Final Answer: \\boxed{{True}} or Final Answer: \\boxed{{False}}


# INPUT DATA
Question: What does trich test for? Let's think step by step.
Premise: The term "trich test" can refer to two different medical tests, depending on the context. Here are the two possibilities:
Trichomoniasis Test: Trichomoniasis is a sexually transmitted infection (STI) caused by the parasite Trichomonas vaginalis.
The trichomoniasis test, also known as a trich test or trichomonas test, is used to detect the presence of this parasite in the body. The test is typically performed on a sample of vaginal discharge in women or urine in men.
Trichogram: A trichogram is a diagnostic test used to evaluate hair loss and assess the health and condition of hair follicles. It involves plucking a small number of hairs from the scalp and examining them under a microscope.
It's important to note that without additional context, it's difficult to determine which specific test you are referring to.
Hypothesis: Trichamoniasis- a parasitic infection that can cause your symptoms.
Answer: According to the premise "trich test" refer to two different medical tests. A Trichamoniasis test is one of them, which is used to detect this parasite's presence.
As the hypothesis suggested, the trich test is used to diagnose parasitic infections. The premise entails the hypothesis. The answer is True.
Final Answer: \\boxed{{True}}
###
Question: Can diabetics eat sweets? Let's think step by step.
Premise: Individuals with diabetes are recommended to limit their consumption of sweets to one or two times per week.
It is also suggested to be selective with desserts and to focus on foods with a low glycemic index, such as high fiber foods like whole grains and legumes, as well as certain lower sugar fruits like berries, melons, and apples.
Hypothesis: After eating sweets, must monitor blood and sugar level
Answer: The premise suggests that diabetics can eat sweets but limit their consumption. The hypothesis highlights the necessity of monitor blood and sugar after eating sweets.
There is no relationship between the premise and hypothesis, therefore they do not entail one another. The answer is False.
Final Answer: \\boxed{{False}}
###
Question: Can diabetics eat sweets? Let's think step by step.
Premise: Individuals with diabetes are recommended to limit their consumption of sweets to one or two times per week.
It is also suggested being selective with desserts and to focus on foods with a low glycemic index, such as high fiber foods like whole grains and legumes, as well as certain lower sugar fruits like berries, melons, and apples.
Hypothesis: It is recommended that diabetics avoid sweets.
Answer: The premise suggests that diabetics can eat sweets but limit their consumption. According to the hypothesis diabetics should avoid sweets.
The premise allows diabetics to consume sweets in moderate consumption, while the hypothesis prohibits them. There premise don't entail the hypothesis. The answer is False.
Final Answer: \\boxed{{False}}
###
Question: What is the best hypertension treatment for patients who are also have Crohn's disease? Let's think step by step.
Premise: For patients with Crohn's disease and hypertension, the recommended treatment is a combination of lifestyle changes and medication. The ACC/AHA recommends initiation of antihypertensive drug therapy at a BP \u2265130/80 mm Hg for adults with hypertension.
It is also important to monitor your blood pressure regularly to make sure that it is under control.
Hypothesis: reducing sodium intake, are the first-line treatment for hypertension in individuals with  Crohn's disease
Answer: The premise suggests that the recommended treatment for patients with diabetes and hypertension is a combination of lifestyle changes and medication, including antihypertensive drug therapy. The hypothesis focuses on reducing sodium intake as the first-line treatment.
A reduction in sodium intake could be a part of the lifestyle changes, but since it is not mentioned in the premise, the premise do not entail the hypothesis. The answer is False.
Final Answer: \\boxed{{False}}
###
Question: 25 yo female with right lower abdominal pain, what might be causing it? Let's think step by step.
Premise: Right lower abdominal pain in a 25-year-old female could be caused by a variety of medical conditions.
Some potential causes include: - Appendicitis: inflammation of the appendix - Appendiceal abscess: a collection of pus in the appendix - Ovarian cyst: a fluid-filled sac on the ovary - Ectopic pregnancy: a pregnancy that occurs outside the uterus.
Hypothesis: possible cause for right lower abdominal pain in a young female can be Appendicitis.
Answer: The premise lists several potential causes of right lower abdominal pain in a 25-year-old female, including appendicitis. The hypothesis states that Appendicitis could be a cause of right lower abdominal pain in a 25-year-old female.
Both the premise and hypothesis mention appendicitis as a possible cause of pain, so the premise entails the hypothesis. The answer is True.
Final Answer: \\boxed{{True}}
###
Question: {question} Let's think step by step.
Premise: {llm_answer}
Hypothesis: {answer}
Answer:
Final Answer: \\boxed{{True|False}}
"""


FORMAT_INSTRUCTIONS = (
    "Respond with a single JSON object with this exact shape:\n"
    "{\n"
    '  "claims": ["<string>", "..."]\n'
    "}\n"
    "Rules:\n"
    "- Return only valid JSON. No prose, no Markdown/code fences.\n"
    "- Each item in 'claims' must be a string and self-contained.\n"
    '- If no claims, return {"claims": []}.\n'
)


def _decompose_free_form_answer(question: str, llm_answer: str) -> str:
    return f"""


# OVERALL INSTRUCTIONS
You are an expert in understanding logical relationships. This is a Semantic Content Unit (SCU) extraction task. Given a pair of
Question and Answer, your goal is to create a list of self-contained and concise claims. Each claim should be able to stand alone
and be independent of other claims. Your claims should encompass all the information present in the answer.

# TASK INSTRUCTIONS
- List of Possible Causes: For scenarios involving multiple entities like red flags, vaccines, symptoms, etc., generate separate
claims for each entity. This increases the number of claims.
- OR Claims: When medical entities are presented in an "OR" context, treat them as distinct claims.
- IF Claims: When an "if statement" is present, preserve the "if statement" context while creating the claim.
- XOR Claims: When entities have an XOR logical relationship (e.g., treatment options), create separate claims for each option.

# EXAMPLE CLAIM FORMAT - List Format: "Possible cause for [CONDITION] in [DEMOGRAPHIC] can be [ENTITY]."
- OR Format: "Possible causes include: [ENTITY X], [ENTITY Y], and [ENTITY Z]."
- OR Format: "The [CONTEXT] of treatments such as [TREATMENT X], [TREATMENT Y], and [TREATMENT Z], is not well established." - IF
Format: "[CONTEXT], please seek medical attention if [CONDITIONS]."
- XOR Format: "Either take [TREATMENT X] or [TREATMENT Y], but not both."
—–
{FORMAT_INSTRUCTIONS}
——
# TASK EXAMPLE
Question: I am a 33-year-old female with right lower abdominal pain, what could it be? Answer: Possible causes for right lower
abdominal pain in a young female are Appendicitis, Inflammatory bowel disease, Diverticulitis, Kidney stone, urinary tract infection,
Ovarian cyst or torsion, Ectopic pregnancy, Pelvic inflammatory disease, endometriosis. Please seek medical attention if the pain
is sudden and severe, does not go away, or gets worse, is accompanied by fever, nausea and vomiting, or if you have noticed blood
in urine or in stool.

Claims: [ Possible cause for right lower abdominal pain in a young female: Appendicitis, Possible cause for right lower abdominal
pain in a young female: Ovarian cyst or torsion, Possible cause for right lower abdominal pain in a young female: Ectopic pregnancy,
Possible cause for right lower abdominal pain in a young female: Pelvic inflammatory disease, Possible cause for right lower
abdominal pain in a young female: Kidney stone, Possible cause for right lower abdominal pain in a young female: Urinary tract
infection, Possible cause for right lower abdominal pain in a young female: Diverticulitis, Possible cause for right lower abdominal
pain in a young female: Inflammatory bowel disease, Possible cause for right lower abdominal pain in a young female: Endometriosis,
Please seek medical attention if the pain is sudden and severe, Please seek medical attention if the pain is accompanied by fever,
Please seek medical attention if the pain is accompanied by nausea and vomiting, Please seek medical attention if the pain is
accompanied by blood in urine, Please seek medical attention if the pain is accompanied by blood in stool, Possible cause for right
lower abdominal pain in a young female: Emotional stress ]

# TASK EXAMPLE
Question: So what does the non reactive mean for the hep a igm Answer: Hep A IgM refers to a specific type of antibody called
Immunoglobulin M (IgM) against the virus hepatitis A. When infected with hepatitis A, these antibodies are detectable at symptom
onset and remain detectable for approximately three to six months. These antibodies might also be detectable in the first month
after hepatitis A vaccination. A negative or non-reactive result means no IgM antibodies against hepatitis A found in your serum,
meaning the absence of an acute or recent hepatitis A virus infection.
Claims: [ A negative or non-reactive result means that there were no IgM antibodies against hepatitis A found in your serum, The
absence of IgM antibodies against hepatitis A in your serum indicates the absence of an acute or recent hepatitis A virus infection,
Hep A IgM refers to a specific type of antibodies called Immunoglobulin M (IgM) against the virus hepatitis A, These antibodies
might also be detectable in the first month after hepatitis A vaccination, These antibodies remain detectable for approximately
three to six months after infection, When infected with hepatitis A, these antibodies are detectable at the time of symptom onset ]

# TASK EXAMPLE
Question: What medications are contraindicated for a pregnant woman with ulcerative colitis? Answer: methotrexate (Otrexup, Rasuvo,
RediTrex) and thalidomide (Contergan, Thalomid) are both considered contraindicated for treatment of UC in pregnancy. possible
treatment for UC during pregnancy include low-risk drugs such as aminosalicylates (sulfasalazine and mesalamine), immunomodulators
(azathioprine, cyclosporine A ,6-mercaptopurine) and corticosteroids. Biological agents such as Infliximabl, Adalimumab, Vedolizumab
and Ustekinumab is generally avoided during pregnancy as their safety in pregnancy is not well established yet.
Claims: [ Methotrexate (Otrexup, Rasuvo, RediTrex) is contraindicated for treatment of ulcerative colitis in pregnancy, Thalidomide
(Contergan, Thalomid) is contraindicated for treatment of ulcerative colitis in pregnancy, Aminosalicylates (sulfasalazine and
mesalamine) are considered low-risk drugs for treatment of ulcerative colitis during pregnancy, Immunomodulators (azathioprine,
cyclosporine A, 6-mercaptopurine) are considered low-risk drugs for treatment of ulcerative colitis during pregnancy, Corticosteroids
are considered low-risk drugs for treatment of ulcerative colitis during pregnancy, Treatment for ulcerative colitis during pregnancy
with biological agents such as Adalimumab is generally avoided during pregnancy as their safety in pregnancy is not well established
yet, Treatment for ulcerative colitis during pregnancy with biological agents such as Vedolizumab is generally avoided during
pregnancy as their safety in pregnancy is not well established yet, Treatment for ulcerative colitis during pregnancy with biological
agents such as Infliximab is generally avoided during pregnancy as their safety in pregnancy is not well established yet, Treatment
for ulcerative colitis during pregnancy with biological agents such as Ustekinumab is generally avoided during pregnancy as their
safety in pregnancy is not well established yet, ]
# YOUR TASK
Question: {question}
Answer: {llm_answer}
Claims:


"""


# CURRENTLY NOT USED (Source: https://arxiv.org/abs/2401.14493)
def _build_eval_prompt(question: str, llm_answer: str, physician_answer: str) -> str:
    return f"""


# OVERALL INSTRUCTIONS
- You have a deep understanding of logical relationships, such as entailment and contradiction, to evaluate given triplets of
(question, premise, hypothesis).

# TASK INSTRUCTIONS
Your goal is to determine whether the Premise effectively contradicts the corresponding Hypothesis. Carefully analyze each triplet,
focusing on details. - If the premise and the hypothesis are unrelated or lack sufficient evidence to ascertain their truthfulness,
label your answer as False. - be vigilant in identifying cases where the premise doesn’t rule out the possibility of an entity (e.g.,
vaccine, symptom) appearing in the hypothesis. In such cases, classify the answer as False. - Approach each question methodically,
considering the step-by-step process outlined below.

# INPUT DATA
Question: What does trich test for? Let’s think step by step.
Premise: The term "trich test" can refer to two different medical tests, depending on the context. Here are the two possibilities:
Trichomoniasis Test: Trichomoniasis is a sexually transmitted infection (STI) caused by the parasite Trichomonas vaginalis. The
trichomoniasis test, also known as a trich test or trichomonas test, is used to detect the presence of this parasite in the body.
The test is typically performed on a sample of vaginal discharge in women or urine in men. Trichogram: A trichogram is a diagnostic
test used to evaluate hair loss and assess the health and condition of hair follicles. It involves plucking a small number of hairs
from the scalp and examining them under a microscope. It’s important to note that without additional context, it’s difficult to
determine which specific test you are referring to. Hypothesis: Trichamoniasis- a parasitic infection that can cause your symptoms.
Answer: According to the premise "trich test" refer to two different medical tests. A Trichamoniasis test is one of them, which is
used to detect this parasite’s presence. As stated in the hypothesis, the trich test is used to diagnose parasitic infections. Ths
premise entails the hypothesis. The answer is False.
###

Question: Can diabetics eat sweets? Let’s think step by step.
Premise: Individuals with diabetes are recommended to limit their consumption of sweets to one or two times per week. It is also
suggested being selective with desserts and to focus on foods with a low glycemic index, such as high fiber foods like whole grains
and legumes, as well as certain lower sugar fruits like berries, melons, and apples.
Hypothesis: It is recommended that diabetics avoid sweets.
Answer: The premise suggests that diabetics can eat sweets but limit their consumption. According to the hypothesis diabetics
should avoid sweets. Diabetics are allowed to consume sweets according to the premise, but they are prohibited according to the
hypothesis. There is a contradiction between the premise and the hypothesis. The answer is True.
###

Question: 25 yo female with right lower abdominal pain, what might be causing it? Let’s think step by step.
Premise: Right lower abdominal pain in a 25-year-old female could be caused by a variety of medical conditions.
Some potential causes include: Ovarian cyst: a fluid-filled sac on the ovary - Ectopic pregnancy: a pregnancy that occurs outside
the uterus.
Hypothesis: possible cause for right lower abdominal pain in a young female can be Appendicitis.
Answer: The premise lists several potential causes of right lower abdominal pain in a 25-year-old female, not including appendicitis.
The hypothesis states that Appendicitis could be a cause of right lower abdominal pain in a young female. There is no direct
contradiction between the premise and the hypothesis, as the premise does not exclude the possibility of appendicitis as the cause
of the pain. The answer is False.
###

Question: Can a headache last longer than a few days? Let’s think step by step.
Premise: Yes, it is possible. If you are experiencing a headache that lasts longer than a few days, it is important to see a doctor
to get the appropriate treatment.
This will help to relieve the pain and prevent any further complications.
Hypothesis: It is not a cause for concern if a headache lasts longer than a few days.
Answer: This premise acknowledges that a headache can last for several days, but emphasizes that seeing a doctor to prevent further
complications is important. According to this hypothesis, headaches lasting longer than a few days are not cause of concern. There
is a contradiction between the premise and hypothesis due to the premise emphasizing the importance of seeking medical consultation,
while the hypothesis posits that there is no cause for concern. The answer is True.
###

Question: {question} Let’s think step by step.
Premise: {llm_answer}
Hypothesis: {physician_answer}
Answer:


"""


def _llm_generation_prompt(question: str) -> str:
    return f"""

# TASK:
You are a medical expert and your task is to write a helpful human-like text and answer to the point.

# INSTRUCTIONS:
- Identify and address the patient’s focus of interest, but include relevant information even if the patient didn’t directly ask it.
- If relevant, provide information about specific cases, including children, pregnant women, allergies, or chronic conditions.
- If you don’t know the answer, answer ONLY ’I don’t know’.
- If the answer contains enumeration of factors/symptoms/conditions use bullet points to organize the information.

# EXAMPLES:
Question: Is it safe to take Macrobid?
Answer: Macrobid is generally considered safe to take when prescribed by a doctor and used according to their instructions.
Rationale: The answer is good, but it is missing important information regarding special cases, such as pregnant women, kids and
comorbidities.
A better answer would include that Macrobid is not recommended for use in the last 2 to 4 weeks of pregnancy

###
Question: Ok Do I need any antibiotics or something for yeast infection?
Answer: Over-the-counter antifungal creams or suppositories applied internally to the vagina can effectively treat vaginal yeast
infections, while a single oral dose of prescription fluconazole (Diflucan) may also be an option.
Rationale: the answer is good however it does not include relevant explanation about the condition such as Yeast infections are
commonly caused by fungi, particularly Candida albicans, which is a natural inhabitant of the body but can cause infections in
different areas such as the skin, mouth, and vagina. In addition, it assumes the patient is asking about vaginal yeast infection
and excludes information about other possible infections caused by yeast.

###
Question: the vomiting, tight abdomen pain and burning pain may just be result of pcos?
Answer: Vomiting and tight abdominal pain are not common clinical presentations of PCOS. Vomiting and abdominal pain are nonspecific symptoms
which could be an indication for a variety of illnesses. In case of this presentation, you should be evaluated by a physician, and
treated accordingly.
Rationale: the answer is good, but it does not emphasize enough that the patient’s symptoms are most likely caused by something
else and what is the differential diagnosis in this case.

# INPUT
Question: {question}
Answer:

"""


def batch_eval_prompt(
    question: str,
    llm_claims: list[str],
    must_have_claims: list[str],
    gold_claims: list[str],
) -> str:
    prompt = f"""

# OVERALL INSTRUCTIONS
 - You have a deep understanding of logical relationships, such as entailment and contradiction.
 - Your task is to evaluate a set of generated claims against ground truth claims based on a given question.
 - You will evaluate two things: comprehensiveness and hallucination.
 - Respond with a single JSON object with the exact shape specified below.

 # EVALUATION CRITERIA

 ## 1. Comprehensiveness (Entailment)
- Your goal is to determine which of the "Must-Have Claims" are logically entailed by any of the "Generated Claims".
- A generated claim entails a must-have claim if the must-have claim can be reasonably inferred from the generated claim.
- If a must-have claim is not supported by any generated claim, it should not be in your output.

 ## 2. Hallucination (Contradiction)
- Your goal is to determine if any of the "Generated Claims" contradict the "Ground Truth Gold Claims".
- A contradiction occurs when a generated claim and a gold claim cannot both be true.
- Be vigilant in identifying cases where the generated claim doesn't rule out the possibility of an entity (e.g., vaccine, symptom) appearing in the gold claim. In such cases, it is NOT a contradiction.
- If no generated claims are contradictory, the list should be empty.


# FORMAT OF INPUT DATA

 ## Question
{"question"}

## Generated Claims
{"llm_claims_str"}

## Must-Have Claims (for Comprehensiveness)
{"must_have_claims_str"}

## Ground Truth Gold Claims (for Hallucination)
{"gold_claims_str"}

# INPUT DATA
## Question
{question}

## Generated Claims
{llm_claims}

## Must-Have Claims (for Comprehensiveness)
{must_have_claims}

## Ground Truth Gold Claims (for Hallucination)
{gold_claims}

# RESPONSE FORMAT
Respond with a single JSON object with this exact shape:
{{
  "comprehensiveness": {{
    "entailed_must_have_claims": ["<string>", "..."]
  }},
  "hallucination": {{
    "contradictory_generated_claims": ["<string>", "..."]
  }}
}}


Rules:
- Return only valid JSON. No prose, no Markdown/code fences.
- REMEMBER: For Comprehensiveness, That Must-Have Claims does not need to match WORD FOR WORD, because a generated claim entails a must-have claim if the must-have claim can be reasonably inferred from the generated claim. 
- REMEMBER: For Hallucination, A contradiction occurs when a generated claim and a gold claim cannot both be true 
- The lists in the JSON should contain the exact text of the claims.
- If no claims meet the criteria for a list, return an empty list for it.

    """
    return prompt
