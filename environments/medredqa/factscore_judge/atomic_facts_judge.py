import re
import os
import verifiers as vf
from openai import AsyncOpenAI
# from factscore_judge.openai_lm import OpenAIModel
from factscore_judge.atomic_facts import AtomicFactGenerator

# Judge prompt template for medical diagnosis evaluation
JUDGE_TEMPLATE = """\
Your job is to evaluate whether the provided opinions and recommendations match the opinions and recommendations of a certified medical professional.

You will be given:
1. A medical question (question)
2. The ground truth opinion and recommendations from a certified medical professional (answer)  
3. Predicted opinion and recommendations (response)

Your task is to determine how similar the predicted opinion and recommendations are to those of a certified medical professional, even if worded differently and provide a similarity score between 0 and 1.

Consider these guidelines:
- Medical terms that refer to the same condition should be considered equivalent
- Different levels of specificity may be acceptable (e.g., "pneumonia" vs "bacterial pneumonia")
- Spelling variations of medical terms should be considered equivalent
- The core medical meaning should match, even if additional details vary
- Consider if predicted opinion and recommendations are similar in implications to ground truth opinion and recommendations

Question: {question}

Certified medical professional's opinion and recommendations: {answer}

Predicted opinion and recommendations: {response}

Provide a score indicating if predicted opinion and recommendations cover the ground truth opinion and recommendations based on following:
- Consider number of unique points that are made in ground truth opinion and recommendations. For each unique point, consider if the predicted opinion and recommendations make a similar point. Use this to compute percentage of points that are made in the ground truth opinion and recommendations that are also made in the predicted opinion and recommendations.
- Respond with number between 0 and 1 based on this percentage, where 1 is the most similar and 0 is the least similar. 
- Give final score in the format **SCORE:actual_score**. For example **SCORE:0.5** or **SCORE:1.0**
""".strip()

JUDGE_TEMPLATE2 = """\
Does the Passage support the Fact? Respond with True or False only

Passage: 
{response}

Fact: 
{answer}
""".strip()


def extract_score(text: str) -> tuple[float, bool]:
    """
    Extracts the numeric value following 'SCORE:' in the given text. Also indicates if score was found
    """
    try:
        match = re.search(r"SCORE:([0-9]*\.?[0-9]+)", text)
        if match:
            return float(match.group(1)), True
    except Exception:
        pass
    return 0.0, False

def extract_boolean_score(text: str) -> tuple[bool, bool]:
    """
    Extract a boolean score from text.
    
    Args:
        text: Input text to parse
        
    Returns:
        A tuple of (parsed_value, found) where:
        - parsed_value: The boolean value extracted (True or False)
        - found: Whether a valid boolean value was found
    """
    # Strip whitespace and newlines, convert to lowercase
    cleaned_text = text.strip().lower()
    
    # Check for 'true' and 'false' in the text
    has_true = 'true' in cleaned_text
    has_false = 'false' in cleaned_text
    
    # If both or neither found, return not found
    if has_true == has_false:  # both True or both False
        return (False, False)
    
    # If only 'true' found
    if has_true:
        return (True, True)
    
    # If only 'false' found
    return (False, True)


async def medical_recommendations_reward_func(
    judge, prompt, completion, answer, state, info=None, **kwargs
) -> float:
    """
    Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
    Expects judge response to be a float between 0 and 1. Returns 0 for invalid numbers.
    """
    # Extract the answer section (handling think tags)
    completion_text = completion if isinstance(completion, str) else str(completion)
    
    # Get judge response using the extracted answer
    judge_response = await judge(prompt, completion_text, answer, state, **kwargs)

    #add judge response to info to be saved
    info["judge_response"] = judge_response

    #parse score and save parsing status
    score, passed_parse = extract_score(judge_response)

    #log to info if unable to parse
    if not passed_parse:
        info["failure_reason"] = "Unable to parse score from judge response"
    
    return score


def create_judge_rubric_old(
    judge_client: AsyncOpenAI | None,
    judge_model: str = "gpt-4o-mini",
) -> vf.JudgeRubric:
    """
    Creates and configures a JudgeRubric for medical recommendations evaluation.
    
    Args:
        judge_client: AsyncOpenAI client for the judge
        judge_model: Model name to use for judging
        
    Returns:
        Configured JudgeRubric instance
    """
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_TEMPLATE,
    )
    
    # Add the reward function to the rubric
    rubric.add_reward_func(medical_recommendations_reward_func, weight=1.0)
    
    return rubric

async def medical_recommendations_atommic_reward_func(
    judge, prompt, completion, answer, state, info=None, **kwargs
) -> float:
    """
    Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
    Expects judge response to be a float between 0 and 1. Returns 0 for invalid numbers.
    """
    judge_client = kwargs.get('judge_client')
    judge_model = kwargs.get('judge_model', 'gpt-4o-mini')
    judge_base_url = kwargs.get('judge_base_url')
    judge_api_key = kwargs.get('judge_api_key')
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
    generator = AtomicFactGenerator(judge_client, model_name=judge_model)
    atomic_facts, para_breaks = await generator.run(answer)

    # Extract the answer section (handling think tags)
    completion_text = completion if isinstance(completion, str) else str(completion)
    print(f"completion_text: {completion_text}")
    print(f"answer: {answer}")
    print(f"prompt: {prompt}")

    

    total_score = 0
    total_facts = 0
    for sen, facts in atomic_facts:
        print(sen)
        for fact in facts:
            total_facts += 1
            judge_response = await judge(prompt, completion_text, fact, state, **kwargs)
            score, passed_parse = extract_boolean_score(judge_response)
            if passed_parse and score:
                total_score += 1
            print(f"Fact: {fact}, Score: {score}, Passed Parse: {passed_parse}")
            
    print(f"Total Score: {total_score}, Total Facts: {total_facts}, Score: {total_score/total_facts}")

    score = 0
    if total_facts > 0:
        score = total_score/total_facts
    return score

def create_judge_rubric(
    judge_client: AsyncOpenAI | None,
    judge_model: str = "gpt-4o-mini",
) -> vf.JudgeRubric:
    """
    Creates and configures a JudgeRubric for medical recommendations evaluation.
    
    Args:
        judge_client: AsyncOpenAI client for the judge
        judge_model: Model name to use for judging
        
    Returns:
        Configured JudgeRubric instance
    """
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_TEMPLATE2,
    )
    
    # Add the reward function to the rubric
    rubric.add_reward_func(medical_recommendations_atommic_reward_func, weight=1.0)
    
    return rubric

def test_medical_recommendations_reward_func(completion_text, answer):
    api_key = os.getenv("OPENAI_API_KEY")
    openai_lm = OpenAIModel("InstructGPT", api_key=api_key)


    generator = AtomicFactGenerator("api.key")
    atomic_facts, para_breaks = generator.run(answer)
    print("##atomic_facts")
    print(atomic_facts)
    print(para_breaks)
    prompt = "Does the Passage support the fact? Respond with True or False only"
    prompt = prompt + "\n\n" + "Passage:\n" + completion_text

    total_score = 0
    total_facts = 0
    for sen, facts in atomic_facts:
        print(sen)
        for fact in facts:
            total_facts += 1
            fact_prompt = prompt + "\n\n" + "Fact:\n" + fact
            output, _ = openai_lm.generate(fact_prompt)
            score, passed_parse = extract_boolean_score(output)
            if passed_parse and score:
                total_score += 1
            print(f"Fact: {fact}, Score: {score}, Passed Parse: {passed_parse}")
            
    print(f"Total Score: {total_score}, Total Facts: {total_facts}, Score: {total_score/total_facts}")

    for atomic_fact, para_break in zip(atomic_facts, para_breaks):
        print(atomic_fact)
        print(para_break)


    # judge = create_judge_rubric(judge_client)
    # prompt = "Please evaluate the following completion text against the following answer: {answer}"
    # completion = completion_text
    # score = await judge(prompt, completion, answer)
    # print(score)

# def main():
#     answer = "I did all my PhD work about AAA, and a 16M with no smoking history having a AAA would be more than rare. It would be writing medical journal case report rare. I wouldn't worry about a AAA."
#     completion_text = "Thank you for sharing your concerns. Its good that you are paying attention to your health and planning to see a doctor. Let me help explain a few things to hopefully ease your worries and guide you on what to do next.\n\n**1. Feeling a pulsation in the abdomen:**\n- It is actually not uncommon for some people, especially young and thin individuals, to feel or even see the pulsation of their abdominal aorta, especially when lying down. This is because the abdominal aorta is a large artery close to the surface in the abdomen.\n- However, you mentioned you are overweight, which can sometimes make feeling the aorta less likely, but its still possible depending on body shape, how your abdominal muscles are, and your heart rate.\n\n**2. Abdominal Aortic Aneurysm (AAA):**\n- AAA is an enlargement of the abdominal aorta that usually occurs in older adults, especially men over 65, smokers, or those with a family history.\n- It is very rare in young people like you (16 years old).\n- AAA usually does not cause pain or visible pulsations until it becomes large or ruptures, which is a medical emergency.\n- Mild, brief abdominal pain and sensations you describe are unlikely to be caused by AAA, especially given your age.\n\n**3. High blood pressure and lifestyle:**\n- High blood pressure at your age should be evaluated by a healthcare provider. It can sometimes be related to lifestyle factors but also may have other causes.\n- Lifestyle changes like healthier eating, exercise, and weight loss are great steps to improve your blood pressure and overall health.\n\n**4. What you should do next:**\n- Its important to see a healthcare provider to get a proper physical exam and possibly some tests (like blood pressure monitoring, abdominal ultrasound if needed).\n- An abdominal ultrasound is a painless, non-invasive way to check for an aneurysm and other abdominal issues. Its a straightforward test that your doctor can order if they think its necessary.\n- Let your parents or guardians know about your symptoms and concerns so they can help you get medical evaluation.\n\n**5. Managing anxiety:**\n- Feeling anxious about health is common, especially when you notice unusual symptoms.\n- Try to focus on facts: AAA is very unlikely at your age.\n- Continue healthy habits and keep track of symptoms.\n- If anxiety feels overwhelming, consider discussing this with your doctor as well.\n\n---\n\n**Summary:**\n\n- Feeling your abdominal pulse can be normal and does not automatically mean you have an aneurysm.\n- AAA is very rare at your age.\n- Mild abdominal pain and sensations are unlikely related to aneurysm.\n- You should see a healthcare provider for evaluation and blood pressure management.\n- An abdominal ultrasound can be done if needed to reassure you.\n- Keep up with healthy lifestyle changes and try to manage anxiety.\n\nIf you experience sudden severe abdominal or back pain, dizziness, or fainting, seek emergency medical care immediately.\n\n---\n\nI hope this helps you feel more informed and less worried. Seeing a doctor soon will give you peace of mind and ensure your health is on the right track. If you have any more questions, feel free to ask!"

#     test_medical_recommendations_reward_func(completion_text, answer)

# if __name__ == "__main__":
#     main()