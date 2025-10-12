import os
from openai import AsyncOpenAI
import asyncio

class AtomicFactGenerator(object):
    def __init__(self, async_openai_client, model_name="gpt-5"):
        #self.nlp = spacy.load("en_core_web_sm")
        self.openai_lm = async_openai_client
        self.model_name = model_name


    async def run(self, generation):
        """Convert the generation into a set of atomic facts"""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return await self.get_atomic_facts(paragraphs)

    async def get_atomic_facts(self, paragraphs):
        para_breaks = []
        #number of paragraphs sent
        num_paragraphs = len(paragraphs)
        atoms_by_para = await self.get_atomic_facts_from_paragraphs(paragraphs)
        #check if the number of atoms is equal to the number of paragraphs
        assert len(atoms_by_para) == num_paragraphs, "number of atoms should be equal to number of paragraphs"
         

        atomic_facts_pairs = []
        print(atoms_by_para)
        for i, sent in enumerate(paragraphs):
            atomic_facts_pairs.append((sent, atoms_by_para[i]))


        return atomic_facts_pairs, para_breaks


    async def get_atomic_facts_from_paragraphs(self, paragraphs):
        """Get atomic facts from the paragraphs."""
        print(f"get_init_atomic_facts_from_sentence: {paragraphs} of len {len(paragraphs)}, model: {self.model_name}")

        prompts = []
        atoms = {}
        for i, paragraph in enumerate(paragraphs):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical language processing assistant. "
                        "Your job is to extract and rewrite *only* independent medical facts or medical advice "
                        "from a sentence. Ignore anything not medically relevant."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                        "Whenever the skin is inflamed it scales during or afterwards."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- Skin can scale when inflamed."
                        "- Skin can scale after being inflamed."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                        "Some conditions may not need antibiotics and fewer conditions require extending antibiotics."
                    )
                },
                    {
                    "role": "assistant",
                    "content": (
                        "- Antibiotics is not needed for some skin conditions."
                        "- Few skin conditions require extending antibiotics."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                        "There are many factors at play that can't be explained in a simple post on reddit."
                    )
                },
                {
                    "role": "assistant",
                    "content": "None"
                },
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                        "I suggest your father has a good conversation with his physicians to determine what next steps align with his goals."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- Father should have a conversation with his physicians."
                        "- Have a conversation with physicians to determine next steps."
                        "- Converse with physicians to determine next steps to align with his goals."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                        "Previous meningitis can make seizures more likely. Drug use also can cause seizures. If she had drugs in her system they almost certainly played a part in the seizures, at least this time around."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- Having previously had meningitis can make seizures more likely."
                        "- Drug use can also cause meningitis."
                        "- If she had drugs in her system they almost certainly contributed to having a seizure."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                        "This warrants evaluation by your doctor. There are many different caused for these kinds of symptoms, some organic and some psychological, but all of them merit treatment."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- This warrants evaluation by your doctor."
                        "- There can be many different causes for these kinds of symptoms."
                        "- These symptoms can be caused by something organic."
                        "- These symptoms can be caused by something psychological."
                        "- These symptoms merit treatment."
                    )
                },
            ]
            messages.append({
                "role": "user",
                "content": (
                    "Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise:"
                    + paragraph
                )
            })
            prompts.append(messages)

        kwargs = {
            "model": self.model_name,
            "messages": messages,
        }

        if self.model_name != "gpt-5":
            kwargs["temperature"] = 0.3
        else:
            kwargs["verbosity"] = "low"


        for i, messages in enumerate(prompts):
            try:
                #print("doing call to gpt")
                response = await self.openai_lm.chat.completions.create(**kwargs)
                #print("done call to gpt")
                print(response.choices[0].message.content)
            
                output = response.choices[0].message.content
                atoms[i] = text_to_atomic_facts(output)
            except Exception as e:
                print(f"Error: {e}")
                continue

            # for key, value in demons.items():
            #     if key not in atoms:
            #       atoms[key] = value
        print("atoms", atoms)
        return atoms


# transform output into atomic facts
def text_to_atomic_facts(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    #print("got messages of len", len(sentences))
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences


async def main():
    judge_model = 'gpt-4o-mini'
    judge_base_url = None
    judge_api_key = None
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
    generator = AtomicFactGenerator(judge_client, judge_model)
    query = "Thank you for sharing your concerns. Its good that you are paying attention to your health and planning to see a doctor. Let me help explain a few things to hopefully ease your worries and guide you on what to do next.\n\n**1. Feeling a pulsation in the abdomen:**\n- It is actually not uncommon for some people, especially young and thin individuals, to feel or even see the pulsation of their abdominal aorta, especially when lying down. This is because the abdominal aorta is a large artery close to the surface in the abdomen.\n- However, you mentioned you are overweight, which can sometimes make feeling the aorta less likely, but its still possible depending on body shape, how your abdominal muscles are, and your heart rate.\n\n**2. Abdominal Aortic Aneurysm (AAA):**\n- AAA is an enlargement of the abdominal aorta that usually occurs in older adults, especially men over 65, smokers, or those with a family history.\n- It is very rare in young people like you (16 years old).\n- AAA usually does not cause pain or visible pulsations until it becomes large or ruptures, which is a medical emergency.\n- Mild, brief abdominal pain and sensations you describe are unlikely to be caused by AAA, especially given your age.\n\n**3. High blood pressure and lifestyle:**\n- High blood pressure at your age should be evaluated by a healthcare provider. It can sometimes be related to lifestyle factors but also may have other causes.\n- Lifestyle changes like healthier eating, exercise, and weight loss are great steps to improve your blood pressure and overall health.\n\n**4. What you should do next:**\n- Its important to see a healthcare provider to get a proper physical exam and possibly some tests (like blood pressure monitoring, abdominal ultrasound if needed).\n- An abdominal ultrasound is a painless, non-invasive way to check for an aneurysm and other abdominal issues. Its a straightforward test that your doctor can order if they think its necessary.\n- Let your parents or guardians know about your symptoms and concerns so they can help you get medical evaluation.\n\n**5. Managing anxiety:**\n- Feeling anxious about health is common, especially when you notice unusual symptoms.\n- Try to focus on facts: AAA is very unlikely at your age.\n- Continue healthy habits and keep track of symptoms.\n- If anxiety feels overwhelming, consider discussing this with your doctor as well.\n\n---\n\n**Summary:**\n\n- Feeling your abdominal pulse can be normal and does not automatically mean you have an aneurysm.\n- AAA is very rare at your age.\n- Mild abdominal pain and sensations are unlikely related to aneurysm.\n- You should see a healthcare provider for evaluation and blood pressure management.\n- An abdominal ultrasound can be done if needed to reassure you.\n- Keep up with healthy lifestyle changes and try to manage anxiety.\n\nIf you experience sudden severe abdominal or back pain, dizziness, or fainting, seek emergency medical care immediately.\n\n---\n\nI hope this helps you feel more informed and less worried. Seeing a doctor soon will give you peace of mind and ensure your health is on the right track. If you have any more questions, feel free to ask!"
    query2 = "This topic can be a little sensitive. Without knowing more of your background, I'd advise going to the doctor to differentiate a UTI from an STI or some other infection or other cutaneous condition like bacterial vaginosis. STIs in particular can progress to pelvic inflammatory disease which has the potential to cause you significant long term issues as well as discomfort. Ultimately, if it is a UTI, it might go away on its own or it might not. The risk is yours to take. Is there any particular reason you don't want to go to a doctor? The basic tests that they might consider ordering (e.g. swabs, urine sample) can often be self-collected. They may recommend a visual inspection +\/- examination of the area. You can very reasonably ask for a female doctor or specialised nurse at any point."
    answer2 = "I did all my PhD work about AAA, and a 16M with no smoking history having a AAA would be more than rare. It would be writing medical journal case report rare. I wouldn't worry about a AAA."
    #answer2 = "I love basketball."
    atomic_facts, para_breaks = await generator.run(answer2)
    text="- 123. \n- 456."
    text="None"
    #print(text_to_sentences(text))
    #print("atomic_facts", atomic_facts)
    #print("para_breaks", para_breaks)

if __name__ == "__main__":
    asyncio.run(main())