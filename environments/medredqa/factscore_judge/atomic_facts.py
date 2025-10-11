import json
import numpy as np
import re
import functools
import string
import spacy
import sys
import nltk
import openai
import os
import time
from nltk.tokenize import sent_tokenize
from openai import AsyncOpenAI
# from factscore_judge.openai_lm import OpenAIModel
import asyncio

nltk.download("punkt")


class AtomicFactGenerator(object):
    def __init__(self, async_openai_client, model_name="gpt-4o-mini"):
        self.nlp = spacy.load("en_core_web_sm")
        self.openai_lm = async_openai_client
        self.model_name = model_name


    async def run(self, generation):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return await self.get_atomic_facts_from_paragraph(paragraphs)

    async def get_atomic_facts_from_paragraph(self, paragraphs):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0 :
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)

            curr_sentences = sent_tokenize(paragraph)
            curr_sentences_2 = sent_tokenize(paragraph)

            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

            # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

            sentences += curr_sentences

        #filters out common AI assistant boilerplate phrases from a list of sentences before extracting atomic facts from them.
        atoms = await self.get_init_atomic_facts_from_sentence([sent for i, sent in enumerate(sentences) if not ( \
                            (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                            (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are"))))])

         

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if ((i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks


    async def get_init_atomic_facts_from_sentence(self, sentences):
        """Get the initial atomic facts from the sentences."""
        print(f"get_init_atomic_facts_from_sentence: {sentences}")

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in sentences:
            prompt = ""
            # if sentence in atoms:
            #     continue
            # top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            # 
            #prompt = "Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s. \n- He made his acting debut in the film. \n- He made his acting debut in The Moon is the Sun's Dream. \n- The Moon is the Sun's Dream is a film. \n- The Moon is the Sun's Dream was released in 1992. \n- After his acting debut, he appeared in small and supporting roles. \n- After his acting debut, he appeared in small and supporting roles throughout the 1990s. \n\nPlease breakdown the following sentence into independent facts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission. \n- Collins became an astronaut. \n- Collins became one of the third group of astronauts. \n- Collins became one of the third group of astronauts selected. \n- Collins became one of the third group of astronauts selected by NASA. \n- Collins became one of the third group of astronauts selected by NASA in 1963. \n- He served as the Command Module Pilot. \n- He served as the back-up Command Module Pilot. \n- He served as the Command Module Pilot for the Gemini 7 mission. \n\nPlease breakdown the following sentence into independent facts: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut. \n- Bateman has acting roles. \n- Bateman has written two short films. \n- Bateman has directed two short films. \n- Bateman has written and directed two short films. \n- Bateman is currently in development on his feature debut. \n\nPlease breakdown the following sentence into independent facts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969. \n- Michael Collins was born on October 31, 1930. \n- Michael Collins is retired. \n- Michael Collins is an American. \n- Michael Collins was an astronaut. \n- Michael Collins was a test pilot. \n- Michael Collins was the Command Module Pilot. \n- Michael Collins was the Command Module Pilot for the Apollo 11 mission. \n- Michael Collins was the Command Module Pilot for the Apollo 11 mission in 1969. \n\nPlease breakdown the following sentence into independent facts: He was an American composer, conductor, and musical director. \n- He was an American. \n- He was a composer. \n- He was a conductor. \n- He was a musical director. \n\nPlease breakdown the following sentence into independent facts: She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019. \n- She currently stars in Love and Destiny. \n- Love and Destiny is a romantic comedy series. \n- Love and Destiny premiered in 2019. \n\nPlease breakdown the following sentence into independent facts: His breakthrough came with the leading role in the acclaimed crime-drama film Memories of Murder in 2003. \n- His breakthrough came with Memories of Murder. \n- He was the leading role in Memories of Murder. \n- Memories of Murder was released in 2003. \n- Memories of Murder is a film. \n- Memories of Murder is an acclaimed crime-drama film.\n\n"
            prompt = "Please breakdown the following sentence into independent medical facts or medical advise. Ignore not medical related facts or advise: Whenever the skin is inflamed it scales during or afterwards. \n- Skin can scale when inflamed. \n- Skin can scale after being inflamed. \n\nPlease breakdown the following sentence into independent medical facts or medical advise. Ignore not medical related facts or advise: Some conditions may not need antibiotics and fewer conditions require extending antibiotics. \n- Antibiotics is not needed for some conditions. \n- Few conditions require extending antibiotics. \n\nPlease breakdown the following sentence into independent medical facts or medical advise. Ignore not medical related facts or advise: Taking benzos in the long term is not advisable. \n- It is not advisable to take benzos long term.\n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: They’re well recognised as causing cognitive issues, melatonin is not. \n- They're know to cause cognitive issues. \n- melatonin is not know to cause cognitive issues. \nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: There are many factors at play that can't be explained in a simple post on reddit. \nNone \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: I suggest your father has a good conversation with his physicians to determine what next steps align with his goals. \n- Father should have a conversation with his physicians. \n- Have a conversation with physicians to determine next steps. \n- Converse with physicians to determine next steps with align with his goals. \n\n Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: Thank you for sharing. \nNone \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: If treatment is not successful, seek care. \n- Seek care if treatment is not successful. \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: Hydrocortisone ointment (not cream) for up to a week can help. \n- Hydrocortisone ointment (not cream) for up to a week can help. \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: Topical corticosteroids are not safe long-term, especially on eyelids. \n- Topical corticosteroids are not safe long-term. \n- Topical corticosteroids are not safe long-term for eyelids. \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: There are many causes of blepharitis and it can be detective work to sort this out. \n- There are many causes of blepharitis. \n- It can take a lot of investigation to determine the cause of blepharitis. \n\n Please breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: You shouldn't feel guilty! \nNone \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: Your healthcare team is here to help YOU. \n- Your healthcare team is here to help you. \n\nPlease breakdown the following sentence into independent medical facts or advise. Ignore not medical related facts or advise: Acetaminophen mixed with alcohol is an issue for your liver, not your kidneys. \n- Acetaminophen mixed with alcohol is an issue for your liver. \n- Acetaminophen mixed with alcohol is not an issue for your kidneys. \n\n"
            # for i in range(n):
            #     prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
            #     for fact in demons[list(demons.keys())[i]]:
            #         prompt = prompt + "- {}\n".format(fact)
            #     prompt = prompt + "\n"

            # for match in top_machings:
            #     prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(match)
            #     for fact in demons[match]:
            #         prompt = prompt + "- {}\n".format(fact)
            #     prompt = prompt + "\n"
            prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(sentence)
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence
        print(f"get_init_atomic_facts_from_sentence prompt_to_sent: {prompt_to_sent}")

        for prompt in prompts:
            try:
                response = await self.openai_lm.completions.create(
                    model=self.model_name,
                    prompt=prompt.strip(),
                    max_tokens=128,
                    temperature=0.7
                )
            
                output = response.choices[0].text
                atoms[prompt_to_sent[prompt]] = text_to_sentences(output)
            except Exception as e:
                print(f"Error: {e}")
                continue

            # for key, value in demons.items():
            #     if key not in atoms:
            #       atoms[key] = value

        return atoms


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


async def main():
    judge_model = 'gpt-4o-mini'
    judge_base_url = None
    judge_api_key = None
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
    generator = AtomicFactGenerator(judge_client)
    query = "Thank you for sharing your concerns. Its good that you are paying attention to your health and planning to see a doctor. Let me help explain a few things to hopefully ease your worries and guide you on what to do next.\n\n**1. Feeling a pulsation in the abdomen:**\n- It is actually not uncommon for some people, especially young and thin individuals, to feel or even see the pulsation of their abdominal aorta, especially when lying down. This is because the abdominal aorta is a large artery close to the surface in the abdomen.\n- However, you mentioned you are overweight, which can sometimes make feeling the aorta less likely, but its still possible depending on body shape, how your abdominal muscles are, and your heart rate.\n\n**2. Abdominal Aortic Aneurysm (AAA):**\n- AAA is an enlargement of the abdominal aorta that usually occurs in older adults, especially men over 65, smokers, or those with a family history.\n- It is very rare in young people like you (16 years old).\n- AAA usually does not cause pain or visible pulsations until it becomes large or ruptures, which is a medical emergency.\n- Mild, brief abdominal pain and sensations you describe are unlikely to be caused by AAA, especially given your age.\n\n**3. High blood pressure and lifestyle:**\n- High blood pressure at your age should be evaluated by a healthcare provider. It can sometimes be related to lifestyle factors but also may have other causes.\n- Lifestyle changes like healthier eating, exercise, and weight loss are great steps to improve your blood pressure and overall health.\n\n**4. What you should do next:**\n- Its important to see a healthcare provider to get a proper physical exam and possibly some tests (like blood pressure monitoring, abdominal ultrasound if needed).\n- An abdominal ultrasound is a painless, non-invasive way to check for an aneurysm and other abdominal issues. Its a straightforward test that your doctor can order if they think its necessary.\n- Let your parents or guardians know about your symptoms and concerns so they can help you get medical evaluation.\n\n**5. Managing anxiety:**\n- Feeling anxious about health is common, especially when you notice unusual symptoms.\n- Try to focus on facts: AAA is very unlikely at your age.\n- Continue healthy habits and keep track of symptoms.\n- If anxiety feels overwhelming, consider discussing this with your doctor as well.\n\n---\n\n**Summary:**\n\n- Feeling your abdominal pulse can be normal and does not automatically mean you have an aneurysm.\n- AAA is very rare at your age.\n- Mild abdominal pain and sensations are unlikely related to aneurysm.\n- You should see a healthcare provider for evaluation and blood pressure management.\n- An abdominal ultrasound can be done if needed to reassure you.\n- Keep up with healthy lifestyle changes and try to manage anxiety.\n\nIf you experience sudden severe abdominal or back pain, dizziness, or fainting, seek emergency medical care immediately.\n\n---\n\nI hope this helps you feel more informed and less worried. Seeing a doctor soon will give you peace of mind and ensure your health is on the right track. If you have any more questions, feel free to ask!"
    query2 = "This topic can be a little sensitive. Without knowing more of your background, I'd advise going to the doctor to differentiate a UTI from an STI or some other infection or other cutaneous condition like bacterial vaginosis. STIs in particular can progress to pelvic inflammatory disease which has the potential to cause you significant long term issues as well as discomfort. Ultimately, if it is a UTI, it might go away on its own or it might not. The risk is yours to take. Is there any particular reason you don't want to go to a doctor? The basic tests that they might consider ordering (e.g. swabs, urine sample) can often be self-collected. They may recommend a visual inspection +\/- examination of the area. You can very reasonably ask for a female doctor or specialised nurse at any point."
    answer2 = "I did all my PhD work about AAA, and a 16M with no smoking history having a AAA would be more than rare. \nIt would be writing medical journal case report rare. I wouldn't worry about a AAA."
    atomic_facts, para_breaks = await generator.run(answer2)

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    asyncio.run(main())