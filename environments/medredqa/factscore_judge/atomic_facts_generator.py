import os
from openai import AsyncOpenAI
import sys
import asyncio


class AtomicFactGenerator:
    """
    Generates atomic facts from medical text using an LLM.
    
    Atomic facts are independent, single-statement medical claims that can be
    verified separately. This class breaks down complex medical text into
    simple, standalone medical facts or advice.
    """
    
    def __init__(self, async_openai_client, model_name="gpt-5"):
        """
        Initialize the AtomicFactGenerator.
        
        Args:
            async_openai_client: AsyncOpenAI client instance for API calls
            model_name: Name of the OpenAI model to use (default: "gpt-5")
        """
        self.openai_lm = async_openai_client
        self.model_name = model_name

    async def run(self, generation):
        """
        Convert generated text into a list of atomic facts.
        
        Args:
            generation: Input text string to be broken down into atomic facts
            
        Returns:
            List of atomic facts (strings) extracted from the input text
            
        Raises:
            AssertionError: If generation is not a string
        """
        assert isinstance(generation, str), "generation must be a string"
        
        # Split text into paragraphs, filtering out empty lines
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        
        return await self.get_atomic_facts(paragraphs)

    async def get_atomic_facts(self, paragraphs):
        """
        Extract atomic facts from a list of paragraphs.
        
        Args:
            paragraphs: List of paragraph strings
            
        Returns:
            Flattened list of atomic facts from all paragraphs
            
        Raises:
            AssertionError: If the number of returned fact lists doesn't match input paragraphs
        """
        # Track number of paragraphs sent to the LLM
        num_paragraphs = len(paragraphs)

        # Query the LLM for atomic facts
        atomic_facts_by_para = await self.query_llm_for_atomic_facts(paragraphs)
        
        # Verify that we received facts for each paragraph
        assert len(atomic_facts_by_para) == num_paragraphs, \
            "number of atomic fact lists should equal number of paragraphs"
         
        # Flatten the atomic facts from all paragraphs into a single list
        atomic_facts = [fact for para in atomic_facts_by_para for fact in para]

        return atomic_facts

    async def query_llm_for_atomic_facts(self, paragraphs):
        """
        Query the LLM to extract atomic facts from paragraphs.
        
        Uses few-shot prompting with medical examples to guide the LLM in
        extracting only medically relevant, independent facts.
        
        Args:
            paragraphs: List of paragraph strings to process
            
        Returns:
            List of lists, where each inner list contains atomic facts for one paragraph
        """
        #Print debug info if verbose specified
        verbose = True if '-v' in sys.argv else False

        prompts = []
        atoms = []
        
        # Prepare prompts for each paragraph with few-shot examples
        for i, paragraph in enumerate(paragraphs):
            # Build the message list with system prompt and few-shot examples
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a medical language processing assistant. "
                        "Your job is to extract and rewrite *only* independent medical facts or medical advice "
                        "from a sentence. Ignore anything not medically relevant."
                    )
                },
                # Example 1: Simple medical facts about skin
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advice. "
                        "Ignore non-medical related facts or advice: "
                        "Whenever the skin is inflamed it scales during or afterwards."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- Skin can scale when inflamed.\n"
                        "- Skin can scale after being inflamed.\n"
                    )
                },
                # Example 2: Facts about antibiotics and skin conditions
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advice. "
                        "Ignore non-medical related facts or advice: "
                        "Some conditions may not need antibiotics and fewer conditions require extending antibiotics."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- Antibiotics is not needed for some skin conditions.\n"
                        "- Few skin conditions require extending antibiotics.\n"
                    )
                },
                # Example 3: Non-medical content (should return None)
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advice. "
                        "Ignore non-medical related facts or advice: "
                        "There are many factors at play that can't be explained in a simple post on reddit."
                    )
                },
                {
                    "role": "assistant",
                    "content": "None \n"
                },
                # Example 4: Medical advice about physician consultation
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advice. "
                        "Ignore non-medical related facts or advice: "
                        "I suggest your father has a good conversation with his physicians to determine "
                        "what next steps align with his goals."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- Father should have a conversation with his physicians.\n"
                        "- Have a conversation with physicians to determine next steps.\n"
                        "- Converse with physicians to determine next steps to align with his goals."
                    )
                },
                # Example 5: Facts about seizures and meningitis
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
                        "- Having previously had meningitis can make seizures more likely.\n"
                        "- Drug use can also cause meningitis.\n"
                        "- If she had drugs in her system they almost certainly contributed to having a seizure.\n"
                    )
                },
                # Example 6: Multiple facts about symptoms and evaluation
                {
                    "role": "user",
                    "content": (
                        "Please breakdown the following sentence into independent medical facts or advice. "
                        "Ignore non-medical related facts or advice: "
                        "This warrants evaluation by your doctor. There are many different causes for these kinds "
                        "of symptoms, some organic and some psychological, but all of them merit treatment."
                    )
                },
                {
                    "role": "assistant",
                    "content": (
                        "- This warrants evaluation by your doctor.\n"
                        "- There can be many different causes for these kinds of symptoms.\n"
                        "- These symptoms can be caused by something organic.\n"
                        "- These symptoms can be caused by something psychological.\n"
                        "- These symptoms merit treatment.\n"
                    )
                },
            ]
            
            # Append the actual paragraph to be processed
            messages.append({
                "role": "user",
                "content": (
                    "Please breakdown the following sentence into independent medical facts or advice. "
                    "Ignore non-medical related facts or advice: "
                    + paragraph
                )
            })
            
            prompts.append(messages)

        # Configure model-specific parameters
        if "gpt-5" not in self.model_name:
            temperature = 0.3
        else:
            # gpt-5 uses different parameters. Instead of temperature, we use verbosity "low"
            temperature = None

        # Process each prompt and extract atomic facts
        for messages in prompts:
            try:
                # Build kwargs dynamically based on model
                kwargs = {
                    "model": self.model_name,
                    "messages": messages,
                }
                
                if temperature is not None:
                    kwargs["temperature"] = temperature
                else:
                    kwargs["verbosity"] = "low"
                
                # Call the LLM API
                response = await self.openai_lm.chat.completions.create(**kwargs)
                output = response.choices[0].message.content

                if verbose:
                    print(f"output: {output}")
                
                # Parse the LLM output into atomic facts
                atoms.append(text_to_atomic_facts(output))
            except Exception as e:
                print(f"Error processing paragraph: {e}")
                # Continue processing remaining paragraphs even if one fails
                continue

        return atoms


def text_to_atomic_facts(text):
    """
    Parse LLM output text into a list of atomic facts.
    
    Expects text in the format:
    - Fact 1
    - Fact 2
    - Fact 3
    
    Args:
        text: String output from the LLM containing atomic facts
        
    Returns:
        List of atomic fact strings, each properly formatted with a period at the end
    """
    # Split by "- " and skip the first empty element
    sentences = text.split("- ")[1:]
    
    # Clean up each sentence by stripping whitespace and newlines
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    
    # Ensure the last sentence ends with a period
    if len(sentences) > 0:
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.'
    
    return sentences


async def test_atomic_facts():
    """
    Example usage of AtomicFactGenerator.
    
    This function demonstrates how to initialize and use the AtomicFactGenerator
    to extract atomic facts from medical text.
    """
    # Configure the LLM client
    judge_model = 'gpt-4o-mini'
    judge_base_url = None
    judge_api_key = None
    
    # Get API key from environment variable if not explicitly provided
    api_key = judge_api_key if judge_api_key else os.getenv("OPENAI_API_KEY")
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key) if api_key else None
    
    # Initialize the generator
    generator = AtomicFactGenerator(judge_client, judge_model)
    
    # Example medical text
    example_text = (
        "I did all my PhD work about AAA, and a 16M with no smoking history having a AAA "
        "would be more than rare. It would be writing medical journal case report rare. "
        "I wouldn't worry about a AAA."
    )
    
    # Extract atomic facts and print them
    atomic_facts = await generator.run(example_text)
    print("atomic_facts", atomic_facts)
    

if __name__ == "__main__":
    asyncio.run(test_atomic_facts())
