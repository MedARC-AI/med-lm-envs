import verifiers as vf
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
from prompts import _build_eval_prompt, _llm_generation_prompt, _decompose_free_form_answer, _prompt_for_has_contradiction, _prompt_for_is_entails
from openai import OpenAI
from datasets import Dataset
from medarc_verifiers.parsers import JSONParser 


def _call_llm(model: str, prompt: str) -> str:
    """for making an OpenAI API call."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    return (response.choices[0].message.content or "").strip()



def _nli_judge( # evaluate the entailment or contradiction between the premise and hypothesis
    question: str, premise: str, hypothesis: str, task: str, judge_model: str
) -> bool:
    """
    Judges whether the premise entails or contradicts the hypothesis.

    Args:
        task: Either "entails" or "contradicts".
    """
    if task == "entails":
        prompt = _prompt_for_is_entails(question, premise, hypothesis)
    elif task == "contradicts":
        prompt = _prompt_for_has_contradiction(question, premise, hypothesis)
    else:
        raise ValueError(f"Unknown NLI task: {task}")

    response_text = _call_llm(judge_model, prompt).lower()
    return "true" in response_text[-20:]



class ClaimExtractorParser(vf.Parser):
    """A parser that extracts atomic claims from a model's completion."""

    def __init__(self, extractor_model: str):
        super().__init__(extract_fn=lambda x: x)  # Use raw text
        self.extractor_model = extractor_model
        self.json_parser = JSONParser(fields=["claims"])

    def parse(self, text: str, **kwargs: Any) -> Dict[str, Any]:
        info: Dict[str, Any] = kwargs.get("info", {}) or {}
        question = info.get("Question", "")
        if not question:
            return {"raw_completion": text, "claims": []}
        prompt = _decompose_free_form_answer(question, text)
        extracted_text = _call_llm(self.extractor_model, prompt)
        parsed_obj = self.json_parser.parse(extracted_text)
        claims = []
        if isinstance(parsed_obj, dict):
            raw_claims = parsed_obj.get("claims") or []
            claims = [str(c).strip() for c in raw_claims if str(c).strip()]
        return {"raw_completion": text, "claims": claims}



# from the paper K-QA 
def comprehensiveness_reward(completion, info, parser, judge_model: str, **_) -> float:
    # get final assistant text
    msgs = parser.get_assistant_messages(completion) if isinstance(completion, list) else []
    text = msgs[-1]["content"] if msgs else (completion if isinstance(completion, str) else "")
    # parse claims using the parser instance , passing info
    parsed = parser.parse(text, info=info) if hasattr(parser, "parse") else {}
    predicted_claims = (parsed or {}).get("claims", []) if isinstance(parsed, dict) else []

    question: str = (info or {}).get("Question", "")
    must_have_claims = (info or {}).get("Must_have", []) or []
    if not must_have_claims:
        return 1.0

    covered = 0
    for must_claim in must_have_claims: #must have claims that the model should have covered
        if any(_nli_judge(question, pred, must_claim, "entails", judge_model) for pred in predicted_claims):
            covered += 1
    return covered / len(must_have_claims)


# from the paper K-QA 
# claims that the model hallucinated
def hallucination_rate_reward(completion, info, parser, judge_model: str, **_) -> float:
    msgs = parser.get_assistant_messages(completion) if isinstance(completion, list) else []
    text = msgs[-1]["content"] if msgs else (completion if isinstance(completion, str) else "")
    parsed = parser.parse(text, info=info) if hasattr(parser, "parse") else {}
    predicted_claims = (parsed or {}).get("claims", []) if isinstance(parsed, dict) else []
    if not predicted_claims:
        return 0.0

    question: str = (info or {}).get("Question", "")
    gold_claims = ((info or {}).get("Must_have", []) or []) + ((info or {}).get("Nice_to_have", []) or [])

    contradictions = 0
    for pred in predicted_claims: # A claim that the model hallucinated and contradicts the gold claims
        if any(_nli_judge(question, gold, pred, "contradicts", judge_model) for gold in gold_claims):
            contradictions += 1
    return contradictions / len(predicted_claims)



def load_environment(
    extractor_model: str = "gpt-4-mini", judge_model: str = "gpt-4-mini", **kwargs
) -> vf.Environment:
    """
    K-QA Environment using a FACTSCORE-style evaluation.
    - `extractor_model`: LLM to use for breaking answers into atomic claims.
    - `judge_model`: LLM to use for NLI-based scoring.
    """
    data_fp = Path(__file__).resolve().parent / "questions_w_answers.jsonl"
    df = pd.read_json(data_fp, orient="records", lines=True)

    # map to vf format
    def _map_to_vf_format(row: pd.Series) -> Dict[str, Any]:
        q = row.get("Question", "")
        return {
            "question": _llm_generation_prompt(q),
            "answer": "",  
            "info": {
                "Question": q,
                "Must_have": list(row.get("Must_have", []) or []), # from the dataset we have both must have and nice to have claims
                "Nice_to_have": list(row.get("Nice_to_have", []) or []),
            },
        }

    dataset = Dataset.from_list(df.apply(_map_to_vf_format, axis=1).tolist())
    parser = ClaimExtractorParser(extractor_model=extractor_model) # parser to extract claims from the model's completion

    rubric = vf.Rubric(
        funcs=[
            lambda **kw: comprehensiveness_reward(judge_model=judge_model, **kw),
            lambda **kw: hallucination_rate_reward(judge_model=judge_model, **kw),
        ],
        names=["comprehensiveness", "hallucination_rate"],
        parser=parser,
    )

    return vf.SingleTurnEnv(
        eval_dataset=dataset,
        parser=parser,
        rubric=rubric,
    )

