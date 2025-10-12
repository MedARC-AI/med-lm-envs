import os
import re

import verifiers as vf
from datasets import Dataset, concatenate_datasets
from openai import AsyncOpenAI
from verifiers.utils.data_utils import BOXED_SYSTEM_PROMPT, THINK_BOXED_SYSTEM_PROMPT, extract_boxed_answer
import pandas as pd


# MedExQA specialties
SPECIALTIES = [
    "biomedical_engineer",
    "clinical_laboratory_scientist",
    "clinical_psychologist",
    "occupational_therapist",
    "speech_pathologist",
]



def _build_question_str(question: str, options: dict[str, str]) -> str:
    """Format question with answer choices, following authors' format with boxed instruction."""
    # Instruction adapted from authors' code https://github.com/knowlab/MedExQA/blob/9a5b34af103b0c8ba0c00906e278f6572249fafa/evaluate_pipe_MedExQA.py#L32
    instruction = (
        "The following is a multiple-choice question. Please choose the most suitable one "
        "among A, B, C and D as the answer to this question. "
        'Put your answer in \\boxed{X} format where X is the letter choice. '
        "Your answer should be paired with an explanation why you chose that answer.\n\n"
    )
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    return f"{instruction}{question}\n{opts}\nAnswer:"


def _to_vf_format(ds: Dataset) -> Dataset:
    """
    Shape each row for SingleTurnEnv's defaults:
      - 'question': formatted question string with options
      - 'answer': gold letter (A/B/C/D)
      - 'info': keep all original fields including explanations
    """
    def _format_row(row: dict) -> dict:
        question = row.get("question", "") or ""

        # Build options dict from A, B, C, D columns
        opts = {
            "A": row.get("A", ""),
            "B": row.get("B", ""),
            "C": row.get("C", ""),
            "D": row.get("D", ""),
        }

        # Get answer letter
        answer_letter = (row.get("answer") or "").strip().upper()
        if answer_letter not in ("A", "B", "C", "D"):
            return None

        question_str = _build_question_str(question, opts)

        # Keep original data in info
        info = dict(row)

        return {
            "question": question_str,
            "answer": answer_letter,
            "info": info,
        }

    return ds.map(_format_row, remove_columns=ds.column_names).filter(lambda row: row is not None)


def load_environment(
    use_think: bool = False,
    use_explanations: bool = False,
    mcq_weight: float = 0.5,
    explanation_weight: float = 0.5,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    **kwargs
) -> vf.Environment:
    """
    Single-turn MedExQA environment using HuggingFace `bluesky333/MedExQA` dataset

    Each example is normalized to the fields expected by `vf.SingleTurnEnv`:
        {
            "question": "<question + formatted options>",  # string used as the user prompt
            "answer": "<A|B|C|D>",                         # top-level gold letter
            "info": { ...original example fields... }      # full source row including exp0, exp1
        }

    - Loads all 5 medical specialties (biomedical engineering, clinical lab science,
      clinical psychology, occupational therapy, speech language pathology)
    - No training split (dataset does not provide one)
    - Test split used as evaluation data (940 total examples)

    - Parser extracts \\boxed{A|B|C|D} from completions

    - Reward looks for exact match between parsed letter and answer letter
    - Optional: Explanation quality evaluation using LLM-as-judge
    """

    # Load all specialties and concatenate
    # Note: MedExQA only has dev and test splits, no train split
    # Load TSV files directly since HF dataset has column name issues
    test_datasets = []

    for specialty in SPECIALTIES:
        try:
            # Download and load TSV file directly
            url = f"https://huggingface.co/datasets/bluesky333/MedExQA/resolve/main/test/{specialty}_test.tsv"

            # Load TSV with pandas (no headers in file)
            df = pd.read_csv(
                url,
                sep='\t',
                header=None,
                names=["question", "A", "B", "C", "D", "exp0", "exp1", "answer"]
            )

            # Add specialty column
            df['specialty'] = specialty

            # Convert to HF dataset
            test_ds = Dataset.from_pandas(df, preserve_index=False)
            test_datasets.append(test_ds)
        except Exception as e:
            print(f"Warning: Could not load {specialty}: {e}")
            continue

    # Concatenate all specialties
    test_combined = concatenate_datasets(test_datasets) if test_datasets else None

    # Format for verifiers - no training dataset available
    test_ds = _to_vf_format(test_combined) if test_combined else None

    # Setup system prompt - use standard boxed prompts since instruction is in question
    # Like M-ARC, we put the instruction in the question itself, so use standard prompts
    system_prompt = THINK_BOXED_SYSTEM_PROMPT if use_think else BOXED_SYSTEM_PROMPT

    # Parser for extracting \\boxed{} answers
    parser = (
        vf.ThinkParser(extract_fn=extract_boxed_answer) if use_think
        else vf.Parser(extract_fn=extract_boxed_answer)
    )

    def correct_answer_reward_func(parser, completion, answer, **kwargs) -> float:
        """Reward function for MCQ accuracy."""
        response = parser.parse_answer(completion) or ""
        return 1.0 if response == answer else 0.0

    # Create rubric based on whether we're evaluating explanations
    if use_explanations:
        # Setup judge for explanation evaluation
        api_key = judge_api_key if judge_api_key else os.getenv("JUDGE_API_KEY")
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key)

        # We construct the judge prompt directly below when calling the judge

        # Important: the JudgeRubric formats only with {question}, {answer}, {response}.
        # To include reference explanations exp0/exp1, we fully format the prompt
        # ourselves and pass it as {question}. Hence, set rubric prompt to "{question}".
        judge_rubric = vf.JudgeRubric(
            judge_client=judge_client,
            judge_model=judge_model,
            judge_prompt="{question}",
        )

        async def combined_reward(
            judge, prompt, completion, answer, state, **kwargs
        ) -> float:
            """Combined reward: MCQ accuracy + explanation quality."""
            # 1. Calculate MCQ accuracy
            
            mcq_score = correct_answer_reward_func(parser, completion, answer)

            # 2. Calculate explanation quality (strictly after the boxed answer)
            completion_text = completion if isinstance(completion, str) else str(completion)
            boxed_pattern = r"\\boxed\{[A-D]\}"
            match = re.search(boxed_pattern, completion_text)

            if match:
                explanation = completion_text[match.end():].strip()
            else:
                explanation = completion_text.strip()
            # If the explanation is too short, set the score to 0.0
            if len(explanation.split()) < 10:
                explanation_score = 0.0
            else:
                info = kwargs.get("info", {})
                if not info:
                    return (mcq_weight * mcq_score)  # no info, skip explanation
                exp0 = info.get("exp0", "")
                exp1 = info.get("exp1", "")
                if not exp0 or not exp1:
                    return (mcq_weight * mcq_score)  # missing refs

                question = info.get("question", "")
                opts = {
                    "A": info.get("A", ""),
                    "B": info.get("B", ""),
                    "C": info.get("C", ""),
                    "D": info.get("D", ""),
                }
                opts_str = "\n".join(f"{k}. {v}" for k, v in opts.items())
                formatted_question = f"{question}\n{opts_str}"

                # Build judge prompt directly to avoid brace-escaping issues
                full_prompt = (
                    "You are evaluating the quality of a medical explanation.\n\n"
                    "**Question:**\n" + formatted_question + "\n\n"
                    "**Correct Answer:** " + str(answer) + "\n\n"
                    "**Reference Explanation 1:**\n" + str(exp0) + "\n\n"
                    "**Reference Explanation 2:**\n" + str(exp1) + "\n\n"
                    "**Model's Response:**\n" + explanation + "\n\n"
                    "Evaluate whether the model's explanation is medically accurate, relevant, and demonstrates understanding of the medical concepts. The explanation should justify why the answer is correct.\n\n"
                    "Compare the model's explanation quality to the reference explanations. Consider:\n"
                    "- Medical accuracy\n"
                    "- Relevance to the question\n"
                    "- Clarity and completeness\n"
                    "- Proper use of medical concepts\n\n"
                    "Respond with a score from 0.0 to 1.0:\n"
                    "- 1.0 = Excellent (as good as or better than references)\n"
                    "- 0.75 = Good (mostly correct with minor issues)\n"
                    "- 0.5 = Acceptable (partially correct)\n"
                    "- 0.25 = Poor (significant errors)\n"
                    "- 0.0 = Wrong or irrelevant\n\n"
                    "Respond with ONLY a number between 0.0 and 1.0."
                )

                judge_response = await judge_rubric.judge(
                    [{"role": "user", "content": full_prompt}],
                    "",  # completion (unused)
                    "",  # answer (unused)
                    state,
                    **kwargs,
                )

                try:
                    score_str = str(judge_response).strip()
                    number_match = re.search(r"(\d+\.?\d*)", score_str)
                    if number_match:
                        explanation_score = float(number_match.group(1))
                        explanation_score = max(0.0, min(1.0, explanation_score))
                    else:
                        explanation_score = 0.0
                except (ValueError, AttributeError):
                    explanation_score = 0.0

            # Return weighted combination
            return (mcq_weight * mcq_score) + (explanation_weight * explanation_score)

        # Add combined reward function
        judge_rubric.add_reward_func(combined_reward, weight=1.0)

        rubric = judge_rubric
    else:
        # MCQ-only evaluation
        rubric = vf.Rubric(
            funcs=[correct_answer_reward_func],
            weights=[1.0],
            parser=parser,
        )

    return vf.SingleTurnEnv(
        dataset=None,  # No training split available
        eval_dataset=test_ds,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
