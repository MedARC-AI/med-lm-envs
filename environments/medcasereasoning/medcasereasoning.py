import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
from medarc_verifiers.parsers.xml_parser import XMLParser
from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers
from openai import AsyncOpenAI
from verifiers.types import Info, State

disable_progress_bar()  # suppress datasets progress indicators

SYSTEM_PROMPT = (
    "Read the following case presentation and give the most likely diagnosis. "
    "First, provide your internal reasoning for the diagnosis within the tags <think> ... </think>. "
    "Then, output the final diagnosis (just the name of the disease/entity) within the tags <answer> ... </answer>."
)


QUESTION_TEMPLATE = """\
----------------------------------------
CASE PRESENTATION
----------------------------------------
{question}
----------------------------------------
OUTPUT TEMPLATE
----------------------------------------
<think>
...your internal reasoning for the diagnosis...
</think>
<answer>
...the name of the disease/entity...
</answer>
""".strip()


JUDGE_TEMPLATE = """\
Is our predicted diagnosis correct (yes/no)?
Predicted diagnosis: {predicted_diagnosis}, True diagnosis: {true_diagnosis}
Answer [yes/no].
""".strip()


def load_environment(
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
) -> vf.Environment:
    """
    MedCaseReasoning environment using LLM-as-a-Judge evaluation.

    This environment loads the MedCaseReasoning dataset and uses an LLM judge
    to evaluate whether model responses are equivalent to the ground truth
    medical diagnoses.
    """
    # Load the MedCaseReasoning dataset
    full_dataset = load_dataset("zou-lab/MedCaseReasoning")

    # Use train split for training, val split for evaluation
    train_dataset = full_dataset["train"].map(
        lambda x: {
            "question": QUESTION_TEMPLATE.format(question=x["case_prompt"]),
            "answer": x["final_diagnosis"],
            "task": "medcasereasoning",
            "info": {"case_prompt": x["case_prompt"]},
        }
    )

    eval_dataset = full_dataset["val"].map(
        lambda x: {
            "question": QUESTION_TEMPLATE.format(question=x["case_prompt"]),
            "answer": x["final_diagnosis"],
            "task": "medcasereasoning",
            "info": {"case_prompt": x["case_prompt"]},
        }
    )

    # System prompt for the task

    # Initialize OpenAI client for judge
    api_key = default_judge_api_key(judge_base_url) if judge_api_key is None else judge_api_key
    sampling_args, default_headers = judge_sampling_args_and_headers(judge_model, judge_base_url)
    judge_client = AsyncOpenAI(base_url=judge_base_url, api_key=api_key, default_headers=default_headers)

    # Create JudgeRubric with custom prompt
    judge_rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="{question}",
        judge_sampling_args=sampling_args,
    )

    parser = XMLParser(fields=["think", "answer"], answer_field="answer")

    async def medical_diagnosis_reward_func(
        completion,
        answer,
        info: Info,
        state: State,
        **kwargs,
    ) -> float:
        """
        Reward function that uses LLM judge to evaluate medical diagnosis equivalence.
        """
        parsed = parser.parse(completion, last=True)
        model_answer = getattr(parsed, "answer", None)

        # Get judge response using the extracted answer
        if model_answer is not None:
            judge_response = await judge_rubric.judge(
                JUDGE_TEMPLATE.format(predicted_diagnosis=model_answer, true_diagnosis=answer),
                model_answer,
                answer,
                state,
            )

            # Parse judge response
            judge_response_clean = judge_response.strip().lower()
        else:
            judge_response_clean = "no"
            judge_response = "no answer"

        info.setdefault("judge_feedback", []).append(
            {
                "parsed": judge_response_clean,
                "raw_judge": str(judge_response),
            }
        )

        # Return 1.0 if equivalent, 0.0 otherwise
        if "yes" in judge_response_clean and "no" not in judge_response_clean:
            return 1.0
        else:
            return 0.0

    # Add the reward function to the rubric
    judge_rubric.add_reward_func(medical_diagnosis_reward_func, weight=1.0)

    # Create the environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=judge_rubric,
    )

    return vf_env
