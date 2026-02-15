from __future__ import annotations

import logging

from medarc_verifiers.judging.judge_core import call_judge_model

logger = logging.getLogger(__name__)


def _format_judge_prompt(rubric, prompt, completion, answer) -> str:
    if isinstance(prompt, list):
        last_msg = prompt[-1]
        if isinstance(last_msg, dict) and "content" in last_msg:
            question = str(last_msg["content"])
        else:
            question = ""
    else:
        question = str(prompt)

    response_text = rubric.parser.parse_answer(completion)
    if response_text is None:
        response_text = ""
    return rubric.judge_prompt.format(question=question, answer=answer, response=response_text)


def _judge_cache_key(rubric) -> str:
    base_url = getattr(rubric.judge_client, "base_url", None)
    base_url_str = str(base_url) if base_url is not None else "default"
    return f"{base_url_str}::{rubric.judge_model}"

def install_cache_patch() -> bool:
    try:
        from verifiers.rubrics.judge_rubric import JudgeRubric

        if not hasattr(JudgeRubric, "_original_judge_unpatched"):
            JudgeRubric._original_judge_unpatched = JudgeRubric.judge

        async def patched_judge(self, prompt, completion, answer, state, **kwargs):
            judge_prompt = _format_judge_prompt(self, prompt, completion, answer)
            if state is None:
                response_text, _ = await call_judge_model(
                    self.judge_client,
                    self.judge_model,
                    judge_prompt,
                    self.judge_sampling_args,
                    self.logger,
                )
                return response_text

            cache = state.get("judge_response")
            if not isinstance(cache, dict):
                cache = {}
                state["judge_response"] = cache
            cache_key = _judge_cache_key(self)
            judge_cache = cache.get(cache_key)
            if not isinstance(judge_cache, dict):
                judge_cache = {}
                cache[cache_key] = judge_cache

            if judge_prompt in judge_cache:
                return judge_cache[judge_prompt]

            response_text, _ = await call_judge_model(
                self.judge_client,
                self.judge_model,
                judge_prompt,
                self.judge_sampling_args,
                self.logger,
            )

            # Write back without clobbering other concurrent updates:
            # we mutate the shared nested dict in-place.
            judge_cache[judge_prompt] = response_text

            return response_text

        JudgeRubric.judge = patched_judge
        logger.debug("Judge cache namespacing patch installed")
        return True
    except Exception as e:
        logger.warning(f"Failed to install judge cache patch: {e}")
        return False
