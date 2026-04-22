"""RL training env: HealthBench-style rubric-judged prompts.

Wraps the existing `healthbench` env's judge-per-criterion reward path so it
can be driven from prime-rl as a training env with a `dataset=` attribute
(not just eval). Each rollout's reward = sum(points for criteria met) /
sum(positive_points_possible), bounded to [0, 1].

Train split comes from HealthBench's `consensus` (3,671 examples) or `hard`
(1,000 examples) datasets. Eval split comes from HealthBench `all` test set.
Choosing a smaller eval slice keeps per-step validation cheap.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar

# Reuse healthbench's judge-plumbing + scoring exactly — single source of truth.
from healthbench import (  # type: ignore[import-not-found]
    HEALTHBENCH_CONSENSUS_CRITERIA_LOOKUP,
    _format_prompt_to_judge,
    _judge_single_criterion,
    _process_healthbench_dataset,
)
from medarc_verifiers.judging import MultiJudge, MultiJudgeRubric
from medarc_verifiers.types import Messages
from verifiers.types import Info, State

disable_progress_bar()


TRAIN_MAPPING = {
    "consensus": ("neuralleap/healthbench-consensus", "train"),
    "hard": ("neuralleap/healthbench-hard", "train"),
}
EVAL_DATASET = ("neuralleap/healthbench-regular", "test")


def load_environment(
    judge_model: str | list[str] = "openai/gpt-oss-20b",
    judge_base_url: str | list[str] | None = None,
    judge_api_key: str | list[str] | None = None,
    train_split: str = "consensus",
    num_train_examples: int | None = None,
    num_eval_examples: int = 100,
    judge_timeout: int | None = 300,
    max_parallel_judges: int | None = 3,
    **kwargs: Any,
) -> vf.Environment:
    """Return a SingleTurnEnv with HealthBench-style rubric reward.

    The reward for a rollout is `sum(points for criteria met)` divided by
    `sum(p for p in points_possible if p > 0)`. Negative points stay negative
    when met, which preserves the penalty structure of the original rubric.

    `judge_model` defaults to gpt-oss-20b (matches the scale-sweep judge on
    n-4:18001). Override via `judge_base_url` for a different endpoint.
    """
    if train_split not in TRAIN_MAPPING:
        raise ValueError(f"Unknown train_split={train_split}; expected one of {list(TRAIN_MAPPING)}")

    train_repo, train_split_name = TRAIN_MAPPING[train_split]
    train_ds = load_dataset(train_repo, split=train_split_name)
    if num_train_examples is not None:
        train_ds = train_ds.select(range(min(num_train_examples, len(train_ds))))
    train_ds = train_ds.map(lambda ex: {"info": _process_healthbench_dataset(ex)})

    eval_repo, eval_split_name = EVAL_DATASET
    eval_ds = load_dataset(eval_repo, split=eval_split_name)
    if num_eval_examples:
        eval_ds = eval_ds.select(range(min(num_eval_examples, len(eval_ds))))
    eval_ds = eval_ds.map(lambda ex: {"info": _process_healthbench_dataset(ex)})

    multi_judge = MultiJudge.from_env_args(
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        judge_api_key=judge_api_key,
        judge_prompt="{question}",
        judge_timeout=judge_timeout,
    )
    rubric = MultiJudgeRubric(multi_judge)

    criteria_parallel = max_parallel_judges if max_parallel_judges is not None else 3

    async def reward_rubric(prompt: Messages, completion: Messages, info: Info, state: State) -> float:
        """Per-rollout reward = fraction of positive-point rubric criteria met.

        Mirrors healthbench.reward_healthbench but normalizes by positive-point
        total so the scalar lives in [0, 1] for RL stability.
        """
        if isinstance(completion, list) and completion:
            raw_completion = completion[-1].get("content", "")
        else:
            raw_completion = str(completion)

        conversation = _format_prompt_to_judge(prompt, raw_completion)
        criteria = info.get("criteria", []) or []
        points_list = info.get("points_list", []) or []
        if not points_list:
            return 0.0

        positive_total = sum(pt for pt in points_list if pt > 0)
        if positive_total <= 0:
            return 0.0

        semaphore = asyncio.Semaphore(criteria_parallel)
        tasks = [
            _judge_single_criterion(
                idx=idx,
                criterion=criterion,
                points_possible=pts,
                conversation=conversation,
                rubric=rubric,
                semaphore=semaphore,
                state=state,
            )
            for idx, (criterion, pts) in enumerate(zip(criteria, points_list))
        ]
        judgments = await asyncio.gather(*tasks)

        earned = 0.0
        for j in judgments:
            if isinstance(j, dict) and j.get("criteria_met"):
                earned += float(j.get("points_possible", 0))
        # Clamp to [0, 1] — negative-point criteria can push earned slightly below 0.
        return max(0.0, min(1.0, earned / positive_total))

    rubric.add_reward_func(reward_rubric, weight=1.0)

    return vf.SingleTurnEnv(
        dataset=train_ds,
        eval_dataset=eval_ds,
        system_prompt="",
        rubric=rubric,
    )
