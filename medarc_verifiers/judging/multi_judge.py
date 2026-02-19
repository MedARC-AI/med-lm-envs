from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Iterable

import verifiers as vf
from openai import AsyncOpenAI

from medarc_verifiers.utils import default_judge_api_key, judge_sampling_args_and_headers


@dataclass(frozen=True)
class JudgeResult:
    name: str
    model: str
    base_url: str | None
    raw: str | None
    error: str | None
    index: int


def normalize_judge_models(judge_model: str | Iterable[str]) -> list[str]:
    if isinstance(judge_model, str):
        models = [judge_model]
    else:
        models = list(judge_model)
    cleaned = [str(model).strip() for model in models if str(model).strip()]
    if not cleaned:
        raise ValueError("judge_model must contain at least one non-empty model name.")
    return cleaned


def normalize_judge_endpoints(
    judge_base_url: str | list[str] | None,
    judge_api_key: str | list[str] | None,
    n_judges: int,
) -> tuple[list[str | None], list[str | None]]:
    if isinstance(judge_base_url, list):
        # Accept len==1 as "broadcast to all judges" (common when CLI uses append semantics).
        if len(judge_base_url) == 1 and n_judges > 1:
            base_urls = [judge_base_url[0] for _ in range(n_judges)]
        elif len(judge_base_url) == n_judges:
            base_urls = [url for url in judge_base_url]
        else:
            raise ValueError("judge_base_url list length must match judge_model list length.")
    else:
        base_urls = [judge_base_url for _ in range(n_judges)]

    if isinstance(judge_api_key, list):
        if len(judge_api_key) == 1 and n_judges > 1:
            api_keys = [judge_api_key[0] for _ in range(n_judges)]
        elif len(judge_api_key) == n_judges:
            api_keys = [key for key in judge_api_key]
        else:
            raise ValueError("judge_api_key list length must match judge_model list length.")
    else:
        api_keys = [judge_api_key for _ in range(n_judges)]

    resolved_keys = []
    for base_url, api_key in zip(base_urls, api_keys):
        resolved_keys.append(default_judge_api_key(base_url) if api_key is None else api_key)
    return base_urls, resolved_keys


class MultiJudge:
    def __init__(
        self,
        judge_models: list[str],
        judge_base_urls: list[str | None],
        judge_api_keys: list[str | None],
        judge_prompt: str = "{question}",
        judge_timeout: int | None = 300,
        completion_parser: vf.Parser | None = None,
    ) -> None:
        if len(judge_models) != len(judge_base_urls) or len(judge_models) != len(judge_api_keys):
            raise ValueError("Judge model, base_url, and api_key lists must have matching lengths.")
        self.judge_models = judge_models
        self.judge_base_urls = judge_base_urls
        self.judge_api_keys = judge_api_keys
        self.judge_prompt = judge_prompt
        self.judge_timeout = judge_timeout
        self.completion_parser = completion_parser

        self.judge_rubrics: list[vf.JudgeRubric] = []
        self.judge_ids: list[str] = []

        for model, base_url, api_key in zip(judge_models, judge_base_urls, judge_api_keys):
            sampling_args, default_headers = judge_sampling_args_and_headers(model, base_url, timeout=judge_timeout)
            judge_client = AsyncOpenAI(base_url=base_url, api_key=api_key, default_headers=default_headers)
            rubric = vf.JudgeRubric(
                parser=completion_parser,
                judge_client=judge_client,
                judge_model=model,
                judge_prompt=judge_prompt,
                judge_sampling_args=sampling_args,
            )
            self.judge_rubrics.append(rubric)
            base_key = base_url if base_url is not None else "default"
            self.judge_ids.append(f"{base_key}::{model}")

    @classmethod
    def from_env_args(
        cls,
        judge_model: str | list[str],
        judge_base_url: str | list[str] | None = None,
        judge_api_key: str | list[str] | None = None,
        judge_prompt: str = "{question}",
        judge_timeout: int | None = 300,
        completion_parser: vf.Parser | None = None,
    ) -> "MultiJudge":
        judge_models = normalize_judge_models(judge_model)
        base_urls, api_keys = normalize_judge_endpoints(judge_base_url, judge_api_key, len(judge_models))
        return cls(
            judge_models=judge_models,
            judge_base_urls=base_urls,
            judge_api_keys=api_keys,
            judge_prompt=judge_prompt,
            judge_timeout=judge_timeout,
            completion_parser=completion_parser,
        )

    async def judge(self, prompt, completion, answer, state) -> list[JudgeResult]:
        async def _run(idx: int, rubric: vf.JudgeRubric) -> JudgeResult:
            try:
                raw = await rubric.judge(prompt, completion, answer, state)
                return JudgeResult(
                    name=self.judge_ids[idx],
                    model=self.judge_models[idx],
                    base_url=self.judge_base_urls[idx],
                    raw=str(raw),
                    error=None,
                    index=idx,
                )
            except Exception as e:
                return JudgeResult(
                    name=self.judge_ids[idx],
                    model=self.judge_models[idx],
                    base_url=self.judge_base_urls[idx],
                    raw=None,
                    error=str(e),
                    index=idx,
                )

        tasks = [_run(idx, rubric) for idx, rubric in enumerate(self.judge_rubrics)]
        return await asyncio.gather(*tasks)

    async def rerun(self, result: JudgeResult, prompt, completion, answer, state) -> JudgeResult:
        idx = result.index
        rubric = self.judge_rubrics[idx]
        try:
            raw = await rubric.judge(prompt, completion, answer, state)
            return JudgeResult(
                name=self.judge_ids[idx],
                model=self.judge_models[idx],
                base_url=self.judge_base_urls[idx],
                raw=str(raw),
                error=None,
                index=idx,
            )
        except Exception as e:
            return JudgeResult(
                name=self.judge_ids[idx],
                model=self.judge_models[idx],
                base_url=self.judge_base_urls[idx],
                raw=None,
                error=str(e),
                index=idx,
            )

    @staticmethod
    def mean(scores: list[float | None]) -> float:
        valid = [score for score in scores if score is not None]
        if not valid:
            return 0.0
        return sum(valid) / len(valid)
