from __future__ import annotations

import verifiers as vf

from verifiers.types import State
from medarc_verifiers.judging.multi_judge import JudgeResult, MultiJudge


class MultiJudgeRubric(vf.Rubric):
    def __init__(
        self,
        multi_judge: MultiJudge,
        *,
        funcs=None,
        weights=None,
        parser: vf.Parser | None = None,
    ) -> None:
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.multi_judge = multi_judge
        self.add_class_object("judge", multi_judge.judge)
        self.add_class_object("multi_judge", multi_judge)

    async def judge(self, prompt: str, completion: str, answer: str, state: State):
        return await self.multi_judge.judge(prompt, completion, answer, state)

    async def rerun_judge(self, result: JudgeResult, prompt: str, completion: str, answer: str, state: State):
        return await self.multi_judge.rerun(result, prompt, completion, answer, state)
