from __future__ import annotations

"""Small service facade for evaluation v2."""

from pathlib import Path
from typing import Callable

from codepilot.runtime import RuntimeService

from .runner import EvaluationRunner
from .schema import EvalCase, EvalResult, EvalRunOptions, EvalSuiteResult


class EvaluationService:
    """Convenience wrapper around :class:`EvaluationRunner`."""

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], RuntimeService] = RuntimeService,
    ) -> None:
        self.runner = EvaluationRunner(runtime_factory=runtime_factory)

    async def run_case(
        self,
        case: EvalCase,
        options: EvalRunOptions,
    ) -> EvalResult:
        return await self.runner.run_case(case, options)

    async def run_suite(
        self,
        suite_path: Path | str,
        options: EvalRunOptions,
    ) -> EvalSuiteResult:
        return await self.runner.run_suite(suite_path, options)


__all__ = ["EvaluationService"]
