from __future__ import annotations

"""Public EvaluationService facade."""

import uuid
from pathlib import Path
from typing import Callable

from codepilot.runtime import RuntimeService

from .artifacts import EvalArtifactStore
from .executor import EvaluationExecutor
from .loader import EvalCaseValidationError, load_eval_suite
from .report import build_suite_summary
from .types import (
    EvalCase,
    EvalResult,
    EvalRunOptions,
    EvalScenario,
    EvalSuiteResult,
)


class EvaluationService:
    def __init__(
        self,
        *,
        runtime_factory: Callable[[], RuntimeService] = RuntimeService,
    ) -> None:
        self.executor = EvaluationExecutor(runtime_factory=runtime_factory)

    async def run_case(
        self,
        case: EvalCase,
        options: EvalRunOptions,
    ) -> EvalResult:
        artifacts = self._artifact_store(options)
        self._initialize_manifest(artifacts, options)
        artifacts.write_case_definition(case.id, case)
        result = await self._run_definition(case, options, artifacts)
        artifacts.write_summary(build_suite_summary([result]))
        return result

    async def run_scenario(
        self,
        scenario: EvalScenario,
        options: EvalRunOptions,
    ) -> EvalResult:
        artifacts = self._artifact_store(options)
        self._initialize_manifest(artifacts, options)
        artifacts.write_case_definition(scenario.id, scenario)
        result = await self._run_definition(scenario, options, artifacts)
        artifacts.write_summary(build_suite_summary([result]))
        return result

    async def run_suite(
        self,
        suite_path: Path,
        options: EvalRunOptions,
    ) -> EvalSuiteResult:
        artifacts = self._artifact_store(options)
        self._initialize_manifest(artifacts, options)
        try:
            definitions = load_eval_suite(suite_path)
        except EvalCaseValidationError as exc:
            result = EvalResult(
                case_id=Path(suite_path).stem or "suite",
                verdict="invalid_case",
                session_id=None,
                run_ids=[],
                verifier_results=[],
                artifact_dir=str(artifacts.root),
                error=str(exc),
            )
            summary = build_suite_summary([result])
            artifacts.write_summary(summary)
            return EvalSuiteResult(
                eval_id=artifacts.root.name,
                results=[result],
                artifact_dir=str(artifacts.root),
                summary=summary,
            )

        results: list[EvalResult] = []
        for definition in definitions:
            artifacts.write_case_definition(definition.id, definition)
            result = await self._run_definition(
                definition,
                options,
                artifacts,
            )
            results.append(result)
        summary = build_suite_summary(results)
        artifacts.write_summary(summary)
        return EvalSuiteResult(
            eval_id=artifacts.root.name,
            results=results,
            artifact_dir=str(artifacts.root),
            summary=summary,
        )

    async def _run_definition(
        self,
        definition: EvalCase | EvalScenario,
        options: EvalRunOptions,
        artifacts: EvalArtifactStore,
    ) -> EvalResult:
        try:
            if isinstance(definition, EvalCase):
                return await self.executor.run_case(
                    definition,
                    options,
                    artifacts,
                )
            return await self.executor.run_scenario(
                definition,
                options,
                artifacts,
            )
        except (FileNotFoundError, ValueError) as exc:
            result = EvalResult(
                case_id=definition.id,
                verdict="invalid_case",
                session_id=None,
                run_ids=[],
                verifier_results=[],
                artifact_dir=str(artifacts.case_dir(definition.id)),
                error=f"{type(exc).__name__}: {exc}",
            )
            artifacts.write_result(result)
            return result

    @staticmethod
    def _artifact_store(options: EvalRunOptions) -> EvalArtifactStore:
        eval_id = options.eval_id or f"eval_{uuid.uuid4().hex[:12]}"
        return EvalArtifactStore(options.artifact_root, eval_id)

    @staticmethod
    def _initialize_manifest(
        artifacts: EvalArtifactStore,
        options: EvalRunOptions,
    ) -> None:
        model = options.session_options.model
        artifacts.initialize_manifest(
            benchmark_name=options.benchmark_name,
            model_id=(
                model.id if model is not None
                else options.session_options.model_id
            ),
            provider=(
                model.provider if model is not None
                else options.session_options.provider
            ),
        )
