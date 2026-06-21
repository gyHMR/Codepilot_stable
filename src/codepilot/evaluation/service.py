from __future__ import annotations

"""评估服务门面：对外暴露的评估 API。"""

import uuid
from pathlib import Path
from typing import Callable

from codepilot.runtime import RuntimeService

from .artifacts import EvalArtifactStore, hash_tree, stable_hash
from .executor import EvaluationExecutor
from .experiment import run_experiment
from .loader import EvalCaseValidationError, load_eval_suite
from .report import build_suite_summary, render_suite_markdown
from .types import (
    EvalCase,
    EvalDefinition,
    EvalResult,
    EvalRunOptions,
    EvalScenario,
    EvalSuiteResult,
)


class EvaluationService:
    """评估服务：协调加载、执行、产物管理和报告生成。"""

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
        self._initialize_manifest(artifacts, options, [case])
        artifacts.write_case_definition(case.id, case)
        result = await self._run_definition(case, options, artifacts)
        self._write_report(artifacts, [result])
        return result

    async def run_scenario(
        self,
        scenario: EvalScenario,
        options: EvalRunOptions,
    ) -> EvalResult:
        artifacts = self._artifact_store(options)
        self._initialize_manifest(artifacts, options, [scenario])
        artifacts.write_case_definition(scenario.id, scenario)
        result = await self._run_definition(
            scenario,
            options,
            artifacts,
        )
        self._write_report(artifacts, [result])
        return result

    async def run_suite(
        self,
        suite_path: Path,
        options: EvalRunOptions,
    ) -> EvalSuiteResult:
        artifacts = self._artifact_store(options)
        try:
            definitions = load_eval_suite(suite_path)
        except EvalCaseValidationError as exc:
            self._initialize_manifest(artifacts, options, [])
            result = self._invalid_result(
                Path(suite_path).stem or "suite",
                artifacts,
                str(exc),
            )
            self._write_report(artifacts, [result])
            return self._suite_result(artifacts, [result])

        self._initialize_manifest(artifacts, options, definitions)
        results = []
        for definition in definitions:
            artifacts.write_case_definition(definition.id, definition)
            results.append(
                await self._run_definition(
                    definition,
                    options,
                    artifacts,
                )
            )
        self._write_report(artifacts, results)
        return self._suite_result(artifacts, results)

    async def run_experiment(
        self,
        suite_path: Path,
        options: EvalRunOptions,
        *,
        module: str,
        repeat: int = 3,
    ) -> dict:
        return await run_experiment(
            self,
            suite_path,
            options,
            module=module,
            repeat=repeat,
        )

    async def _run_definition(
        self,
        definition: EvalDefinition,
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
            result = self._invalid_result(
                definition.id,
                artifacts,
                f"{type(exc).__name__}: {exc}",
            )
            artifacts.write_result(result)
            return result
        except Exception as exc:
            result = EvalResult(
                case_id=definition.id,
                overall="execution_error",
                session_id=None,
                run_ids=[],
                dimensions=[],
                failure_categories=["evaluation.execution_error"],
                metrics={},
                artifact_dir=str(artifacts.case_dir(definition.id)),
                error=f"{type(exc).__name__}: {exc}",
            )
            artifacts.write_result(result)
            return result

    @staticmethod
    def _invalid_result(
        case_id: str,
        artifacts: EvalArtifactStore,
        error: str,
    ) -> EvalResult:
        return EvalResult(
            case_id=case_id,
            overall="invalid_case",
            session_id=None,
            run_ids=[],
            dimensions=[],
            failure_categories=["evaluation.invalid_case"],
            metrics={},
            artifact_dir=str(artifacts.case_dir(case_id)),
            error=error,
        )

    @staticmethod
    def _artifact_store(options: EvalRunOptions) -> EvalArtifactStore:
        eval_id = options.eval_id or f"eval_{uuid.uuid4().hex[:12]}"
        return EvalArtifactStore(options.artifact_root, eval_id)

    @staticmethod
    def _initialize_manifest(
        artifacts: EvalArtifactStore,
        options: EvalRunOptions,
        definitions: list[EvalDefinition],
    ) -> None:
        model = options.session_options.model
        artifacts.initialize_manifest(
            benchmark_name=options.benchmark_name,
            model_id=(
                model.id
                if model is not None
                else options.session_options.model_id
            ),
            provider=(
                model.provider
                if model is not None
                else options.session_options.provider
            ),
            benchmark_snapshot=stable_hash(definitions),
            fixture_snapshot=hash_tree(options.fixtures_root),
        )

    @staticmethod
    def _write_report(
        artifacts: EvalArtifactStore,
        results: list[EvalResult],
    ) -> None:
        summary = build_suite_summary(results)
        artifacts.write_summary(
            summary,
            markdown=render_suite_markdown(results, summary),
        )

    @staticmethod
    def _suite_result(
        artifacts: EvalArtifactStore,
        results: list[EvalResult],
    ) -> EvalSuiteResult:
        return EvalSuiteResult(
            eval_id=artifacts.root.name,
            results=results,
            artifact_dir=str(artifacts.root),
            summary=build_suite_summary(results),
        )
