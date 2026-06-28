from __future__ import annotations

"""评估用例和状态化场景的执行引擎。"""

import asyncio
import shutil
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Callable

from codepilot.observability import AuditBundle, build_audit_report
from codepilot.runtime import RuntimeService
from codepilot.runtime.types import CreateAgentSessionOptions, UserInput
from codepilot.tools.approval import ApprovalDecision

from .artifacts import EvalArtifactStore
from .assertions import (
    build_dimension_results,
    failure_categories,
    required_assertions_passed,
    run_assertions,
    run_metric_assertions,
)
from .outcome_assertions import capture_workspace_baseline
from .metrics import calculate_case_metrics
from .types import (
    AssertionResult,
    AssertionSpec,
    EvalCase,
    EvalBudgets,
    EvalEvidence,
    EvalResult,
    EvalRunOptions,
    EvalRuntimeProfile,
    EvalScenario,
    ScenarioStep,
)


class EvaluationExecutor:
    """评估执行器：负责准备 workspace、运行 prompt/scenario、收集证据并评估断言。"""

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], RuntimeService] = RuntimeService,
    ) -> None:
        self.runtime_factory = runtime_factory

    async def run_case(
        self,
        case: EvalCase,
        options: EvalRunOptions,
        artifacts: EvalArtifactStore,
    ) -> EvalResult:
        started = time.perf_counter()
        workspace = self._prepare_workspace(
            case.id,
            case.fixture,
            options,
            artifacts,
        )
        evidence = EvalEvidence(
            workspace=workspace,
            baseline=capture_workspace_baseline(workspace),
        )
        runtime = self.runtime_factory()
        handle = runtime.create_session(
            replace(
                self._apply_runtime_profile(
                    options.session_options,
                    replace(case.runtime, **options.runtime_overrides),
                ),
                workspace_dir=workspace,
                session_id=None,
                max_task_replans_per_run=case.budgets.max_replans,
            )
        )
        evidence.session_id = handle.session_id
        error: str | None = None
        try:
            await self._execute_prompt(
                runtime,
                handle.session_id,
                case.prompt,
                case.timeout_seconds,
                evidence,
            )
            evidence.freshness_history.append(
                runtime.get_session_freshness(handle.session_id)
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._refresh_audit_evidence(
                runtime,
                handle.session_id,
                evidence,
            )
        finally:
            await runtime.aclose_all()

        result = self._evaluate(
            case.id,
            case.assertions,
            case.budgets,
            evidence,
            artifacts,
            metric_names=case.metrics,
            expected=case.expected,
            error=error,
            started=started,
        )
        artifacts.write_case_result(result, evidence)
        if not options.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        return result

    async def run_scenario(
        self,
        scenario: EvalScenario,
        options: EvalRunOptions,
        artifacts: EvalArtifactStore,
    ) -> EvalResult:
        started = time.perf_counter()
        workspace = self._prepare_workspace(
            scenario.id,
            scenario.fixture,
            options,
            artifacts,
        )
        evidence = EvalEvidence(
            workspace=workspace,
            baseline=capture_workspace_baseline(workspace),
        )
        runtime = self.runtime_factory()
        session_options = replace(
            self._apply_runtime_profile(
                options.session_options,
                replace(scenario.runtime, **options.runtime_overrides),
            ),
            workspace_dir=workspace,
            session_id=None,
            max_task_replans_per_run=scenario.budgets.max_replans,
        )
        handle = runtime.create_session(session_options)
        session_id = handle.session_id
        evidence.session_id = session_id
        step_assertions: list[AssertionResult] = []
        pending_task: asyncio.Task[None] | None = None
        pending_events: list[dict] = []
        error: str | None = None

        try:
            for index, step in enumerate(scenario.steps):
                if step.type == "prompt":
                    text = str(step.options["text"])
                    if bool(step.options.get("background", False)):
                        if pending_task is not None and not pending_task.done():
                            raise RuntimeError(
                                "A background prompt is already running"
                            )
                        pending_events = []
                        pending_task = asyncio.create_task(
                            self._consume_prompt_stream(
                                runtime,
                                session_id,
                                text,
                                pending_events,
                            )
                        )
                        await self._wait_until_running(runtime, session_id)
                        evidence.step_results.append(
                            {
                                "step": index,
                                "type": "prompt",
                                "background": True,
                            }
                        )
                    else:
                        await self._execute_prompt(
                            runtime,
                            session_id,
                            text,
                            scenario.timeout_seconds,
                            evidence,
                        )
                        evidence.step_results.append(
                            {
                                "step": index,
                                "type": "prompt",
                                "completed": True,
                            }
                        )
                elif step.type == "cancel":
                    cancelled = await runtime.cancel_run(session_id)
                    if pending_task is not None:
                        with suppress(asyncio.CancelledError):
                            await pending_task
                        pending_task = None
                        pending_events = []
                    self._refresh_audit_evidence(
                        runtime,
                        session_id,
                        evidence,
                    )
                    evidence.step_results.append(
                        {
                            "step": index,
                            "type": "cancel",
                            "cancelled": cancelled,
                        }
                    )
                elif step.type == "modify_file":
                    self._modify_file(
                        workspace,
                        scenario.fixture,
                        step,
                        options,
                    )
                    freshness = runtime.get_session_freshness(session_id)
                    evidence.freshness_history.append(freshness)
                    evidence.step_results.append(
                        {
                            "step": index,
                            "type": "modify_file",
                            "path": step.options["path"],
                            "freshness": freshness,
                        }
                    )
                elif step.type == "restart":
                    if pending_task is not None and not pending_task.done():
                        raise RuntimeError(
                            "Cannot restart with a background prompt"
                        )
                    await runtime.aclose_all()
                    runtime = self.runtime_factory()
                    runtime.create_session(
                        replace(session_options, session_id=session_id)
                    )
                    evidence.step_results.append(
                        {
                            "step": index,
                            "type": "restart",
                            "session_id": session_id,
                        }
                    )
                elif step.type == "continue":
                    await asyncio.wait_for(
                        self._consume_continue(runtime, session_id),
                        timeout=scenario.timeout_seconds,
                    )
                    self._refresh_audit_evidence(
                        runtime,
                        session_id,
                        evidence,
                    )
                    evidence.step_results.append(
                        {
                            "step": index,
                            "type": "continue",
                            "completed": True,
                        }
                    )
                elif step.type == "verify":
                    assertion = step.options["assertion"]
                    if not isinstance(assertion, AssertionSpec):
                        raise TypeError(
                            "verify step assertion was not normalized"
                        )
                    step_assertions.extend(
                        run_assertions([assertion], evidence)
                    )
                    evidence.step_results.append(
                        {"step": index, "type": "verify"}
                    )
                elif step.type == "inspect":
                    state = self._inspect_runtime(runtime, session_id)
                    evidence.step_results.append(
                        {
                            "step": index,
                            "type": "inspect",
                            "state": state,
                        }
                    )
            evidence.freshness_history.append(
                runtime.get_session_freshness(session_id)
            )
            self._refresh_audit_evidence(runtime, session_id, evidence)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._refresh_audit_evidence(runtime, session_id, evidence)
        finally:
            if pending_task is not None and not pending_task.done():
                await runtime.cancel_run(session_id)
                pending_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pending_task
                self._refresh_audit_evidence(
                    runtime,
                    session_id,
                    evidence,
                )
            await runtime.aclose_all()

        result = self._evaluate(
            scenario.id,
            scenario.assertions,
            scenario.budgets,
            evidence,
            artifacts,
            metric_names=scenario.metrics,
            expected=scenario.expected,
            error=error,
            started=started,
            precomputed=step_assertions,
        )
        artifacts.write_case_result(result, evidence)
        if not options.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        return result

    def _evaluate(
        self,
        case_id: str,
        assertions: list[AssertionSpec],
        budgets: EvalBudgets,
        evidence: EvalEvidence,
        artifacts: EvalArtifactStore,
        *,
        metric_names: list[str] | None = None,
        expected: dict | None = None,
        error: str | None,
        started: float,
        precomputed: list[AssertionResult] | None = None,
    ) -> EvalResult:
        metric_assertions = [
            spec for spec in assertions if spec.type == "metric"
        ]
        regular_assertions = [
            spec for spec in assertions if spec.type != "metric"
        ]
        assertion_results = [
            *(precomputed or []),
            *run_assertions(regular_assertions, evidence),
            *self._budget_assertions(budgets, evidence),
        ]
        metric_names_to_calculate = list(
            dict.fromkeys(
                [
                    *(metric_names or []),
                    *[
                        str(spec.options.get("metric"))
                        for spec in metric_assertions
                        if spec.options.get("metric")
                    ],
                ]
            )
        )
        metrics = {
            **self._metrics(evidence, assertion_results),
            **calculate_case_metrics(
                metric_names_to_calculate,
                expected or {},
                evidence,
                assertion_results,
            ),
        }
        assertion_results.extend(
            run_metric_assertions(metric_assertions, metrics)
        )
        dimensions = build_dimension_results(assertion_results)
        if error:
            overall = "execution_error"
        elif not assertion_results:
            overall = "invalid_case"
        elif required_assertions_passed(assertion_results):
            overall = "passed"
        else:
            overall = "failed"
        return EvalResult(
            case_id=case_id,
            overall=overall,
            session_id=evidence.session_id,
            run_ids=evidence.run_ids,
            dimensions=dimensions,
            failure_categories=failure_categories(assertion_results),
            metrics=metrics,
            artifact_dir=str(artifacts.case_dir(case_id)),
            error=error,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _execute_prompt(
        self,
        runtime: RuntimeService,
        session_id: str,
        text: str,
        timeout_seconds: int,
        evidence: EvalEvidence,
    ) -> None:
        events: list[dict] = []
        try:
            await asyncio.wait_for(
                self._consume_prompt_stream(
                    runtime,
                    session_id,
                    text,
                    events,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            await runtime.cancel_run(session_id)
            raise TimeoutError(
                f"Prompt timed out after {timeout_seconds}s"
            ) from None
        self._refresh_audit_evidence(runtime, session_id, evidence)

    @staticmethod
    async def _consume_prompt_stream(
        runtime: RuntimeService,
        session_id: str,
        text: str,
        events: list[dict],
    ) -> None:
        async for event in runtime.send_message(
            session_id,
            UserInput(text=text),
        ):
            events.append(dict(event))

    @staticmethod
    async def _consume_continue(
        runtime: RuntimeService,
        session_id: str,
    ) -> None:
        async for _ in runtime.continue_session(session_id):
            pass

    @staticmethod
    async def _wait_until_running(
        runtime: RuntimeService,
        session_id: str,
    ) -> None:
        for _ in range(100):
            if runtime.get_session_status(session_id).is_running:
                return
            await asyncio.sleep(0.01)
        raise TimeoutError("Background prompt did not start")

    @staticmethod
    def _refresh_audit_evidence(
        runtime: RuntimeService,
        session_id: str,
        evidence: EvalEvidence,
    ) -> None:
        by_id = {bundle.run_id: bundle for bundle in evidence.audit_bundles}
        for result in runtime.list_runs(session_id):
            run_id = result.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            events = runtime.get_run_events(session_id, run_id)
            if hasattr(runtime, "get_run_audit_bundle"):
                bundle = runtime.get_run_audit_bundle(session_id, run_id)
            else:
                state = {
                    "run_id": run_id,
                    "session_id": session_id,
                    "status": result.get("status"),
                    "stop_reason": result.get("stop_reason"),
                    "workspace_path": str(evidence.workspace),
                }
                bundle = AuditBundle(
                    run_id=run_id,
                    session_id=session_id,
                    events=events,
                    state=state,
                    result=result,
                    report=build_audit_report(
                        result,
                        events=events,
                        state=state,
                    ),
                    workspace=evidence.workspace,
                )
            by_id[run_id] = bundle
        evidence.audit_bundles = list(by_id.values())

    @staticmethod
    def _inspect_runtime(
        runtime: RuntimeService,
        session_id: str,
    ) -> dict:
        if hasattr(runtime, "get_session_recovery_state"):
            return runtime.get_session_recovery_state(session_id)
        return {
            "session_id": session_id,
            "freshness": runtime.get_session_freshness(session_id),
            "run_ids": [
                result.get("run_id")
                for result in runtime.list_runs(session_id)
            ],
        }

    @staticmethod
    def _metrics(
        evidence: EvalEvidence,
        assertion_results: list[AssertionResult],
    ) -> dict:
        run_summaries = [
            bundle.report.get("run", {}).get("summary", {})
            for bundle in evidence.audit_bundles
        ]
        return {
            "runtime.run_count": len(evidence.audit_bundles),
            "runtime.assertion_count": len(assertion_results),
            "runtime.assertions_passed": sum(
                result.status == "passed" for result in assertion_results
            ),
            "runtime.model_attempts": sum(
                int(summary.get("model_attempts", 0) or 0)
                for summary in run_summaries
            ),
            "runtime.tool_calls": sum(
                int(summary.get("tool_calls", 0) or 0)
                for summary in run_summaries
            ),
            "runtime.changed_path_count": len(evidence.changes),
        }

    @staticmethod
    def _budget_assertions(
        budgets: EvalBudgets,
        evidence: EvalEvidence,
    ) -> list[AssertionResult]:
        summaries = [
            bundle.report.get("run", {}).get("summary", {})
            for bundle in evidence.audit_bundles
        ]
        actual = {
            "model_attempts": sum(
                int(item.get("model_attempts", 0) or 0)
                for item in summaries
            ),
            "tool_calls": sum(
                int(item.get("tool_calls", 0) or 0)
                for item in summaries
            ),
            "replans": sum(
                int(
                    bundle.report.get("task", {})
                    .get("decision_counts", {})
                    .get("replan", 0)
                    or 0
                )
                for bundle in evidence.audit_bundles
            ),
        }
        configured = {
            "model_attempts": budgets.max_model_attempts,
            "tool_calls": budgets.max_tool_calls,
            "replans": budgets.max_replans,
        }
        results = []
        for metric, limit in configured.items():
            if limit is None:
                continue
            passed = actual[metric] <= limit
            results.append(
                AssertionResult(
                    name=f"budget_{metric}",
                    dimension="efficiency",
                    status="passed" if passed else "failed",
                    summary=(
                        f"{metric} within budget: {actual[metric]} <= {limit}"
                        if passed
                        else f"{metric} exceeded budget: {actual[metric]} > {limit}"
                    ),
                    expected={"maximum": limit},
                    actual={"value": actual[metric]},
                    evidence_refs=[
                        f"run:{bundle.run_id}"
                        for bundle in evidence.audit_bundles
                    ],
                    required=False,
                    role="diagnostic",
                )
            )
        return results

    def _prepare_workspace(
        self,
        case_id: str,
        fixture: str,
        options: EvalRunOptions,
        artifacts: EvalArtifactStore,
    ) -> Path:
        fixtures_root = Path(options.fixtures_root).resolve()
        source = (fixtures_root / fixture).resolve()
        if source != fixtures_root and fixtures_root not in source.parents:
            raise ValueError(f"Fixture escapes fixtures root: {fixture}")
        if not source.is_dir():
            raise FileNotFoundError(
                f"Fixture directory not found: {source}"
            )
        workspace = artifacts.workspace_dir(case_id)
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(
            source,
            workspace,
            ignore=shutil.ignore_patterns(
                ".git",
                ".codepilot",
                "__pycache__",
                ".pytest_cache",
            ),
        )
        return workspace

    @staticmethod
    def _apply_runtime_profile(
        session_options: CreateAgentSessionOptions,
        profile: EvalRuntimeProfile,
    ) -> CreateAgentSessionOptions:
        """应用评估运行时配置（不修改调用方的选项）。"""

        return replace(
            session_options,
            tool_permission_mode=profile.permission_mode,
            # Eval security cases need mutating tools to remain visible so
            # PermissionPolicy can emit denied/approval evidence instead of
            # turning the call into "Tool not found".
            read_only_mode=False,
            context_governance_enabled=profile.context_governance_enabled,
            memory_enabled=profile.memory_enabled,
            task_control_enabled=profile.task_control_enabled,
            approval_provider=(
                _EvalAutoApprovalProvider()
                if profile.auto_approve
                else session_options.approval_provider
            ),
        )

    @staticmethod
    def _modify_file(
        workspace: Path,
        fixture: str,
        step: ScenarioStep,
        options: EvalRunOptions,
    ) -> None:
        target = _safe_child(workspace, str(step.options["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        if "content" in step.options:
            target.write_text(
                str(step.options["content"]),
                encoding="utf-8",
            )
            return
        fixture_root = (
            Path(options.fixtures_root).resolve() / fixture
        ).resolve()
        source = _safe_child(
            fixture_root,
            str(step.options["source"]),
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"Mutation source not found: {source}"
            )
        shutil.copyfile(source, target)


def _safe_child(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"Path escapes root: {relative}")
    return target


class _EvalAutoApprovalProvider:
    """评估专用审批器：用于无人值守 coding-fix benchmark。

    安全阻断仍由 PermissionPolicy 先执行；只有策略判定为
    approval_required 的调用会来到这里。
    """

    async def request_approval(
        self,
        request: object,
        metadata: object,
        decision: object,
    ) -> ApprovalDecision:
        return ApprovalDecision(approved=True, reason="eval_auto_approved")
