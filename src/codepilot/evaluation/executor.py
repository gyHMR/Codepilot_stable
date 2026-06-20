from __future__ import annotations

"""Execution engine for Eval cases and recovery scenarios."""

import asyncio
import shutil
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Callable

from codepilot.runtime import RuntimeService, UserInput

from .artifacts import EvalArtifactStore
from .types import (
    EvalCase,
    EvalResult,
    EvalRunOptions,
    EvalScenario,
    EvaluationEvidence,
    ScenarioStep,
    VerifierResult,
    VerifierSpec,
)
from .verifier import capture_workspace_baseline, run_verifiers


class EvaluationExecutor:
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
        workspace = self._prepare_workspace(case.id, case.fixture, options, artifacts)
        evidence = EvaluationEvidence(
            workspace=workspace,
            baseline=capture_workspace_baseline(workspace),
        )
        runtime = self.runtime_factory()
        session_options = replace(
            options.session_options,
            workspace_dir=workspace,
            session_id=None,
        )
        handle = runtime.create_session(session_options)
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
            evidence.freshness = runtime.get_session_freshness(
                handle.session_id
            )
            evidence.freshness_history.append(dict(evidence.freshness))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            await runtime.aclose_all()

        verifier_results = run_verifiers(case.verifiers, evidence)
        verdict = _case_verdict(case, verifier_results, error)
        result = EvalResult(
            case_id=case.id,
            verdict=verdict,
            session_id=evidence.session_id,
            run_ids=evidence.run_ids,
            verifier_results=verifier_results,
            artifact_dir=str(artifacts.case_dir(case.id)),
            error=error,
            duration_ms=int((time.perf_counter() - started) * 1000),
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
        evidence = EvaluationEvidence(
            workspace=workspace,
            baseline=capture_workspace_baseline(workspace),
        )
        runtime = self.runtime_factory()
        session_options = replace(
            options.session_options,
            workspace_dir=workspace,
            session_id=None,
        )
        handle = runtime.create_session(session_options)
        session_id = handle.session_id
        evidence.session_id = session_id
        step_verifiers: list[VerifierResult] = []
        pending_task: asyncio.Task[None] | None = None
        pending_events: list[dict] = []
        error: str | None = None

        try:
            for index, step in enumerate(scenario.steps):
                if step.type == "prompt":
                    text = str(step.options["text"])
                    background = bool(step.options.get("background", False))
                    if background:
                        if pending_task is not None and not pending_task.done():
                            raise RuntimeError("A background prompt is already running")
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
                            {"step": index, "type": "prompt", "background": True}
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
                            {"step": index, "type": "prompt", "completed": True}
                        )
                elif step.type == "cancel":
                    cancelled = await runtime.cancel_run(session_id)
                    if pending_task is not None:
                        with suppress(asyncio.CancelledError):
                            await pending_task
                        self._record_captured_events(
                            pending_events,
                            evidence,
                        )
                        pending_task = None
                        pending_events = []
                    self._refresh_run_evidence(runtime, session_id, evidence)
                    evidence.step_results.append(
                        {"step": index, "type": "cancel", "cancelled": cancelled}
                    )
                elif step.type == "modify_file":
                    self._modify_file(
                        workspace,
                        scenario.fixture,
                        step,
                        options,
                    )
                    evidence.freshness = runtime.get_session_freshness(session_id)
                    evidence.freshness_history.append(
                        dict(evidence.freshness)
                    )
                    evidence.step_results.append(
                        {
                            "step": index,
                            "type": "modify_file",
                            "path": step.options["path"],
                            "freshness": evidence.freshness,
                        }
                    )
                elif step.type == "restart":
                    if pending_task is not None and not pending_task.done():
                        raise RuntimeError("Cannot restart with a background prompt")
                    await runtime.aclose_all()
                    runtime = self.runtime_factory()
                    restored_options = replace(
                        session_options,
                        session_id=session_id,
                    )
                    runtime.create_session(restored_options)
                    evidence.step_results.append(
                        {"step": index, "type": "restart", "session_id": session_id}
                    )
                elif step.type == "continue":
                    events = await asyncio.wait_for(
                        self._consume_continue(runtime, session_id),
                        timeout=scenario.timeout_seconds,
                    )
                    self._record_captured_events(events, evidence)
                    self._refresh_run_evidence(runtime, session_id, evidence)
                    evidence.step_results.append(
                        {"step": index, "type": "continue", "completed": True}
                    )
                elif step.type == "verify":
                    raw = step.options["verifier"]
                    verifier_spec = VerifierSpec(
                        type=raw["type"],
                        options={
                            key: value for key, value in raw.items()
                            if key != "type"
                        },
                    )
                    step_verifiers.extend(run_verifiers([verifier_spec], evidence))
                    evidence.step_results.append(
                        {"step": index, "type": "verify"}
                    )
            evidence.freshness = runtime.get_session_freshness(session_id)
            evidence.freshness_history.append(dict(evidence.freshness))
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if pending_task is not None and not pending_task.done():
                await runtime.cancel_run(session_id)
                pending_task.cancel()
                with suppress(asyncio.CancelledError):
                    await pending_task
                self._record_captured_events(pending_events, evidence)
            await runtime.aclose_all()

        verifier_results = [
            *step_verifiers,
            *run_verifiers(scenario.verifiers, evidence),
        ]
        verdict = _scenario_verdict(verifier_results, error)
        result = EvalResult(
            case_id=scenario.id,
            verdict=verdict,
            session_id=session_id,
            run_ids=evidence.run_ids,
            verifier_results=verifier_results,
            artifact_dir=str(artifacts.case_dir(scenario.id)),
            error=error,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        artifacts.write_case_result(result, evidence)
        if not options.keep_workspace:
            shutil.rmtree(workspace, ignore_errors=True)
        return result

    async def _execute_prompt(
        self,
        runtime: RuntimeService,
        session_id: str,
        text: str,
        timeout_seconds: int,
        evidence: EvaluationEvidence,
    ) -> None:
        events = await self._consume_prompt(
            runtime,
            session_id,
            text,
            timeout_seconds,
        )
        self._record_captured_events(events, evidence)
        self._refresh_run_evidence(runtime, session_id, evidence)

    async def _consume_prompt(
        self,
        runtime: RuntimeService,
        session_id: str,
        text: str,
        timeout_seconds: int,
    ) -> list[dict]:
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
        return events

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

    async def _consume_continue(
        self,
        runtime: RuntimeService,
        session_id: str,
    ) -> list[dict]:
        events: list[dict] = []
        async for event in runtime.continue_session(session_id):
            events.append(dict(event))
        return events

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
    def _record_captured_events(
        events: list[dict],
        evidence: EvaluationEvidence,
    ) -> None:
        by_run: dict[str, list[dict]] = {}
        for event in events:
            run_id = event.get("runId") or event.get("run_id")
            if isinstance(run_id, str) and run_id:
                by_run.setdefault(run_id, []).append(event)
        for run_id, run_events in by_run.items():
            if run_id not in evidence.run_ids:
                evidence.run_ids.append(run_id)
            evidence.run_events.setdefault(run_id, []).extend(run_events)

    @staticmethod
    def _refresh_run_evidence(
        runtime: RuntimeService,
        session_id: str,
        evidence: EvaluationEvidence,
    ) -> None:
        for result in runtime.list_runs(session_id):
            run_id = result.get("run_id")
            if not isinstance(run_id, str) or not run_id:
                continue
            if run_id not in evidence.run_ids:
                evidence.run_ids.append(run_id)
            evidence.run_results[run_id] = result
            evidence.run_events[run_id] = runtime.get_run_events(
                session_id,
                run_id,
            )

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
            raise FileNotFoundError(f"Fixture directory not found: {source}")
        workspace = artifacts.workspace_dir(case_id)
        if workspace.exists():
            shutil.rmtree(workspace)
        shutil.copytree(
            source,
            workspace,
            ignore=shutil.ignore_patterns(".git", ".codepilot", "__pycache__"),
        )
        return workspace

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
            target.write_text(str(step.options["content"]), encoding="utf-8")
            return
        fixture_root = (
            Path(options.fixtures_root).resolve() / fixture
        ).resolve()
        source = _safe_child(fixture_root, str(step.options["source"]))
        if not source.is_file():
            raise FileNotFoundError(f"Mutation source not found: {source}")
        shutil.copyfile(source, target)


def _safe_child(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if target != resolved_root and resolved_root not in target.parents:
        raise ValueError(f"Path escapes root: {relative}")
    return target


def _case_verdict(
    case: EvalCase,
    results: list[VerifierResult],
    error: str | None,
):
    if error:
        return "harness_failed"
    if not results or all(item.status == "skipped" for item in results):
        return "invalid_case"
    failed = [item for item in results if item.status in {"failed", "error"}]
    if not failed:
        return "passed"
    if case.category == "harness" or any(
        item.name in {"run", "trace"} for item in failed
    ):
        return "harness_failed"
    return "task_failed"


def _scenario_verdict(
    results: list[VerifierResult],
    error: str | None,
):
    if not results or all(item.status == "skipped" for item in results):
        return "invalid_case"
    if error or any(
        item.status in {"failed", "error"} for item in results
    ):
        return "recovery_failed"
    return "passed"
