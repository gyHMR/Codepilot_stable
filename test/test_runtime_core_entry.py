from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from codepilot.core.contracts import CoreOutcome, CoreRunInput, CoreReason, ModelEntry
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AssistantMessage, Model, TextContent, UserMessage
from codepilot.runtime.environment import RunEnvironment, RunResourceScope
from codepilot.runtime.executor import RunExecutionCompleted, RunExecutor


def _model() -> Model:
    return Model(
        id="unit",
        name="Unit",
        api="unit",
        provider="unit",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def _core_input(run_id: str = "run_core_entry") -> CoreRunInput:
    return CoreRunInput(
        session_id="session_core_entry",
        run_id=run_id,
        entry=ModelEntry(),
        messages=(UserMessage(content="inspect the project"),),
        state=CoreState.new("inspect the project"),
        mode="read",
        model=ModelDescriptor(provider="unit", model_id="unit"),
    )


def _environment(run_id: str = "run_core_entry") -> RunEnvironment:
    resources = RunResourceScope()
    return RunEnvironment(
        run_id=run_id,
        session_id="session_core_entry",
        trigger="prompt",
        model=SimpleNamespace(stream=lambda _request: None),
        tools=None,
        context=None,
        state=None,
        cancellation=resources.cancellation,
        deadline_at_ms=None,
        event_sink=None,
        resources=resources,
    )


def test_run_executor_invokes_run_core_directly(monkeypatch) -> None:
    from codepilot.runtime import executor as executor_module

    received: list[CoreRunInput] = []

    async def execute(input_value, _ports):
        received.append(input_value)
        return CoreOutcome(
            status="completed",
            reason=CoreReason("task.completed"),
            state=input_value.state,
            final_message=AssistantMessage(content=[TextContent(text="done")]),
        )

    monkeypatch.setattr(executor_module, "run_core", execute, raising=False)
    core_input = _core_input()

    async def run_case():
        updates = [
            update
            async for update in RunExecutor().execute(
                _environment(),
                SimpleNamespace(loop_input=core_input),
            )
        ]
        return next(
            update.outcome
            for update in updates
            if isinstance(update, RunExecutionCompleted)
        )

    outcome = asyncio.run(run_case())

    assert received == [core_input]
    assert outcome.status == "completed"


def test_main_run_preparation_produces_core_input(tmp_path) -> None:
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
    from codepilot.sessions.contracts import SessionOptions, SessionRunIntent

    async def run_case():
        coordinator = RuntimeSessionCoordinator(
            SessionOptions(
                model=_model(),
                workspace_dir=tmp_path,
                session_id="session_main_core_entry",
                memory_enabled=False,
            )
        )
        try:
            return await coordinator._prepare_run(  # noqa: SLF001
                SessionRunIntent(text="inspect the project", request_id="request_1"),
                run_id="run_main_core_entry",
                model=ModelDescriptor(provider="unit", model_id="unit"),
            )
        finally:
            coordinator.close()

    prepared = asyncio.run(run_case())

    assert isinstance(prepared.loop_input, CoreRunInput)
    assert isinstance(prepared.loop_input.entry, ModelEntry)
    assert prepared.context_port is not None
    assert prepared.context_port.__class__.__name__ == "ContextService"
    assert not hasattr(prepared.loop_input, "approval_id")
    assert not hasattr(prepared.loop_input, "retry_policy")


def test_subagent_preparation_produces_core_input(monkeypatch, tmp_path) -> None:
    from codepilot.core.contracts import CoreOutcome, CoreReason
    from codepilot.runtime.subagents.runner import ExplorationTask, SubagentRunner

    captured: list[object] = []
    report = json.dumps(
        {
            "status": "completed",
            "summary": "found the target",
            "findings": [],
            "relevant_files": [],
            "evidence": [],
            "risks": [],
            "suggested_plan_notes": [],
            "open_questions": [],
            "confidence": 1.0,
        }
    )

    async def execute(_self, _environment, prepared):
        _environment.lifecycle.transition("preparing")
        _environment.lifecycle.transition("executing")
        captured.append(prepared.loop_input)
        yield RunExecutionCompleted(
            CoreOutcome(
                status="completed",
                reason=CoreReason("task.completed"),
                state=prepared.loop_input.state,
                final_message=AssistantMessage(content=[TextContent(text=report)]),
            )
        )

    monkeypatch.setattr(RunExecutor, "execute", execute)
    runner = SubagentRunner(
        workspace=tmp_path,
        session_id="session_subagent_core_entry",
        model=ModelDescriptor(provider="unit", model_id="unit"),
        model_port=SimpleNamespace(),
        tool_port=SimpleNamespace(),
    )
    task = ExplorationTask(
        task_id="task_1",
        subagent_id="reader",
        purpose="inspect",
        instruction="inspect the project",
        scope_key="scope_1",
    )

    result = asyncio.run(runner.run(task, peer_assignments=[]))

    assert result["status"] == "completed"
    assert len(captured) == 1
    assert isinstance(captured[0], CoreRunInput)
