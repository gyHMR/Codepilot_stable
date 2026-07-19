from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest

from codepilot.runtime.contracts import (
    external_status,
    external_stop_reason,
    project_core_counters,
    project_core_domain_event,
    project_core_signals,
    terminal_outcome_for_status,
)
from codepilot.runtime.environment import RunEnvironment, RunResourceScope
from codepilot.runtime.errors import runtime_error_info, runtime_error_payload
from codepilot.runtime.executor import RunExecutionCompleted, RunExecutor
from codepilot.runtime.lifecycle import RuntimeLifecycle
from codepilot.core.contracts import (
    CoreOutcome,
    CoreReason,
    CoreWait,
)
from codepilot.core.errors import (
    CoreBoundaryCommitError,
    CoreContractError,
    CoreInvariantError,
)
from codepilot.core.events import CoreDomainEvent
from codepilot.core.state import CoreCounters, CoreState, RunFacts, WorkspaceFacts
from codepilot.protocols import AssistantMessage, TextContent


def _environment() -> RunEnvironment:
    resources = RunResourceScope()
    return RunEnvironment(
        run_id="run_1",
        session_id="session_1",
        trigger="prompt",
        model=None,
        tools=None,
        context=None,
        state=None,
        cancellation=resources.cancellation,
        deadline_at_ms=None,
        event_sink=None,
        resources=resources,
    )


async def _collect_execution(executor, environment, prepared):
    updates = [update async for update in executor.execute(environment, prepared)]
    completed = [
        update for update in updates if isinstance(update, RunExecutionCompleted)
    ]
    assert len(completed) == 1
    return completed[0].outcome


def _completed_core_outcome() -> CoreOutcome:
    return CoreOutcome(
        status="completed",
        reason=CoreReason("task.completed"),
        state=CoreState.new("test Runtime execution"),
        final_message=AssistantMessage(content=[TextContent(text="done")]),
    )


def test_runtime_lifecycle_accepts_normal_and_resume_paths() -> None:
    lifecycle = RuntimeLifecycle("run_1")
    lifecycle.transition("preparing")
    lifecycle.transition("executing")
    lifecycle.transition("waiting")

    with pytest.raises(ValueError, match="waiting -> executing"):
        lifecycle.transition("executing")

    lifecycle.transition("resuming")
    lifecycle.transition("executing")
    lifecycle.transition("finalizing")
    lifecycle.transition("terminal", terminal_outcome="completed")
    lifecycle.mark_released()
    lifecycle.mark_released()

    assert lifecycle.state == "released"
    assert lifecycle.terminal_outcome == "completed"


def test_runtime_lifecycle_requires_terminal_commit_semantics() -> None:
    lifecycle = RuntimeLifecycle("run_1")
    lifecycle.transition("preparing")
    lifecycle.transition("executing")
    lifecycle.transition("finalizing")

    with pytest.raises(ValueError, match="requires terminal_outcome"):
        lifecycle.transition("terminal")
    with pytest.raises(ValueError, match="finalizing -> released"):
        lifecycle.mark_released()

    lifecycle.transition("terminal", terminal_outcome="failed")
    with pytest.raises(ValueError, match="terminal -> finalizing"):
        lifecycle.transition("finalizing")


def test_runtime_lifecycle_uses_single_cancellation_path() -> None:
    lifecycle = RuntimeLifecycle("run_1")
    lifecycle.transition("preparing")
    lifecycle.transition("executing")
    lifecycle.transition("cancelling")
    lifecycle.transition("finalizing")
    lifecycle.transition("terminal", terminal_outcome="cancelled")

    assert lifecycle.terminal_outcome == "cancelled"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("completed", "completed"),
        ("failed", "failed"),
        ("cancelled", "cancelled"),
        ("aborted", "cancelled"),
        ("waiting_approval", None),
        ("waiting_user", None),
    ],
)
def test_terminal_outcome_mapping_has_one_runtime_vocabulary(status, expected) -> None:
    assert terminal_outcome_for_status(status) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("task.completed", "final_answer"),
        ("tool.approval_required", "approval_required"),
        ("plan.confirmation_required", "plan_approval_required"),
        ("run.max_model_turns", "max_iterations"),
        ("run.max_tool_calls", "tool_call_limit"),
        ("run.repeated_tool_call", "repeated_tool_call"),
        ("tool.unavailable", "tool_unavailable"),
        ("run.cancelled", "aborted"),
        ("verification.failed", "completion_blocked"),
    ],
)
def test_structured_core_reason_maps_without_parsing_message(code, expected) -> None:
    assert external_stop_reason(CoreReason(code, message="localized text")) == expected


def test_runtime_projects_external_views_from_core_outcome() -> None:
    state = CoreState(
        task=CoreState.new("fix app").task,
        facts=RunFacts(
            counters=CoreCounters(model_turns=2, tool_iterations=1, tool_calls=3),
            workspace=WorkspaceFacts(
                revision=1,
                changed=True,
                affected_paths=("src/app.py",),
            ),
        ),
    )
    wait = CoreWait(
        "tool_approval",
        "approval_1",
        CoreReason("tool.approval_required", source="tools", recoverable=True),
    )

    outcome = CoreOutcome(
        status="waiting",
        reason=wait.reason,
        state=state,
        wait=wait,
    )

    assert external_status(outcome) == "waiting_approval"
    assert external_stop_reason(outcome.reason) == "approval_required"
    assert project_core_counters(outcome).model_attempts == 2
    assert outcome.state.facts.workspace.affected_paths == ("src/app.py",)
    assert project_core_signals(outcome).approval_required is True


def test_runtime_event_projection_adds_envelope_to_domain_event() -> None:
    event = CoreDomainEvent(
        "verification_recorded",
        {"status": "passed"},
        evidence_refs=("call_test",),
    )

    projected = project_core_domain_event(
        event,
        event_id="run_1:core:0",
        run_id="run_1",
        session_id="session_1",
    )

    assert projected == {
        "event_id": "run_1:core:0",
        "run_id": "run_1",
        "session_id": "session_1",
        "type": "verification_recorded",
        "status": "passed",
        "evidence_refs": ["call_test"],
    }


def test_runtime_model_conversion_drops_orphan_tool_results() -> None:
    from codepilot.protocols import ToolCall, ToolResultMessage, UserMessage
    from codepilot.runtime.model import convert_to_llm

    messages = [
        UserMessage(content="continue"),
        ToolResultMessage(
            tool_call_id="missing_call",
            tool_name="read_file",
            content=[TextContent(text="orphan output")],
        ),
        AssistantMessage(
            content=[
                TextContent(text="I can continue."),
                ToolCall(
                    id="kept_call",
                    name="read_file",
                    arguments={"path": "README.md"},
                ),
                ToolCall(
                    id="dropped_call",
                    name="read_file",
                    arguments={"path": "old.md"},
                ),
            ],
            stop_reason="toolUse",
        ),
        ToolResultMessage(
            tool_call_id="kept_call",
            tool_name="read_file",
            content=[TextContent(text="paired output")],
        ),
    ]

    converted = convert_to_llm(messages)

    assert not any(
        isinstance(message, ToolResultMessage)
        and message.tool_call_id == "missing_call"
        for message in converted
    )
    assistant = next(
        message for message in converted if isinstance(message, AssistantMessage)
    )
    tool_calls = [block for block in assistant.content if isinstance(block, ToolCall)]
    assert [call.id for call in tool_calls] == ["kept_call"]
    assert isinstance(converted[-1], ToolResultMessage)
    assert converted[-1].tool_call_id == "kept_call"


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (CoreContractError("bad input"), "runtime.core_contract_error"),
        (CoreInvariantError("bad state"), "runtime.core_invariant_error"),
    ],
)
def test_runtime_error_adapter_distinguishes_core_faults(error, code) -> None:
    info = runtime_error_info(error)

    assert info is not None
    assert info.code == code
    assert info.details["error_type"] == type(error).__name__


def test_runtime_error_adapter_preserves_failed_core_reason() -> None:
    reason = CoreReason(
        "plan.reconciliation_exhausted",
        message="The active Task Plan still has incomplete steps.",
        evidence_refs=("plan:1:step:1",),
        details={"pending_steps": 1},
    )

    info = runtime_error_info(reason)
    payload = runtime_error_payload(reason)

    assert info is not None
    assert info.code == "plan.reconciliation_exhausted"
    assert info.source == "core"
    assert info.details == {
        "pending_steps": 1,
        "evidence_refs": ["plan:1:step:1"],
    }
    assert payload == {
        "code": "plan.reconciliation_exhausted",
        "message": "The active Task Plan still has incomplete steps.",
        "source": "core",
        "retryable": False,
        "details": {
            "pending_steps": 1,
            "evidence_refs": ["plan:1:step:1"],
        },
    }


def test_runtime_error_adapter_maps_core_domain_sources() -> None:
    info = runtime_error_info(
        CoreReason("tool.failed", source="tools", message="Tool execution failed")
    )

    assert info is not None
    assert info.source == "tool"
    assert info.details["reason_source"] == "tools"


def test_failed_outcome_uses_core_reason_for_persistence_and_runtime_frame() -> None:
    from codepilot.runtime.actions import FailedFrame
    from codepilot.runtime.coordinator import RunCoordinator
    from codepilot.runtime.gateway import RuntimeGateway

    outcome = CoreOutcome(
        status="failed",
        reason=CoreReason(
            "plan.reconciliation_exhausted",
            message="Plan reconciliation was not completed.",
        ),
        state=CoreState.new("finish the plan"),
    )
    coordinator = RunCoordinator(SimpleNamespace(session_id="session_1"))
    result = coordinator._agent_result_from_outcome(  # noqa: SLF001
        SimpleNamespace(run_id="run_1", input_messages=[]),
        outcome,
    )

    async def collect_frames():
        gateway = RuntimeGateway()
        return [
            frame
            async for frame in gateway._frames_from_outcome(  # noqa: SLF001
                SimpleNamespace(controller=SimpleNamespace()),
                outcome,
                SimpleNamespace(run_id="run_1"),
            )
        ]

    frames = asyncio.run(collect_frames())

    assert result.error is not None
    assert result.error.code == "plan.reconciliation_exhausted"
    assert len(frames) == 1
    assert isinstance(frames[0], FailedFrame)
    assert frames[0].error["code"] == "plan.reconciliation_exhausted"
    assert frames[0].error["source"] == "core"


def test_run_executor_preserves_structured_core_fault_code(monkeypatch) -> None:
    async def broken(_input, _ports):
        raise CoreInvariantError("state revision moved backwards")

    monkeypatch.setattr("codepilot.runtime.executor.run_core", broken)

    outcome = asyncio.run(
        _collect_execution(
            RunExecutor(),
            _environment(),
            SimpleNamespace(loop_input=SimpleNamespace(state=CoreState.new("test"))),
        )
    )

    assert outcome.status == "failed"
    assert outcome.error["code"] == "runtime.core_invariant_error"


def test_run_executor_does_not_terminalize_boundary_commit_failure(monkeypatch) -> None:
    async def broken(_input, _ports):
        raise CoreBoundaryCommitError("before_tools", RuntimeError("store unavailable"))

    monkeypatch.setattr("codepilot.runtime.executor.run_core", broken)

    with pytest.raises(CoreBoundaryCommitError, match="before_tools"):
        asyncio.run(
            _collect_execution(
                RunExecutor(),
                _environment(),
                SimpleNamespace(loop_input=SimpleNamespace(state=CoreState.new("test"))),
            )
        )


def test_run_executor_preserves_core_state_when_deadline_wins(monkeypatch) -> None:
    latest = CoreState.new("latest")
    environment = _environment()

    async def cancelled(_input, _ports):
        environment.cancellation.cancel("deadline_exceeded")
        return CoreOutcome(
            status="cancelled",
            reason=CoreReason("run.cancelled"),
            state=latest,
            error={"code": "run.cancelled", "message": "cancelled"},
        )

    monkeypatch.setattr("codepilot.runtime.executor.run_core", cancelled)
    prepared = SimpleNamespace(
        loop_input=SimpleNamespace(state=CoreState.new("initial"))
    )

    outcome = asyncio.run(_collect_execution(RunExecutor(), environment, prepared))

    assert outcome.status == "failed"
    assert outcome.state is latest
    assert external_stop_reason(outcome.reason) == "deadline_exceeded"


def test_runtime_projects_provider_attempt_count_separately_from_model_turns() -> None:
    state = CoreState.new("count attempts")
    state = replace(
        state,
        facts=replace(
            state.facts,
            counters=replace(
                state.facts.counters,
                model_turns=1,
                model_attempts=3,
            ),
        ),
    )
    outcome = CoreOutcome(
        status="completed",
        reason=CoreReason("task.completed"),
        state=state,
        final_message=AssistantMessage(content=[TextContent(text="done")]),
    )

    assert project_core_counters(outcome).model_attempts == 3


def test_run_executor_uses_same_core_entry_for_resume(monkeypatch) -> None:
    received = []

    async def execute(input_value, _ports):
        received.append(input_value)
        return _completed_core_outcome()

    monkeypatch.setattr("codepilot.runtime.executor.run_core", execute)
    loop_input = SimpleNamespace(entry="resume")
    environment = _environment()

    outcome = asyncio.run(
        _collect_execution(
            RunExecutor(), environment, SimpleNamespace(loop_input=loop_input)
        )
    )

    assert received == [loop_input]
    assert outcome.status == "completed"


def test_run_executor_binds_tool_checkpoint_reader_to_runtime_adapter(
    monkeypatch,
) -> None:
    class StateAdapter:
        reader = None

        def bind_tool_state(self, reader):
            self.reader = reader

        def commit(self, _boundary):
            return None

    class Tools:
        def checkpoint_state(self):
            return {"pending": "attempt_1"}

        def catalog_snapshot(self, *, mode=None):
            del mode

        def prepare_batch(self, _requests):
            return None

        async def execute_prepared(self, _batch_id):
            return ()

    state = StateAdapter()
    environment = _environment()
    environment = RunEnvironment(
        run_id=environment.run_id,
        session_id=environment.session_id,
        trigger=environment.trigger,
        model=environment.model,
        tools=Tools(),
        context=environment.context,
        state=state,
        cancellation=environment.cancellation,
        deadline_at_ms=environment.deadline_at_ms,
        event_sink=environment.event_sink,
        resources=environment.resources,
    )

    async def execute(_input, _ports):
        assert state.reader is not None
        assert state.reader() == {"pending": "attempt_1"}
        return _completed_core_outcome()

    monkeypatch.setattr("codepilot.runtime.executor.run_core", execute)

    outcome = asyncio.run(
        _collect_execution(
            RunExecutor(),
            environment,
            SimpleNamespace(loop_input=SimpleNamespace(state=CoreState.new("test"))),
        )
    )

    assert outcome.status == "completed"


def test_run_executor_isolates_live_event_sink_failures(monkeypatch) -> None:
    async def execute(_input, ports):
        assert ports.live_events is not None
        ports.live_events({"type": "model_delta", "text": "partial"})
        return _completed_core_outcome()

    def broken_sink(_event):
        raise RuntimeError("interface disconnected")

    monkeypatch.setattr("codepilot.runtime.executor.run_core", execute)
    base = _environment()
    environment = RunEnvironment(
        run_id=base.run_id,
        session_id=base.session_id,
        trigger=base.trigger,
        model=base.model,
        tools=base.tools,
        context=base.context,
        state=base.state,
        cancellation=base.cancellation,
        deadline_at_ms=base.deadline_at_ms,
        event_sink=broken_sink,
        resources=base.resources,
    )

    outcome = asyncio.run(
        _collect_execution(
            RunExecutor(),
            environment,
            SimpleNamespace(loop_input=SimpleNamespace(state=CoreState.new("test"))),
        )
    )

    assert outcome.status == "completed"
