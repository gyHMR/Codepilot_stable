from __future__ import annotations

from dataclasses import replace

import pytest

from codepilot.core.contracts import (
    CallModel,
    CoreBoundary,
    CoreDirective,
    CoreLimits,
    CoreOutcome,
    CorePorts,
    CoreReason,
    CoreRunInput,
    CoreWait,
    ModelEntry,
    PreparedModelContext,
    ToolResultEntry,
)
from codepilot.core.errors import CoreContractError
from codepilot.core.events import CoreDomainEvent
from codepilot.core.state import CORE_STATE_SCHEMA_VERSION, CoreState, load_core_state
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import AssistantMessage, TextContent, UserMessage
from codepilot.tools.results import ToolResult


class _ModelPort:
    async def stream(self, _request):
        if False:
            yield None


class _ContextPort:
    async def prepare(self, request):
        return PreparedModelContext(
            system_prompt="",
            messages=request.messages,
            tools=(),
            projection_ref="context:test",
        )


class _BoundaryPort:
    async def commit(self, _boundary):
        return None


def _state() -> CoreState:
    return CoreState.new("inspect the repository")


def _input(**changes) -> CoreRunInput:
    values = {
        "session_id": "session_contracts",
        "run_id": "run_contracts",
        "entry": ModelEntry(),
        "messages": (UserMessage(content="inspect the repository"),),
        "state": _state(),
        "mode": "build",
        "model": ModelDescriptor(provider="unit-test", model_id="test-model"),
        "limits": CoreLimits(),
        "context_seed": {"source": "session"},
    }
    values.update(changes)
    return CoreRunInput(**values)


def test_core_run_input_requires_a_typed_entry_and_freezes_inputs() -> None:
    run_input = _input()

    assert isinstance(run_input.entry, ModelEntry)
    assert run_input.session_id == "session_contracts"
    assert run_input.state.schema_version == CORE_STATE_SCHEMA_VERSION
    assert run_input.context_seed == {"source": "session"}

    with pytest.raises(CoreContractError, match="entry"):
        _input(entry="prompt")
    with pytest.raises(ValueError, match="session_id"):
        _input(session_id="")
    with pytest.raises(TypeError):
        run_input.context_seed["source"] = "changed"  # type: ignore[index]


def test_tool_result_entry_accepts_only_final_canonical_results() -> None:
    result = ToolResult(
        tool_call_id="call_1",
        tool_name="read",
        status="success",
        registration_id="read@1",
    )

    entry = ToolResultEntry(results=(result,))

    assert entry.results == (result,)
    with pytest.raises(CoreContractError, match="at least one"):
        ToolResultEntry()
    with pytest.raises(CoreContractError, match="canonical ToolResult"):
        ToolResultEntry(results=(object(),))  # type: ignore[arg-type]


def test_current_sessions_v2_payload_is_loaded_into_the_new_schema() -> None:
    legacy = {
        "counters": {
            "model_attempts": 2,
            "tool_iterations": 1,
            "tool_calls": 1,
        },
        "workspace_changed": True,
        "affected_paths": ["src/app.py"],
        "verification_status": "passed",
        "verification": [
            {
                "tool_call_id": "call_test",
                "tool_name": "shell",
                "status": "passed",
                "command": "pytest -q",
            }
        ],
        "seen_tool_call_ids": ["call_test"],
    }

    run_input = _input(
        state=legacy,
        messages=(UserMessage(content="fix app"),),
    )
    state = run_input.state

    assert state.schema_version == CORE_STATE_SCHEMA_VERSION
    assert state.task.original_request == "fix app"
    assert state.facts.counters.model_turns == 2
    assert state.facts.workspace.affected_paths == ("src/app.py",)
    assert state.facts.verification.status == "passed"
    assert state.facts.verification.verified_revision == 1
    assert "workspace_changed" not in state.to_dict()


def test_unknown_serialized_core_schema_is_not_treated_as_legacy() -> None:
    with pytest.raises(CoreContractError, match="schema"):
        load_core_state(
            {"schema_version": 99},
            original_request="inspect",
        )


def test_core_wait_and_reason_are_structured_and_immutable() -> None:
    reason = CoreReason(
        code="tool.approval_required",
        source="tools",
        recoverable=True,
        details={"risk": "high"},
    )
    wait = CoreWait(
        kind="tool_approval",
        request_id="approval_1",
        reason=reason,
        payload={"tool_name": "shell"},
    )

    assert wait.reason.details == {"risk": "high"}
    with pytest.raises(TypeError):
        wait.payload["tool_name"] = "write"  # type: ignore[index]
    with pytest.raises(ValueError, match="Unknown wait kind"):
        replace(wait, kind="approval")  # type: ignore[arg-type]


def test_core_boundary_contains_only_core_authority() -> None:
    event = CoreDomainEvent("verification_recorded", {"status": "passed"})
    boundary = CoreBoundary(
        kind="after_tools",
        state=_state(),
        domain_events=(event,),
    )

    assert boundary.domain_events == (event,)
    assert not hasattr(boundary, "tool_recovery_state")
    assert not hasattr(boundary, "context_checkpoint")
    assert not hasattr(boundary, "workspace_checkpoint")

    with pytest.raises(CoreContractError, match="Waiting boundary"):
        CoreBoundary(kind="waiting", state=_state())
    with pytest.raises(CoreContractError, match="only valid"):
        CoreBoundary(
            kind="after_model",
            state=_state(),
            wait=CoreWait(
                "continuation",
                "continue_1",
                CoreReason("run.max_model_turns", recoverable=True),
            ),
        )
    with pytest.raises(CoreContractError, match="boundary kind"):
        CoreBoundary(kind="before_finalization", state=_state())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Runtime envelope"):
        CoreDomainEvent("bad_event", {"run_id": "run_1"})


def test_core_outcome_enforces_waiting_and_completion_invariants() -> None:
    wait = CoreWait(
        "user_input",
        "question_1",
        CoreReason("task.user_input_required", recoverable=True),
    )
    waiting = CoreOutcome(
        status="waiting",
        reason=wait.reason,
        state=_state(),
        wait=wait,
    )
    final = AssistantMessage(content=[TextContent(text="done")])
    completed = CoreOutcome(
        status="completed",
        reason=CoreReason("task.completed"),
        state=_state(),
        new_messages=(final,),
        final_message=final,
    )

    assert waiting.wait is wait
    assert completed.final_message is final
    with pytest.raises(CoreContractError, match="waiting outcome"):
        replace(waiting, wait=None)
    with pytest.raises(CoreContractError, match="only valid"):
        replace(completed, wait=wait)
    with pytest.raises(CoreContractError, match="final_message"):
        replace(completed, final_message=None)


def test_core_ports_require_model_context_and_boundary_capabilities() -> None:
    ports = CorePorts(
        model=_ModelPort(),
        context=_ContextPort(),
        boundary=_BoundaryPort(),
    )

    assert ports.tools is None
    with pytest.raises(CoreContractError, match="model"):
        replace(ports, model=None)  # type: ignore[arg-type]
    with pytest.raises(CoreContractError, match="context"):
        replace(ports, context=None)  # type: ignore[arg-type]
    with pytest.raises(CoreContractError, match="boundary"):
        replace(ports, boundary=None)  # type: ignore[arg-type]


def test_decision_value_objects_are_owned_by_core_contracts() -> None:
    decision = CallModel(
        purpose="reasoning",
        directive=CoreDirective("core.reasoning"),
        reason=CoreReason("reasoning.continue", recoverable=True),
    )

    assert decision.directive.code == "core.reasoning"
