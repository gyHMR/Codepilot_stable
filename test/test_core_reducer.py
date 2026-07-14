from __future__ import annotations

from codepilot.core.commands import ReportVerificationUnavailable
from codepilot.core.observations import (
    CoreCommandObservation,
    ModelObservation,
    ToolBatchObservation,
)
from codepilot.core.reducer import ReductionContext, reduce_observation
from codepilot.core.state import CoreState
from codepilot.protocols import AssistantMessage, TextContent
from codepilot.tools.results import ToolError, ToolResult
from codepilot.tools.security import ApprovalChallenge, ToolEffect, ToolResource


def _context() -> ReductionContext:
    return ReductionContext(run_id="run_1", mode="build", now_ms=100)


def test_reducer_applies_observation_idempotently() -> None:
    state = CoreState.new("answer the question")
    observation = ModelObservation(
        observation_id="model_1",
        message=AssistantMessage(content=[TextContent(text="answer")]),
    )

    first = reduce_observation(state, observation, _context())
    second = reduce_observation(first.state, observation, _context())

    assert first.state.facts.counters.model_turns == 1
    assert second.state == first.state
    assert second.events == ()


def test_model_observation_tracks_logical_turn_and_provider_attempts() -> None:
    reduction = reduce_observation(
        CoreState.new("answer the question"),
        ModelObservation(
            observation_id="model_1",
            message=AssistantMessage(content=[TextContent(text="answer")]),
            attempts=3,
        ),
        _context(),
    )

    assert reduction.state.facts.counters.model_turns == 1
    assert reduction.state.facts.counters.model_attempts == 3


def test_tool_batch_mutation_makes_same_batch_verification_stale() -> None:
    state = CoreState.new("change the file")
    result = ToolResult(
        tool_call_id="tool_1",
        tool_name="write",
        status="success",
        data={
            "verification": {
                "status": "passed",
                "command": "pytest focused",
                "summary": "passed",
            }
        },
        effects=(
            ToolEffect(
                kind="filesystem_write",
                resource=ToolResource("workspace:///src/example.py"),
                operation="write",
                status="completed",
                certainty="observed",
            ),
        ),
        registration_id="reg_1",
    )

    reduction = reduce_observation(
        state,
        ToolBatchObservation(observation_id="tools_1", results=(result,)),
        _context(),
    )

    assert reduction.state.facts.workspace.revision == 1
    assert reduction.state.facts.workspace.affected_paths == ("src/example.py",)
    assert reduction.state.facts.verification.status == "stale"
    assert reduction.state.facts.verification.verified_revision is None


def test_later_verification_records_current_workspace_revision() -> None:
    changed = reduce_observation(
        CoreState.new("change the file"),
        ToolBatchObservation(
            observation_id="tools_1",
            results=(
                ToolResult(
                    tool_call_id="write_1",
                    tool_name="write",
                    status="success",
                    effects=(
                        ToolEffect(
                            kind="filesystem_write",
                            resource=ToolResource("workspace:///src/example.py"),
                            operation="write",
                            status="completed",
                            certainty="observed",
                        ),
                    ),
                    registration_id="reg_1",
                ),
            ),
        ),
        _context(),
    ).state
    verified = reduce_observation(
        changed,
        ToolBatchObservation(
            observation_id="tools_2",
            results=(
                ToolResult(
                    tool_call_id="verify_1",
                    tool_name="shell",
                    status="success",
                    data={
                        "verification": {
                            "status": "passed",
                            "command": "pytest focused",
                            "summary": "passed",
                        }
                    },
                    registration_id="reg_2",
                ),
            ),
        ),
        _context(),
    ).state

    assert verified.facts.verification.status == "passed"
    assert verified.facts.verification.verified_revision == 1


def test_reducer_records_unavailable_verification_from_command() -> None:
    state = CoreState.new("verify the change")
    command = ReportVerificationUnavailable(
        command_id="command_1",
        reason="test runner is not installed",
        attempted_checks=("pytest focused",),
        evidence_refs=("tool_1",),
    )

    reduction = reduce_observation(
        state,
        CoreCommandObservation(observation_id="command_obs_1", command=command),
        _context(),
    )

    assert reduction.state.facts.verification.status == "unavailable"
    assert reduction.state.facts.verification.verified_revision == 0
    assert reduction.command_results[0].status == "applied"


def test_reducer_does_not_apply_the_same_command_id_twice() -> None:
    command = ReportVerificationUnavailable(
        command_id="command_1",
        reason="test runner is not installed",
        attempted_checks=("pytest focused",),
        evidence_refs=("tool_1",),
    )
    first = reduce_observation(
        CoreState.new("verify the change"),
        CoreCommandObservation(observation_id="command_obs_1", command=command),
        _context(),
    )

    second = reduce_observation(
        first.state,
        CoreCommandObservation(observation_id="command_obs_2", command=command),
        _context(),
    )

    assert second.state.facts.verification == first.state.facts.verification
    assert second.events == ()
    assert second.command_results == ()


def test_reducer_rejects_unavailable_verification_without_evidence() -> None:
    state = CoreState.new("verify the change")
    command = ReportVerificationUnavailable(
        command_id="command_1",
        reason="cannot verify",
    )

    reduction = reduce_observation(
        state,
        CoreCommandObservation(observation_id="command_obs_1", command=command),
        _context(),
    )

    assert reduction.state.facts.verification.status == "none"
    assert reduction.command_results[0].status == "rejected"
    assert reduction.command_results[0].reason == "verification_evidence_required"
    assert reduction.events[0].kind == "command_rejected"


def test_tool_failure_records_failure_and_blocker() -> None:
    state = CoreState.new("use the required tool")
    result = ToolResult(
        tool_call_id="tool_1",
        tool_name="missing",
        status="error",
        error=ToolError(
            code="tool_not_found",
            kind="unavailable",
            message="tool is unavailable",
            retryable=False,
        ),
        registration_id="missing",
    )

    reduction = reduce_observation(
        state,
        ToolBatchObservation(observation_id="tools_1", results=(result,)),
        _context(),
    )

    assert reduction.state.facts.failures.latest is not None
    assert reduction.state.facts.failures.latest.code == "tool_not_found"
    assert reduction.state.facts.failures.count_for("tool_not_found") == 1
    assert [item.kind for item in reduction.state.task.blockers] == ["tool_unavailable"]


def test_successful_alternative_tool_clears_recoverable_tool_blockers() -> None:
    failed = reduce_observation(
        CoreState.new("use the required tool"),
        ToolBatchObservation(
            observation_id="tools_1",
            results=(
                ToolResult(
                    tool_call_id="tool_1",
                    tool_name="missing",
                    status="error",
                    error=ToolError(
                        code="tool_not_found",
                        kind="unavailable",
                        message="tool is unavailable",
                        retryable=True,
                    ),
                    registration_id="missing",
                ),
            ),
        ),
        _context(),
    ).state

    recovered = reduce_observation(
        failed,
        ToolBatchObservation(
            observation_id="tools_2",
            results=(
                ToolResult(
                    tool_call_id="tool_2",
                    tool_name="read",
                    status="success",
                    registration_id="read@1",
                ),
            ),
        ),
        _context(),
    ).state

    assert not recovered.task.blockers


def test_fifth_repeated_failure_requires_replan() -> None:
    state = CoreState.new("repair the implementation")
    for index in range(1, 6):
        state = reduce_observation(
            state,
            ToolBatchObservation(
                observation_id=f"tools_{index}",
                results=(
                    ToolResult(
                        tool_call_id=f"tool_{index}",
                        tool_name="command",
                        status="error",
                        error=ToolError(
                            code="command_exit_nonzero",
                            kind="execution",
                            message="tests failed",
                            retryable=True,
                        ),
                        registration_id="command@1",
                    ),
                ),
            ),
            _context(),
        ).state

    assert state.facts.failures.count_for("command_exit_nonzero") == 5
    assert [item.kind for item in state.task.blockers] == ["replan_required"]


def test_tool_batch_deduplicates_repeated_result_ids_within_the_batch() -> None:
    result = ToolResult(
        tool_call_id="tool_1",
        tool_name="read",
        status="success",
        registration_id="reg_1",
    )

    reduction = reduce_observation(
        CoreState.new("inspect"),
        ToolBatchObservation(
            observation_id="tools_1",
            results=(result, result),
        ),
        _context(),
    )

    assert reduction.state.facts.counters.tool_calls == 1
    assert reduction.state.facts.loop_guards.seen_tool_call_ids == ("tool_1",)


def test_final_result_after_approval_is_not_dropped_as_duplicate_call() -> None:
    challenge = ApprovalChallenge(
        approval_id="approval_1",
        request_fingerprint="sha256:test",
        run_id="run_1",
        session_id="session_1",
        tool_call_id="tool_1",
        tool_name="shell",
        registration_id="reg_1",
        actions=("shell",),
        resources=(ToolResource("workspace:///"),),
        effects=frozenset({"process_spawn"}),
        risk="medium",
        reason="requires approval",
        safe_preview={},
    )
    waiting = reduce_observation(
        CoreState.new("run command"),
        ToolBatchObservation(
            observation_id="tools_waiting",
            results=(
                ToolResult(
                    tool_call_id="tool_1",
                    tool_name="shell",
                    status="approval_required",
                    approval=challenge,
                    registration_id="reg_1",
                ),
            ),
        ),
        _context(),
    ).state

    resumed = reduce_observation(
        waiting,
        ToolBatchObservation(
            observation_id="tools_resumed",
            results=(
                ToolResult(
                    tool_call_id="tool_1",
                    tool_name="shell",
                    status="success",
                    data={"verification": {"status": "passed", "command": "check"}},
                    registration_id="reg_1",
                ),
            ),
        ),
        _context(),
    ).state

    assert resumed.facts.counters.tool_calls == 1
    assert resumed.facts.verification.status == "passed"
