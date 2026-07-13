from __future__ import annotations

from dataclasses import replace

from codepilot.core.observations import (
    ModelObservation,
    ToolBatchObservation,
    UserInputObservation,
)
from codepilot.core.contracts import (
    CallModel,
    CoreLimits,
    ExecuteTools,
    Terminate,
    Wait,
)
from codepilot.core.policy import CorePolicy, PolicyContext
from codepilot.core.reducer import ReductionContext, apply_decision
from codepilot.core.state import (
    CoreCounters,
    CoreState,
    FailureFacts,
    FailureRecord,
    RunFacts,
    TaskBlocker,
    VerificationFacts,
    WorkspaceFacts,
)
from codepilot.protocols import AssistantMessage, TextContent, ToolCall
from codepilot.tools.results import ToolResult
from codepilot.tools.security import ApprovalChallenge, ToolResource


def _policy_context(**limits) -> PolicyContext:
    return PolicyContext(
        mode="build",
        limits=CoreLimits(**limits),
    )


def test_policy_calls_model_for_new_user_input() -> None:
    observation = UserInputObservation(
        observation_id="user_1",
        text="inspect the repository",
    )

    decision = CorePolicy.decide(
        CoreState.new("inspect the repository"),
        observation,
        _policy_context(),
    )

    assert isinstance(decision, CallModel)
    assert decision.purpose == "reasoning"


def test_policy_executes_model_tool_calls() -> None:
    observation = ModelObservation(
        observation_id="model_1",
        message=AssistantMessage(
            content=[ToolCall(id="call_1", name="read", arguments={})]
        ),
    )

    decision = CorePolicy.decide(
        CoreState.new("inspect"),
        observation,
        _policy_context(),
    )

    assert isinstance(decision, ExecuteTools)
    assert [call.id for call in decision.calls] == ["call_1"]


def test_policy_waits_for_tool_approval() -> None:
    challenge = ApprovalChallenge(
        approval_id="approval_1",
        request_fingerprint="sha256:test",
        run_id="run_1",
        session_id="session_1",
        tool_call_id="call_1",
        tool_name="shell",
        registration_id="reg_1",
        actions=("shell",),
        resources=(ToolResource("workspace:///"),),
        effects=frozenset({"process_spawn"}),
        risk="medium",
        reason="requires approval",
        safe_preview={},
    )
    observation = ToolBatchObservation(
        observation_id="tools_1",
        results=(
            ToolResult(
                tool_call_id="call_1",
                tool_name="shell",
                status="approval_required",
                approval=challenge,
                registration_id="reg_1",
            ),
        ),
    )

    decision = CorePolicy.decide(
        CoreState.new("run command"),
        observation,
        _policy_context(),
    )

    assert isinstance(decision, Wait)
    assert decision.wait.kind == "tool_approval"
    assert decision.wait.request_id == "approval_1"


def test_policy_requires_fresh_verification_before_completion() -> None:
    state = CoreState(
        task=CoreState.new("change code").task,
        facts=RunFacts(
            workspace=WorkspaceFacts(revision=1, changed=True),
            verification=VerificationFacts(status="stale"),
        ),
    )
    observation = ModelObservation(
        observation_id="model_1",
        message=AssistantMessage(content=[TextContent(text="done")]),
    )

    decision = CorePolicy.decide(state, observation, _policy_context())

    assert isinstance(decision, CallModel)
    assert decision.purpose == "verification"


def test_policy_completes_with_fresh_verification() -> None:
    state = CoreState(
        task=CoreState.new("change code").task,
        facts=RunFacts(
            workspace=WorkspaceFacts(revision=1, changed=True),
            verification=VerificationFacts(
                status="passed",
                verified_revision=1,
            ),
        ),
    )
    observation = ModelObservation(
        observation_id="model_1",
        message=AssistantMessage(content=[TextContent(text="done")]),
    )

    decision = CorePolicy.decide(state, observation, _policy_context())

    assert isinstance(decision, Terminate)
    assert decision.status == "completed"


def test_resolved_recoverable_failure_does_not_block_completion() -> None:
    state = CoreState(
        task=CoreState.new("change code").task,
        facts=RunFacts(
            workspace=WorkspaceFacts(revision=1, changed=True),
            verification=VerificationFacts(status="passed", verified_revision=1),
            failures=FailureFacts(
                latest=FailureRecord(
                    code="verification.failed",
                    source="verification",
                    message="previous check failed",
                    recoverable=True,
                    evidence_refs=("verify_1",),
                )
            ),
        ),
    )
    observation = ModelObservation(
        observation_id="model_2",
        message=AssistantMessage(content=[TextContent(text="fixed and verified")]),
    )

    decision = CorePolicy.decide(state, observation, _policy_context())

    assert isinstance(decision, Terminate)
    assert decision.status == "completed"


def test_policy_replans_after_recovery_budget_is_exhausted() -> None:
    base = CoreState.new("repair the implementation")
    state = replace(
        base,
        task=replace(
            base.task,
            blockers=(
                TaskBlocker(
                    kind="replan_required",
                    reason="current path has no progress",
                    evidence_refs=("tool_3",),
                    recoverable=True,
                ),
            ),
        ),
        facts=RunFacts(
            counters=CoreCounters(total_recoveries=2),
            failures=FailureFacts(
                latest=FailureRecord(
                    code="tool.execution_failed",
                    source="tools",
                    message="failed",
                    recoverable=True,
                    evidence_refs=("tool_3",),
                )
            ),
        ),
    )
    observation = ToolBatchObservation(observation_id="tools_3")

    decision = CorePolicy.decide(
        state,
        observation,
        _policy_context(max_recovery_attempts=2),
    )

    assert isinstance(decision, CallModel)
    assert decision.purpose == "replan"


def test_policy_recovers_before_replanning_while_budget_remains() -> None:
    base = CoreState.new("repair the implementation")
    state = replace(
        base,
        task=replace(
            base.task,
            blockers=(
                TaskBlocker(
                    kind="replan_required",
                    reason="current path has no progress",
                    evidence_refs=("tool_1",),
                    recoverable=True,
                ),
            ),
        ),
        facts=RunFacts(
            failures=FailureFacts(
                latest=FailureRecord(
                    code="tool.execution_failed",
                    source="tools",
                    message="failed",
                    recoverable=True,
                    evidence_refs=("tool_1",),
                )
            )
        ),
    )

    decision = CorePolicy.decide(
        state,
        ToolBatchObservation(observation_id="tools_1"),
        _policy_context(max_recovery_attempts=2),
    )

    assert isinstance(decision, CallModel)
    assert decision.purpose == "recovery"


def test_policy_fails_on_nonrecoverable_task_failure() -> None:
    state = CoreState(
        task=CoreState.new("answer").task,
        facts=RunFacts(
            failures=FailureFacts(
                latest=FailureRecord(
                    code="model.unavailable",
                    source="model",
                    message="provider unavailable",
                    recoverable=False,
                )
            )
        ),
    )
    observation = ModelObservation(
        observation_id="model_1",
        message=AssistantMessage(content=[TextContent(text="answer")]),
    )

    decision = CorePolicy.decide(state, observation, _policy_context())

    assert isinstance(decision, Terminate)
    assert decision.status == "failed"
    assert decision.reason.code == "model.unavailable"


def test_read_mode_workspace_change_fails_instead_of_requesting_verification() -> None:
    state = CoreState(
        task=CoreState.new("inspect only").task,
        facts=RunFacts(workspace=WorkspaceFacts(revision=1, changed=True)),
    )
    observation = ModelObservation(
        observation_id="model_1",
        message=AssistantMessage(content=[TextContent(text="inspection complete")]),
    )

    decision = CorePolicy.decide(
        state,
        observation,
        PolicyContext(mode="read"),
    )

    assert isinstance(decision, Terminate)
    assert decision.status == "failed"
    assert decision.reason.code == "read.workspace_changed"


def test_apply_completed_decision_marks_task_satisfied() -> None:
    state = CoreState.new("answer")
    decision = Terminate(status="completed", reason_code="task.completed")

    reduction = apply_decision(
        state,
        decision,
        ReductionContext(run_id="run_1", mode="read", now_ms=100),
    )

    assert reduction.state.task.status == "satisfied"
    assert reduction.events[0].kind == "task_satisfied"
