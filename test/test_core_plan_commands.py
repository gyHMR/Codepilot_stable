from __future__ import annotations

from dataclasses import replace

from codepilot.core.commands import (
    AbandonPlan,
    ApprovePlan,
    ApprovePlanRevision,
    CommandResult,
    ProposePlanRevision,
    RequestPlanClose,
    SubmitPlan,
    UpdatePlanProgress,
)
from codepilot.core.contracts import CallModel, Terminate, Wait
from codepilot.core.observations import (
    ModelObservation,
    ToolBatchObservation,
    UserInputObservation,
)
from codepilot.core.plan import PlanDefinition, PlanStepDefinition, PlanStepUpdate
from codepilot.core.policy import CorePolicy, PolicyContext
from codepilot.core.reducer import ReductionContext, apply_core_command, apply_decision
from codepilot.core.state import (
    CoreState,
    FailureCount,
    FailureFacts,
    LoopGuardFacts,
)
from codepilot.core.tool_step import project_core_command_results, project_core_commands
from codepilot.protocols import AssistantMessage, TextContent
from codepilot.tools.results import ToolResult


def _definition(summary: str = "完成登录重构") -> PlanDefinition:
    return PlanDefinition(
        summary=summary,
        completion_criteria=("登录测试通过",),
        task_understanding="收敛登录职责。",
        current_implementation="认证和会话逻辑耦合。",
        target_design="认证和会话边界清晰。",
        impact_scope="登录服务。",
        risks_and_open_questions=("保持现有接口。",),
        verification_plan="运行登录测试。",
    )


def _steps() -> tuple[PlanStepDefinition, ...]:
    return (
        PlanStepDefinition("修改登录服务", "拆分职责。", "运行登录单测。"),
        PlanStepDefinition("验证登录流程", "执行回归。", "运行登录回归。"),
    )


def _context(mode: str = "build") -> ReductionContext:
    return ReductionContext(run_id="run-1", mode=mode, now_ms=100)


def _submit(mode: str = "build"):
    state = CoreState.new("重构登录模块")
    reduction = apply_core_command(
        state,
        SubmitPlan("submit-1", _definition(), _steps()),
        _context(mode),
    )
    assert reduction.command_results[0].status == "applied"
    assert reduction.state.task.plan is not None
    return reduction.state


def test_submit_plan_is_deterministic_and_mode_controls_approval() -> None:
    proposed = _submit("plan")
    active = _submit("build")

    assert proposed.task.plan is not None
    assert proposed.task.plan.plan_id == "plan:submit-1"
    assert proposed.task.plan.origin == "plan_mode"
    assert proposed.task.plan.status == "proposed"
    assert proposed.task.plan.revision == 1
    assert proposed.task.plan.steps[0].step_id == "plan:submit-1:step:1"
    assert active.task.plan is not None
    assert active.task.plan.origin == "build_mode"
    assert active.task.plan.status == "active"

    decision = CorePolicy.decide(
        proposed,
        ToolBatchObservation("tools-1"),
        PolicyContext("plan"),
    )
    assert isinstance(decision, Wait)
    assert decision.wait.kind == "plan_confirmation"


def test_read_mode_rejects_plan_writes_and_duplicate_commands_are_idempotent() -> None:
    state = CoreState.new("理解登录模块")
    command = SubmitPlan("submit-read", _definition(), _steps())

    rejected = apply_core_command(state, command, _context("read"))
    repeated = apply_core_command(rejected.state, command, _context("read"))

    assert rejected.command_results[0].status == "rejected"
    assert rejected.command_results[0].reason == "plan.read_only"
    assert repeated.state == rejected.state
    assert repeated.command_results == ()


def test_unapproved_proposal_can_be_revised_after_user_feedback() -> None:
    state = _submit("plan")
    decision = CorePolicy.decide(
        state,
        UserInputObservation(
            "feedback-1",
            text="请增加兼容性验证",
            current_goal=state.task.current_goal,
        ),
        PolicyContext("plan"),
    )
    assert isinstance(decision, CallModel)
    assert decision.purpose == "replan"

    revised = apply_core_command(
        state,
        SubmitPlan(
            "submit-2",
            _definition("补充兼容性验证"),
            (
                *_steps(),
                PlanStepDefinition("验证兼容性", "覆盖旧接口。", "运行兼容测试。"),
            ),
        ),
        _context("plan"),
    )

    assert revised.state.task.plan is not None
    assert revised.state.task.plan.plan_id == "plan:submit-1"
    assert revised.state.task.plan.revision == 2
    assert revised.state.task.plan.definition.summary == "补充兼容性验证"


def test_progress_is_delta_based_revision_checked_and_evidence_backed() -> None:
    state = _submit()
    assert state.task.plan is not None
    first, second = state.task.plan.steps
    state = replace(
        state,
        facts=replace(
            state.facts,
            loop_guards=LoopGuardFacts(seen_tool_call_ids=("tool-edit",)),
        ),
    )

    updated = apply_core_command(
        state,
        UpdatePlanProgress(
            "progress-1",
            expected_revision=1,
            updates=(
                PlanStepUpdate(
                    first.step_id,
                    "completed",
                    completion_note="登录职责已经拆分。",
                    evidence_refs=("tool-edit",),
                ),
                PlanStepUpdate(second.step_id, "in_progress"),
            ),
        ),
        _context(),
    )

    assert updated.command_results[0].status == "applied"
    assert updated.state.task.plan is not None
    assert updated.state.task.plan.revision == 2
    assert [step.status for step in updated.state.task.plan.steps] == [
        "completed",
        "in_progress",
    ]

    stale = apply_core_command(
        updated.state,
        UpdatePlanProgress(
            "progress-stale",
            expected_revision=1,
            updates=(PlanStepUpdate(second.step_id, "completed", "已验证"),),
        ),
        _context(),
    )
    assert stale.command_results[0].reason == "plan.revision_conflict"

    unknown_evidence = apply_core_command(
        updated.state,
        UpdatePlanProgress(
            "progress-unknown",
            expected_revision=2,
            updates=(
                PlanStepUpdate(
                    second.step_id,
                    "completed",
                    "已验证",
                    ("missing-evidence",),
                ),
            ),
        ),
        _context(),
    )
    assert unknown_evidence.command_results[0].reason == "plan.unknown_evidence"


def test_plan_mode_revision_waits_for_approval_and_preserves_completed_steps() -> None:
    proposed = _submit("plan")
    assert proposed.task.plan is not None
    approved = apply_core_command(
        proposed,
        ApprovePlan("approve-1", expected_revision=1),
        _context("plan"),
    ).state
    assert approved.task.plan is not None
    first = approved.task.plan.steps[0]
    approved = replace(
        approved,
        facts=replace(
            approved.facts,
            loop_guards=LoopGuardFacts(seen_tool_call_ids=("tool-edit",)),
        ),
    )
    progressed = apply_core_command(
        approved,
        UpdatePlanProgress(
            "progress-1",
            expected_revision=2,
            updates=(
                PlanStepUpdate(first.step_id, "completed", "实现完成", ("tool-edit",)),
            ),
        ),
        _context(),
    ).state
    assert progressed.task.plan is not None

    proposed_revision = apply_core_command(
        progressed,
        ProposePlanRevision(
            "revision-1",
            expected_revision=3,
            reason="user_request",
            definition=_definition("按新范围完成登录重构"),
            steps=(
                _steps()[0],
                PlanStepDefinition("补充兼容验证", "验证旧接口。", "运行兼容测试。"),
            ),
        ),
        _context(),
    )

    assert proposed_revision.state.task.plan is not None
    assert proposed_revision.state.task.plan.pending_revision is not None
    assert proposed_revision.state.task.plan.definition.summary == "完成登录重构"

    applied = apply_core_command(
        proposed_revision.state,
        ApprovePlanRevision("approve-revision-1", expected_revision=4),
        _context(),
    )
    assert applied.state.task.plan is not None
    assert applied.state.task.plan.definition.summary == "按新范围完成登录重构"
    assert applied.state.task.plan.steps[0].step_id == first.step_id
    assert applied.state.task.plan.steps[0].status == "completed"


def test_build_plan_auto_revision_requires_five_qualified_failures() -> None:
    state = _submit()
    state = replace(
        state,
        facts=replace(
            state.facts,
            failures=FailureFacts(counts=(FailureCount("verification.failed", 5),)),
        ),
    )

    revised = apply_core_command(
        state,
        ProposePlanRevision(
            "revision-auto",
            expected_revision=1,
            reason="repeated_execution_failure",
            definition=_definition("改用兼容实现"),
            steps=(PlanStepDefinition("兼容实现", "替换阻塞路径。", "运行兼容测试。"),),
        ),
        _context(),
    )

    assert revised.state.task.plan is not None
    assert revised.state.task.plan.pending_revision is None
    assert revised.state.task.plan.definition.summary == "改用兼容实现"


def test_close_is_a_request_and_completion_policy_owns_terminal_transition() -> None:
    state = _submit()
    assert state.task.plan is not None
    first, second = state.task.plan.steps
    state = replace(
        state,
        facts=replace(
            state.facts,
            loop_guards=LoopGuardFacts(
                seen_tool_call_ids=("tool-edit", "tool-test")
            ),
        ),
    )
    progressed = apply_core_command(
        state,
        UpdatePlanProgress(
            "progress-all",
            expected_revision=1,
            updates=(
                PlanStepUpdate(first.step_id, "completed", "实现完成", ("tool-edit",)),
                PlanStepUpdate(second.step_id, "completed", "验证通过", ("tool-test",)),
            ),
        ),
        _context(),
    ).state

    before_close = CorePolicy.decide(
        progressed,
        ToolBatchObservation("tools-before-close"),
        PolicyContext("build"),
    )
    assert isinstance(before_close, CallModel)
    assert before_close.purpose == "plan_closeout"

    requested = apply_core_command(
        progressed,
        RequestPlanClose(
            "close-1",
            expected_revision=2,
            summary="计划步骤和验证均已完成。",
            evidence_refs=("tool-test",),
        ),
        _context(),
    ).state
    assert requested.task.plan is not None
    assert requested.task.plan.status == "active"
    assert requested.task.plan.close_request is not None

    decision = CorePolicy.decide(
        requested,
        ToolBatchObservation("tools-after-close"),
        PolicyContext("build"),
    )
    assert isinstance(decision, CallModel)
    assert decision.purpose == "final_response"

    terminal = CorePolicy.decide(
        requested,
        ModelObservation(
            "model-final",
            message=AssistantMessage(content=[TextContent(text="计划已完成。")]),
            purpose="final_response",
        ),
        PolicyContext("build"),
    )
    assert isinstance(terminal, Terminate)
    assert terminal.status == "completed"

    completed = apply_decision(requested, terminal, _context()).state
    assert completed.task.status == "satisfied"
    assert completed.task.plan is not None
    assert completed.task.plan.status == "completed"


def test_abandon_plan_is_an_external_command_not_a_plan_state_method() -> None:
    state = _submit()
    abandoned = apply_core_command(
        state,
        AbandonPlan("abandon-1", expected_revision=1, reason="user_abandoned"),
        _context(),
    )

    assert abandoned.state.task.plan is not None
    assert abandoned.state.task.plan.status == "abandoned"
    assert not hasattr(abandoned.state.task.plan, "abandon")


def test_active_plan_cannot_be_finished_by_plain_model_text() -> None:
    state = _submit()
    decision = CorePolicy.decide(
        state,
        ModelObservation("model-1"),
        PolicyContext("build"),
    )

    assert isinstance(decision, CallModel)
    assert decision.purpose == "reasoning"


def test_only_canonical_plan_tool_results_are_projected_as_core_commands() -> None:
    payload = {
        "kind": "submit_plan",
        "command_id": "call-plan",
        "definition": _definition().to_dict(),
        "steps": [item.to_dict() for item in _steps()],
    }
    plan_result = ToolResult(
        tool_call_id="call-plan",
        tool_name="create_build_plan",
        status="success",
        data={"core_command": payload},
        registration_id="core-plan",
    )
    untrusted_result = ToolResult(
        tool_call_id="call-shell",
        tool_name="shell",
        status="success",
        data={"core_command": payload},
        registration_id="shell",
    )

    commands = project_core_commands((plan_result, untrusted_result))

    assert len(commands) == 1
    assert isinstance(commands[0], SubmitPlan)
    assert commands[0].command_id == "call-plan"

    visible = project_core_command_results(
        (plan_result,),
        (CommandResult("call-plan", "rejected", "plan.revision_conflict"),),
    )
    assert visible[0].status == "error"
    assert visible[0].error is not None
    assert visible[0].error.code == "plan.revision_conflict"
    assert visible[0].data["core_command_result"]["status"] == "rejected"
