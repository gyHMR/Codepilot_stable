from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from codepilot.core.contracts import TaskStrategy


def test_task_control_contracts_define_planning_budget_and_state() -> None:
    from codepilot.core.task import (
        PlanningBudgetUsage,
        PlanningDiscoveryReport,
        TaskPlanningState,
        budget_for_profile,
        ensure_plan_source,
        ensure_planning_budget_profile,
    )

    budget = budget_for_profile("balanced")
    usage = PlanningBudgetUsage(
        model_rounds=2,
        tool_calls=3,
        estimated_tokens=400,
        stop_reason="sufficient_evidence",
    )
    report = PlanningDiscoveryReport(
        status="completed",
        facts=["TaskController owns completion checks", ""],
        relevant_files=["src/codepilot/core/task_controller.py"],
        risks=["verification can be stale"],
        verification_hints=["python -m pytest test/test_task_planning.py -q"],
        open_questions=[],
        evidence_refs=["tool:read_1", "tool:read_1"],
        budget=usage,
    )
    planning = TaskPlanningState(
        phase="execution",
        source="llm_with_discovery",
        budget=budget,
        discovery=report,
    )

    assert budget.profile == "balanced"
    assert budget.max_model_rounds == 4
    assert budget.max_tool_calls == 12
    assert report.facts == ("TaskController owns completion checks",)
    assert report.evidence_refs == ("tool:read_1",)
    assert planning.to_signal()["source"] == "llm_with_discovery"
    assert ensure_planning_budget_profile("wide") == "wide"
    assert ensure_plan_source("recovered") == "recovered"


def test_task_step_owns_basic_state_transitions() -> None:
    from codepilot.core import TaskStep

    step = TaskStep(id="step_1", title="运行验证")

    step.mark_in_progress()
    step.add_evidence_refs(["tool:read_1", "tool:read_1"])
    step.record_failure("验证失败，需要修复", evidence_refs=["tool:test_1"])
    step.block("等待用户审批", evidence_refs=["tool:write_1", "tool:write_1"])
    step.complete(
        summary="验证通过",
        evidence_refs=["verification:test_2"],
        progress_state="verified",
    )

    assert step.status == "completed"
    assert step.failure_count == 1
    assert step.note is None
    assert step.summary == "验证通过"
    assert step.progress_state == "verified"
    assert step.evidence_refs == [
        "tool:read_1",
        "tool:test_1",
        "tool:write_1",
        "verification:test_2",
    ]


def test_task_state_records_reject_unknown_enum_values() -> None:
    import pytest
    from codepilot.core import (
        AttemptRecord,
        ChangeSet,
        CompletionCheck,
        ExecutionDecision,
        TaskState,
        TaskStep,
    )

    with pytest.raises(ValueError, match="Unknown task step status"):
        TaskStep(id="step_1", title="bad status", status="paused")

    with pytest.raises(ValueError, match="Unknown task step kind"):
        TaskStep(id="step_1", title="bad kind", kind="deploy")

    with pytest.raises(ValueError, match="Unknown task progress state"):
        TaskStep(id="step_1", title="bad progress", progress_state="almost_done")

    with pytest.raises(ValueError, match="Unknown task phase"):
        TaskState(task_id="task_1", goal="bad phase", phase="paused")

    with pytest.raises(ValueError, match="Unknown execution action"):
        ExecutionDecision(action="retry", reason="bad action")

    with pytest.raises(ValueError, match="Unknown completion reason"):
        CompletionCheck(satisfied=False, reason="almost_done")

    with pytest.raises(ValueError, match="Unknown attempt status"):
        AttemptRecord(
            attempt_id="attempt_1",
            step_id="step_1",
            action_intent="edit_file",
            status="partial",
        )

    with pytest.raises(ValueError, match="Unknown change set status"):
        ChangeSet(
            change_id="change_1",
            attempt_id="attempt_1",
            step_id="step_1",
            status="needs_review",
        )


def test_task_state_owns_step_navigation_and_status_projections() -> None:
    from codepilot.core import TaskState, TaskStep

    task = TaskState(
        task_id="task_1",
        goal="修复任务推进",
        steps=[
            TaskStep(id="step_1", title="定位问题", status="completed"),
            TaskStep(id="step_2", title="修改实现"),
            TaskStep(id="step_3", title="等待确认", status="blocked"),
        ],
        current_step_id="missing",
        next_action="旧动作",
    )

    assert task.current_step() is None

    next_step = task.advance_to_next_open_step()

    assert next_step is task.steps[1]
    assert task.current_step() is next_step
    assert task.current_step_id == "step_2"
    assert task.next_action == "修改实现"
    assert task.completed_step_titles() == ["定位问题"]
    assert task.pending_step_titles() == ["修改实现"]
    assert task.blocked_step_titles() == ["等待确认"]

    task.steps[1].complete()
    assert task.advance_to_next_open_step() is None
    assert task.current_step_id is None
    assert task.next_action is None


def test_session_runtime_records_task_recovery_warning_separately_from_memory(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.contracts import SessionOptions
    from codepilot.sessions.prepare import SessionRuntime, begin_task_recovery

    class BrokenTaskRecovery:
        def begin_task(self, text: str, *, run_id: str | None = None):
            _ = text, run_id
            raise RuntimeError("task recovery write failed")

    session = SessionRuntime(
        SessionOptions(
            model=_task_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
            memory_enabled=False,
        )
    )
    session.task_recovery = BrokenTaskRecovery()  # type: ignore[assignment]

    try:
        begin_task_recovery(session, "修复任务推进", run_id="run_1")

        events = session.store.load_events()
        assert not any(
            event.get("type") == "task_recovery_warning"
            for event in events
        )
        assert not any(
            event.get("type") == "memory_warning"
            and event.get("operation") == "task_recovery_begin"
            for event in events
        )
    finally:
        session._close()


def test_task_planner_parses_json_plan_from_llm_message() -> None:
    from codepilot.core import TaskPlanner
    from codepilot.protocols import AssistantMessage, TextContent

    message = AssistantMessage(
        content=[
            TextContent(
                text="""
                {
                  "goal": "修复任务推进",
                  "steps": [
                    {
                      "title": "定位任务模块",
                      "kind": "investigate",
                      "acceptance": "找到 TaskController 调用链",
                      "verification_hint": null
                    },
                    {
                      "title": "修改 step 推进逻辑",
                      "kind": "edit",
                      "acceptance": "验证通过后只推进当前步骤",
                      "verification_hint": "python -m pytest test/test_task_planning.py -q"
                    }
                  ]
                }
                """
            )
        ]
    )

    draft = TaskPlanner().parse_plan_message(message, fallback_goal="fallback")

    assert draft.goal == "修复任务推进"
    assert [step.title for step in draft.steps] == [
        "定位任务模块",
        "修改 step 推进逻辑",
    ]
    assert draft.steps[0].kind == "investigate"
    assert draft.steps[0].acceptance == "找到 TaskController 调用链"
    assert draft.steps[1].verification_hint == "python -m pytest test/test_task_planning.py -q"


def test_task_plan_draft_owns_planner_output_invariants() -> None:
    import pytest
    from codepilot.core import PlannedTaskStep, TaskPlanDraft

    step = PlannedTaskStep(
        title="  定位\n任务模块  ",
        kind="investigate",
        acceptance="  ",
        verification_hint="  python -m pytest test/test_task_planning.py -q  ",
    )
    draft = TaskPlanDraft(goal="  修复\n任务推进  ", steps=[step], source=" llm ")

    assert step.title == "定位 任务模块"
    assert step.acceptance is None
    assert step.verification_hint == "python -m pytest test/test_task_planning.py -q"
    assert draft.goal == "修复 任务推进"
    assert draft.source == "llm"
    assert draft.steps == (step,)

    with pytest.raises(ValueError, match="step title"):
        PlannedTaskStep(title="")

    with pytest.raises(ValueError, match="Unknown task step kind"):
        PlannedTaskStep(title="部署", kind="deploy")

    with pytest.raises(ValueError, match="plan goal"):
        TaskPlanDraft(goal=" ", steps=[step], source="llm")

    with pytest.raises(ValueError, match="at least one step"):
        TaskPlanDraft(goal="修复任务推进", steps=[], source="llm")

    with pytest.raises(ValueError, match="plan source"):
        TaskPlanDraft(goal="修复任务推进", steps=[step], source="manual")


def test_task_planner_records_fallback_reason_when_generation_fails() -> None:
    asyncio.run(_task_planner_fallback_reason_case())


def test_task_planner_accepts_common_plan_aliases_and_string_steps() -> None:
    from codepilot.core import TaskPlanner
    from codepilot.protocols import AssistantMessage, TextContent

    message = AssistantMessage(
        content=[
            TextContent(
                text=(
                    '{"goal":"修复 planner","plan":['
                    '"定位 planner 输出",'
                    '{"title":"运行验证","kind":"verify",'
                    '"acceptance":"测试通过",'
                    '"verification_hint":"python -m pytest test/test_task_planning.py -q"}'
                    ']}'
                )
            )
        ]
    )

    draft = TaskPlanner().parse_plan_message(message, fallback_goal="fallback")

    assert draft.source == "llm"
    assert [step.title for step in draft.steps] == ["定位 planner 输出", "运行验证"]
    assert draft.steps[0].kind == "investigate"
    assert draft.steps[1].kind == "verify"
    assert draft.steps[1].verification_hint == "python -m pytest test/test_task_planning.py -q"


def test_task_planner_fallback_keeps_parse_diagnostics() -> None:
    from codepilot.core import TaskPlanner
    from codepilot.protocols import AssistantMessage, TextContent

    message = AssistantMessage(
        content=[
            TextContent(
                text='{"goal":"修复 planner","unexpected":["定位","验证"]}'
            )
        ]
    )

    draft = TaskPlanner().parse_plan_message(message, fallback_goal="fallback")

    assert draft.source == "fallback"
    assert draft.fallback_reason == "missing_steps"
    assert draft.fallback_output_preview == '{"goal":"修复 planner","unexpected":["定位","验证"]}'
    assert draft.fallback_parsed_keys == ("goal", "unexpected")


async def _task_planner_fallback_reason_case() -> None:
    from codepilot.core import TaskPlanner
    from codepilot.llm.ports import LLMFailed, LLMRequest, ModelDescriptor
    from codepilot.protocols import UserMessage

    class BrokenModelPort:
        async def stream(self, _request: LLMRequest):
            yield LLMFailed(error=RuntimeError("planner unavailable"))

    draft = await TaskPlanner().generate(
        model=ModelDescriptor(provider="unit-test", model_id="task-test"),
        messages=[UserMessage(content="修复任务规划")],
        model_port=BrokenModelPort(),
        fallback_goal="修复任务规划",
    )

    assert draft.source == "fallback"
    assert draft.fallback_reason == "RuntimeError: planner unavailable"


def test_task_controller_initializes_from_planned_steps_and_exports_details() -> None:
    from codepilot.core import TaskController
    from codepilot.core import PlannedTaskStep
    from codepilot.protocols import UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="实现 plan and execute")],
        proposed_steps=[
            PlannedTaskStep(
                title="定位任务模块",
                kind="investigate",
                acceptance="找到 TaskController 调用链",
            ),
            PlannedTaskStep(
                title="修改 step 推进逻辑",
                kind="edit",
                acceptance="验证通过后只推进当前步骤",
                verification_hint="python -m pytest test/test_task_planning.py -q",
            ),
        ],
    )

    rendered = controller.render_context(task)
    summary = controller.summarize(task)

    assert task.steps[0].kind == "investigate"
    assert task.steps[0].acceptance == "找到 TaskController 调用链"
    assert task.steps[1].verification_hint == "python -m pytest test/test_task_planning.py -q"
    assert "Acceptance: 找到 TaskController 调用链" in rendered
    assert "Verification hint: python -m pytest test/test_task_planning.py -q" in rendered
    assert summary.step_details["定位任务模块"]["kind"] == "investigate"
    assert summary.step_details["修改 step 推进逻辑"]["verification_hint"] == (
        "python -m pytest test/test_task_planning.py -q"
    )


def test_task_controller_coerces_unknown_raw_step_kind() -> None:
    from codepilot.core import TaskController
    from codepilot.protocols import UserMessage

    task = TaskController().initialize(
        [UserMessage(content="部署并验证")],
        proposed_steps=[
            {
                "title": "部署预览",
                "kind": "deploy",
                "acceptance": "预览环境可访问",
            }
        ],
    )

    assert task.steps[0].kind == "other"
    assert task.steps[0].acceptance == "预览环境可访问"


def test_task_controller_normalizes_steps_and_updates_from_tool_results() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复配置加载失败并验证")],
        proposed_steps=[
            "定位配置加载调用链",
            "",
            "定位配置加载调用链",
            "修改实现",
            "运行相关测试",
            "多余步骤 1",
            "多余步骤 2",
            "多余步骤 3",
        ],
    )
    assert [step.title for step in task.steps] == [
        "定位配置加载调用链",
        "修改实现",
        "运行相关测试",
        "多余步骤 1",
        "多余步骤 2",
        "多余步骤 3",
    ]
    assert task.current_step_id == "step_1"

    read = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        content=[TextContent(text="config.py lines")],
        status="success",
    )
    run = RunState(run_id="run_1", session_id="session_1")
    run.collect_tool_results([read])
    decision = controller.after_tool_results(task, run, [read])

    assert decision.action == "continue"
    assert task.steps[0].status == "in_progress"
    assert task.steps[0].progress_state == "evidence_collected"
    assert task.steps[0].evidence_refs == ["tool:read_1"]
    assert task.steps[1].status == "pending"

    failed_verification = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="error",
        is_error=True,
        verification={
            "status": "failed",
            "command": "python -m pytest test/test_config.py -q",
            "exit_code": 1,
            "summary": "failed",
        },
    )
    run.collect_tool_results([failed_verification])
    decision = controller.after_tool_results(task, run, [failed_verification])

    assert decision.action == "repair"
    assert decision.next_action is not None
    assert "python -m pytest test/test_config.py -q" in decision.next_action
    assert "修复" in decision.next_action
    assert task.steps[0].failure_count == 1
    assert task.steps[0].status == "in_progress"
    assert task.recent_error_code == "verification_failed"
    assert task.action_intent == "debug_failure"


def test_completion_gate_requires_fresh_verification_after_workspace_change() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修改实现并运行测试")],
        proposed_steps=["修改实现", "运行相关测试"],
    )
    run = RunState(run_id="run_1", session_id="session_1")

    edit = ToolResultMessage(
        tool_call_id="edit_1",
        tool_name="edit",
        content=[TextContent(text="Edited file")],
        affected_paths=["src/app.py"],
        workspace_changed=True,
        status="success",
    )
    run.collect_tool_results([edit])
    controller.after_tool_results(task, run, [edit])
    assert task.steps[0].status == "in_progress"
    assert task.steps[0].progress_state == "changed"

    missing = controller.check_completion(task, run)
    assert missing.satisfied is False
    assert missing.reason == "modified_without_fresh_verification"
    assert missing.can_continue is True
    assert "fresh_verification" in missing.missing

    passed = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="success",
        verification={
            "status": "passed",
            "command": "python -m pytest test -q",
            "exit_code": 0,
            "summary": "passed",
        },
    )
    run.collect_tool_results([passed])
    controller.after_tool_results(task, run, [passed])

    ok = controller.check_completion(task, run)
    assert ok.satisfied is True
    assert ok.reason == "all_steps_completed"
    assert all(step.status == "completed" for step in task.steps)


def test_passed_verification_completes_current_step_and_advances() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="按步骤执行")],
        proposed_steps=["修改实现", "总结结果"],
    )
    run = RunState(run_id="run_1", session_id="session_1")
    passed = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="success",
        verification={
            "status": "passed",
            "command": "python -m pytest test/test_task_planning.py -q",
            "exit_code": 0,
            "summary": "passed",
        },
    )

    run.collect_tool_results([passed])
    decision = controller.after_tool_results(task, run, [passed])

    assert decision.action == "continue"
    assert task.steps[0].status == "completed"
    assert task.steps[0].progress_state == "verified"
    assert task.steps[1].status == "in_progress"
    assert task.current_step_id == "step_2"
    assert task.phase == "acting"


def test_passed_verification_keeps_acting_phase_after_fresh_verification() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修改实现后总结")],
        proposed_steps=["修改实现", "总结结果"],
    )
    run = RunState(run_id="run_1", session_id="session_1")
    edit = ToolResultMessage(
        tool_call_id="edit_1",
        tool_name="edit",
        content=[TextContent(text="edited")],
        affected_paths=["src/app.py"],
        workspace_changed=True,
        status="success",
    )
    run.collect_tool_results([edit])
    controller.after_tool_results(task, run, [edit])
    passed = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="success",
        verification={
            "status": "passed",
            "command": "python -m pytest test/test_task_planning.py -q",
            "exit_code": 0,
            "summary": "passed",
        },
    )

    run.collect_tool_results([passed])
    decision = controller.after_tool_results(task, run, [passed])

    assert decision.action == "continue"
    assert run.workspace_changed is True
    assert run.fresh_verification_passed is True
    assert task.steps[0].status == "completed"
    assert task.steps[1].status == "in_progress"
    assert task.phase == "acting"


def test_completion_gate_treats_unavailable_tool_as_blocked() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="使用 write 修改 state.txt")],
    )
    run = RunState(run_id="run_1", session_id="session_1")
    missing_tool = ToolResultMessage(
        tool_call_id="write_1",
        tool_name="write",
        content=[TextContent(text="Tool write not found")],
        status="error",
        is_error=True,
    )

    decision = controller.after_tool_results(task, run, [missing_tool])
    check = controller.check_completion(task, run)

    assert decision.action == "stop"
    assert decision.reason == "tool_unavailable"
    assert check.satisfied is False
    assert check.reason == "blocked_steps"
    assert task.steps[0].status == "blocked"
    assert task.steps[0].note == "工具不可用"


def test_permission_blocked_steps_keep_tool_evidence() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize([UserMessage(content="写入文件")])
    run = RunState(run_id="run_1", session_id="session_1")
    denied = ToolResultMessage(
        tool_call_id="write_1",
        tool_name="write",
        content=[TextContent(text="blocked")],
        status="denied",
        is_error=True,
        error_code="read_only_mode",
    )

    decision = controller.after_tool_results(task, run, [denied])

    assert decision.action == "replan"
    assert task.steps[0].status == "blocked"
    assert "tool:write_1" in task.steps[0].evidence_refs

    task = controller.initialize([UserMessage(content="部署")])
    approval = ToolResultMessage(
        tool_call_id="deploy_1",
        tool_name="deploy",
        content=[TextContent(text="approval required")],
        status="approval_required",
        is_error=True,
        approved=False,
        approval_id="approval_1",
        error_code="approval_required",
    )

    decision = controller.after_tool_results(task, run, [approval])

    assert decision.action == "wait_approval"
    assert task.steps[0].status == "blocked"
    assert "tool:deploy_1" in task.steps[0].evidence_refs
    assert "approval:approval_1" in task.steps[0].evidence_refs


def test_replan_preserves_completed_steps_and_stops_after_limit() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复失败测试")],
        proposed_steps=["定位失败", "修改实现", "运行验证"],
    )
    run = RunState(run_id="run_1", session_id="session_1")

    read = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        status="success",
    )
    run.collect_tool_results([read])
    controller.after_tool_results(task, run, [read])
    assert task.steps[0].status == "in_progress"
    assert task.steps[0].progress_state == "evidence_collected"

    first = _failed_verification("test_1")
    second = _failed_verification("test_2")
    run.collect_tool_results([first])
    assert controller.after_tool_results(task, run, [first]).action == "repair"
    run.collect_tool_results([second])
    decision = controller.after_tool_results(task, run, [second])

    assert decision.action == "stop"
    assert decision.reason == "revision_needed"
    assert task.replan_count == 0
    assert task.steps[0].status == "blocked"
    assert task.steps[0].note == "revision_needed"
    assert task.next_action == "报告连续验证失败并等待用户决定是否回到 plan"


def test_task_controller_respects_configured_replan_limit() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复失败测试")],
        proposed_steps=["定位失败", "修改实现", "运行验证"],
        max_replans_per_run=1,
    )
    run = RunState(run_id="run_1", session_id="session_1")

    for call_id in ["test_1", "test_2"]:
        failed = _failed_verification(call_id)
        run.collect_tool_results([failed])
        decision = controller.after_tool_results(task, run, [failed])

    assert decision.action == "stop"
    assert decision.reason == "revision_needed"
    assert task.replan_count == 0
    assert task.max_replans_per_run == 1


def test_repeated_failed_verification_after_change_proposes_revert() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修改实现并验证")],
        proposed_steps=["修改实现", "运行验证"],
    )
    run = RunState(run_id="run_1", session_id="session_1")
    edit = ToolResultMessage(
        tool_call_id="edit_1",
        tool_name="edit",
        content=[TextContent(text="edited")],
        affected_paths=["src/app.py"],
        workspace_changed=True,
        metadata={
            "change_evidence": {
                "change_kind": "update",
                "before_hashes": {"src/app.py": "old"},
                "after_hashes": {"src/app.py": "new"},
                "affected_paths": ["src/app.py"],
                "effect_detection": "direct",
                "effect_detection_confidence": "high",
                "safe_revert_available": False,
            }
        },
    )
    run.collect_tool_results([edit])
    controller.after_tool_results(task, run, [edit])

    first = _failed_verification("test_1")
    second = _failed_verification("test_2")
    run.collect_tool_results([first])
    assert controller.after_tool_results(task, run, [first]).action == "repair"
    run.collect_tool_results([second])
    decision = controller.after_tool_results(task, run, [second])

    assert decision.action == "stop"
    assert decision.reason == "revision_needed"
    assert task.rollback_required is False
    assert task.rollback_targets == []
    assert task.change_sets
    assert task.change_sets[-1].status == "failed"


def test_task_controller_exports_control_signal_and_attempts() -> None:
    from codepilot.core.state import RunState
    from codepilot.core import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize([UserMessage(content="读取文件")], mode="read")
    run = RunState(run_id="run_1", session_id="session_1")
    result = ToolResultMessage(
        tool_call_id="read_1",
        tool_name="read",
        content=[TextContent(text="content")],
        status="success",
    )

    run.collect_tool_results([result])
    controller.after_tool_results(task, run, [result])
    signal = controller.control_signal(task)
    summary = controller.summarize(task)

    assert signal["task_id"] == task.task_id
    assert signal["mode"] == "read"
    assert "Read mode" in controller.render_context(task)
    assert signal["action_intent"] == "read_context"
    assert signal["last_decision"] == "continue"
    assert task.attempts[-1].tool_call_ids == ["read_1"]
    assert summary.control_signal["action_intent"] == "read_context"
    assert summary.control_signal["mode"] == "read"
    assert summary.attempts[-1]["attempt_id"].startswith("attempt_")


def test_task_controller_rebuilds_task_state_from_memory_projection() -> None:
    from codepilot.core import TaskController
    from codepilot.protocols import UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="继续修复失败测试")],
        task_recovery_projection={
            "schema_version": 1,
            "goal": "修复失败测试",
            "current_mode": "plan",
            "planning": {"phase": "recovered", "source": "recovered"},
            "current_step_id": "step_3",
            "steps": [
                {"id": "step_1", "title": "定位失败", "status": "completed"},
                {"id": "step_2", "title": "根据最新失败证据调整方案", "status": "blocked"},
                {
                    "id": "step_3",
                    "title": "重新运行相关验证",
                    "status": "pending",
                    "kind": "verify",
                    "acceptance": "验证失败已修复",
                    "verification_hint": "python -m pytest test/test_task.py -q",
                },
            ],
            "verification_status": "revision_needed",
            "blocked_reason": "replan_limit_exceeded",
            "next_action": "报告连续失败并等待用户指示",
        },
    )

    assert task.goal == "修复失败测试"
    assert task.mode == "plan"
    assert task.planning.source == "recovered"
    assert [step.status for step in task.steps] == [
        "completed",
        "blocked",
        "in_progress",
    ]
    assert [step.title for step in task.steps] == [
        "定位失败",
        "根据最新失败证据调整方案",
        "重新运行相关验证",
    ]
    assert task.current_step_id == "step_3"
    assert task.steps[2].kind == "verify"
    assert task.steps[2].acceptance == "验证失败已修复"
    assert task.steps[2].verification_hint == "python -m pytest test/test_task.py -q"
    assert task.next_action == "报告连续失败并等待用户指示"
    assert task.completion_reason == "replan_limit_exceeded"


def test_task_recovery_projection_mapping_builds_task_state() -> None:
    from codepilot.core import build_task_state_from_recovery_projection
    from codepilot.protocols import UserMessage

    task = build_task_state_from_recovery_projection(
        [UserMessage(content="继续修复")],
        {
            "schema_version": 1,
            "goal": "恢复任务",
            "current_mode": "read",
            "current_step_id": "step_3",
            "steps": [
                {"id": "step_1", "title": "定位失败", "status": "completed"},
                {"id": "step_2", "title": "等待审批", "status": "blocked"},
                {
                    "id": "step_3",
                    "title": "重新运行验证",
                    "status": "pending",
                    "kind": "verify",
                    "acceptance": "验证通过",
                    "verification_hint": "pytest task",
                },
            ],
            "verification_status": "revision_needed",
            "blocked_reason": "blocked_steps",
            "next_action": "继续验证",
        },
    )

    assert task is not None
    assert task.goal == "恢复任务"
    assert task.mode == "read"
    assert [(step.title, step.status) for step in task.steps] == [
        ("定位失败", "completed"),
        ("等待审批", "blocked"),
        ("重新运行验证", "in_progress"),
    ]
    assert task.current_step_id == "step_3"
    assert task.phase == "acting"
    assert task.next_action == "继续验证"
    assert task.steps[2].kind == "verify"
    assert task.steps[2].acceptance == "验证通过"
    assert task.steps[2].verification_hint == "pytest task"


def test_task_recovery_projection_coerces_unknown_step_kind() -> None:
    from codepilot.core import build_task_state_from_recovery_projection
    from codepilot.protocols import UserMessage

    task = build_task_state_from_recovery_projection(
        [UserMessage(content="继续旧任务")],
        {
            "schema_version": 1,
            "goal": "恢复旧任务",
            "current_mode": "build",
            "current_step_id": "step_1",
            "steps": [
                {
                    "id": "step_1",
                    "title": "部署预览",
                    "status": "pending",
                    "kind": "deploy",
                    "acceptance": "预览环境可访问",
                    "verification_hint": "curl localhost",
                },
            ],
        },
    )

    assert task is not None
    assert task.steps[0].kind == "other"
    assert task.steps[0].acceptance == "预览环境可访问"
    assert task.steps[0].verification_hint == "curl localhost"


def test_task_controller_rebuilds_from_authoritative_task_state() -> None:
    from codepilot.core import build_task_state_from_recovery_projection
    from codepilot.protocols import UserMessage

    task = build_task_state_from_recovery_projection(
        [UserMessage(content="继续")],
        {
            "schema_version": 1,
            "task_id": "task_existing",
            "raw_user_request": "修复任务恢复",
            "current_mode": "build",
            "approval_state": "approved",
            "goal": "修复任务恢复",
            "approved_plan": {"steps": [{"id": "step_1", "title": "阅读代码"}]},
            "current_step_id": "step_2",
            "steps": [
                {
                    "id": "step_1",
                    "title": "阅读代码",
                    "status": "completed",
                    "kind": "investigate",
                    "evidence_refs": ["tool:read_1"],
                },
                {
                    "id": "step_2",
                    "title": "补充恢复测试",
                    "status": "in_progress",
                    "kind": "verify",
                    "acceptance": "恢复测试通过",
                    "verification_hint": "python -m pytest test/test_task_planning.py -q",
                },
            ],
            "verification_status": "unknown",
            "evidence_refs": ["tool:read_1"],
            "blocked_reason": None,
            "recovery_summary": "Goal: 修复任务恢复",
        },
    )

    assert task is not None
    assert task.task_id == "task_existing"
    assert task.goal == "修复任务恢复"
    assert task.mode == "build"
    assert task.current_step_id == "step_2"
    assert [(step.title, step.status) for step in task.steps] == [
        ("阅读代码", "completed"),
        ("补充恢复测试", "in_progress"),
    ]
    assert task.steps[1].kind == "verify"
    assert task.steps[1].verification_hint == "python -m pytest test/test_task_planning.py -q"


def test_v2_agent_loop_emits_task_events_and_result_summary() -> None:
    asyncio.run(_v2_agent_loop_task_summary_case())


def test_v2_agent_loop_uses_plan_strategy_steps_in_context() -> None:
    asyncio.run(_v2_agent_loop_plan_strategy_case())


def test_v2_agent_loop_complete_task_step_advances_plan_execution() -> None:
    asyncio.run(_v2_agent_loop_complete_step_advances_case())


def test_v2_agent_loop_uses_recovered_task_projection_in_context() -> None:
    asyncio.run(_v2_agent_loop_recovered_task_context_case())


def test_v2_agent_loop_allows_one_final_verification_at_iteration_limit() -> None:
    asyncio.run(_v2_agent_loop_final_verification_grace_case())


def test_v2_agent_loop_does_not_complete_when_completion_gate_is_unsatisfied() -> None:
    asyncio.run(_v2_agent_loop_unverified_completion_gate_case())


def test_v2_agent_loop_reports_blocked_task_instead_of_completed_after_denied_tool() -> None:
    asyncio.run(_v2_agent_loop_denied_tool_blocks_completion_case())


def test_v2_agent_loop_preserves_cancelled_stop_reason() -> None:
    asyncio.run(_v2_agent_loop_cancelled_stop_reason_case())


def test_v2_agent_loop_does_not_complete_after_generic_tool_error() -> None:
    asyncio.run(_v2_agent_loop_generic_tool_error_case())


def test_v2_agent_loop_waits_for_user_when_revert_is_proposed() -> None:
    asyncio.run(_v2_agent_loop_propose_revert_case())


def test_v2_agent_loop_stops_when_replan_limit_is_exceeded() -> None:
    asyncio.run(_v2_agent_loop_replan_limit_case())


async def _v2_agent_loop_task_summary_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.ports import ToolObservation

    model = _TaskScriptedModel(
        lambda request, _calls: AssistantMessage(
            content=[TextContent(text="done")]
            if any(isinstance(message, ToolResultMessage) for message in request.messages)
            else [ToolCall(id="read_1", name="read_test", arguments={})],
            stop_reason="stop" if any(isinstance(message, ToolResultMessage) for message in request.messages) else "toolUse",
        )
    )
    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        _v2_loop_input("run_task_summary", prompt="解释这个文件"),
        AgentLoopPorts(
            model=model,
            tools=_TaskToolPort(
                {
                    "read_test": ToolObservation(
                        tool_call_id="read_1",
                        name="read_test",
                        status="success",
                        content=(TextContent(text="read result"),),
                    )
                }
            ),
            events=events.append,
        ),
    )

    assert result.status == "completed"
    assert result.task is not None
    assert result.task.goal == "解释这个文件"
    assert result.task.completed_steps == ["完成当前请求"]
    assert result.task.completion_satisfied is True
    assert any(event["type"] == "task_plan_created" for event in events)
    assert any(event["type"] == "task_step_updated" for event in events)
    assert any(event["type"] == "completion_checked" for event in events)


async def _v2_agent_loop_plan_strategy_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent

    prompts: list[str] = []

    class CapturingModel:
        async def stream(self, request):
            from codepilot.llm.ports import LLMCompleted

            prompts.append(request.system_prompt)
            yield LLMCompleted(message=AssistantMessage(content=[TextContent(text="done")]))

    result = await run_agent_loop(
        _v2_loop_input(
            "run_plan_strategy",
            prompt="实现 planner",
            task_strategy=TaskStrategy(
                enabled=True,
                mode="plan",
                planning_budget_profile="wide",
                steps=[
                    {
                        "title": "定位任务模块",
                        "kind": "investigate",
                        "acceptance": "找到 TaskController",
                    },
                    {
                        "title": "修改执行逻辑",
                        "kind": "edit",
                        "acceptance": "按 step 推进",
                        "verification_hint": "pytest task",
                    },
                ],
            ),
        ),
        AgentLoopPorts(model=CapturingModel(), tools=None),
    )

    assert result.task is not None
    assert result.task.control_signal["mode"] == "plan"
    assert result.task.control_signal["planning"]["budget"]["profile"] == "wide"
    assert result.task.step_details["定位任务模块"]["acceptance"] == "找到 TaskController"
    assert result.task.step_details["修改执行逻辑"]["verification_hint"] == "pytest task"
    assert "Current step: 定位任务模块" in prompts[0]


async def _v2_agent_loop_complete_step_advances_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.ports import ToolObservation

    execution_contexts: list[str] = []

    def response(request, _calls):
        execution_contexts.append(request.system_prompt)
        if not any(isinstance(message, ToolResultMessage) for message in request.messages):
            return AssistantMessage(
                content=[ToolCall(id="read_1", name="read_test", arguments={})],
                stop_reason="toolUse",
            )
        if "Current step: 定位任务模块" in request.system_prompt:
            return AssistantMessage(
                content=[
                    ToolCall(
                        id="complete_1",
                        name="complete_task_step",
                        arguments={
                            "summary": "已定位 TaskController",
                            "evidence_refs": ["tool:read_1"],
                        },
                    )
                ],
                stop_reason="toolUse",
            )
        if any(
            isinstance(message, ToolResultMessage) and message.tool_name == "edit_test"
            for message in request.messages
        ):
            return AssistantMessage(content=[TextContent(text="修改完成，等待验证")])
        assert "Current step: 修改执行逻辑" in request.system_prompt
        return AssistantMessage(
            content=[ToolCall(id="edit_1", name="edit_test", arguments={})],
            stop_reason="toolUse",
        )

    result = await run_agent_loop(
        _v2_loop_input(
            "run_complete_step",
            prompt="实现 planner",
            task_strategy=TaskStrategy(
                enabled=True,
                mode="plan",
                steps=[
                    {"title": "定位任务模块", "kind": "investigate", "acceptance": "找到 TaskController"},
                    {"title": "修改执行逻辑", "kind": "edit", "acceptance": "按 step 推进"},
                ],
            ),
        ),
        AgentLoopPorts(
            model=_TaskScriptedModel(response),
            tools=_TaskToolPort(
                {
                    "read_test": ToolObservation(
                        tool_call_id="read_1",
                        name="read_test",
                        status="success",
                        content=(TextContent(text="TaskController source"),),
                    ),
                    "complete_task_step": ToolObservation(
                        tool_call_id="complete_1",
                        name="complete_task_step",
                        status="success",
                        content=(TextContent(text="Current task step completed: 已定位 TaskController"),),
                        metadata={
                            "task_control": {
                                "action": "complete_step",
                                "summary": "已定位 TaskController",
                                "evidence_refs": ["tool:read_1"],
                            }
                        },
                    ),
                    "edit_test": ToolObservation(
                        tool_call_id="edit_1",
                        name="edit_test",
                        status="success",
                        content=(TextContent(text="edited"),),
                        workspace_changed=True,
                        affected_paths=("src/codepilot/core/task_controller.py",),
                    ),
                }
            ),
        ),
    )

    assert any("Current step: 修改执行逻辑" in item for item in execution_contexts)
    assert result.task is not None
    assert "定位任务模块" in result.task.completed_steps
    assert result.task.pending_steps == ["修改执行逻辑"]
    assert result.workspace_effects.changed is True


async def _v2_agent_loop_recovered_task_context_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent

    prompts: list[str] = []

    class CapturingModel:
        async def stream(self, request):
            from codepilot.llm.ports import LLMCompleted

            prompts.append(request.system_prompt)
            yield LLMCompleted(message=AssistantMessage(content=[TextContent(text="done")]))

    result = await run_agent_loop(
        _v2_loop_input(
            "run_recovered_task",
            prompt="继续旧任务",
            task_strategy=TaskStrategy(
                enabled=True,
                mode="build",
                recovery_projection={
                    "schema_version": 1,
                    "goal": "恢复旧任务",
                    "current_mode": "build",
                    "current_step_id": "step_3",
                    "steps": [
                        {"id": "step_1", "title": "定位失败", "status": "completed"},
                        {
                            "id": "step_2",
                            "title": "根据最新失败证据调整方案",
                            "status": "blocked",
                        },
                        {
                            "id": "step_3",
                            "title": "重新运行相关验证",
                            "status": "pending",
                        },
                    ],
                    "verification_status": "revision_needed",
                    "blocked_reason": "replan_limit_exceeded",
                    "next_action": "报告连续失败并等待用户指示",
                },
            ),
        ),
        AgentLoopPorts(model=CapturingModel(), tools=None),
    )

    assert result.task is not None
    assert result.task.completed_steps == ["定位失败"]
    assert result.task.pending_steps == ["重新运行相关验证"]
    assert result.task.blocked_steps == ["根据最新失败证据调整方案"]
    assert "恢复旧任务" in prompts[0]


async def _v2_agent_loop_final_verification_grace_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall
    from codepilot.tools.ports import ToolObservation

    def response(_request, calls):
        if calls == 1:
            return AssistantMessage(
                content=[ToolCall(id="edit_1", name="edit_test", arguments={})],
                stop_reason="toolUse",
            )
        return AssistantMessage(
            content=[
                ToolCall(
                    id="verify_1",
                    name="bash",
                    arguments={"command": "python -m pytest test/test_task_planning.py -q"},
                )
            ],
            stop_reason="toolUse",
        )

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        _v2_loop_input(
            "run_final_verification_grace",
            prompt="修改代码并运行验证",
            limits=AgentLoopLimits(max_model_turns=3, max_tool_iterations=1, repeated_tool_call_limit=20),
        ),
        AgentLoopPorts(
            model=_TaskScriptedModel(response),
            tools=_TaskToolPort(
                {
                    "edit_test": ToolObservation(
                        tool_call_id="edit_1",
                        name="edit_test",
                        status="success",
                        content=(TextContent(text="edited"),),
                        workspace_changed=True,
                        affected_paths=("src/codepilot/core/task_controller.py",),
                    ),
                    "bash": ToolObservation(
                        tool_call_id="verify_1",
                        name="bash",
                        status="success",
                        content=(TextContent(text="passed"),),
                        verification=(
                            RunVerification(
                                tool_call_id="verify_1",
                                tool_name="bash",
                                status="passed",
                                command="python -m pytest test/test_task_planning.py -q",
                                exit_code=0,
                                summary="passed",
                            ),
                        ),
                    ),
                }
            ),
            events=events.append,
        ),
    )

    assert result.status == "completed"
    assert result.stop_reason == "final_answer"
    assert result.counters.tool_iterations == 2
    assert result.task is not None
    assert result.task.completion_satisfied is True
    assert any(event.get("type") == "tool_execution_grace" for event in events)


async def _v2_agent_loop_unverified_completion_gate_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.tools.ports import ToolObservation

    model = _TaskScriptedModel(
        lambda request, calls: AssistantMessage(
            content=[ToolCall(id="edit_1", name="edit_test", arguments={})]
            if calls == 1
            else [TextContent(text="done without verification")],
            stop_reason="toolUse" if calls == 1 else "stop",
        )
    )
    result = await run_agent_loop(
        _v2_loop_input(
            "run_unverified_gate",
            prompt="修改代码",
            limits=AgentLoopLimits(max_model_turns=2),
        ),
        AgentLoopPorts(
            model=model,
            tools=_TaskToolPort(
                {
                    "edit_test": ToolObservation(
                        tool_call_id="edit_1",
                        name="edit_test",
                        status="success",
                        workspace_changed=True,
                        affected_paths=("src/app.py",),
                    )
                }
            ),
        ),
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_incomplete"
    assert result.task is not None
    assert result.task.completion_satisfied is False
    assert result.task.completion_reason == "modified_without_fresh_verification"


async def _v2_agent_loop_denied_tool_blocks_completion_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall
    from codepilot.tools.ports import ToolObservation

    result = await run_agent_loop(
        _v2_loop_input("run_denied_tool", prompt="写入文件", limits=AgentLoopLimits(max_model_turns=2)),
        AgentLoopPorts(
            model=_TaskScriptedModel(
                lambda _request, calls: AssistantMessage(
                    content=[ToolCall(id="write_1", name="write_test", arguments={})]
                    if calls == 1
                    else [TextContent(text="final")],
                    stop_reason="toolUse" if calls == 1 else "stop",
                )
            ),
            tools=_TaskToolPort(
                {
                    "write_test": ToolObservation(
                        tool_call_id="write_1",
                        name="write_test",
                        status="denied",
                        metadata={"error_code": "read_only_mode"},
                    )
                }
            ),
        ),
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_blocked"
    assert result.task is not None
    assert result.task.blocked_steps == ["完成当前请求"]
    assert result.task.completion_satisfied is False


async def _v2_agent_loop_cancelled_stop_reason_case() -> None:
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, ToolCall
    from codepilot.tools.ports import ToolObservation

    result = await run_agent_loop(
        _v2_loop_input("run_cancelled", prompt="运行命令"),
        AgentLoopPorts(
            model=_TaskScriptedModel(
                lambda _request, _calls: AssistantMessage(
                    content=[ToolCall(id="bash_1", name="bash_test", arguments={})],
                    stop_reason="toolUse",
                )
            ),
            tools=_TaskToolPort(
                {
                    "bash_test": ToolObservation(
                        tool_call_id="bash_1",
                        name="bash_test",
                        status="cancelled",
                    )
                }
            ),
        ),
    )

    assert result.status == "aborted"
    assert result.stop_reason == "aborted"


async def _v2_agent_loop_generic_tool_error_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall
    from codepilot.tools.ports import ToolObservation

    result = await run_agent_loop(
        _v2_loop_input("run_generic_error", prompt="做一个需要工具的任务", limits=AgentLoopLimits(max_model_turns=2)),
        AgentLoopPorts(
            model=_TaskScriptedModel(
                lambda _request, calls: AssistantMessage(
                    content=[ToolCall(id="tool_1", name="custom_tool", arguments={})]
                    if calls == 1
                    else [TextContent(text="final")],
                    stop_reason="toolUse" if calls == 1 else "stop",
                )
            ),
            tools=_TaskToolPort(
                {
                    "custom_tool": ToolObservation(
                        tool_call_id="tool_1",
                        name="custom_tool",
                        status="error",
                        metadata={"error_code": "tool_exception"},
                    )
                }
            ),
        ),
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_incomplete"
    assert result.task is not None
    assert result.task.completion_satisfied is False
    assert result.task.completion_reason == "incomplete_steps"


async def _v2_agent_loop_propose_revert_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall
    from codepilot.tools.ports import ToolObservation

    def response(_request, calls):
        if calls == 1:
            call = ToolCall(id="edit_1", name="edit_test", arguments={})
        elif calls == 2:
            call = ToolCall(id="test_1", name="test_tool", arguments={})
        else:
            call = ToolCall(id="test_2", name="test_tool", arguments={})
        return AssistantMessage(content=[call], stop_reason="toolUse")

    result = await run_agent_loop(
        _v2_loop_input(
            "run_propose_revert",
            prompt="修改实现并验证",
            limits=AgentLoopLimits(max_model_turns=4, repeated_tool_call_limit=20),
        ),
        AgentLoopPorts(
            model=_TaskScriptedModel(response),
            tools=_TaskToolPort(
                {
                    "edit_test": ToolObservation(
                        tool_call_id="edit_1",
                        name="edit_test",
                        status="success",
                        workspace_changed=True,
                        affected_paths=("src/app.py",),
                        metadata={
                            "change_evidence": {
                                "change_kind": "update",
                                "before_hashes": {"src/app.py": "old"},
                                "after_hashes": {"src/app.py": "new"},
                                "affected_paths": ["src/app.py"],
                                "effect_detection": "direct",
                                "effect_detection_confidence": "high",
                                "safe_revert_available": False,
                            }
                        },
                    ),
                    "test_tool": ToolObservation(
                        tool_call_id="test_1",
                        name="test_tool",
                        status="error",
                        verification=(
                            RunVerification(
                                tool_call_id="test_1",
                                tool_name="test_tool",
                                status="failed",
                                command="python -m pytest test/test_task.py -q",
                                exit_code=1,
                                summary="failed",
                            ),
                        ),
                    ),
                }
            ),
        ),
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_blocked"
    assert result.task is not None
    assert result.task.control_signal["rollback_required"] is False
    assert result.task.control_signal["rollback_targets"] == []
    assert result.task.next_action == "报告连续验证失败并等待用户决定是否回到 plan"


async def _v2_agent_loop_replan_limit_case() -> None:
    from codepilot.core.contracts import AgentLoopLimits, AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.protocols import AssistantMessage, RunVerification, ToolCall
    from codepilot.tools.ports import ToolObservation

    result = await run_agent_loop(
        _v2_loop_input(
            "run_replan_limit",
            prompt="修复失败测试",
            limits=AgentLoopLimits(max_model_turns=4, max_tool_iterations=20, repeated_tool_call_limit=20),
            task_strategy=TaskStrategy(
                enabled=True,
                mode="build",
                max_replans_per_run=1,
            ),
        ),
        AgentLoopPorts(
            model=_TaskScriptedModel(
                lambda _request, calls: AssistantMessage(
                    content=[ToolCall(id=f"test_{calls}", name="bash_test", arguments={"attempt": calls})],
                    stop_reason="toolUse",
                )
            ),
            tools=_TaskToolPort(
                {
                    "bash_test": ToolObservation(
                        tool_call_id="test_1",
                        name="bash_test",
                        status="error",
                        verification=(
                            RunVerification(
                                tool_call_id="test_1",
                                tool_name="bash_test",
                                status="failed",
                                command="python -m pytest test/test_task.py -q",
                                exit_code=1,
                                summary="failed",
                            ),
                        ),
                    )
                }
            ),
        ),
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_blocked"
    assert result.task is not None
    assert result.task.blocked_steps == ["完成当前请求"]
    assert result.task.next_action == "报告连续验证失败并等待用户决定是否回到 plan"
    assert any(
        event.get("type") == "task_decision"
        and event.get("decision", {}).get("reason") == "revision_needed"
        for event in result.events
    )


def _v2_loop_input(
    run_id: str,
    *,
    prompt: str,
    limits: Any | None = None,
    task_strategy: TaskStrategy | None = None,
):
    from codepilot.core.contracts import AgentLoopInput, AgentLoopLimits, RunCorrelation
    from codepilot.llm.ports import ModelDescriptor

    return AgentLoopInput(
        run_id=run_id,
        correlation=RunCorrelation(session_id="session_1"),
        user_prompt=prompt,
        context={"system_prompt": "rules"},
        model=ModelDescriptor(provider="unit-test", model_id="task-test"),
        limits=limits or AgentLoopLimits(max_model_turns=4),
        task_strategy=task_strategy or TaskStrategy(enabled=True, mode="build"),
    )


class _TaskScriptedModel:
    def __init__(self, factory):
        self._factory = factory
        self.calls = 0

    async def stream(self, request):
        from codepilot.llm.ports import LLMCompleted

        self.calls += 1
        yield LLMCompleted(message=self._factory(request, self.calls))


class _TaskToolPort:
    def __init__(self, observations: dict[str, Any]):
        self._observations = observations
        self.calls: list[str] = []

    def catalog(self):
        return {"tools": sorted(self._observations)}

    async def execute(self, invocation):
        from dataclasses import replace

        self.calls.append(invocation.name)
        observation = self._observations.get(invocation.name)
        if observation is None:
            from codepilot.protocols import TextContent
            from codepilot.tools.ports import ToolObservation

            return ToolObservation(
                tool_call_id=invocation.tool_call_id,
                name=invocation.name,
                status="error",
                content=(TextContent(text=f"Tool {invocation.name} not found"),),
                metadata={"error_code": "tool_not_found"},
            )
        return replace(
            observation,
            tool_call_id=invocation.tool_call_id,
            name=invocation.name,
        )
def _failed_verification(tool_call_id: str):
    from codepilot.protocols import ToolResultMessage

    return ToolResultMessage(
        tool_call_id=tool_call_id,
        tool_name="bash",
        status="error",
        is_error=True,
        verification={
            "status": "failed",
            "command": "python -m pytest test/test_task.py -q",
            "exit_code": 1,
            "summary": "failed",
        },
    )


def _task_test_model():
    from codepilot.protocols import Model

    return Model(
        id="task-test",
        name="Task Test",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )
