from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


def test_task_step_owns_basic_state_transitions() -> None:
    from codepilot.core.task_state import TaskStep

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
    from codepilot.core.task_state import (
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
    from codepilot.core.task_state import TaskState, TaskStep

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


def test_agent_session_records_task_recovery_warning_separately_from_memory(
    tmp_path: Path,
) -> None:
    from codepilot.runtime.types import AgentSessionOptions
    from codepilot.sessions.session import AgentSession

    class BrokenTaskRecovery:
        def begin_task(self, text: str, *, run_id: str | None = None):
            _ = text, run_id
            raise RuntimeError("task recovery write failed")

    session = AgentSession(
        AgentSessionOptions(
            model=_task_test_model(),
            workspace_dir=tmp_path,
            system_prompt="sys",
            memory_enabled=False,
        )
    )
    session.task_recovery = BrokenTaskRecovery()  # type: ignore[assignment]

    try:
        session._begin_task_recovery("修复任务推进", run_id="run_1")

        events = session.store.load_events()
        recovery_warnings = [
            event
            for event in events
            if event.get("operation") == "task_recovery_begin"
        ]
        assert recovery_warnings[-1]["type"] == "task_recovery_warning"
        assert not any(
            event.get("type") == "memory_warning"
            and event.get("operation") == "task_recovery_begin"
            for event in events
        )
    finally:
        session.close()


def test_task_planner_parses_json_plan_from_llm_message() -> None:
    from codepilot.core.task_planner import TaskPlanner
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
    from codepilot.core.task_planner import PlannedTaskStep, TaskPlanDraft

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


def test_task_controller_initializes_from_planned_steps_and_exports_details() -> None:
    from codepilot.core.task_controller import TaskController
    from codepilot.core.task_planner import PlannedTaskStep
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
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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

    assert decision.action == "replan"
    assert decision.reason == "repeated_step_failure"
    assert task.replan_count == 1
    assert task.steps[0].title == "根据最新失败证据调整方案"
    assert task.steps[0].status == "in_progress"
    assert task.steps[1].title == "重新运行相关验证"
    assert task.replans
    assert task.replans[-1].trigger == "verification_failed"

    for call_id in ["test_3", "test_4", "test_5", "test_6"]:
        failed = _failed_verification(call_id)
        run.collect_tool_results([failed])
        decision = controller.after_tool_results(task, run, [failed])

    assert decision.action == "stop"
    assert decision.reason == "replan_limit_exceeded"
    assert task.steps[0].status == "blocked"
    assert task.next_action == "报告连续失败并等待用户指示"


def test_task_controller_respects_configured_replan_limit() -> None:
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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

    assert decision.action == "replan"
    assert task.replan_count == 1
    assert task.max_replans_per_run == 1

    for call_id in ["test_3", "test_4"]:
        failed = _failed_verification(call_id)
        run.collect_tool_results([failed])
        decision = controller.after_tool_results(task, run, [failed])

    assert decision.action == "stop"
    assert decision.reason == "replan_limit_exceeded"


def test_repeated_failed_verification_after_change_proposes_revert() -> None:
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
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

    assert decision.action == "propose_revert"
    assert task.rollback_required is True
    assert task.rollback_targets == ["src/app.py"]
    assert task.change_sets
    assert task.change_sets[-1].status == "revert_required"


def test_task_controller_exports_control_signal_and_attempts() -> None:
    from codepilot.core.run_state import RunState
    from codepilot.core.task_controller import TaskController
    from codepilot.protocols import TextContent, ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize([UserMessage(content="读取文件")])
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
    assert signal["action_intent"] == "read_context"
    assert signal["last_decision"] == "continue"
    assert task.attempts[-1].tool_call_ids == ["read_1"]
    assert summary.control_signal["action_intent"] == "read_context"
    assert summary.attempts[-1]["attempt_id"].startswith("attempt_")


def test_task_controller_rebuilds_task_state_from_memory_projection() -> None:
    from codepilot.core.task_controller import TaskController
    from codepilot.protocols import UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="继续修复失败测试")],
        task_recovery_projection={
            "goal": "修复失败测试",
            "task_progress": {
                "completed_steps": ["定位失败"],
                "pending_steps": ["重新运行相关验证"],
                "blocked_steps": ["根据最新失败证据调整方案"],
                "completion_satisfied": False,
                "completion_reason": "replan_limit_exceeded",
                "step_details": {
                    "重新运行相关验证": {
                        "kind": "verify",
                        "acceptance": "验证失败已修复",
                        "verification_hint": "python -m pytest test/test_task.py -q",
                    }
                },
            },
            "next_action": "报告连续失败并等待用户指示",
        },
    )

    assert task.goal == "修复失败测试"
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
    from codepilot.core.task_controller import build_task_state_from_recovery_projection
    from codepilot.protocols import UserMessage

    task = build_task_state_from_recovery_projection(
        [UserMessage(content="继续修复")],
        {
            "goal": "恢复任务",
            "task_progress": {
                "completed_steps": ["定位失败"],
                "blocked_steps": ["等待审批"],
                "pending_steps": ["重新运行验证"],
                "completion_satisfied": False,
                "completion_reason": "blocked_steps",
                "step_details": {
                    "重新运行验证": {
                        "kind": "verify",
                        "acceptance": "验证通过",
                        "verification_hint": "pytest task",
                    }
                },
            },
            "next_action": "继续验证",
        },
    )

    assert task is not None
    assert task.goal == "恢复任务"
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
    from codepilot.core.task_controller import build_task_state_from_recovery_projection
    from codepilot.protocols import UserMessage

    task = build_task_state_from_recovery_projection(
        [UserMessage(content="继续旧任务")],
        {
            "goal": "恢复旧任务",
            "task_progress": {
                "pending_steps": ["部署预览"],
                "step_details": {
                    "部署预览": {
                        "kind": "deploy",
                        "acceptance": "预览环境可访问",
                        "verification_hint": "curl localhost",
                    }
                },
            },
        },
    )

    assert task is not None
    assert task.steps[0].kind == "other"
    assert task.steps[0].acceptance == "预览环境可访问"
    assert task.steps[0].verification_hint == "curl localhost"


def test_agent_loop_emits_task_events_and_result_summary() -> None:
    asyncio.run(_agent_loop_task_summary_case())


def test_agent_loop_can_plan_before_react_execution() -> None:
    asyncio.run(_agent_loop_llm_planner_case())


def test_agent_loop_complete_task_step_advances_plan_execution() -> None:
    asyncio.run(_agent_loop_complete_step_advances_case())


def test_agent_loop_uses_recovered_task_projection_in_context() -> None:
    asyncio.run(_agent_loop_recovered_task_context_case())


async def _agent_loop_complete_step_advances_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    execution_contexts: list[str] = []

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        system_prompt = context.system_prompt or ""
        if "Task Planner" in system_prompt:
            stream.end(
                AssistantMessage(
                    content=[
                        TextContent(
                            text=(
                                '{"goal":"实现 planner","steps":['
                                '{"title":"定位任务模块","kind":"investigate",'
                                '"acceptance":"找到 TaskController","verification_hint":null},'
                                '{"title":"修改执行逻辑","kind":"edit",'
                                '"acceptance":"按 step 推进","verification_hint":null}'
                                ']}'
                            )
                        )
                    ]
                )
            )
            return stream

        execution_contexts.append(system_prompt)
        if not any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="read_1", name="read_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
            return stream
        if "Current step: 定位任务模块" in system_prompt:
            stream.end(
                AssistantMessage(
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
            )
            return stream
        assert "Current step: 修改执行逻辑" in system_prompt
        if any(
            isinstance(message, ToolResultMessage) and message.tool_name == "edit_test"
            for message in context.messages
        ):
            stream.end(AssistantMessage(content=[TextContent(text="修改完成，等待验证")]))
            return stream
        stream.end(
            AssistantMessage(
                content=[ToolCall(id="edit_1", name="edit_test", arguments={})],
                stop_reason="toolUse",
            )
        )
        return stream

    async def read_tool(*_args):
        return AgentToolResult(content=[TextContent(text="TaskController source")])

    async def edit_tool(*_args):
        return AgentToolResult(
            content=[TextContent(text="edited")],
            workspace_changed=True,
            affected_paths=["src/codepilot/core/task_controller.py"],
        )

    result = await run_agent_loop(
        prompts=[UserMessage(content="实现 planner")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="read_test",
                    label="read",
                    description="read",
                    parameters={},
                    execute=read_tool,
                ),
                AgentTool(
                    name="edit_test",
                    label="edit",
                    description="edit",
                    parameters={},
                    execute=edit_tool,
                ),
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
            task_planner_enabled=True,
            repeated_tool_call_limit=20,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert any("Current step: 修改执行逻辑" in item for item in execution_contexts)
    assert result.task is not None
    assert "定位任务模块" in result.task.completed_steps
    assert result.task.pending_steps == ["修改执行逻辑"]
    assert result.workspace_changed is True


async def _agent_loop_llm_planner_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall, ToolResultMessage, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    calls: list[str] = []

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        system_prompt = context.system_prompt or ""
        if "Task Planner" in system_prompt:
            calls.append("plan")
            stream.end(
                AssistantMessage(
                    content=[
                        TextContent(
                            text=(
                                '{"goal":"实现 planner","steps":['
                                '{"title":"定位任务模块","kind":"investigate",'
                                '"acceptance":"找到 TaskController","verification_hint":null},'
                                '{"title":"修改执行逻辑","kind":"edit",'
                                '"acceptance":"按 step 推进","verification_hint":"pytest task"}'
                                ']}'
                            )
                        )
                    ]
                )
            )
            return stream
        calls.append("execute")
        assert "## Current Task" in system_prompt
        assert "定位任务模块" in system_prompt
        assert "Acceptance: 找到 TaskController" in system_prompt
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="read_1", name="read_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def read_tool(*_args):
        return AgentToolResult(content=[TextContent(text="read result")])

    result = await run_agent_loop(
        prompts=[UserMessage(content="实现 planner")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="read_test",
                    label="read",
                    description="read",
                    parameters={},
                    execute=read_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
            task_planner_enabled=True,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert calls[:2] == ["plan", "execute"]
    assert result.task is not None
    assert result.task.goal == "实现 planner"
    assert result.task.step_details["定位任务模块"]["acceptance"] == "找到 TaskController"


async def _agent_loop_recovered_task_context_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, TextContent, UserMessage

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        system_prompt = context.system_prompt or ""
        assert "定位失败" in system_prompt
        assert "根据最新失败证据调整方案" in system_prompt
        assert "Next action: 报告连续失败并等待用户指示" in system_prompt
        stream.end(AssistantMessage(content=[TextContent(text="继续")]))
        return stream

    result = await run_agent_loop(
        prompts=[UserMessage(content="继续修复失败测试")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            task_recovery_projection={
                "goal": "修复失败测试",
                "task_progress": {
                    "completed_steps": ["定位失败"],
                    "pending_steps": ["重新运行相关验证"],
                    "blocked_steps": ["根据最新失败证据调整方案"],
                    "completion_satisfied": False,
                    "completion_reason": "replan_limit_exceeded",
                },
                "next_action": "报告连续失败并等待用户指示",
            },
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.task is not None
    assert result.task.completed_steps == ["定位失败"]
    assert result.task.pending_steps == ["重新运行相关验证"]
    assert result.task.blocked_steps == ["根据最新失败证据调整方案"]


def test_agent_loop_stops_when_replan_limit_is_exceeded() -> None:
    asyncio.run(_agent_loop_replan_limit_case())


def test_agent_loop_does_not_complete_when_completion_gate_is_unsatisfied() -> None:
    asyncio.run(_agent_loop_unverified_completion_gate_case())


async def _agent_loop_unverified_completion_gate_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    attempts = 0

    async def fake_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        if attempts == 1:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="edit_1", name="edit_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
        else:
            stream.end(
                AssistantMessage(content=[TextContent(text="done without verification")])
            )
        return stream

    async def edit_tool(*_args):
        return AgentToolResult(
            status="success",
            workspace_changed=True,
            affected_paths=["src/app.py"],
        )

    result = await run_agent_loop(
        prompts=[UserMessage(content="修改代码")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="edit_test",
                    label="edit",
                    description="edit",
                    parameters={},
                    execute=edit_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=_task_test_model(),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_incomplete"
    assert result.task is not None
    assert result.task.completion_satisfied is False
    assert result.task.completion_reason == "modified_without_fresh_verification"


def test_agent_loop_reports_blocked_task_instead_of_completed_after_denied_tool() -> None:
    asyncio.run(_agent_loop_denied_tool_blocks_completion_case())


async def _agent_loop_denied_tool_blocks_completion_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    attempts = 0

    async def fake_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        if attempts == 1:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="write_1", name="write_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
        else:
            stream.end(AssistantMessage(content="final"))
        return stream

    async def denied_tool(*_args):
        return AgentToolResult(
            status="denied",
            is_error=True,
            error_code="read_only_mode",
        )

    result = await run_agent_loop(
        prompts=[UserMessage(content="写入文件")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="write_test",
                    label="write",
                    description="write",
                    parameters={},
                    execute=denied_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=_task_test_model(),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_blocked"
    assert result.task is not None
    assert result.task.blocked_steps == ["完成当前请求"]
    assert result.task.completion_satisfied is False


def test_agent_loop_preserves_cancelled_stop_reason() -> None:
    asyncio.run(_agent_loop_cancelled_stop_reason_case())


async def _agent_loop_cancelled_stop_reason_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    async def fake_stream(_model, _context, _options):
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[ToolCall(id="bash_1", name="bash_test", arguments={})],
                stop_reason="toolUse",
            )
        )
        return stream

    async def cancelled_tool(*_args):
        return AgentToolResult(status="cancelled", is_error=True)

    result = await run_agent_loop(
        prompts=[UserMessage(content="运行命令")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="bash_test",
                    label="bash",
                    description="bash",
                    parameters={},
                    execute=cancelled_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=_task_test_model(),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.status == "aborted"
    assert result.stop_reason == "aborted"


def test_agent_loop_does_not_complete_after_generic_tool_error() -> None:
    asyncio.run(_agent_loop_generic_tool_error_case())


async def _agent_loop_generic_tool_error_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    attempts = 0

    async def fake_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        if attempts == 1:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="tool_1", name="custom_tool", arguments={})],
                    stop_reason="toolUse",
                )
            )
        else:
            stream.end(AssistantMessage(content="final"))
        return stream

    async def failing_tool(*_args):
        return AgentToolResult(
            status="error",
            is_error=True,
            error_code="tool_exception",
        )

    result = await run_agent_loop(
        prompts=[UserMessage(content="做一个需要工具的任务")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="custom_tool",
                    label="custom",
                    description="custom",
                    parameters={},
                    execute=failing_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=_task_test_model(),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_incomplete"
    assert result.task is not None
    assert result.task.completion_satisfied is False
    assert result.task.completion_reason == "incomplete_steps"


def test_agent_loop_waits_for_user_when_revert_is_proposed() -> None:
    asyncio.run(_agent_loop_propose_revert_case())


async def _agent_loop_propose_revert_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    attempts = 0

    async def fake_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        if attempts == 1:
            call = ToolCall(id="edit_1", name="edit_test", arguments={})
        elif attempts == 2:
            call = ToolCall(id="test_1", name="test_tool", arguments={})
        elif attempts == 3:
            call = ToolCall(id="test_2", name="test_tool", arguments={})
        else:
            stream.end(AssistantMessage(content=[TextContent(text="final")]))
            return stream
        stream.end(AssistantMessage(content=[call], stop_reason="toolUse"))
        return stream

    async def edit_tool(*_args):
        return AgentToolResult(
            status="success",
            workspace_changed=True,
            affected_paths=["src/app.py"],
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

    async def failed_test(*_args):
        return AgentToolResult(
            status="error",
            is_error=True,
            verification={
                "status": "failed",
                "command": "python -m pytest test/test_task.py -q",
                "exit_code": 1,
                "summary": "failed",
            },
        )

    result = await run_agent_loop(
        prompts=[UserMessage(content="修改实现并验证")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="edit_test",
                    label="edit",
                    description="edit",
                    parameters={},
                    execute=edit_tool,
                ),
                AgentTool(
                    name="test_tool",
                    label="test",
                    description="test",
                    parameters={},
                    execute=failed_test,
                ),
            ],
        ),
        config=AgentLoopConfig(
            model=_task_test_model(),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
            repeated_tool_call_limit=20,
        ),
        emit=lambda _event: None,
        stream_fn=fake_stream,
    )

    assert result.status == "waiting_user"
    assert result.stop_reason == "task_blocked"
    assert result.task is not None
    assert result.task.control_signal["rollback_required"] is True
    assert result.task.control_signal["rollback_targets"] == ["src/app.py"]
    assert result.task.next_action == "报告可能需要撤销的变更并等待用户确认"


async def _agent_loop_replan_limit_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, Model, ToolCall, UserMessage
    from codepilot.tools import AgentTool, AgentToolResult

    attempts = 0

    async def fake_stream(_model, _context, _options):
        nonlocal attempts
        attempts += 1
        stream = AssistantMessageEventStream()
        stream.end(
            AssistantMessage(
                content=[
                    ToolCall(
                        id=f"test_{attempts}",
                        name="bash_test",
                        arguments={"attempt": attempts},
                    )
                ],
                stop_reason="toolUse",
            )
        )
        return stream

    async def test_tool(*_args):
        return AgentToolResult(
            status="error",
            is_error=True,
            verification={
                "status": "failed",
                "command": "python -m pytest test/test_task.py -q",
                "exit_code": 1,
                "summary": "failed",
            },
        )

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="修复失败测试")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="bash_test",
                    label="bash",
                    description="bash",
                    parameters={},
                    execute=test_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
            repeated_tool_call_limit=20,
            max_tool_iterations=20,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.status == "failed"
    assert result.stop_reason == "replan_limit"
    assert result.task is not None
    assert result.task.blocked_steps == ["根据最新失败证据调整方案"]
    assert result.task.next_action == "报告连续失败并等待用户指示"
    assert any(
        event.get("type") == "task_decision"
        and event.get("decision", {}).get("reason") == "replan_limit_exceeded"
        for event in events
    )


async def _agent_loop_task_summary_case() -> None:
    from codepilot.core import AgentContext, AgentLoopConfig, run_agent_loop
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import (
        AssistantMessage,
        Model,
        TextContent,
        ToolCall,
        ToolResultMessage,
        UserMessage,
    )
    from codepilot.tools import AgentTool, AgentToolResult

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        assert "## Current Task" in (context.system_prompt or "")
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[ToolCall(id="read_1", name="read_test", arguments={})],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def read_tool(*_args):
        return AgentToolResult(content=[TextContent(text="read result")])

    events: list[dict[str, Any]] = []
    result = await run_agent_loop(
        prompts=[UserMessage(content="解释这个文件")],
        context=AgentContext(
            system_prompt="rules",
            messages=[],
            tools=[
                AgentTool(
                    name="read_test",
                    label="read",
                    description="read",
                    parameters={},
                    execute=read_tool,
                )
            ],
        ),
        config=AgentLoopConfig(
            model=Model(
                id="task-test",
                name="Task Test",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            convert_to_llm=lambda items: items,
            allow_unmanaged_tools=True,
        ),
        emit=events.append,
        stream_fn=fake_stream,
    )

    assert result.task is not None
    assert result.task.goal == "解释这个文件"
    assert result.task.completed_steps == ["完成当前请求"]
    assert result.task.completion_satisfied is True
    assert any(event["type"] == "task_plan_created" for event in events)
    assert any(event["type"] == "task_step_updated" for event in events)
    assert events[-1]["result"].task is result.task


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
