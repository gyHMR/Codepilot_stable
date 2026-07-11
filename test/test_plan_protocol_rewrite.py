from __future__ import annotations

import asyncio

import pytest

from codepilot.core.plan import (
    PlanSnapshot,
    PlanState,
    PlanValidationError,
    apply_plan_snapshot,
)
from codepilot.protocols import (
    CLOSE_PLAN_TOOL,
    CREATE_BUILD_PLAN_TOOL,
    PROPOSE_PLAN_TOOL,
    ToolHookContextSnapshot,
    UPDATE_PLAN_PROGRESS_TOOL,
)


def _snapshot(
    *,
    raw_user_request: str | None = "用户要求优化登录逻辑并先给出方案。",
    interpreted_goal: str | None = "重构登录模块并验证登录流程。",
    status: str | None = None,
    change_reason: str | None = None,
    include_plan_sections: bool = True,
) -> PlanSnapshot:
    payload = {
        "raw_user_request": raw_user_request,
        "interpreted_goal": interpreted_goal,
        "task_understanding": "用户希望先审批方案，再由 Build 修改登录逻辑并验证行为。",
        "current_implementation": "已定位登录服务和注册测试作为主要证据。",
        "target_design": "保持接口兼容，收敛登录服务职责并补充回归验证。",
        "impact_scope": "影响登录服务、调用入口和注册相关测试。",
        "risks_and_open_questions": ["需要确认登录边界是否覆盖第三方认证。"],
        "verification_plan": "运行登录模块测试和注册回归测试。",
        "summary": "完成登录模块重构。",
        "completion_criteria": ["相关测试通过", "登录流程可以正常使用"],
        "items": [
            {
                "step": "重构登录服务",
                "details": "整理登录服务的职责边界。",
                "verification": "运行登录模块测试。",
                "status": "in_progress",
            },
            {
                "step": "验证登录流程",
                "details": "执行回归验证。",
                "verification": "运行完整测试。",
                "status": "pending",
            },
        ],
        "status": status,
        "change_reason": change_reason,
        "explanation": "基于当前仓库事实更新计划。",
    }
    if raw_user_request is None:
        del payload["raw_user_request"]
    if interpreted_goal is None:
        del payload["interpreted_goal"]
    if not include_plan_sections:
        for key in [
            "task_understanding",
            "current_implementation",
            "target_design",
            "impact_scope",
            "risks_and_open_questions",
            "verification_plan",
        ]:
            del payload[key]
    return PlanSnapshot.from_mapping(
        payload
    )


def test_plan_mode_creates_pending_proposal_with_request_and_goal() -> None:
    state = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="plan",
        run_id="run_plan",
        operation="propose_plan",
    )

    assert state.status == "proposed"
    assert state.raw_user_request == "用户要求优化登录逻辑并先给出方案。"
    assert state.interpreted_goal == "重构登录模块并验证登录流程。"
    assert state.task_understanding == "用户希望先审批方案，再由 Build 修改登录逻辑并验证行为。"
    assert state.current_implementation == "已定位登录服务和注册测试作为主要证据。"
    assert state.target_design == "保持接口兼容，收敛登录服务职责并补充回归验证。"
    assert state.impact_scope == "影响登录服务、调用入口和注册相关测试。"
    assert state.risks_and_open_questions == ("需要确认登录边界是否覆盖第三方认证。",)
    assert state.verification_plan == "运行登录模块测试和注册回归测试。"
    assert state.completion_criteria == ("相关测试通过", "登录流程可以正常使用")
    assert [item.status for item in state.items] == ["pending", "pending"]


def test_plan_mode_requires_structured_proposal_sections() -> None:
    with pytest.raises(PlanValidationError, match="task_understanding"):
        apply_plan_snapshot(
            None,
            _snapshot(include_plan_sections=False),
            mode="plan",
            run_id="run_plan",
            operation="propose_plan",
        )


def test_build_mode_creates_lightweight_active_plan_without_approval() -> None:
    state = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
        operation="create_build_plan",
    )

    assert state.status == "active"
    assert state.origin_mode == "build"
    assert state.raw_user_request == "用户要求优化登录逻辑并先给出方案。"
    assert state.interpreted_goal == "重构登录模块并验证登录流程。"


def test_build_create_plan_rejects_existing_current_plan() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
        operation="create_build_plan",
    )

    with pytest.raises(PlanValidationError, match="current Task Plan"):
        apply_plan_snapshot(
            active,
            _snapshot(),
            mode="build",
            run_id="run_build",
            operation="create_build_plan",
        )


def test_plan_approved_active_plan_rejects_structural_replacement() -> None:
    proposed = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="plan",
        run_id="run_plan",
        operation="propose_plan",
    )
    active = proposed.approve()
    replacement = PlanSnapshot.from_mapping(
        {
            "summary": "改用新的实现路径。",
            "completion_criteria": ["相关测试通过"],
            "items": [
                {
                    "id": active.items[0].id,
                    "step": "替换认证实现",
                    "details": "迁移到新的认证边界。",
                    "verification": "运行认证测试。",
                    "status": "in_progress",
                }
            ],
            "change_reason": "user_request",
            "explanation": "直接替换步骤。",
        }
    )

    with pytest.raises(PlanValidationError, match="Plan-mode approved"):
        apply_plan_snapshot(
            active,
            replacement,
            mode="build",
            run_id="run_after_approval",
            operation="update_plan_progress",
        )


def test_plan_approved_active_plan_can_update_status_and_close_across_runs() -> None:
    proposed = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="plan",
        run_id="run_plan",
        operation="propose_plan",
    )
    active = proposed.approve()
    progress = PlanSnapshot.from_mapping(
        {
            "summary": active.summary,
            "completion_criteria": list(active.completion_criteria),
            "items": [
                {**active.items[0].to_dict(), "status": "completed"},
                {**active.items[1].to_dict(), "status": "in_progress"},
            ],
            "explanation": "第一步已经完成，继续验证。",
        }
    )

    updated = apply_plan_snapshot(
        active,
        progress,
        mode="build",
        run_id="run_after_approval",
        operation="update_plan_progress",
    )

    assert updated.plan_id == active.plan_id
    assert [item.status for item in updated.items] == ["completed", "in_progress"]

    closeout = PlanSnapshot.from_mapping(
        {
            "summary": updated.summary,
            "completion_criteria": list(updated.completion_criteria),
            "items": [
                {**item.to_dict(), "status": "completed"}
                for item in updated.items
            ],
            "status": "completed",
            "explanation": "最终完成标准已核对。",
        }
    )
    completed = apply_plan_snapshot(
        updated,
        closeout,
        mode="build",
        run_id="run_after_approval",
        operation="close_plan",
    )

    assert completed.status == "completed"
    assert completed.completion_source == "model_closeout"


def test_five_qualified_failures_allow_active_plan_revision() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
        operation="create_build_plan",
    )
    replacement = PlanSnapshot.from_mapping(
        {
            "summary": "改用兼容实现。",
            "completion_criteria": ["相关测试通过"],
            "items": [
                {
                    "step": "改用兼容认证实现",
                    "details": "绕开当前实现阻塞。",
                    "verification": "运行认证测试。",
                    "status": "in_progress",
                }
            ],
            "change_reason": "repeated_execution_failure",
            "explanation": "当前步骤已发生五次有效验证失败。",
        }
    )

    revised = apply_plan_snapshot(
        active,
        replacement,
        mode="build",
        run_id="run_build",
        qualified_failure_count=5,
        operation="update_plan_progress",
    )

    assert revised.revision == active.revision + 1
    assert revised.items[0].step == "改用兼容认证实现"


def test_close_plan_can_complete_even_when_prior_steps_were_not_all_marked_done() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
        operation="create_build_plan",
    )
    completed = PlanSnapshot.from_mapping(
        {
            "summary": active.summary,
            "completion_criteria": list(active.completion_criteria),
            "items": [
                {
                    **item.to_dict(),
                    "status": "completed",
                }
                for item in active.items
            ],
            "status": "completed",
            "explanation": "全部完成标准均已核对。",
        }
    )

    result = apply_plan_snapshot(
        active,
        completed,
        mode="build",
        run_id="run_build",
        operation="close_plan",
    )

    assert result.status == "completed"
    assert result.completion_source == "model_closeout"


def test_close_plan_can_leave_plan_active_when_task_is_obviously_unfinished() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
        operation="create_build_plan",
    )
    unfinished = PlanSnapshot.from_mapping(
        {
            "summary": active.summary,
            "completion_criteria": list(active.completion_criteria),
            "items": [item.to_dict() for item in active.items],
            "status": "active",
            "explanation": "验证命令仍失败，保留剩余步骤。",
        }
    )

    result = apply_plan_snapshot(
        active,
        unfinished,
        mode="build",
        run_id="run_build",
        operation="close_plan",
    )

    assert result.status == "active"
    assert result.completed_at is None


def test_old_plan_schema_is_rejected() -> None:
    with pytest.raises(PlanValidationError, match="unsupported plan state schema"):
        PlanState.from_mapping(
            {
                "schema_version": 4,
                "plan_id": "plan_old",
                "owner_run_id": "run_old",
                "status": "active",
                "objective": "旧字段",
            }
        )


def test_new_plan_requires_raw_user_request_and_interpreted_goal() -> None:
    with pytest.raises(PlanValidationError, match="raw_user_request"):
        apply_plan_snapshot(
            None,
            _snapshot(raw_user_request=None),
            mode="plan",
            run_id="run_plan",
            operation="propose_plan",
        )
    with pytest.raises(PlanValidationError, match="interpreted_goal"):
        apply_plan_snapshot(
            None,
            _snapshot(interpreted_goal=None),
            mode="build",
            run_id="run_build",
            operation="create_build_plan",
        )


def test_active_plan_cannot_replace_request_or_goal() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
        operation="create_build_plan",
    )

    with pytest.raises(PlanValidationError, match="raw_user_request"):
        apply_plan_snapshot(
            active,
            _snapshot(raw_user_request="新的用户请求"),
            mode="build",
            run_id="run_build",
            operation="update_plan_progress",
        )


def test_semantic_plan_tools_emit_canonical_operation_metadata() -> None:
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    tools = {tool.name: tool for tool in create_plan_tools(allow=lambda name: True)}
    assert set(tools) == {
        PROPOSE_PLAN_TOOL,
        CREATE_BUILD_PLAN_TOOL,
        UPDATE_PLAN_PROGRESS_TOOL,
        CLOSE_PLAN_TOOL,
    }
    tool = tools[PROPOSE_PLAN_TOOL]
    assert "only authoritative way to publish" in tool.description
    assert "ordinary assistant text is not an approvable Task Plan" in tool.description
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_plan",
                tool_call_id="call_plan",
                name=PROPOSE_PLAN_TOOL,
                current_mode="plan",
                arguments={
                    "raw_user_request": "用户要求优化登录逻辑并先给出方案。",
                    "interpreted_goal": "重构登录模块并验证登录流程。",
                    "summary": "批准后完成登录模块重构。",
                    "task_understanding": "用户希望优化登录逻辑并先审批方案。",
                    "current_implementation": "已确认登录服务和注册测试是主要修改边界。",
                    "target_design": "保持现有接口，调整登录服务内部职责。",
                    "impact_scope": "影响登录服务和注册回归测试。",
                    "risks_and_open_questions": ["暂无阻塞风险。"],
                    "verification_plan": "运行登录模块测试。",
                    "completion_criteria": ["相关测试通过"],
                    "items": [
                        {
                            "step": "重构登录服务",
                            "details": "整理登录服务职责。",
                            "verification": "运行登录测试。",
                        }
                    ],
                },
            )
        )
    )

    assert result.status == "success"
    assert set(result.metadata) == {"plan_operation", "plan_snapshot"}
    assert result.metadata["plan_operation"] == "propose_plan"
    assert result.metadata["plan_snapshot"]["items"][0]["status"] == "pending"
    assert "id" not in result.metadata["plan_snapshot"]["items"][0]


def test_propose_plan_schema_does_not_expose_framework_owned_item_fields() -> None:
    from codepilot.tools.builtins.plan import create_plan_tools

    tool = {tool.name: tool for tool in create_plan_tools(allow=lambda name: True)}[
        PROPOSE_PLAN_TOOL
    ]
    item_schema = tool.parameters["properties"]["items"]["items"]

    assert set(item_schema["properties"]) == {"step", "details", "verification"}
    assert item_schema["required"] == ["step", "details", "verification"]
    assert "raw_user_request" in tool.parameters["required"]
    assert "interpreted_goal" in tool.parameters["required"]


def test_propose_plan_tool_rejects_framework_owned_item_fields() -> None:
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    tool = {tool.name: tool for tool in create_plan_tools(allow=lambda name: True)}[
        PROPOSE_PLAN_TOOL
    ]
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_plan",
                tool_call_id="call_plan",
                name=PROPOSE_PLAN_TOOL,
                current_mode="plan",
                arguments={
                    "raw_user_request": "用户要求优化登录逻辑并先给出方案。",
                    "interpreted_goal": "重构登录模块并验证登录流程。",
                    "summary": "批准后完成登录模块重构。",
                    "task_understanding": "用户希望优化登录逻辑并先审批方案。",
                    "current_implementation": "已确认登录服务和注册测试是主要修改边界。",
                    "target_design": "保持现有接口，调整登录服务内部职责。",
                    "impact_scope": "影响登录服务和注册回归测试。",
                    "risks_and_open_questions": ["暂无阻塞风险。"],
                    "verification_plan": "运行登录模块测试。",
                    "completion_criteria": ["相关测试通过"],
                    "items": [
                        {
                            "step": "重构登录服务",
                            "details": "整理登录服务职责。",
                            "verification": "运行登录测试。",
                            "status": "pending",
                        }
                    ],
                },
            )
        )
    )

    assert result.status == "error"
    assert result.error_code == "invalid_plan_snapshot"
    assert "items[0] has unknown fields: status" in result.content[0].text


def test_propose_plan_normalizes_string_lists_and_allows_no_known_risks() -> None:
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    tool = {tool.name: tool for tool in create_plan_tools(allow=lambda name: True)}[
        PROPOSE_PLAN_TOOL
    ]
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_plan",
                tool_call_id="call_plan",
                name=PROPOSE_PLAN_TOOL,
                current_mode="plan",
                arguments={
                    "raw_user_request": "用户要求优化登录逻辑并先给出方案。",
                    "interpreted_goal": "重构登录模块并验证登录流程。",
                    "summary": "批准后完成登录模块重构。",
                    "task_understanding": "用户希望优化登录逻辑并先审批方案。",
                    "current_implementation": "已确认登录服务和注册测试是主要修改边界。",
                    "target_design": "保持现有接口，调整登录服务内部职责。",
                    "impact_scope": "影响登录服务和注册回归测试。",
                    "risks_and_open_questions": [],
                    "verification_plan": "运行登录模块测试。",
                    "completion_criteria": "相关测试通过",
                    "items": [
                        {
                            "step": "重构登录服务",
                            "details": "整理登录服务职责。",
                            "verification": "运行登录测试。",
                        }
                    ],
                },
            )
        )
    )

    assert result.status == "success"
    snapshot = result.metadata["plan_snapshot"]
    assert snapshot["risks_and_open_questions"] == []
    assert snapshot["completion_criteria"] == ["相关测试通过"]
    assert snapshot["items"][0]["status"] == "pending"


def test_propose_plan_revision_repeats_request_and_goal_for_schema_consistency() -> None:
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    proposed = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="plan",
        run_id="run_plan",
        operation="propose_plan",
    )
    tool = {tool.name: tool for tool in create_plan_tools(allow=lambda name: True)}[
        PROPOSE_PLAN_TOOL
    ]
    request = ToolCallRequest(
        run_id="run_plan",
        tool_call_id="call_revision",
        name=PROPOSE_PLAN_TOOL,
        current_mode="plan",
        context=ToolHookContextSnapshot(metadata={"plan_state": proposed.to_dict()}),
            arguments={
                "raw_user_request": "用户要求优化登录逻辑并先给出方案。",
                "interpreted_goal": "重构登录模块并验证登录流程。",
                "task_understanding": "用户反馈要求先补测试再调整登录逻辑。",
                "current_implementation": "已确认登录服务和测试边界仍然适用。",
                "target_design": "先补回归测试，再进行登录服务调整。",
                "impact_scope": "影响登录测试和登录服务。",
                "risks_and_open_questions": ["暂无新的待确认项。"],
                "verification_plan": "运行登录测试。",
                "summary": "批准后先补登录测试，再重构登录模块。",
                "completion_criteria": ["相关测试通过"],
                "items": [
                {
                    "step": "补充登录测试",
                    "details": "先覆盖关键登录边界。",
                    "verification": "运行登录测试。",
                }
            ],
        },
    )

    result = asyncio.run(tool.execute(request))

    assert result.status == "success"
    assert result.metadata["plan_snapshot"]["raw_user_request"] == "用户要求优化登录逻辑并先给出方案。"
    assert result.metadata["plan_snapshot"]["interpreted_goal"] == "重构登录模块并验证登录流程。"


def test_first_propose_plan_requires_request_and_goal_even_if_schema_is_soft() -> None:
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    tool = {tool.name: tool for tool in create_plan_tools(allow=lambda name: True)}[
        PROPOSE_PLAN_TOOL
    ]
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_plan",
                tool_call_id="call_plan",
                name=PROPOSE_PLAN_TOOL,
                current_mode="plan",
                arguments={
                    "summary": "批准后完成登录模块重构。",
                    "completion_criteria": ["相关测试通过"],
                    "items": [
                        {
                            "step": "重构登录服务",
                            "details": "整理登录服务职责。",
                            "verification": "运行登录测试。",
                            "status": "pending",
                        }
                    ],
                },
            )
        )
    )

    assert result.status == "error"
    assert result.error_code == "invalid_plan_snapshot"
    assert "raw_user_request is required" in result.content[0].text
