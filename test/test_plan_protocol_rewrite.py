from __future__ import annotations

import pytest

from codepilot.core.plan import (
    PlanSnapshot,
    PlanState,
    PlanValidationError,
    apply_plan_snapshot,
)
from codepilot.protocols import (
    PROPOSE_PLAN_TOOL,
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



def test_propose_plan_schema_does_not_expose_framework_owned_item_fields() -> None:
    from codepilot.core.tool_adapters.plan import create_plan_registrations

    registration = {
        item.spec.name: item
        for item in create_plan_registrations(
            service=None,  # type: ignore[arg-type]
            allow=lambda name: True,
        )
    }[PROPOSE_PLAN_TOOL]
    parameters = registration.spec.input_schema
    item_schema = parameters["properties"]["items"]["items"]

    assert set(item_schema["properties"]) == {"step", "details", "verification"}
    assert item_schema["required"] == ("step", "details", "verification")
    assert "raw_user_request" in parameters["required"]
    assert "interpreted_goal" in parameters["required"]
