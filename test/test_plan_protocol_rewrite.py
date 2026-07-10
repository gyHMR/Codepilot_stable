from __future__ import annotations

import asyncio

import pytest

from codepilot.core.plan import (
    PlanSnapshot,
    PlanState,
    PlanValidationError,
    apply_plan_snapshot,
)


def _snapshot(
    *,
    execution_objective: str | None = "重构登录模块",
    status: str | None = None,
    change_reason: str | None = None,
) -> PlanSnapshot:
    payload = {
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
    if execution_objective is not None:
        payload["execution_objective"] = execution_objective
    return PlanSnapshot.from_mapping(
        payload
    )


def test_plan_mode_creates_pending_proposal_with_completion_criteria() -> None:
    state = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="plan",
        run_id="run_plan",
    )

    assert state.status == "proposed"
    assert state.objective == "重构登录模块"
    assert state.completion_criteria == ("相关测试通过", "登录流程可以正常使用")
    assert [item.status for item in state.items] == ["pending", "pending"]


def test_active_plan_rejects_structure_replacement_without_explicit_reason() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
    )
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
            "explanation": "直接替换步骤。",
        }
    )

    with pytest.raises(PlanValidationError, match="structure"):
        apply_plan_snapshot(
            active,
            replacement,
            mode="build",
            run_id="run_build",
        )


def test_five_qualified_failures_allow_active_plan_revision() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
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
    )

    assert revised.revision == active.revision + 1
    assert revised.items[0].step == "改用兼容认证实现"


def test_active_plan_requires_explicit_completed_snapshot_before_closeout() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
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
    )

    assert result.status == "completed"
    assert result.completion_source == "model_closeout"


def test_old_plan_schema_is_rejected() -> None:
    with pytest.raises(PlanValidationError, match="unsupported plan state schema"):
        PlanState.from_mapping(
            {
                "schema_version": 3,
                "plan_id": "plan_old",
                "owner_run_id": "run_old",
                "status": "active",
                "approval_state": "approved",
            }
        )


def test_new_plan_requires_an_execution_objective() -> None:
    with pytest.raises(PlanValidationError, match="execution_objective"):
        apply_plan_snapshot(
            None,
            _snapshot(execution_objective=None),
            mode="plan",
            run_id="run_plan",
        )


def test_active_plan_cannot_replace_its_execution_objective() -> None:
    active = apply_plan_snapshot(
        None,
        _snapshot(),
        mode="build",
        run_id="run_build",
    )

    with pytest.raises(PlanValidationError, match="execution_objective"):
        apply_plan_snapshot(
            active,
            _snapshot(execution_objective="改造权限模块"),
            mode="build",
            run_id="run_build",
        )


def test_update_plan_tool_emits_only_canonical_snapshot_metadata() -> None:
    from codepilot.protocols import UPDATE_PLAN_TOOL
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    tool = create_plan_tools(allow=lambda name: name == UPDATE_PLAN_TOOL)[0]
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_plan",
                tool_call_id="call_plan",
                name=UPDATE_PLAN_TOOL,
                current_mode="plan",
                arguments={
                    "execution_objective": "重构登录模块",
                    "summary": "批准后完成登录模块重构。",
                    "completion_criteria": ["相关测试通过"],
                    "items": [
                        {
                            "step": "重构登录服务",
                            "details": "整理登录服务职责。",
                            "verification": "运行登录测试。",
                            "status": "in_progress",
                        }
                    ],
                },
            )
        )
    )

    assert result.status == "success"
    assert set(result.metadata) == {"plan_snapshot"}
    assert result.metadata["plan_snapshot"]["items"][0]["status"] == "pending"
