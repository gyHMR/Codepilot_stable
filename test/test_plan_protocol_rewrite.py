from __future__ import annotations

import pytest

from codepilot.core.plan import (
    PLAN_STATE_SCHEMA_VERSION,
    PendingPlanRevision,
    PlanCloseRequest,
    PlanDefinition,
    PlanState,
    PlanStep,
    PlanValidationError,
    load_plan_state,
)


def _definition(summary: str = "重构登录模块") -> PlanDefinition:
    return PlanDefinition(
        summary=summary,
        completion_criteria=("相关测试通过", "登录流程保持兼容"),
        task_understanding="先确认当前实现，再收敛登录边界。",
        current_implementation="登录服务同时承担认证和会话写入。",
        target_design="认证与会话写入通过清晰接口协作。",
        impact_scope="登录服务和对应测试。",
        risks_and_open_questions=("第三方认证边界需要保持兼容。",),
        verification_plan="运行登录模块的聚焦测试。",
    )


def _state() -> PlanState:
    return PlanState(
        plan_id="plan:submit-1",
        origin="plan_mode",
        status="active",
        revision=3,
        definition=_definition(),
        steps=(
            PlanStep(
                step_id="plan:submit-1:step:1",
                step="收敛登录职责",
                details="拆分认证和会话写入。",
                verification="运行登录服务测试。",
                status="completed",
                completion_note="职责边界已经收敛。",
                evidence_refs=("tool-1",),
            ),
            PlanStep(
                step_id="plan:submit-1:step:2",
                step="验证登录流程",
                details="执行聚焦回归。",
                verification="运行登录流程测试。",
                status="in_progress",
            ),
        ),
        pending_revision=PendingPlanRevision(
            reason="user_request",
            definition=_definition("调整登录重构范围"),
            steps=(
                PlanStep(
                    step_id="pending:step:1",
                    step="收敛登录职责",
                    details="拆分认证和会话写入。",
                    verification="运行登录服务测试。",
                ),
            ),
            proposed_at_revision=2,
        ),
        close_request=PlanCloseRequest(
            summary="等待最终验证。",
            evidence_refs=("tool-1",),
            requested_at_revision=2,
        ),
    )


def test_plan_state_round_trip_uses_only_core_owned_shape() -> None:
    state = _state()

    payload = state.to_dict()
    restored = PlanState.from_mapping(payload)

    assert restored == state
    assert set(payload) == {
        "schema_version",
        "plan_id",
        "origin",
        "status",
        "revision",
        "definition",
        "steps",
        "pending_revision",
        "close_request",
    }
    assert "raw_user_request" not in payload
    assert "interpreted_goal" not in payload
    assert "owner_run_id" not in payload


def test_plan_state_rejects_unknown_fields_and_multiple_in_progress_steps() -> None:
    payload = _state().to_dict()
    payload["raw_user_request"] = "不应重复保存的请求"

    with pytest.raises(PlanValidationError, match="unknown plan state fields"):
        PlanState.from_mapping(payload)

    with pytest.raises(PlanValidationError, match="at most one in_progress"):
        PlanState(
            plan_id="plan:bad",
            origin="build_mode",
            status="active",
            revision=1,
            definition=_definition(),
            steps=(
                PlanStep("step-1", "第一步", "实现第一步", "验证第一步", "in_progress"),
                PlanStep("step-2", "第二步", "实现第二步", "验证第二步", "in_progress"),
            ),
        )


def test_load_plan_state_migrates_current_schema_without_retaining_task_identity() -> None:
    legacy = {
        "schema_version": 6,
        "plan_id": "plan_old",
        "owner_run_id": "run_old",
        "status": "active",
        "origin_mode": "plan",
        "raw_user_request": "重构登录模块",
        "interpreted_goal": "完成登录重构",
        "task_understanding": "理解当前登录实现。",
        "current_implementation": "旧实现。",
        "target_design": "新实现。",
        "impact_scope": "登录模块。",
        "risks_and_open_questions": ["兼容性。"],
        "verification_plan": "运行测试。",
        "summary": "重构登录模块",
        "completion_criteria": ["测试通过"],
        "items": [
            {
                "id": "item_1",
                "step": "修改实现",
                "details": "调整登录服务。",
                "verification": "运行登录测试。",
                "status": "pending",
            }
        ],
        "revision": 2,
        "explanation": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "completed_at": None,
        "completion_source": None,
    }

    migrated = load_plan_state(legacy)

    assert migrated is not None
    assert migrated.schema_version == PLAN_STATE_SCHEMA_VERSION
    assert migrated.origin == "plan_mode"
    assert migrated.steps[0].step_id == "item_1"
    assert "raw_user_request" not in migrated.to_dict()


def test_plan_definition_requires_bounded_completion_criteria() -> None:
    with pytest.raises(PlanValidationError, match="between 1 and 5"):
        PlanDefinition(summary="无效计划", completion_criteria=())
