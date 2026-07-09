from __future__ import annotations

import json
from pathlib import Path

import pytest


def _canonical_plan(**updates):
    state = {
        "schema_version": 2,
        "plan_id": "plan_1",
        "owner_run_id": "run_1",
        "status": "active",
        "approval_state": "approved",
        "origin_mode": "build",
        "objective": "重构任务编排",
        "summary": "统一任务级 Run 和计划状态。",
        "items": [
            {
                "id": "item_1",
                "step": "实现 soft PlanState",
                "details": "实现结构化计划持久化。",
                "verification": "运行计划状态测试。",
                "status": "in_progress",
            }
        ],
        "revision": 1,
        "explanation": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "completed_at": None,
        "completion_source": None,
    }
    state.update(updates)
    return state


def test_plan_state_store_rejects_legacy_task_fields(tmp_path: Path) -> None:
    from codepilot.core.plan import PlanValidationError
    from codepilot.sessions.plan_state import PlanStateStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_plan")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    session_store.plan_state_file.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "task_id": "task_legacy",
                "goal": {"value": "旧任务状态"},
                "steps": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(PlanValidationError):
        PlanStateStore(session_store).load()


def test_plan_state_store_writes_schema_v2_without_task_fields(tmp_path: Path) -> None:
    from codepilot.sessions.plan_state import PlanStateStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_plan")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = PlanStateStore(session_store)

    saved = store.save(_canonical_plan())

    raw = json.loads(session_store.plan_state_file.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert raw == saved
    assert "task_id" not in raw
    assert "goal" not in raw
    assert "steps" not in raw


def test_plan_state_store_only_marks_proposed_or_active_as_current(tmp_path: Path) -> None:
    from codepilot.sessions.plan_state import PlanStateStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_plan")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = PlanStateStore(session_store)

    store.save(_canonical_plan(status="proposed", approval_state="pending"))
    assert session_store.read_meta()["active_plan_id"] == "plan_1"

    store.save(
        _canonical_plan(
            status="completed",
            items=[
                {
                    "id": "item_1",
                    "step": "实现 soft PlanState",
                    "details": "实现结构化计划持久化。",
                    "verification": "运行计划状态测试。",
                    "status": "completed",
                }
            ],
            completed_at="2026-01-01T01:00:00+00:00",
            completion_source="run_finalized",
        )
    )
    assert session_store.read_meta()["active_plan_id"] is None

    store.save(_canonical_plan(status="rejected", approval_state="rejected"))
    assert session_store.read_meta()["active_plan_id"] is None

    store.save(_canonical_plan(status="abandoned"))
    assert session_store.read_meta()["active_plan_id"] is None


def test_plan_state_mode_defaults_do_not_complete_from_item_statuses() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem

    plan_mode = PlanState.new(objective="先给方案", origin_mode="plan", run_id="run_plan")
    proposed = plan_mode.apply_update(
        PlanUpdate(
            summary="先说明方案。",
            items=(
                PlanUpdateItem(
                    step="说明方案",
                    details="说明关键实现决策。",
                    verification="用户能够审阅完整方案。",
                    status="pending",
                ),
            ),
        ),
        mode="plan",
        run_id="run_plan",
    )
    build_mode = PlanState.new(
        objective="直接实现",
        origin_mode="build",
        run_id="run_build",
    )
    active_plan = build_mode.apply_update(
        PlanUpdate(
            summary="实现后总结。",
            items=(
                PlanUpdateItem(
                    step="总结",
                    details="整理实现结果。",
                    verification="输出包含验证证据。",
                    status="completed",
                ),
            ),
        ),
        mode="build",
        run_id="run_build",
    )

    assert proposed.status == "proposed"
    assert proposed.approval_state == "pending"
    assert proposed.items[0].status == "pending"
    assert active_plan.status == "active"
    assert active_plan.approval_state == "not_required"
    assert active_plan.items[0].status == "completed"


def test_plan_state_and_update_tool_allow_larger_coding_plans() -> None:
    import asyncio

    from codepilot.core.plan import MAX_PLAN_ITEMS, PlanState, PlanUpdate, PlanUpdateItem
    from codepilot.protocols import UPDATE_PLAN_TOOL
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    assert MAX_PLAN_ITEMS >= 20
    items = tuple(
        PlanUpdateItem(
            step=f"步骤 {index}",
            details=f"执行步骤 {index}",
            verification=f"验证步骤 {index}",
            status="pending",
        )
        for index in range(MAX_PLAN_ITEMS)
    )
    state = PlanState.new(
        objective="大型重构",
        origin_mode="build",
        run_id="run_large_plan",
    ).apply_update(
        PlanUpdate(summary="分阶段完成大型重构。", items=items),
        mode="build",
        run_id="run_large_plan",
    )
    assert len(state.items) == MAX_PLAN_ITEMS

    tool = create_plan_tools(allow=lambda name: name == UPDATE_PLAN_TOOL)[0]
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_large_plan",
                tool_call_id="call_plan",
                name=UPDATE_PLAN_TOOL,
                arguments={
                    "summary": "分阶段完成大型重构。",
                    "plan": [
                        {
                            "step": f"步骤 {index}",
                            "details": f"执行步骤 {index}",
                            "verification": f"验证步骤 {index}",
                            "status": "pending",
                        }
                        for index in range(MAX_PLAN_ITEMS)
                    ]
                },
            )
        )
    )

    assert result.status == "success"
    assert len(result.metadata["plan_update"]["plan"]) == MAX_PLAN_ITEMS


def test_update_plan_tool_normalizes_proposed_items_in_plan_mode() -> None:
    import asyncio

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
                    "summary": "阅读实现后提出待审批方案。",
                    "plan": [
                        {
                            "step": "阅读实现",
                            "details": "阅读相关实现。",
                            "verification": "确认关键调用链。",
                            "status": "completed",
                        },
                        {
                            "step": "提出方案",
                            "details": "形成结构化方案。",
                            "verification": "方案包含实现与验证。",
                            "status": "in_progress",
                        },
                        {
                            "step": "等待审批",
                            "details": "发布方案等待用户决定。",
                            "verification": "PlanState 为 proposed。",
                            "status": "pending",
                        },
                    ]
                },
            )
        )
    )

    assert result.status == "success"
    assert result.metadata["plan_normalized"] is True
    assert result.metadata["original_plan_statuses"] == [
        "completed",
        "in_progress",
        "pending",
    ]
    assert [
        item["status"]
        for item in result.metadata["plan_update"]["plan"]
    ] == ["pending", "pending", "pending"]


def test_rejected_plan_cannot_be_changed_by_model_update() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem, PlanValidationError

    rejected = PlanState.new(
        objective="方案被拒绝",
        origin_mode="plan",
        run_id="run_bad",
    ).apply_update(
        PlanUpdate(
            summary="待拒绝方案。",
            items=(
                PlanUpdateItem(
                    step="提出方案",
                    details="形成待审批方案。",
                    verification="用户审阅方案。",
                    status="pending",
                ),
            ),
        ),
        mode="plan",
        run_id="run_bad",
    ).reject()

    with pytest.raises(PlanValidationError, match="rejected plan"):
        rejected.apply_update(
            PlanUpdate(
                summary="强行激活。",
                items=(
                    PlanUpdateItem(
                        step="强行激活",
                        details="尝试修改 rejected plan。",
                        verification="应被拒绝。",
                        status="in_progress",
                    ),
                ),
            ),
            mode="build",
            run_id="run_bad",
        )


def test_agent_context_uses_plan_state_and_run_signals_only() -> None:
    from codepilot.core.contracts import AgentContext

    context = AgentContext(
        system_prompt="sys",
        messages=[],
        plan_state=_canonical_plan(),
        run_signals={"verification_status": "unknown"},
    )

    assert context.plan_state is not None
    assert context.plan_state["plan_id"] == "plan_1"
    assert context.run_signals == {"verification_status": "unknown"}
    assert not hasattr(context, "task_state")
    assert not hasattr(context, "task_signal")

    with pytest.raises(TypeError):
        AgentContext(system_prompt="sys", messages=[], task_state={})  # type: ignore[call-arg]


def test_plan_state_v2_is_a_structured_execution_contract() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem

    proposed = PlanState.new(
        objective="重构任务编排",
        origin_mode="plan",
        run_id="run_task",
    ).apply_update(
        PlanUpdate(
            summary="以任务级 Run 统一计划和工具审批恢复。",
            explanation="完成仓库分析后发布方案",
            items=(
                PlanUpdateItem(
                    step="统一 continuation 协议",
                    details="Plan 和工具审批都从 checkpoint 恢复原 run。",
                    verification="审批前后 run_id 保持一致。",
                    status="in_progress",
                ),
            ),
        ),
        mode="plan",
        run_id="run_task",
    )

    payload = proposed.to_dict()
    assert payload["schema_version"] == 2
    assert payload["owner_run_id"] == "run_task"
    assert payload["status"] == "proposed"
    assert payload["approval_state"] == "pending"
    assert payload["origin_mode"] == "plan"
    assert payload["summary"] == "以任务级 Run 统一计划和工具审批恢复。"
    assert payload["revision"] == 1
    assert payload["items"] == [
        {
            "id": "item_1",
            "step": "统一 continuation 协议",
            "details": "Plan 和工具审批都从 checkpoint 恢复原 run。",
            "verification": "审批前后 run_id 保持一致。",
            "status": "pending",
        }
    ]
    assert payload["completed_at"] is None
    assert payload["completion_source"] is None


def test_plan_revision_preserves_origin_and_runtime_completion_keeps_soft_items() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem

    proposed = PlanState.new(
        objective="优化注册逻辑",
        origin_mode="plan",
        run_id="run_task",
    ).apply_update(
        PlanUpdate(
            summary="先整理校验边界，再实施修改。",
            items=(
                PlanUpdateItem(
                    step="整理校验边界",
                    details="识别重复校验和调用关系。",
                    verification="列出受影响入口。",
                    status="pending",
                ),
            ),
        ),
        mode="plan",
        run_id="run_task",
    )
    revised = proposed.apply_update(
        PlanUpdate(
            summary="先补回归测试，再整理校验边界。",
            explanation="根据用户反馈调整顺序",
            items=(
                PlanUpdateItem(
                    step="补回归测试",
                    details="覆盖注册成功和重复用户名。",
                    verification="目标测试先失败后通过。",
                    status="pending",
                ),
            ),
        ),
        mode="plan",
        run_id="run_task",
    )
    completed = revised.approve().complete(source="run_finalized")

    assert revised.plan_id == proposed.plan_id
    assert revised.revision == 2
    assert revised.origin_mode == "plan"
    assert completed.status == "completed"
    assert completed.approval_state == "approved"
    assert completed.items[0].status == "pending"
    assert completed.completed_at is not None
    assert completed.completion_source == "run_finalized"


def test_build_and_read_plans_do_not_pretend_to_be_approved() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem

    for mode in ("build", "read"):
        state = PlanState.new(
            objective="处理复杂任务",
            origin_mode=mode,
            run_id=f"run_{mode}",
        ).apply_update(
            PlanUpdate(
                summary="按两个可验证阶段推进。",
                items=(
                    PlanUpdateItem(
                        step="完成当前阶段",
                        details="执行当前模式允许的动作。",
                        verification="检查阶段输出。",
                        status="in_progress",
                    ),
                ),
            ),
            mode=mode,
            run_id=f"run_{mode}",
        )

        assert state.status == "active"
        assert state.approval_state == "not_required"


def test_update_plan_tool_publishes_structured_execution_contract() -> None:
    import asyncio

    from codepilot.protocols import UPDATE_PLAN_TOOL
    from codepilot.tools.builtins.plan import create_plan_tools
    from codepilot.tools.contracts import ToolCallRequest

    tool = create_plan_tools(allow=lambda name: name == UPDATE_PLAN_TOOL)[0]
    result = asyncio.run(
        tool.execute(
            ToolCallRequest(
                run_id="run_contract",
                tool_call_id="call_contract",
                name=UPDATE_PLAN_TOOL,
                current_mode="plan",
                arguments={
                    "summary": "统一任务级 Run 的暂停和恢复。",
                    "explanation": "发布待审批方案",
                    "plan": [
                        {
                            "step": "实现 continuation",
                            "details": "所有审批都恢复原 run。",
                            "verification": "审批前后 run_id 一致。",
                            "status": "in_progress",
                        }
                    ],
                },
            )
        )
    )

    assert result.status == "success"
    assert result.metadata["plan_update"] == {
        "summary": "统一任务级 Run 的暂停和恢复。",
        "explanation": "发布待审批方案",
        "plan": [
            {
                "step": "实现 continuation",
                "details": "所有审批都恢复原 run。",
                "verification": "审批前后 run_id 一致。",
                "status": "pending",
            }
        ],
    }
