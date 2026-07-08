from __future__ import annotations

import json
from pathlib import Path

import pytest


def _canonical_plan(**updates):
    state = {
        "schema_version": 1,
        "plan_id": "plan_1",
        "status": "active",
        "approval_state": "approved",
        "origin_mode": "build",
        "objective": "重构任务编排",
        "items": [
            {
                "id": "item_1",
                "step": "实现 soft PlanState",
                "status": "in_progress",
            }
        ],
        "explanation": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "last_update_run_id": "run_1",
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


def test_plan_state_store_writes_schema_v1_without_task_fields(tmp_path: Path) -> None:
    from codepilot.sessions.plan_state import PlanStateStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_plan")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = PlanStateStore(session_store)

    saved = store.save(_canonical_plan())

    raw = json.loads(session_store.plan_state_file.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 1
    assert raw == saved
    assert "task_id" not in raw
    assert "goal" not in raw
    assert "steps" not in raw


def test_plan_state_mode_defaults_are_soft_not_completion_gate() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem

    plan_mode = PlanState.new(objective="先给方案", origin_mode="plan")
    proposed = plan_mode.apply_update(
        PlanUpdate(items=(PlanUpdateItem(step="说明方案", status="completed"),)),
        mode="plan",
        run_id="run_plan",
    )
    build_mode = PlanState.new(objective="直接实现", origin_mode="build")
    completed_plan = build_mode.apply_update(
        PlanUpdate(items=(PlanUpdateItem(step="总结", status="completed"),)),
        mode="build",
        run_id="run_build",
    )

    assert proposed.status == "proposed"
    assert proposed.approval_state == "proposed"
    assert completed_plan.status == "completed"
    assert completed_plan.items[0].status == "completed"


def test_rejected_plan_cannot_be_changed_by_model_update() -> None:
    from codepilot.core.plan import PlanState, PlanUpdate, PlanUpdateItem, PlanValidationError

    rejected = PlanState.new(objective="方案被拒绝", origin_mode="plan")
    rejected = PlanState(
        plan_id=rejected.plan_id,
        status="rejected",
        approval_state="rejected",
        origin_mode=rejected.origin_mode,
        objective=rejected.objective,
        items=rejected.items,
        created_at=rejected.created_at,
        updated_at=rejected.updated_at,
    )

    with pytest.raises(PlanValidationError, match="rejected plan"):
        rejected.apply_update(
            PlanUpdate(items=(PlanUpdateItem(step="强行激活", status="in_progress"),)),
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
