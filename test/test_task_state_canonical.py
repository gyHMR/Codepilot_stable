from __future__ import annotations

import json
from pathlib import Path

import pytest


def _canonical_state(**updates):
    state = {
        "schema_version": 2,
        "task_id": "task_1",
        "raw_user_request": "重构任务规划",
        "current_mode": "build",
        "approval_state": "none",
        "goal": {
            "value": "重构任务规划",
            "source": "user",
            "confidence": "explicit",
        },
        "user_constraints": [],
        "proposed_plan": None,
        "approved_plan": None,
        "current_step_id": "step_1",
        "steps": [
            {
                "id": "step_1",
                "title": "实现 canonical Task State",
                "kind": "edit",
                "status": "in_progress",
                "acceptance": "task_state.json 为唯一任务真相",
                "verification_hint": "python -m pytest test/test_task_state_canonical.py -q",
                "summary": None,
                "evidence_refs": [],
                "failure_count": 0,
            }
        ],
        "verification_status": "unknown",
        "evidence_refs": [],
        "blocked_reason": None,
        "recovery_summary": "",
        "source_run_id": "run_1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    state.update(updates)
    return state


def test_task_state_store_rejects_legacy_task_fields(tmp_path: Path) -> None:
    from codepilot.sessions.storage import SessionStore
    from codepilot.sessions.task_state import TaskStateStore, TaskStateValidationError

    session_store = SessionStore(tmp_path, "session_task")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    session_store.task_state_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_id": "task_legacy",
                "task_mode": "plan",
                "task_progress": {"completion_satisfied": False},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(TaskStateValidationError, match="legacy task state"):
        TaskStateStore(session_store).load()


def test_task_state_store_writes_schema_v2_without_legacy_fields(tmp_path: Path) -> None:
    from codepilot.sessions.storage import SessionStore
    from codepilot.sessions.task_state import TaskStateStore

    session_store = SessionStore(tmp_path, "session_task")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = TaskStateStore(session_store)

    saved = store.save(_canonical_state())

    raw = json.loads(session_store.task_state_file.read_text(encoding="utf-8"))
    assert saved["schema_version"] == 2
    assert raw == saved
    assert "task_mode" not in raw
    assert "task_progress" not in raw
    assert "recovery_projection" not in raw


def test_task_state_store_accepts_structured_proposed_plan_steps(tmp_path: Path) -> None:
    from codepilot.sessions.storage import SessionStore
    from codepilot.sessions.task_state import TaskStateStore

    session_store = SessionStore(tmp_path, "session_task")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = TaskStateStore(session_store)
    store.save(_canonical_state(current_mode="plan", approval_state="none"))

    state = store.apply_event(
        {
            "type": "planner_proposed_plan",
            "proposed_plan": {
                "plan_id": "plan_1",
                "source": "planner",
                "status": "proposed",
                "steps": [
                    {
                        "id": "step_1",
                        "title": "阅读任务控制链路",
                        "kind": "read",
                    }
                ],
            },
        }
    )

    assert state is not None
    assert state["current_mode"] == "plan"
    assert state["approval_state"] == "proposed"
    assert state["proposed_plan"]["steps"][0]["title"] == "阅读任务控制链路"


def test_task_recovery_symbols_are_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        __import__("codepilot.sessions.history.task_recovery")

    with pytest.raises(ImportError):
        from codepilot.core import build_task_state_from_recovery_projection  # noqa: F401


def test_agent_context_uses_structured_task_state_only() -> None:
    from codepilot.core.contracts import AgentContext

    context = AgentContext(
        system_prompt="sys",
        messages=[],
        task_state=_canonical_state(),
        task_signal={"current_mode": "build"},
    )

    assert context.task_state["task_id"] == "task_1"
    assert not hasattr(context, "current_task")
    assert not hasattr(context, "task_recovery_projection")

    with pytest.raises(TypeError):
        AgentContext(system_prompt="sys", messages=[], current_task="## Current Task")
