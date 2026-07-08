from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_run_modes_use_build_instead_of_edit() -> None:
    from codepilot.core.plan import ensure_run_mode

    assert ensure_run_mode("build") == "build"
    assert ensure_run_mode("plan") == "plan"

    with pytest.raises(ValueError, match="Unknown run mode"):
        ensure_run_mode("edit")


def test_session_store_persists_plan_state_in_dedicated_file(tmp_path: Path) -> None:
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, "session_plan")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")

    plan_state = {
        "schema_version": 1,
        "plan_id": "plan_1",
        "status": "proposed",
        "approval_state": "proposed",
        "origin_mode": "plan",
        "objective": "先规划再执行",
        "items": [
            {"id": "item_1", "step": "阅读实现", "status": "in_progress"},
            {"id": "item_2", "step": "给出方案", "status": "pending"},
        ],
        "explanation": "准备方案",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "last_update_run_id": "run_1",
    }

    store.save_plan_state(plan_state)

    session_dir = tmp_path / ".codepilot" / "sessions" / "session_plan"
    assert (session_dir / "plan_state.json").exists()
    assert not (session_dir / "task_state.json").exists()
    assert store.load_plan_state() == plan_state

    forked = store.fork_to("session_fork")
    assert forked.load_plan_state() == plan_state


def test_repeated_build_verification_failure_keeps_model_in_control() -> None:
    from codepilot.core.run_guard import RunGuard
    from codepilot.core.state import RunState
    from codepilot.protocols import AssistantMessage, TextContent, ToolResultMessage

    run = RunState(run_id="run_1", session_id="session_1")
    failed = ToolResultMessage(
        tool_call_id="test_1",
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

    run.collect_tool_results([failed])
    first = RunGuard().check(
        assistant=AssistantMessage(content=[TextContent(text="已完成")]),
        signals=run.summary(),
        mode="build",
    )
    run.collect_tool_results([failed])
    second = RunGuard().check(
        assistant=AssistantMessage(content=[TextContent(text="已完成")]),
        signals=run.summary(),
        mode="build",
    )

    assert first.action == "continue_with_instruction"
    assert first.reason == "verification_failed"
    assert second.action == "continue_with_instruction"
    assert second.reason == "verification_failed"
    assert run.summary().verification_status == "failed"


def test_update_plan_is_soft_progress_without_evidence_requirements() -> None:
    from codepilot.core.plan import PlanState, apply_plan_update_metadata

    state = PlanState.new(objective="按步骤执行", origin_mode="build", run_id="run_1")
    updated = apply_plan_update_metadata(
        state,
        {
            "plan_update": {
                "explanation": "阅读完成，继续修改",
                "plan": [
                    {"step": "阅读代码", "status": "completed"},
                    {"step": "修改实现", "status": "in_progress"},
                ],
            }
        },
        mode="build",
        objective="按步骤执行",
        run_id="run_1",
    )

    assert updated is not None
    assert updated.status == "active"
    assert [item.status for item in updated.items] == ["completed", "in_progress"]
    assert updated.explanation == "阅读完成，继续修改"


def test_passed_verification_is_run_signal_not_plan_completion_proof() -> None:
    from codepilot.core.state import RunState
    from codepilot.protocols import ToolResultMessage

    run = RunState(run_id="run_1", session_id="session_1")
    passed = ToolResultMessage(
        tool_call_id="test_1",
        tool_name="bash",
        status="success",
        verification={
            "status": "passed",
            "command": "python -m pytest test/test_tri_module_closure.py -q",
        },
    )

    run.collect_tool_results([passed])

    assert run.summary().verification_status == "passed"
    assert run.summary().affected_paths == []


def test_plan_state_store_begins_authoritative_plan_shape(tmp_path: Path) -> None:
    from codepilot.sessions.store import SessionStore
    from codepilot.sessions.plan_state import PlanStateStore

    store = SessionStore(tmp_path, "session_plan_state")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    state = PlanStateStore(store).begin("修复运行编排链路", run_id="run_1")
    stored = store.load_plan_state()

    assert stored is not None
    assert state == stored
    assert stored["schema_version"] == 1
    assert stored["objective"] == "修复运行编排链路"
    assert stored["origin_mode"] == "build"
    assert stored["status"] == "none"
    assert stored["approval_state"] == "none"
    assert stored["items"] == []
    assert stored["last_update_run_id"] == "run_1"


def test_runtime_context_port_passes_plan_and_run_signals_to_context_governor() -> None:
    from codepilot.core.contracts import PreparedAgentContext
    from codepilot.protocols import ContextReport, ContextView
    from codepilot.sessions.runtime import RuntimeSessionContextPort

    captured = {}

    class Model:
        context_window = 4000
        max_tokens = 500

    class Conversation:
        model = Model()

    class Store:
        def append_event(self, event):
            captured.setdefault("events", []).append(event)

    class Session:
        session_id = "session_1"
        conversation = Conversation()
        latest_context_report = None
        memory_enabled = False
        store = Store()

        def current_plan_state(self):
            return {"plan_id": "plan_1", "items": []}

        def context_plan_state(self):
            return self.current_plan_state()

        async def prepare_context(self, context, request):
            captured["run_signals"] = context.run_signals
            captured["plan_state"] = context.plan_state
            return PreparedAgentContext(
                system_prompt=context.system_prompt,
                messages=context.messages,
                tools=context.tools,
                report=ContextReport(
                    context_id="ctx_1",
                    repository_fingerprint="repo",
                    total_budget_tokens=100,
                    estimated_tokens_before=1,
                    estimated_tokens_after=1,
                    context_view=ContextView(
                        system=[],
                        task_plan=[],
                        working_set=[],
                        memory=[],
                        conversation=[],
                    ),
                ),
            )

    asyncio.run(
        RuntimeSessionContextPort(Session()).prepare(
            {
                "system_prompt": "sys",
                "messages": [],
                "tools": [],
                "context": {
                    "run_signals": {"verification_status": "failed"},
                },
            }
        )
    )

    assert captured["plan_state"] == {"plan_id": "plan_1", "items": []}
    assert captured["run_signals"]["verification_status"] == "failed"


def test_structured_memory_record_supports_candidate_and_conflict_fields() -> None:
    from codepilot.sessions.memory import MemoryRecord

    record = MemoryRecord(
        id="mem_1",
        type="constraint",
        scope="project",
        subject="test_command",
        predicate="is",
        value="pytest -q",
        content="本项目默认使用 pytest -q 运行测试。",
        keywords=["pytest", "测试"],
        paths=["test/"],
        status="candidate",
        source="user_correction",
        confidence="explicit",
        priority=3,
        created_by_session_id="session_1",
        created_by_run_id="run_1",
        evidence_refs=["tool:test_1"],
        supersedes=["mem_old"],
    )

    assert record.status == "candidate"
    assert record.subject == "test_command"
    assert record.to_dict()["schema_version"] == 4


def test_memory_writer_uses_single_log_and_candidates_for_ordinary_corrections(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriteContext, MemoryWriter
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_memory")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    candidate_result = writer.admit_prompt_memory(
        "纠正：测试命令是 python -m pytest -q",
        context=MemoryWriteContext(
            session_id="session_memory",
            run_id="run_1",
            source_event_id="event_1",
            evidence_refs=["event:event_1"],
        ),
    )
    active_result = writer.admit_prompt_memory(
        "以后默认测试命令是 python -m pytest -q",
        context=MemoryWriteContext(
            session_id="session_memory",
            run_id="run_2",
            source_event_id="event_2",
            evidence_refs=["event:event_2"],
        ),
    )

    assert candidate_result is not None
    candidate, _ = candidate_result
    assert candidate.status == "candidate"
    assert candidate.source == "user_correction"
    assert active_result is not None
    active, _ = active_result
    assert active.status == "active"
    assert active.created_by_session_id == "session_memory"
    assert active.created_by_run_id == "run_2"
    assert (tmp_path / ".codepilot" / "memory" / "memories.jsonl").exists()
    assert not (tmp_path / ".codepilot" / "sessions" / "session_memory" / "memory.json").exists()


def test_memory_recall_filters_conflicts_and_dedupes_subjects(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRecord, MemoryRetriever, MemoryStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_memory")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = MemoryStore(session_store)
    store.update(
        MemoryRecord(
            id="mem_low",
            type="constraint",
            scope="project",
            subject="test_command",
            predicate="is",
            value="pytest",
            content="Use pytest.",
            keywords=["pytest"],
            status="active",
            source="user_explicit",
            confidence="explicit",
            priority=1,
            created_by_session_id="session_memory",
            evidence_refs=["session:session_memory"],
        )
    )
    store.update(
        MemoryRecord(
            id="mem_high",
            type="constraint",
            scope="project",
            subject="test_command",
            predicate="is",
            value="python -m pytest -q",
            content="Use python -m pytest -q.",
            keywords=["pytest"],
            status="active",
            source="user_explicit",
            confidence="explicit",
            priority=5,
            created_by_session_id="session_memory",
            evidence_refs=["session:session_memory"],
        )
    )

    recall = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(latest_user_message="pytest test command", active_paths=[], limit=5)
    )
    assert [item.record.id for item in recall.retrieved] == ["mem_high"]

    conflicted = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(latest_user_message="纠正 test_command 不要用 pytest", active_paths=[], limit=5)
    )
    assert not conflicted.retrieved
    assert conflicted.dropped["mem_high"] == "conflict:latest_instruction"


def test_memory_management_commands_update_status_and_supersede(tmp_path: Path) -> None:
    from codepilot.sessions.commands import (
        approve_memory,
        delete_memory,
        disable_memory,
        edit_memory,
        search_memory_records,
        supersede_memory,
    )
    from codepilot.sessions.memory import MemoryRecord, MemoryStore, MemoryWriter
    from codepilot.sessions.store import SessionStore

    class Session:
        session_id = "session_memory"
        workspace_dir = tmp_path

        def __init__(self) -> None:
            self.store = SessionStore(tmp_path, self.session_id)
            self.store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
            self.memory_store = MemoryStore(self.store)
            self.memory_writer = MemoryWriter(store=self.memory_store, workspace_dir=tmp_path)

    session = Session()
    session.memory_store.update(
        MemoryRecord(
            id="mem_1",
            type="constraint",
            scope="project",
            subject="test_command",
            predicate="is",
            value="pytest",
            content="Use pytest.",
            keywords=["pytest"],
            status="candidate",
            source="user_correction",
            confidence="explicit",
            priority=1,
            created_by_session_id="session_memory",
            evidence_refs=["session:session_memory"],
        )
    )

    assert approve_memory(session, "mem_1") == "mem_1"
    assert session.memory_store.get("mem_1").status == "active"
    edited = edit_memory(session, "mem_1", "Use python -m pytest -q.")
    assert edited != "mem_1"
    assert "python -m pytest" in search_memory_records(session, "python pytest")[0]["text"]
    assert disable_memory(session, edited) == edited
    assert session.memory_store.get(edited).status == "disabled"
    assert delete_memory(session, edited) == edited
    assert session.memory_store.get(edited).status == "deleted"

    replacement = supersede_memory(session, edited, "Use uv run pytest -q.")
    assert session.memory_store.get(edited).status == "superseded"
    assert session.memory_store.get(replacement).supersedes == [edited]
