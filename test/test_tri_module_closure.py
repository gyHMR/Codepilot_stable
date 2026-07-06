from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def test_task_modes_use_build_instead_of_edit() -> None:
    from codepilot.core.task import ensure_task_mode, policy_for_mode

    assert ensure_task_mode("build") == "build"
    assert policy_for_mode("build").mode == "build"

    with pytest.raises(ValueError, match="Unknown task mode"):
        ensure_task_mode("edit")


def test_session_store_persists_task_state_in_dedicated_file(tmp_path: Path) -> None:
    from codepilot.sessions.storage import SessionStore

    store = SessionStore(tmp_path, "session_task")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")

    task_state = {
        "schema_version": 1,
        "task_id": "task_1",
        "raw_user_request": "先规划再执行",
        "current_mode": "plan",
        "approval_state": "proposed",
        "goal": {"value": "重构任务规划", "source": "planner", "confidence": "inferred"},
        "proposed_plan": {"steps": [{"id": "step_1", "title": "阅读现状"}]},
        "approved_plan": None,
        "current_step_id": None,
        "steps": [],
        "verification_status": "unknown",
        "evidence_refs": [],
        "blocked_reason": None,
        "recovery_summary": "",
        "task_mode": "plan",
        "task_progress": {"completion_satisfied": False},
    }

    store.save_task_state(task_state)
    expected = dict(task_state)
    expected.pop("task_mode")
    expected.pop("task_progress")

    session_dir = tmp_path / ".codepilot" / "sessions" / "session_task"
    assert (session_dir / "task_state.json").exists()
    assert store.load_task_state() == expected
    assert "task_recovery" not in (store.read_meta() or {})

    forked = store.fork_to("session_fork")
    assert forked.load_task_state() == expected


def test_repeated_build_verification_failure_blocks_without_auto_replan() -> None:
    from codepilot.core import TaskController
    from codepilot.core.state import RunState
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="修复失败测试")],
        mode="build",
        proposed_steps=["修改实现", "运行验证"],
        max_replans_per_run=3,
    )
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
    first = controller.after_tool_results(task, run, [failed])
    run.collect_tool_results([failed])
    second = controller.after_tool_results(task, run, [failed])

    assert first.action == "repair"
    assert second.action == "stop"
    assert second.reason == "revision_needed"
    assert task.replan_count == 0
    assert task.current_step() is not None
    assert task.current_step().status == "blocked"


def test_task_update_requires_current_step_and_real_evidence() -> None:
    from codepilot.core import TaskController
    from codepilot.core.state import RunState
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="按步骤执行")],
        mode="build",
        proposed_steps=["阅读代码", "修改实现"],
    )
    run = RunState(run_id="run_1", session_id="session_1")
    rejected = ToolResultMessage(
        tool_call_id="task_update_1",
        tool_name="task_update",
        status="success",
        metadata={
            "task_control": {
                "action": "update_step",
                "step_id": "step_1",
                "proposed_status": "completed",
                "summary": "已阅读代码",
                "evidence_refs": [],
            }
        },
    )

    decision = controller.after_tool_results(task, run, [rejected])

    assert decision.action == "continue"
    assert task.current_step_id == "step_1"
    assert task.current_step() is not None
    assert task.current_step().status == "in_progress"
    assert task.current_step().note == "task_update rejected: missing_evidence"


def test_passed_verification_records_tool_evidence_on_completed_step() -> None:
    from codepilot.core import TaskController
    from codepilot.core.state import RunState
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="运行验证后完成")],
        mode="build",
        proposed_steps=["运行验证"],
    )
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

    decision = controller.after_tool_results(task, run, [passed])

    assert decision.action == "finish"
    assert task.steps[0].status == "completed"
    assert "tool:test_1" in task.steps[0].evidence_refs
    assert "verification:test_1" in task.steps[0].evidence_refs


def test_legacy_complete_task_step_requires_real_evidence_like_task_update() -> None:
    from codepilot.core import TaskController
    from codepilot.core.state import RunState
    from codepilot.protocols import ToolResultMessage, UserMessage

    controller = TaskController()
    task = controller.initialize(
        [UserMessage(content="按步骤执行")],
        mode="build",
        proposed_steps=["阅读代码", "修改实现"],
    )
    run = RunState(run_id="run_1", session_id="session_1")
    rejected = ToolResultMessage(
        tool_call_id="complete_1",
        tool_name="complete_task_step",
        status="success",
        metadata={
            "task_control": {
                "action": "complete_step",
                "summary": "声称已经阅读代码",
                "evidence_refs": [],
            }
        },
    )

    decision = controller.after_tool_results(task, run, [rejected])

    assert decision.action == "continue"
    assert task.current_step_id == "step_1"
    assert task.current_step() is not None
    assert task.current_step().status == "in_progress"
    assert task.current_step().note == "task_update rejected: missing_evidence"


def test_task_recovery_writes_authoritative_task_state_shape(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore
    from codepilot.sessions.storage import SessionStore

    store = SessionStore(tmp_path, "session_task_state")
    store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    recovery = TaskRecoveryStore(store)
    recovery.begin_task("修复任务控制链路", run_id="run_1")

    projection = recovery.update_from_result(
        AgentRunResult(
            run_id="run_1",
            session_id="session_task_state",
            status="completed",
            stop_reason="final_answer",
            task=TaskSummary(
                task_id="task_1",
                goal="修复任务控制链路",
                completed_steps=["阅读任务控制代码"],
                pending_steps=["补充恢复测试"],
                next_action="补充恢复测试",
                completion_satisfied=False,
                completion_reason="incomplete_steps",
                control_signal={
                    "mode": "build",
                    "current_step_id": "step_2",
                    "phase": "acting",
                    "recent_error_code": "verification_failed",
                },
                step_details={
                    "补充恢复测试": {
                        "kind": "verify",
                        "acceptance": "恢复测试覆盖 canonical task_state",
                        "verification_hint": "python -m pytest test/test_tri_module_closure.py -q",
                    }
                },
            ),
        )
    )
    stored = store.load_task_state()

    assert projection is not None
    assert stored is not None
    assert stored["schema_version"] == 1
    assert stored["task_id"] == "task_1"
    assert stored["raw_user_request"] == "修复任务控制链路"
    assert stored["current_mode"] == "build"
    assert stored["approval_state"] == "none"
    assert stored["current_step_id"] == "step_2"
    assert stored["verification_status"] == "failed"
    assert stored["blocked_reason"] == "verification_failed"
    assert "task_progress" not in stored
    assert "task_mode" not in stored
    assert stored["steps"][1]["status"] == "in_progress"
    assert stored["steps"][1]["kind"] == "verify"


def test_task_recovery_projection_defaults_removed_edit_mode_to_build() -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.history.task_recovery import build_task_recovery_projection

    projection = build_task_recovery_projection(
        AgentRunResult(
            run_id="run_1",
            session_id="session_1",
            status="completed",
            stop_reason="final_answer",
            task=TaskSummary(
                task_id="task_1",
                goal="继续任务",
                pending_steps=["补测试"],
            ),
        ),
        current_projection={},
    )

    assert projection is not None
    assert "task_mode" not in projection
    assert "task_progress" not in projection
    assert projection["current_mode"] == "build"


def test_runtime_context_port_passes_task_signal_to_context_governor() -> None:
    from codepilot.core.contracts import PreparedAgentContext
    from codepilot.protocols import ContextReport, ContextView
    from codepilot.sessions.prepare import RuntimeSessionContextPort

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

        def _active_task_recovery_projection(self):
            return None

        async def prepare_context(self, context, request):
            captured["task_signal"] = context.task_signal
            captured["current_task"] = context.current_task
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
                        stable_rules=[],
                        task_state=[],
                        working_set=[],
                        recalled_memory=[],
                        conversation=[],
                        tools=[],
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
                    "current_task": "Goal: fix tests",
                    "task_control_signal": {
                        "phase": "acting",
                        "action_intent": "debug_failure",
                        "recent_error_code": "verification_failed",
                    },
                },
            }
        )
    )

    assert captured["current_task"] == "Goal: fix tests"
    assert captured["task_signal"]["action_intent"] == "debug_failure"


def test_structured_memory_record_supports_candidate_and_conflict_fields() -> None:
    from codepilot.sessions.memory import MemoryRecord

    record = MemoryRecord(
        id="mem_1",
        type="project_convention",
        scope="project:Codepilot",
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
    assert record.to_dict()["schema_version"] == 3


def test_memory_writer_uses_single_log_and_candidates_for_ordinary_corrections(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.storage import SessionStore

    session_store = SessionStore(tmp_path, "session_memory")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    candidate = writer.admit_prompt_memory(
        "纠正：测试命令是 python -m pytest -q",
        run_id="run_1",
    )
    active = writer.admit_prompt_memory(
        "以后默认测试命令是 python -m pytest -q",
        run_id="run_2",
    )

    assert candidate is not None
    assert candidate.status == "candidate"
    assert candidate.source == "user_correction"
    assert active is not None
    assert active.status == "active"
    assert active.created_by_session_id == "session_memory"
    assert active.created_by_run_id == "run_2"
    assert (tmp_path / ".codepilot" / "memory" / "memories.jsonl").exists()
    assert not (tmp_path / ".codepilot" / "sessions" / "session_memory" / "memory.json").exists()


def test_memory_recall_filters_conflicts_and_dedupes_subjects(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRecord, MemoryRetriever, MemoryStore
    from codepilot.sessions.storage import SessionStore

    session_store = SessionStore(tmp_path, "session_memory")
    session_store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
    store = MemoryStore(session_store)
    store.update(
        MemoryRecord(
            id="mem_low",
            type="project_convention",
            scope="project:Codepilot",
            subject="test_command",
            predicate="is",
            value="pytest",
            content="Use pytest.",
            keywords=["pytest"],
            status="active",
            source="user_explicit",
            confidence="explicit",
            priority=1,
        )
    )
    store.update(
        MemoryRecord(
            id="mem_high",
            type="project_convention",
            scope="project:Codepilot",
            subject="test_command",
            predicate="is",
            value="python -m pytest -q",
            content="Use python -m pytest -q.",
            keywords=["pytest"],
            status="active",
            source="user_explicit",
            confidence="explicit",
            priority=5,
        )
    )

    recall = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(text="pytest test command", active_paths=[], limit=5)
    )
    assert [item.record.id for item in recall.retrieved] == ["mem_high"]

    conflicted = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(text="纠正 test_command 不要用 pytest", active_paths=[], limit=5)
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
    from codepilot.sessions.memory import MemoryRecord, MemoryStore
    from codepilot.sessions.storage import SessionStore

    class Session:
        session_id = "session_memory"
        workspace_dir = tmp_path

        def __init__(self) -> None:
            self.store = SessionStore(tmp_path, self.session_id)
            self.store.ensure_initialized(model_id="m", provider="p", system_prompt="sys")
            self.memory_store = MemoryStore(self.store)

    session = Session()
    session.memory_store.update(
        MemoryRecord(
            id="mem_1",
            type="project_convention",
            scope="project:Codepilot",
            subject="test_command",
            predicate="is",
            value="pytest",
            content="Use pytest.",
            keywords=["pytest"],
            status="candidate",
            source="user_correction",
            confidence="explicit",
            priority=1,
        )
    )

    assert approve_memory(session, "mem_1") == "mem_1"
    assert session.memory_store.get("mem_1").status == "active"
    assert edit_memory(session, "mem_1", "Use python -m pytest -q.") == "mem_1"
    assert "python -m pytest" in search_memory_records(session, "python pytest")[0]["text"]
    assert disable_memory(session, "mem_1") == "mem_1"
    assert session.memory_store.get("mem_1").status == "disabled"
    assert delete_memory(session, "mem_1") == "mem_1"
    assert session.memory_store.get("mem_1").status == "deleted"

    replacement = supersede_memory(session, "mem_1", "Use uv run pytest -q.")
    assert session.memory_store.get("mem_1").status == "superseded"
    assert session.memory_store.get(replacement).supersedes == ["mem_1"]
