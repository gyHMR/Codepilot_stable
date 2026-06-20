from __future__ import annotations

import asyncio
from pathlib import Path


def _model():
    from codepilot.protocols import Model

    return Model(
        id="memory-test",
        name="Memory Test",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def test_memory_store_persists_session_and_project_records(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryRecord, MemoryStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_memory")
    session_store.ensure_initialized(
        model_id="test",
        provider="test",
        system_prompt="",
    )
    store = MemoryStore(session_store)
    session_record = MemoryRecord(
        id="mem_session",
        kind="task",
        scope="session",
        content={"goal": "implement memory"},
        source="test",
        trust="user_given",
    )
    project_record = MemoryRecord(
        id="mem_project",
        kind="project",
        scope="project",
        content={"knowledge": "run pytest"},
        source="test",
        trust="verified",
    )

    store.update(session_record)
    store.update(project_record)
    project_record.status = "deleted"
    store.update(project_record)

    assert store.load_session()[0].content["goal"] == "implement memory"
    assert store.load_project()[0].status == "deleted"
    payload = session_store.memory_file.read_text(encoding="utf-8")
    assert '"schema_version": 1' in payload


def test_memory_writer_tracks_task_file_versions_and_failures(tmp_path: Path) -> None:
    from codepilot.protocols import TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.store import SessionStore
    from codepilot.tools.sandbox import file_state_for_path

    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8", newline="\n")
    session_store = SessionStore(tmp_path, "session_writer")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    task = writer.remember_task("Fix service.py API_KEY=secret-value", run_id="run_1")
    state_v1 = file_state_for_path(tmp_path, "service.py")
    first = writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="read_1",
            tool_name="read",
            content=[TextContent(text="value = 1")],
            details={"file_state": state_v1},
        ),
        run_id="run_1",
    )[0]

    assert "[REDACTED]" in task.content["goal"]
    assert first.source_hashes["service.py"] == state_v1["sha256"]

    target.write_text("value = 2\n", encoding="utf-8", newline="\n")
    writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="edit_1",
            tool_name="edit",
            content=[TextContent(text="edited")],
            affected_paths=["service.py"],
            workspace_changed=True,
        ),
        run_id="run_1",
    )
    assert store.get(first.id).status == "stale"

    state_v2 = file_state_for_path(tmp_path, "service.py")
    updated = writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="read_2",
            tool_name="read",
            content=[TextContent(text="value = 2")],
            details={"file_state": state_v2},
        ),
        run_id="run_1",
    )[0]
    assert updated.status == "active"
    assert updated.source_hashes["service.py"] == state_v2["sha256"]

    error = ToolResultMessage(
        tool_call_id="edit_error",
        tool_name="edit",
        content=[TextContent(text="old_text was not unique")],
        status="error",
        is_error=True,
        error_code="multiple_matches",
        affected_paths=["service.py"],
    )
    lesson = writer.observe_tool_result(error, run_id="run_1")[0]
    lesson = writer.observe_tool_result(error, run_id="run_1")[0]
    assert lesson.content["occurrence_count"] == 2


def test_memory_writer_projects_unfinished_task_summary(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.memory import MemoryStore, MemoryWriter, render_memory
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_task_projection")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)
    writer.remember_task("修复失败测试", run_id="run_1")

    result = AgentRunResult(
        run_id="run_1",
        session_id="session_task_projection",
        status="failed",
        stop_reason="replan_limit",
        task=TaskSummary(
            task_id="task_1",
            goal="修复失败测试",
            completed_steps=["定位失败"],
            pending_steps=["重新运行相关验证"],
            blocked_steps=["根据最新失败证据调整方案"],
            next_action="报告连续失败并等待用户指示",
            completion_satisfied=False,
            completion_reason="replan_limit_exceeded",
        ),
    )

    [record] = writer.finalize_run(result)

    assert record.content["next_action"] == "报告连续失败并等待用户指示"
    assert "Completed step: 定位失败" in record.content["confirmed_findings"]
    assert "Blocked step: 根据最新失败证据调整方案" in record.content["blocked_on"]
    assert record.content["task_progress"] == {
        "completed_steps": ["定位失败"],
        "pending_steps": ["重新运行相关验证"],
        "blocked_steps": ["根据最新失败证据调整方案"],
        "completion_satisfied": False,
        "completion_reason": "replan_limit_exceeded",
    }
    rendered = render_memory(record)
    assert "Next: 报告连续失败并等待用户指示" in rendered
    assert "Pending: 重新运行相关验证" in rendered


def test_memory_writer_clears_next_action_only_when_completion_is_satisfied(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_task_done")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)
    writer.remember_task("完成修复", run_id="run_done")

    [record] = writer.finalize_run(
        AgentRunResult(
            run_id="run_done",
            session_id="session_task_done",
            status="completed",
            stop_reason="final_answer",
            task=TaskSummary(
                task_id="task_done",
                goal="完成修复",
                completed_steps=["修改实现", "运行验证"],
                next_action="不应保留",
                completion_satisfied=True,
                completion_reason="all_steps_completed",
            ),
        )
    )

    assert record.content["next_action"] is None
    assert record.content["task_progress"]["completion_satisfied"] is True


def test_memory_retriever_excludes_stale_and_explains_selection(tmp_path: Path) -> None:
    from codepilot.sessions.memory import (
        MemoryQuery,
        MemoryRecord,
        MemoryRetriever,
        MemoryStore,
    )
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_retriever")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    store.update(
        MemoryRecord(
            id="task",
            kind="task",
            scope="session",
            content={"goal": "refactor RuntimeService"},
            source="user",
            trust="user_given",
        )
    )
    store.update(
        MemoryRecord(
            id="file",
            kind="file",
            scope="session",
            content={"path": "runtime/service.py", "summary": "runtime facade"},
            source="read",
            related_paths=["runtime/service.py"],
            source_hashes={"runtime/service.py": "old"},
            trust="observed",
            status="stale",
        )
    )

    retrieved = MemoryRetriever(store=store, workspace_dir=tmp_path).retrieve(
        MemoryQuery(
            text="continue RuntimeService refactor",
            active_paths=["runtime/service.py"],
        )
    )

    assert [item.record.id for item in retrieved] == ["task"]
    assert "task_memory" in retrieved[0].reasons


def test_context_compiler_reads_pinned_memory_dynamically(tmp_path: Path) -> None:
    asyncio.run(_dynamic_pinned_memory_case(tmp_path))


async def _dynamic_pinned_memory_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.runtime.context_compiler import ContextCompiler
    from codepilot.sessions.context_state import SessionContextState
    from codepilot.sessions.memory import MemoryRetriever, MemoryStore
    from codepilot.sessions.store import SessionStore

    session_store = SessionStore(tmp_path, "session_context_memory")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    retriever = MemoryRetriever(
        store=MemoryStore(session_store),
        workspace_dir=tmp_path,
    )
    compiler = ContextCompiler(
        workspace=str(tmp_path),
        state=SessionContextState(workspace_dir=tmp_path),
        memory_retriever=retriever,
    )
    memory_file = tmp_path / ".codepilot" / "MEMORY.md"
    memory_file.write_text("Always run focused tests first.", encoding="utf-8", newline="\n")

    prepared = await compiler.compile(
        AgentContext(system_prompt="rules", messages=[UserMessage(content="test")]),
        ContextPreparationRequest(
            session_id="session_context_memory",
            model_context_window=4000,
            model_max_output_tokens=500,
        ),
    )

    assert "Always run focused tests first." in prepared.system_prompt
    memory_section = next(
        section for section in prepared.report.sections if section.name == "memory"
    )
    assert memory_section.selected_items == 1


def test_fork_copies_session_memory_then_evolves_independently(tmp_path: Path) -> None:
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions

    original = AgentSession(
        AgentSessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            system_prompt="rules",
        )
    )
    forked = None
    try:
        original_record = original.memory_writer.remember_task("original goal")
        forked = original.fork_session()
        assert forked.memory_store.get(original_record.id) is not None

        forked.memory_writer.remember_task("fork goal")

        assert original.memory_store.get(original_record.id).content["goal"] == "original goal"
        assert forked.memory_store.get(original_record.id).content["goal"] == "fork goal"
    finally:
        if forked is not None:
            forked.close()
        original.close()


def test_agent_session_writes_task_and_file_memory_from_run(tmp_path: Path) -> None:
    asyncio.run(_session_memory_run_case(tmp_path))


async def _session_memory_run_case(tmp_path: Path) -> None:
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent, ToolCall, ToolResultMessage
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions
    from codepilot.tools import AgentTool, AgentToolResult, ToolMetadata, ToolRegistry, ToolRuntime
    from codepilot.tools.sandbox import file_state_for_path

    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8", newline="\n")

    async def fake_stream(_model, context, _options):
        stream = AssistantMessageEventStream()
        if any(isinstance(message, ToolResultMessage) for message in context.messages):
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
        else:
            stream.end(
                AssistantMessage(
                    content=[
                        ToolCall(
                            id="read_1",
                            name="read",
                            arguments={"path": "service.py"},
                        )
                    ],
                    stop_reason="toolUse",
                )
            )
        return stream

    async def read_tool(*_args):
        return AgentToolResult(
            content=[TextContent(text="value = 1")],
            details={"file_state": file_state_for_path(tmp_path, "service.py")},
        )

    registry = ToolRegistry()
    registry.register(
        AgentTool(
            name="read",
            label="read",
            description="read",
            parameters={},
            execute=read_tool,
        ),
        metadata=ToolMetadata(
            name="read",
            category="filesystem",
            read_only=True,
            concurrency_safe=True,
            exclusive=False,
            requires_approval=False,
            risk_level="low",
            resource_scope=("workspace",),
        ),
    )
    session = AgentSession(
        AgentSessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            system_prompt="rules",
            tools=ToolRuntime(registry).as_agent_tools(),
            stream_fn=fake_stream,
        )
    )
    try:
        await session.run("inspect service.py", run_id="run_memory")
        records = session.memory_store.load_session()
        assert any(
            record.kind == "task"
            and record.content["goal"] == "inspect service.py"
            for record in records
        )
        assert any(
            record.kind == "file"
            and record.related_paths == ["service.py"]
            for record in records
        )
        assert any(
            event.get("type") == "memory_updated"
            for event in session.store.load_events()
        )
    finally:
        session.close()


def test_agent_session_restores_task_progress_from_memory(tmp_path: Path) -> None:
    asyncio.run(_session_task_recovery_case(tmp_path))


async def _session_task_recovery_case(tmp_path: Path) -> None:
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions

    async def fake_stream(_model, context, _options):
        system_prompt = context.system_prompt or ""
        assert "定位失败" in system_prompt
        assert "Pending: 重新运行相关验证" in system_prompt or "重新运行相关验证" in system_prompt
        assert "Next action: 报告连续失败并等待用户指示" in system_prompt
        stream = AssistantMessageEventStream()
        stream.end(AssistantMessage(content=[TextContent(text="继续处理")]))
        return stream

    session = AgentSession(
        AgentSessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            system_prompt="rules",
            stream_fn=fake_stream,
        )
    )
    try:
        record = session.memory_writer.remember_task("继续修复失败测试", run_id="old_run")
        record.content["task_progress"] = {
            "completed_steps": ["定位失败"],
            "pending_steps": ["重新运行相关验证"],
            "blocked_steps": [],
            "completion_satisfied": False,
            "completion_reason": "replan_limit_exceeded",
        }
        record.content["next_action"] = "报告连续失败并等待用户指示"
        session.memory_store.update(record)

        result = await session.run("继续修复失败测试", run_id="recovered_run")

        assert result.task is not None
        assert result.task.completed_steps == ["定位失败", "重新运行相关验证"]
        assert result.task.pending_steps == []
    finally:
        session.close()
