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
    from codepilot.sessions.persistence.store import SessionStore

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


def test_memory_writer_keeps_transient_task_file_and_failure_out_of_memory(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.persistence.store import SessionStore
    from codepilot.tools.sandbox import file_state_for_path

    target = tmp_path / "service.py"
    target.write_text("value = 1\n", encoding="utf-8", newline="\n")
    session_store = SessionStore(tmp_path, "session_writer")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    task = writer.remember_task("Fix service.py API_KEY=secret-value", run_id="run_1")
    state_v1 = file_state_for_path(tmp_path, "service.py")
    read_records = writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="read_1",
            tool_name="read",
            content=[TextContent(text="value = 1")],
            details={"file_state": state_v1},
        ),
        run_id="run_1",
    )

    assert task is None
    assert read_records == []
    assert store.load_session() == []

    target.write_text("value = 2\n", encoding="utf-8", newline="\n")
    edit_records = writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="edit_1",
            tool_name="edit",
            content=[TextContent(text="edited")],
            affected_paths=["service.py"],
            workspace_changed=True,
        ),
        run_id="run_1",
    )
    assert edit_records == []

    state_v2 = file_state_for_path(tmp_path, "service.py")
    read_again_records = writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="read_2",
            tool_name="read",
            content=[TextContent(text="value = 2")],
            details={"file_state": state_v2},
        ),
        run_id="run_1",
    )
    assert read_again_records == []

    error = ToolResultMessage(
        tool_call_id="edit_error",
        tool_name="edit",
        content=[TextContent(text="old_text was not unique")],
        status="error",
        is_error=True,
        error_code="multiple_matches",
        affected_paths=["service.py"],
    )
    assert writer.observe_tool_result(error, run_id="run_1") == []
    assert writer.observe_tool_result(error, run_id="run_1") == []
    assert store.load_session() == []


def test_memory_writer_promotes_project_constraint_from_user_correction(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_project_constraint")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    writer.remember_task("这个项目是学生学习和求职展示项目，不要做生产级复杂设计")

    project_records = store.load_project()
    assert any(
        record.kind == "project"
        and record.content.get("category") == "project_constraint"
        and "清晰" in record.content.get("knowledge", "")
        and "生产级" in record.content.get("knowledge", "")
        for record in project_records
    )


def test_task_recovery_store_projects_unfinished_task_summary(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.persistence.store import SessionStore
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

    session_store = SessionStore(tmp_path, "session_task_projection")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    recovery = TaskRecoveryStore(session_store)
    recovery.begin_task("修复失败测试", run_id="run_1")

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

    projection = recovery.update_from_result(result)

    assert projection["next_action"] == "报告连续失败并等待用户指示"
    assert projection["task_progress"] == {
        "completed_steps": ["定位失败"],
        "pending_steps": ["重新运行相关验证"],
        "blocked_steps": ["根据最新失败证据调整方案"],
        "completion_satisfied": False,
        "completion_reason": "replan_limit_exceeded",
        "step_details": {},
    }
    assert recovery.load_projection() == projection


def test_task_recovery_store_projects_task_step_details(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.persistence.store import SessionStore
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

    session_store = SessionStore(tmp_path, "session_task_step_details")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    recovery = TaskRecoveryStore(session_store)
    recovery.begin_task("实现 planner", run_id="run_plan")

    projection = recovery.update_from_result(
        AgentRunResult(
            run_id="run_plan",
            session_id="session_task_step_details",
            status="completed",
            stop_reason="final_answer",
            task=TaskSummary(
                task_id="task_plan",
                goal="实现 planner",
                completed_steps=["定位任务模块"],
                pending_steps=["修改执行逻辑"],
                completion_satisfied=False,
                completion_reason="incomplete_steps",
                step_details={
                    "定位任务模块": {
                        "kind": "investigate",
                        "acceptance": "找到 TaskController",
                        "verification_hint": None,
                    },
                    "修改执行逻辑": {
                        "kind": "edit",
                        "acceptance": "按 step 推进",
                        "verification_hint": "pytest task",
                    },
                },
            ),
        )
    )

    progress = projection["task_progress"]
    assert progress["step_details"]["定位任务模块"]["kind"] == "investigate"
    assert progress["step_details"]["修改执行逻辑"]["verification_hint"] == "pytest task"


def test_memory_writer_extracts_verified_experience_from_edit_repair_loop(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary, TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter, render_memory
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_experience")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    result = AgentRunResult(
        run_id="run_1",
        session_id="session_experience",
        status="completed",
        stop_reason="final_answer",
        messages=[
            ToolResultMessage(
                tool_call_id="edit_bad",
                tool_name="edit",
                status="error",
                is_error=True,
                error_code="multiple_matches",
                content=[TextContent(text="Multiple matches found")],
                affected_paths=["src/app.py"],
                metadata={
                    "recovery_hint": {
                        "category": "refine_edit",
                        "suggested_action_intent": "edit_file",
                    }
                },
            ),
            ToolResultMessage(
                tool_call_id="edit_good",
                tool_name="edit",
                status="success",
                content=[TextContent(text="Edited file")],
                affected_paths=["src/app.py"],
                workspace_changed=True,
            ),
            ToolResultMessage(
                tool_call_id="test_good",
                tool_name="bash",
                status="success",
                verification={
                    "status": "passed",
                    "command": "python -m pytest test/test_app.py -q",
                    "exit_code": 0,
                    "summary": "passed",
                },
            ),
        ],
        task=TaskSummary(
            task_id="task_1",
            goal="修复 edit 失败",
            completed_steps=["修复并验证"],
            completion_satisfied=True,
            completion_reason="all_steps_completed",
        ),
    )

    writer.finalize_run(result)
    experiences = [
        record for record in store.load_session() if record.kind == "experience"
    ]

    assert len(experiences) == 1
    experience = experiences[0]
    assert experience.content["failure_signal"] == "multiple_matches"
    assert experience.content["maturity"] == "verified"
    assert "intent:edit_file" in experience.content["applies_when"]
    assert "先 read" in experience.content["better_action"]
    assert "Experience[verified]" in render_memory(experience)

    writer.finalize_run(result)
    experiences = [
        record for record in store.load_session() if record.kind == "experience"
    ]
    assert len(experiences) == 1
    assert experiences[0].content["occurrence_count"] == 2


def test_memory_writer_does_not_extract_experience_from_out_of_order_evidence(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import AgentRunResult, TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_experience_order")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    writer.finalize_run(
        AgentRunResult(
            run_id="run_order",
            session_id="session_experience_order",
            status="completed",
            stop_reason="final_answer",
            messages=[
                ToolResultMessage(
                    tool_call_id="test_old_pass",
                    tool_name="bash",
                    status="success",
                    verification={
                        "status": "passed",
                        "command": "python -m pytest test/test_app.py -q",
                        "exit_code": 0,
                    },
                ),
                ToolResultMessage(
                    tool_call_id="edit_bad",
                    tool_name="edit",
                    status="error",
                    is_error=True,
                    error_code="multiple_matches",
                    content=[TextContent(text="Multiple matches found")],
                    affected_paths=["src/app.py"],
                ),
                ToolResultMessage(
                    tool_call_id="edit_good",
                    tool_name="edit",
                    status="success",
                    content=[TextContent(text="Edited unrelated file")],
                    affected_paths=["src/app.py"],
                    workspace_changed=True,
                ),
                ToolResultMessage(
                    tool_call_id="test_new_fail",
                    tool_name="bash",
                    status="error",
                    is_error=True,
                    verification={
                        "status": "failed",
                        "command": "python -m pytest test/test_app.py -q",
                        "exit_code": 1,
                    },
                ),
            ],
        )
    )

    assert [
        record for record in store.load_session() if record.kind == "experience"
    ] == []


def test_memory_writer_does_not_store_single_tool_failure(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_failure_resolution")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    assert writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="bash_bad",
            tool_name="bash",
            status="error",
            is_error=True,
            error_code="dangerous_command",
            content=[TextContent(text="rm -rf is blocked")],
        ),
        run_id="run_failure",
    ) == []
    assert writer.observe_tool_result(
        ToolResultMessage(
            tool_call_id="bash_good",
            tool_name="bash",
            status="success",
            content=[TextContent(text="pytest passed")],
        ),
        run_id="run_failure",
    ) == []

    assert store.load_session() == []


def test_memory_writer_does_not_infer_project_constraint_from_production_request(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_project_constraint_negative")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    assert writer.remember_task("这个项目现在要按生产级复杂设计推进", run_id="run_prod") is None

    assert not any(
        record.kind == "project"
        and record.content.get("category") == "project_constraint"
        for record in store.load_project()
    )


def test_memory_sanitizer_redacts_authorization_bearer() -> None:
    from codepilot.sessions.memory import sanitize_memory_text

    sanitized = sanitize_memory_text(
        "Authorization: Bearer secret-token-value",
        limit=200,
    )

    assert "secret-token-value" not in sanitized
    assert "[REDACTED]" in sanitized


def test_memory_retriever_uses_phase_intent_and_error_without_hard_filter(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import (
        MemoryQuery,
        MemoryRecord,
        MemoryRetriever,
        MemoryStore,
    )
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_experience_retrieval")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    store.update(
        MemoryRecord(
            id="exp_verified",
            kind="experience",
            scope="session",
            content={
                "lesson_type": "tool_usage",
                "situation": "edit old_text 不唯一",
                "failed_attempt": "直接短文本 edit",
                "failure_signal": "multiple_matches",
                "better_action": "先 read 目标区域，再使用唯一 old_text",
                "applies_when": ["phase:repair", "intent:edit_file", "error:multiple_matches"],
                "avoid_when": [],
                "evidence_refs": ["tool:edit_bad", "tool:edit_good", "verification:test_good"],
                "maturity": "verified",
                "occurrence_count": 1,
                "fingerprint": "fp1",
            },
            source="experience_extractor",
            trust="verified",
        )
    )
    store.update(
        MemoryRecord(
            id="exp_candidate",
            kind="experience",
            scope="session",
            content={
                "lesson_type": "tool_usage",
                "situation": "candidate",
                "better_action": "unverified",
                "applies_when": ["intent:edit_file"],
                "maturity": "candidate",
            },
            source="experience_extractor",
            trust="observed",
        )
    )
    retriever = MemoryRetriever(store=store, workspace_dir=tmp_path)

    with_signal = retriever.retrieve(
        MemoryQuery(
            text="",
            active_paths=[],
            task_phase="repair",
            action_intent="edit_file",
            recent_error="multiple_matches",
        )
    )
    assert [item.record.id for item in with_signal] == ["exp_verified"]
    assert "phase:repair" in with_signal[0].reasons
    assert "intent:edit_file" in with_signal[0].reasons
    assert "error:multiple_matches" in with_signal[0].reasons

    without_intent = retriever.retrieve(
        MemoryQuery(text="old_text", active_paths=[])
    )
    assert [item.record.id for item in without_intent] == ["exp_verified"]


def test_memory_retriever_uses_qa_mode_to_surface_decision_memory(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import (
        MemoryQuery,
        MemoryRecord,
        MemoryRetriever,
        MemoryStore,
    )
    from codepilot.sessions.persistence.store import SessionStore

    session_store = SessionStore(tmp_path, "session_qa_retrieval")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    store = MemoryStore(session_store)
    store.update(
        MemoryRecord(
            id="decision_context",
            kind="decision",
            scope="session",
            content={"decision": "上下文模块采用轻量预算治理，而不是向量数据库"},
            source="user",
            trust="user_given",
        )
    )
    retriever = MemoryRetriever(store=store, workspace_dir=tmp_path)

    retrieved = retriever.retrieve(
        MemoryQuery(text="", active_paths=[], retrieval_mode="qa")
    )

    assert [item.record.id for item in retrieved] == ["decision_context"]
    assert "mode:qa_decision_memory" in retrieved[0].reasons


def test_task_recovery_clears_next_action_only_when_completion_is_satisfied(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.persistence.store import SessionStore
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

    session_store = SessionStore(tmp_path, "session_task_done")
    session_store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    recovery = TaskRecoveryStore(session_store)
    recovery.begin_task("完成修复", run_id="run_done")

    projection = recovery.update_from_result(
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

    assert projection["next_action"] is None
    assert projection["task_progress"]["completion_satisfied"] is True


def test_memory_retriever_excludes_legacy_state_and_explains_selection(tmp_path: Path) -> None:
    from codepilot.sessions.memory import (
        MemoryQuery,
        MemoryRecord,
        MemoryRetriever,
        MemoryStore,
    )
    from codepilot.sessions.persistence.store import SessionStore

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
    store.update(
        MemoryRecord(
            id="constraint",
            kind="project",
            scope="project",
            content={"category": "project_constraint", "knowledge": "保持学习项目边界"},
            source="user",
            trust="user_given",
        )
    )

    retrieved = MemoryRetriever(store=store, workspace_dir=tmp_path).retrieve(
        MemoryQuery(
            text="continue RuntimeService refactor",
            active_paths=["runtime/service.py"],
        )
    )

    assert [item.record.id for item in retrieved] == ["constraint"]
    assert "project_memory" in retrieved[0].reasons


def test_context_compiler_reads_pinned_memory_dynamically(tmp_path: Path) -> None:
    asyncio.run(_dynamic_pinned_memory_case(tmp_path))


async def _dynamic_pinned_memory_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context.compiler import ContextCompiler
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory import MemoryRetriever, MemoryStore
    from codepilot.sessions.persistence.store import SessionStore

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


def test_fork_copies_task_recovery_then_evolves_independently(tmp_path: Path) -> None:
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

    original = AgentSession(
        AgentSessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            system_prompt="rules",
        )
    )
    forked = None
    try:
        original_recovery = TaskRecoveryStore(original.store)
        original_recovery.begin_task("original goal", run_id="run_original")
        forked = original.fork_session()
        forked_recovery = TaskRecoveryStore(forked.store)
        assert forked_recovery.load_projection()["goal"] == "original goal"

        forked_recovery.begin_task("fork goal", run_id="run_fork")

        assert original_recovery.load_projection()["goal"] == "original goal"
        assert forked_recovery.load_projection()["goal"] == "fork goal"
    finally:
        if forked is not None:
            forked.close()
        original.close()


def test_agent_session_keeps_run_context_out_of_durable_memory(tmp_path: Path) -> None:
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
        assert records == []
        assert any(
            event.get("type") == "task_recovery_updated"
            for event in session.store.load_events()
        )
    finally:
        session.close()


def test_agent_session_restores_task_progress_from_task_recovery(tmp_path: Path) -> None:
    asyncio.run(_session_task_recovery_case(tmp_path))


async def _session_task_recovery_case(tmp_path: Path) -> None:
    from codepilot.llm.event_stream import AssistantMessageEventStream
    from codepilot.protocols import AssistantMessage, TextContent
    from codepilot.sessions.session import AgentSession
    from codepilot.sessions.types import AgentSessionOptions
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

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
        TaskRecoveryStore(session.store).save_projection(
            {
                "goal": "继续修复失败测试",
                "task_progress": {
                    "completed_steps": ["定位失败"],
                    "pending_steps": ["重新运行相关验证"],
                    "blocked_steps": [],
                    "completion_satisfied": False,
                    "completion_reason": "replan_limit_exceeded",
                    "step_details": {},
                },
                "next_action": "报告连续失败并等待用户指示",
            },
            run_id="old_run",
        )

        result = await session.run("继续修复失败测试", run_id="recovered_run")

        assert result.task is not None
        assert result.task.completed_steps == ["定位失败", "重新运行相关验证"]
        assert result.task.pending_steps == []
    finally:
        session.close()
