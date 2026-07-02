from __future__ import annotations

import asyncio
from pathlib import Path

import pytest


def _session_store(tmp_path: Path, session_id: str = "session_memory"):
    from codepilot.sessions.persistence.store import SessionStore

    store = SessionStore(tmp_path, session_id)
    store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    return store


def _memory_record(
    memory_id: str,
    *,
    scope: str = "session",
    kind: str = "experience",
    key: str | None = None,
    text: str | None = None,
    triggers: list[str] | None = None,
    status: str = "active",
):
    from codepilot.sessions.memory import MemoryRecord

    return MemoryRecord(
        id=memory_id,
        scope=scope,
        kind=kind,
        status=status,
        key=key or f"{kind}:test:{memory_id}",
        text=text or f"{kind} memory {memory_id}",
        triggers=triggers or [],
        related_paths=["src/app.py"] if kind == "experience" else [],
        evidence_refs=["tool:verify"] if kind == "experience" else [],
        source="run" if kind == "experience" else "user",
    )


def _verified_edit_repair_result(run_id: str):
    from codepilot.protocols import AgentRunResult, TextContent, ToolResultMessage

    return AgentRunResult(
        run_id=run_id,
        session_id="session_experience",
        status="completed",
        stop_reason="final_answer",
        messages=[
            ToolResultMessage(
                tool_call_id=f"{run_id}_bad",
                tool_name="edit",
                status="error",
                is_error=True,
                error_code="multiple_matches",
                content=[TextContent(text="old_text was not unique")],
                affected_paths=["src/app.py"],
            ),
            ToolResultMessage(
                tool_call_id=f"{run_id}_good",
                tool_name="edit",
                status="success",
                content=[TextContent(text="Edited file")],
                affected_paths=["src/app.py"],
                workspace_changed=True,
            ),
            ToolResultMessage(
                tool_call_id=f"{run_id}_verify",
                tool_name="shell",
                status="success",
                content=[TextContent(text="passed")],
                verification={"status": "passed", "command": "python -m pytest -q"},
            ),
        ],
    )


def test_memory_store_persists_session_and_project_records(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore

    store = MemoryStore(_session_store(tmp_path))
    session_record = _memory_record(
        "mem_session",
        scope="session",
        kind="experience",
        key="experience:edit:multiple_matches",
        text="Use a unique old_text when edit reports multiple matches.",
    )
    project_record = _memory_record(
        "mem_project",
        scope="project",
        kind="constraint",
        key="constraint:project_boundary",
        text="Keep Codepilot as a student learning and portfolio project.",
        triggers=["always"],
    )

    store.update(session_record)
    store.update(project_record)

    assert store.load_session() == [session_record]
    assert [record.id for record in store.load_project()] == ["mem_project"]
    assert store.load_project()[0].to_dict()["schema_version"] == 2


def test_memory_store_prunes_session_records_to_capacity(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore

    store = MemoryStore(_session_store(tmp_path), max_session_records=2)

    store.save_session(
        [
            _memory_record("old", kind="decision", text="old", triggers=[]),
            _memory_record("always", kind="constraint", text="keep", triggers=["always"]),
            _memory_record("new", kind="experience", text="new", triggers=[]),
        ]
    )

    records = store.load_session()
    ids = {record.id for record in records}
    assert len(records) == 2
    assert "always" in ids


def test_memory_store_compacts_project_log_to_capacity(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore

    store = MemoryStore(
        _session_store(tmp_path),
        max_project_records=2,
        project_compact_after_lines=3,
    )

    for index in range(4):
        store.update(
            _memory_record(
                f"project_{index}",
                scope="project",
                kind="constraint",
                key=f"constraint:{index}",
                text=f"knowledge {index}",
            )
        )

    records = store.load_project()
    assert len(records) == 2
    assert {record.id for record in records} == {"project_2", "project_3"}


def test_memory_record_v2_schema_rejects_legacy_values() -> None:
    from codepilot.sessions.memory import MemoryRecord

    with pytest.raises(ValueError, match="Unknown memory kind"):
        _memory_record("bad", kind="task")
    with pytest.raises(ValueError, match="Unknown memory status"):
        _memory_record("bad_status", status="stale")
    with pytest.raises(ValueError, match="Unsupported memory schema_version"):
        MemoryRecord.from_dict(
            {
                "schema_version": 1,
                "id": "legacy",
                "scope": "session",
                "kind": "task",
                "key": "task:legacy",
                "text": "legacy task progress",
                "source": "user",
            }
        )


def test_memory_writer_admits_explicit_remember_prompt(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    record = writer.admit_prompt_memory(
        "请记住：通知修复 checkpoint 是 notify-close-42。",
        run_id="run_explicit",
    )

    assert record is not None
    assert record.scope == "project"
    assert record.kind == "constraint"
    assert record.source == "user"
    assert "always" in record.triggers
    assert "notify-close-42" in record.text


def test_memory_writer_records_user_correction_and_supersedes_conflict(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path))
    old = _memory_record(
        "old_constraint",
        scope="project",
        kind="constraint",
        key="constraint:context_design",
        text="Use the old context design.",
    )
    store.update(old)

    writer = MemoryWriter(store=store, workspace_dir=tmp_path)
    record = writer.admit_prompt_memory(
        "纠正一下：上下文治理必须按 ContextGovernor v2 方案来。",
        run_id="run_correction",
    )

    assert record is not None
    assert record.kind == "correction"
    assert record.key == "constraint:context_design"
    assert store.get(old.id).status == "superseded"


def test_memory_writer_does_not_store_ordinary_or_production_prompt(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    assert writer.admit_prompt_memory("修复 service.py 并运行测试") is None
    assert writer.admit_prompt_memory("这个项目现在要按生产级复杂设计推进") is None
    assert store.load_project() == []


def test_prompt_memory_admission_explains_durable_boundary() -> None:
    from codepilot.sessions.memory import decide_prompt_memory_admission

    ordinary = decide_prompt_memory_admission("修复 service.py 并运行测试")
    explicit = decide_prompt_memory_admission("以后调用测试都使用 python -m pytest")
    correction = decide_prompt_memory_admission("不是旧压缩方案，而是 ContextGovernor v2")

    assert ordinary.should_store is False
    assert ordinary.reason == "ordinary_task_prompt"
    assert explicit.should_store is True
    assert explicit.kind == "constraint"
    assert "always" in explicit.triggers
    assert correction.should_store is True
    assert correction.kind == "correction"


def test_memory_writer_extracts_verified_experience_from_edit_repair_loop(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter, render_memory

    store = MemoryStore(_session_store(tmp_path, "session_experience"))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    records = writer.finalize_run(_verified_edit_repair_result("run_1"))

    assert len(records) == 1
    experience = records[0]
    assert experience.scope == "session"
    assert experience.kind == "experience"
    assert experience.key == "experience:edit:multiple_matches"
    assert "error:multiple_matches" in experience.triggers
    assert "path:src/app.py" in experience.triggers
    assert "tool:run_1_bad" in experience.evidence_refs
    assert "old_text is not unique" in render_memory(experience).lower()


def test_memory_writer_merges_and_promotes_repeated_verified_experience(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path, "session_experience"))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    writer.finalize_run(_verified_edit_repair_result("run_1"))
    second = writer.finalize_run(_verified_edit_repair_result("run_2"))[0]

    assert second.occurrences == 2
    assert [record.kind for record in store.load_project()] == ["experience"]
    assert store.load_project()[0].source == "promoted"


def test_memory_writer_does_not_extract_experience_without_verified_loop(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import AgentRunResult, TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    store = MemoryStore(_session_store(tmp_path, "session_no_experience"))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)
    result = AgentRunResult(
        run_id="run_fail",
        session_id="session_no_experience",
        status="failed",
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
            )
        ],
    )

    assert writer.finalize_run(result) == []
    assert store.load_session() == []


def test_memory_sanitizer_redacts_authorization_bearer() -> None:
    from codepilot.sessions.memory import sanitize_memory_text

    sanitized = sanitize_memory_text(
        "Authorization: Bearer secret-token-value",
        limit=200,
    )

    assert "secret-token-value" not in sanitized
    assert "[REDACTED]" in sanitized


def test_memory_retriever_orders_layers_and_excludes_inactive(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRetriever, MemoryStore

    store = MemoryStore(_session_store(tmp_path))
    (tmp_path / ".codepilot").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".codepilot" / "MEMORY.md").write_text(
        "Pinned rule: use UTF-8.",
        encoding="utf-8",
        newline="\n",
    )
    store.update(
        _memory_record(
            "constraint",
            scope="project",
            kind="constraint",
            key="constraint:project_boundary",
            text="Keep the learning-project boundary.",
            triggers=["always"],
        )
    )
    store.update(
        _memory_record(
            "decision",
            scope="project",
            kind="decision",
            key="decision:context",
            text="Context design uses layered projection.",
            triggers=["topic:context"],
        )
    )
    store.update(
        _memory_record(
            "experience",
            scope="session",
            kind="experience",
            key="experience:edit:multiple_matches",
            text="Read the target file before retrying edit after multiple matches.",
            triggers=["intent:edit_file", "error:multiple_matches"],
        )
    )
    store.update(
        _memory_record(
            "deleted",
            scope="project",
            kind="constraint",
            status="deleted",
            text="Do not recall this.",
        )
    )

    recall = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(
            text="context edit repair",
            active_paths=["src/app.py"],
            retrieval_mode="repair",
            action_intent="edit_file",
            recent_error="multiple_matches",
        )
    )

    assert recall.pinned_text == "Pinned rule: use UTF-8."
    assert [item.record.id for item in recall.always] == ["constraint"]
    assert [item.record.id for item in recall.selected] == ["decision", "experience"]
    assert recall.dropped["deleted"] == "status:deleted"


def test_memory_score_explains_single_record_relevance() -> None:
    from codepilot.sessions.memory import MemoryQuery
    from codepilot.sessions.memory.retriever import score_memory_record

    record = _memory_record(
        "exp_verify",
        kind="experience",
        key="experience:verification:failed_then_passed",
        text="When pytest verification fails, reproduce the failure before editing.",
        triggers=[
            "phase:acting",
            "intent:debug_failure",
            "error:verification_failed",
        ],
    )

    retrieved = score_memory_record(
        record,
        MemoryQuery(
            text="pytest verification failed",
            active_paths=[],
            task_phase="acting",
            action_intent="debug_failure",
            recent_error="verification_failed",
            retrieval_mode="verify",
        ),
    )

    assert retrieved is not None
    assert "phase:acting" in retrieved.reasons
    assert "intent:debug_failure" in retrieved.reasons
    assert "error:verification_failed" in retrieved.reasons
    assert "mode:verify_experience" in retrieved.reasons


def test_context_governor_reads_pinned_memory_dynamically(tmp_path: Path) -> None:
    asyncio.run(_dynamic_pinned_memory_case(tmp_path))


async def _dynamic_pinned_memory_case(tmp_path: Path) -> None:
    from codepilot.core import AgentContext, ContextPreparationRequest
    from codepilot.protocols import UserMessage
    from codepilot.sessions.context.governor import ContextGovernor
    from codepilot.sessions.context.state import SessionContextState
    from codepilot.sessions.memory import MemoryRetriever, MemoryStore

    session_store = _session_store(tmp_path, "session_context_memory")
    retriever = MemoryRetriever(
        store=MemoryStore(session_store),
        workspace_dir=tmp_path,
    )
    governor = ContextGovernor(
        workspace_dir=tmp_path,
        session_id="session_context_memory",
        state=SessionContextState(workspace_dir=tmp_path),
        memory_retriever=retriever,
    )
    memory_file = tmp_path / ".codepilot" / "MEMORY.md"
    memory_file.write_text(
        "Always run focused tests first.",
        encoding="utf-8",
        newline="\n",
    )

    prepared = await governor.prepare(
        AgentContext(system_prompt="rules", messages=[UserMessage(content="test")]),
        ContextPreparationRequest(
            session_id="session_context_memory",
            model_context_window=4000,
            model_max_output_tokens=500,
        ),
    )

    assert "Always run focused tests first." in prepared.system_prompt
    assert prepared.report.context_view is not None
    assert prepared.report.context_view.recalled_memory == [
        "[Pinned memory] Always run focused tests first."
    ]
    assert prepared.report.retrieved_memory_ids == []


def test_task_recovery_store_projects_unfinished_task_summary(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

    recovery = TaskRecoveryStore(_session_store(tmp_path, "session_task_recovery"))
    recovery.begin_task("修复上下文治理", run_id="run_1")

    projection = recovery.update_from_result(
        AgentRunResult(
            run_id="run_1",
            session_id="session_task_recovery",
            status="completed",
            stop_reason="final_answer",
            task=TaskSummary(
                task_id="task_1",
                goal="修复上下文治理",
                completed_steps=["阅读代码"],
                pending_steps=["补测试"],
                next_action="继续补测试",
                completion_satisfied=False,
                completion_reason="needs_follow_up",
            ),
        )
    )

    assert projection["goal"] == "修复上下文治理"
    assert projection["task_progress"]["completed_steps"] == ["阅读代码"]
    assert projection["next_action"] == "继续补测试"
    assert recovery.active_projection() is not None


def test_task_recovery_clears_next_action_when_completion_is_satisfied(
    tmp_path: Path,
) -> None:
    from codepilot.protocols import AgentRunResult, TaskSummary
    from codepilot.sessions.history.task_recovery import TaskRecoveryStore

    recovery = TaskRecoveryStore(_session_store(tmp_path, "session_task_done"))
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
    assert recovery.active_projection() is None
