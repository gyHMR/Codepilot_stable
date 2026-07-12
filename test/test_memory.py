from __future__ import annotations

from pathlib import Path

import pytest


def _session_store(tmp_path: Path, session_id: str = "session_memory"):
    class EventStore:
        def __init__(self) -> None:
            self.events = []

        def append_event(self, event):
            self.events.append(event)

        def load_events(self):
            return list(self.events)

    return EventStore()


def _write_context(session_id: str = "session_memory", run_id: str = "run_1"):
    from codepilot.sessions.memory import MemoryWriteContext

    return MemoryWriteContext(
        session_id=session_id,
        run_id=run_id,
        source_event_id="event_memory",
        evidence_refs=["event:event_memory", f"run:{run_id}"],
    )


def _record(memory_id: str, **overrides):
    from codepilot.sessions.memory import MemoryRecord

    payload = {
        "id": memory_id,
        "type": "constraint",
        "scope": "project",
        "subject": "test_command",
        "predicate": "is",
        "value": "python -m pytest -q",
        "content": "Use python -m pytest -q for verification.",
        "keywords": ["pytest", "intent:verify"],
        "paths": ["test/"],
        "status": "active",
        "source": "user_explicit",
        "confidence": "explicit",
        "priority": 4,
        "created_by_session_id": "session_memory",
        "created_by_run_id": "run_1",
        "source_message_id": None,
        "source_event_id": "event_1",
        "evidence_refs": ["event:event_1"],
        "supersedes": [],
        "superseded_by": None,
        "occurrences": 1,
    }
    payload.update(overrides)
    return MemoryRecord(**payload)


def test_memory_store_uses_single_workspace_jsonl(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore

    from codepilot.sessions.memory import MemoryRepository

    store = MemoryStore(MemoryRepository(tmp_path))
    store.append(_record("mem_1"))

    assert (tmp_path / ".codepilot" / "memory" / "memories.jsonl").is_file()
    assert store.all_records()[0].to_dict()["schema_version"] == 4
    assert not (tmp_path / ".codepilot" / "sessions" / "session_memory" / "memory.json").exists()


def test_memory_record_v4_rejects_old_schema_and_fields() -> None:
    from codepilot.sessions.memory import MemoryRecord, validate_memory_record_payload

    with pytest.raises(ValueError, match="Unsupported memory schema_version"):
        validate_memory_record_payload({"schema_version": 3, "id": "legacy"})
    with pytest.raises(ValueError, match="legacy memory fields"):
        MemoryRecord.from_dict({**_record("mem_1").to_dict(), "kind": "constraint"})
    with pytest.raises(TypeError):
        MemoryRecord(
            id="legacy",
            kind="constraint",  # type: ignore[call-arg]
            scope="project",
            subject="legacy",
            predicate="is",
            value="legacy",
            content="legacy",
            source="user_explicit",
            created_by_session_id="session_memory",
        )


def test_memory_writer_admits_explicit_and_candidate_correction(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    from codepilot.sessions.memory import MemoryRepository

    store = MemoryStore(MemoryRepository(tmp_path))
    writer = MemoryWriter(store=store, workspace_dir=tmp_path)

    explicit = writer.admit_prompt_memory(
        "请记住：本项目默认使用 python -m pytest -q。",
        context=_write_context(),
    )
    correction = writer.admit_prompt_memory(
        "纠正：测试命令是 uv run pytest -q。",
        context=_write_context(run_id="run_2"),
    )

    assert explicit is not None
    explicit_record, explicit_decision = explicit
    assert explicit_decision.reason == "user_memory_requested"
    assert explicit_record.status == "active"
    assert explicit_record.source == "user_explicit"
    assert correction is not None
    correction_record, correction_decision = correction
    assert correction_decision.reason == "user_correction_observed"
    assert correction_record.status == "candidate"
    assert store.get(correction_record.id) is not None


def test_memory_writer_rejects_active_without_source_context(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryStore, MemoryWriteContext, MemoryWriter

    from codepilot.sessions.memory import MemoryRepository

    writer = MemoryWriter(store=MemoryStore(MemoryRepository(tmp_path)), workspace_dir=tmp_path)

    with pytest.raises(ValueError, match="source context|requires evidence"):
        writer.admit_prompt_memory(
            "请记住：默认运行 pytest。",
            context=MemoryWriteContext(),
        )


def test_verified_experience_becomes_candidate_only(tmp_path: Path) -> None:
    from codepilot.protocols import AgentRunResult, TextContent, ToolResultMessage
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    result = AgentRunResult(
        run_id="run_1",
        session_id="session_memory",
        status="completed",
        stop_reason="final_answer",
        messages=[
            ToolResultMessage(
                tool_call_id="edit_bad",
                tool_name="edit",
                status="error",
                is_error=True,
                error_code="multiple_matches",
                content=[TextContent(text="multiple matches")],
                affected_paths=["src/app.py"],
            ),
            ToolResultMessage(
                tool_call_id="edit_ok",
                tool_name="edit",
                status="success",
                content=[TextContent(text="ok")],
                affected_paths=["src/app.py"],
            ),
            ToolResultMessage(
                tool_call_id="pytest_ok",
                tool_name="shell",
                status="success",
                content=[TextContent(text="passed")],
                verification={"status": "passed"},
            ),
        ],
    )
    from codepilot.sessions.memory import MemoryRepository

    writer = MemoryWriter(store=MemoryStore(MemoryRepository(tmp_path)), workspace_dir=tmp_path)

    records = writer.finalize_run(result, context=_write_context())

    assert [record.status for record in records] == ["candidate"]
    assert records[0].type == "experience"


def test_memory_retriever_drops_candidates_conflicts_and_dedupes_subject(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRetriever, MemoryStore

    from codepilot.sessions.memory import MemoryRepository

    store = MemoryStore(MemoryRepository(tmp_path))
    store.append(_record("candidate", status="candidate"))
    store.append(_record("low", subject="package_manager", value="pnpm", content="Use pnpm.", priority=1))
    store.append(_record("high", subject="package_manager", value="pnpm", content="Use pnpm.", priority=5))
    recall = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(
            latest_user_message="这次不要用 pnpm，改成 npm。",
            raw_user_request="install dependencies",
            goal="install dependencies",
            active_paths=[],
        )
    )

    assert recall.retrieved == []
    assert recall.dropped["candidate"] == "status:candidate"
    assert recall.dropped["low"] == "conflict:latest_instruction"
    assert recall.dropped["high"] == "conflict:latest_instruction"


def test_memory_commands_edit_supersedes_old_record(tmp_path: Path) -> None:
    from codepilot.runtime.commands import edit_memory
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    class Session:
        session_id = "session_memory"
        workspace_dir = tmp_path

        def __init__(self) -> None:
            self.event_store = _session_store(tmp_path, self.session_id)
            from codepilot.sessions.memory import MemoryRepository

            self.memory_store = MemoryStore(MemoryRepository(tmp_path))
            self.memory_writer = MemoryWriter(store=self.memory_store, workspace_dir=tmp_path)

        def append_event(self, event):
            self.event_store.append_event(event)

    session = Session()
    session.memory_store.append(_record("mem_old"))

    new_id = edit_memory(session, "mem_old", "Use uv run pytest -q.")

    old = session.memory_store.get("mem_old")
    new = session.memory_store.get(new_id)
    assert old is not None and old.status == "superseded"
    assert old.superseded_by == new_id
    assert new is not None and new.supersedes == ["mem_old"]
    assert session.event_store.load_events()[-1]["type"] == "memory_record_edited"
