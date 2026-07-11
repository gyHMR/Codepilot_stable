from __future__ import annotations

import json
from pathlib import Path

import pytest


def _session_store(tmp_path: Path, session_id: str = "session_memory_v4"):
    from codepilot.sessions.store import SessionStore

    store = SessionStore(tmp_path, session_id)
    store.ensure_initialized(model_id="test", provider="test", system_prompt="")
    return store


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
        "created_by_session_id": "session_memory_v4",
        "created_by_run_id": "run_1",
        "source_message_id": "msg_1",
        "source_event_id": "event_1",
        "evidence_refs": ["message:msg_1"],
        "supersedes": [],
        "superseded_by": None,
        "occurrences": 1,
    }
    payload.update(overrides)
    return MemoryRecord(**payload)


def test_memory_record_v4_rejects_legacy_schema_and_fields() -> None:
    from codepilot.sessions.memory import MemoryRecord

    record = _record("mem_v4")

    assert record.to_dict()["schema_version"] == 4
    assert record.source_message_id == "msg_1"
    assert record.superseded_by is None

    with pytest.raises(ValueError, match="Unsupported memory schema_version"):
        MemoryRecord.from_dict({"schema_version": 3, "id": "legacy"})

    with pytest.raises(ValueError, match="legacy memory fields"):
        MemoryRecord.from_dict(
            {
                **record.to_dict(),
                "kind": "constraint",
            }
        )

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
        )


def test_memory_store_uses_single_workspace_jsonl_and_rejects_legacy_rows(
    tmp_path: Path,
) -> None:
    from codepilot.sessions.memory import MemoryStore

    session_store = _session_store(tmp_path)
    store = MemoryStore(session_store)
    record = _record("mem_store")

    store.update(record)

    memory_file = tmp_path / ".codepilot" / "memory" / "memories.jsonl"
    assert memory_file.is_file()
    assert not (tmp_path / ".codepilot" / "sessions" / session_store.session_id / "memory.json").exists()
    assert store.all_records()[0].to_dict()["schema_version"] == 4

    memory_file.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "id": "legacy",
                "kind": "constraint",
                "key": "legacy",
                "text": "legacy",
                "triggers": ["always"],
                "related_paths": ["src/app.py"],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="Unsupported memory schema_version"):
        store.all_records()


def test_candidate_and_conflicting_memory_are_dropped_from_recall(tmp_path: Path) -> None:
    from codepilot.sessions.memory import MemoryQuery, MemoryRetriever, MemoryStore

    store = MemoryStore(_session_store(tmp_path))
    store.update(_record("candidate", status="candidate", subject="candidate_rule"))
    store.update(_record("active", subject="package_manager", value="pnpm", content="Use pnpm."))

    recall = MemoryRetriever(store=store, workspace_dir=tmp_path).recall(
        MemoryQuery(
            latest_user_message="这次不要用 pnpm，改成 npm。",
            raw_user_request="run install",
            goal="run install",
            current_mode="build",
            active_paths=[],
            changed_paths=[],
        )
    )

    assert recall.retrieved == []
    assert recall.dropped["candidate"] == "status:candidate"
    assert recall.dropped["active"] == "conflict:latest_instruction"


def test_memory_edit_supersedes_old_record_and_writes_session_event(tmp_path: Path) -> None:
    from codepilot.sessions.commands import edit_memory
    from codepilot.sessions.memory import MemoryStore, MemoryWriter

    class Session:
        session_id = "session_memory_v4"
        workspace_dir = tmp_path

        def __init__(self) -> None:
            self.store = _session_store(tmp_path, self.session_id)
            self.memory_store = MemoryStore(self.store)
            self.memory_writer = MemoryWriter(
                store=self.memory_store,
                workspace_dir=tmp_path,
            )

    session = Session()
    session.memory_store.update(_record("mem_old"))

    new_id = edit_memory(session, "mem_old", "Use uv run pytest -q.")

    old = session.memory_store.get("mem_old")
    new = session.memory_store.get(new_id)
    assert old is not None and old.status == "superseded"
    assert old.superseded_by == new_id
    assert new is not None and new.status == "active"
    assert new.supersedes == ["mem_old"]
    events = session.store.load_events()
    assert events[-1]["type"] == "memory_record_edited"
