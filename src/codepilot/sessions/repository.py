from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .contracts import (
    MessageRecord,
    RunState,
    SessionState,
    SessionStateConflictError,
    validate_run_transition,
)
from .filesystem import (
    SessionFileLayout,
    append_jsonl,
    atomic_write_json,
    read_json_object,
    read_jsonl,
)
from .serde import (
    message_record_from_dict,
    message_record_to_dict,
    run_state_from_dict,
    run_state_to_dict,
    session_state_from_dict,
    session_state_to_dict,
)


class FileSessionRepository:
    """Typed filesystem access for Sessions v2 state."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.layout = SessionFileLayout(self.workspace_dir)

    def create_session(self, state: SessionState) -> SessionState:
        path = self.layout.session_file(state.session_id)
        if path.exists():
            raise SessionStateConflictError(f"Session already exists: {state.session_id}")
        atomic_write_json(path, session_state_to_dict(state))
        self.layout.messages_file(state.session_id).touch(exist_ok=True)
        self.layout.session_events_file(state.session_id).touch(exist_ok=True)
        return state

    def load_session(self, session_id: str) -> SessionState | None:
        payload = read_json_object(self.layout.session_file(session_id))
        return session_state_from_dict(payload) if payload is not None else None

    def list_sessions(self) -> tuple[SessionState, ...]:
        sessions_dir = self.layout.codepilot_dir / "sessions"
        if not sessions_dir.is_dir():
            return ()
        sessions = [
            session
            for entry in sessions_dir.iterdir()
            if entry.is_dir()
            for session in [self.load_session(entry.name)]
            if session is not None
        ]
        return tuple(sorted(sessions, key=lambda item: item.updated_at, reverse=True))

    def delete_session(self, session_id: str) -> bool:
        if not session_id or Path(session_id).name != session_id:
            raise ValueError("Invalid session_id")
        sessions_dir = (self.layout.codepilot_dir / "sessions").resolve()
        target = self.layout.session_dir(session_id).resolve()
        if target.parent != sessions_dir:
            raise ValueError("Session path escapes the sessions directory")
        if not target.is_dir():
            return False
        shutil.rmtree(target)
        return True

    def update_session(
        self,
        state: SessionState,
        *,
        expected_revision: int,
    ) -> SessionState:
        current = self.load_session(state.session_id)
        if current is None:
            raise FileNotFoundError(f"Session not found: {state.session_id}")
        if current.revision != expected_revision:
            raise SessionStateConflictError(
                f"Session revision conflict: expected {expected_revision}, got {current.revision}"
            )
        if state.revision != expected_revision + 1:
            raise SessionStateConflictError("Session revision must increase by exactly one")
        atomic_write_json(self.layout.session_file(state.session_id), session_state_to_dict(state))
        return state

    def create_run(self, state: RunState) -> RunState:
        path = self.layout.run_file(state.run_id)
        if path.exists():
            raise SessionStateConflictError(f"Run already exists: {state.run_id}")
        atomic_write_json(path, run_state_to_dict(state))
        self.layout.run_events_file(state.run_id).touch(exist_ok=True)
        return state

    def load_run(self, run_id: str) -> RunState | None:
        payload = read_json_object(self.layout.run_file(run_id))
        return run_state_from_dict(payload) if payload is not None else None

    def update_run(
        self,
        state: RunState,
        *,
        expected_revision: int,
    ) -> RunState:
        current = self.load_run(state.run_id)
        if current is None:
            raise FileNotFoundError(f"Run not found: {state.run_id}")
        if current.revision != expected_revision:
            raise SessionStateConflictError(
                f"Run revision conflict: expected {expected_revision}, got {current.revision}"
            )
        validate_run_transition(current, state)
        atomic_write_json(self.layout.run_file(state.run_id), run_state_to_dict(state))
        return state

    def append_message(self, record: MessageRecord) -> MessageRecord:
        path = self.layout.messages_file(record.session_id)
        existing = {item.message_id: item for item in self.load_message_records(record.session_id)}
        current = existing.get(record.message_id)
        if current is not None:
            if message_record_to_dict(current) == message_record_to_dict(record):
                return current
            raise SessionStateConflictError(f"Message id already exists: {record.message_id}")
        append_jsonl(path, message_record_to_dict(record))
        return record

    def load_message_records(self, session_id: str) -> list[MessageRecord]:
        return [
            message_record_from_dict(row)
            for row in read_jsonl(self.layout.messages_file(session_id))
        ]

    def load_message_chain(
        self,
        session_id: str,
        *,
        leaf_id: str | None = None,
    ) -> tuple[MessageRecord, ...]:
        records = self.load_message_records(session_id)
        if not records:
            return ()
        by_id = {record.message_id: record for record in records}
        current = leaf_id
        if current is None:
            session = self.load_session(session_id)
            current = session.leaf_message_id if session is not None else None
        if current is None:
            return ()
        if current not in by_id:
            raise ValueError(f"Message leaf not found: {current}")
        chain: list[MessageRecord] = []
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise ValueError(f"Message chain contains a cycle at: {current}")
            seen.add(current)
            record = by_id.get(current)
            if record is None:
                raise ValueError(f"Message parent not found: {current}")
            chain.append(record)
            current = record.parent_id
        chain.reverse()
        return tuple(chain)

    def append_event(self, event: dict[str, Any]) -> None:
        legacy_keys = {"eventId", "sessionId", "runId"}.intersection(event)
        if legacy_keys:
            raise ValueError(
                "Event uses legacy field names: " + ", ".join(sorted(legacy_keys))
            )
        _event_text(event, "event_id")
        session_id = _event_text(event, "session_id")
        append_jsonl(self.layout.session_events_file(session_id), event)
        run_id = event.get("run_id")
        if isinstance(run_id, str) and run_id:
            append_jsonl(self.layout.run_events_file(run_id), event)

    def load_events(
        self,
        session_id: str,
        *,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> tuple[dict[str, Any], ...]:
        path = (
            self.layout.run_events_file(run_id)
            if run_id is not None
            else self.layout.session_events_file(session_id)
        )
        events = read_jsonl(path)
        if run_id is not None:
            events = [event for event in events if event.get("session_id") == session_id]
        return tuple(events[-limit:] if limit is not None else events)

    def write_result_artifact(self, run_id: str, payload: dict[str, Any]) -> str:
        path = self.layout.run_artifacts_dir(run_id) / "result.json"
        atomic_write_json(path, payload)
        return "artifacts/result.json"


def _event_text(event: dict[str, Any], key: str) -> str:
    value = event.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event {key} is required")
    try:
        json.dumps(event, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Event must be JSON serializable") from exc
    return value.strip()


__all__ = ["FileSessionRepository"]
