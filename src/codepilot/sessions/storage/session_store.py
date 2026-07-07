from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.observability import EventRecorder
from codepilot.protocols import AgentRunResult, Message

from .layout import SessionLayout
from .run_store import RunStore
from .serde import message_from_dict, message_to_dict


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


class SessionStore:
    """Store session metadata, transcript tree, task state, and session events."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.session_dir
        self.session_file = self.layout.session_file
        self.messages_file = self.layout.messages_file
        self.events_file = self.layout.session_events_file
        self.task_state_file = self.layout.task_state_file
        self.event_recorder = EventRecorder(self.events_file)
        self.run_store = RunStore(self.workspace_dir, self.session_id)

    def ensure_initialized(self, *, model_id: str, provider: str, system_prompt: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.session_file.exists():
            self._write_session_state(
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "parent_session_id": None,
                    "model_id": model_id,
                    "provider": provider,
                    "system_prompt": system_prompt,
                    "system_prompt_hash": _hash_text(system_prompt),
                    "leaf_id": None,
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                }
            )
        if not self.messages_file.exists():
            self.messages_file.write_text("", encoding="utf-8", newline="\n")

    def read_meta(self) -> dict[str, Any] | None:
        if not self.session_file.exists():
            return None
        data = json.loads(self.session_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    def update_meta(self, updates: dict[str, Any]) -> dict[str, Any]:
        state = self.read_meta() or {
            "schema_version": 1,
            "session_id": self.session_id,
            "created_at": _utc_now_iso(),
        }
        state.update(updates)
        state["session_id"] = self.session_id
        state["updated_at"] = _utc_now_iso()
        self._write_session_state(state)
        return state

    def touch_updated_at(self) -> None:
        state = self.read_meta()
        if state is not None:
            state["updated_at"] = _utc_now_iso()
            self._write_session_state(state)

    def append_message(self, message: Message) -> str:
        lines = self._read_message_lines()
        parent_id = (self.read_meta() or {}).get("leaf_id")
        entry_id = uuid.uuid4().hex[:8]
        lines.append(
            {
                "type": "message",
                "id": entry_id,
                "parent_id": parent_id if isinstance(parent_id, str) else None,
                "timestamp": _utc_now_iso(),
                "message": message_to_dict(message),
            }
        )
        self._write_message_lines(lines)
        self.update_meta({"leaf_id": entry_id})
        return entry_id

    def rewrite_session_messages(self, messages: list[Message]) -> None:
        entries: list[dict[str, Any]] = []
        parent_id: str | None = None
        for message in messages:
            entry_id = uuid.uuid4().hex[:8]
            entries.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parent_id": parent_id,
                    "timestamp": _utc_now_iso(),
                    "message": message_to_dict(message),
                }
            )
            parent_id = entry_id
        self._write_message_lines(entries)
        self.update_meta({"leaf_id": parent_id})

    def load_session_messages(self, *, leaf_id: str | None = None) -> list[Message]:
        entries = [line for line in self._read_message_lines() if line.get("type") == "message"]
        if not entries:
            return []
        by_id = {str(entry["id"]): entry for entry in entries if isinstance(entry.get("id"), str)}
        current = leaf_id or (self.read_meta() or {}).get("leaf_id")
        if not isinstance(current, str) or current not in by_id:
            current = str(entries[-1].get("id"))

        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            entry = by_id[current]
            chain.append(entry)
            parent_id = entry.get("parent_id")
            current = parent_id if isinstance(parent_id, str) else None
        chain.reverse()

        messages: list[Message] = []
        for entry in chain:
            payload = entry.get("message")
            if isinstance(payload, dict):
                messages.append(message_from_dict(payload))
        return messages

    def list_entry_ids(self) -> list[str]:
        return [
            str(line["id"])
            for line in self._read_message_lines()
            if line.get("type") == "message" and isinstance(line.get("id"), str)
        ]

    def list_entries(self) -> list[dict[str, Any]]:
        leaf_id = self.get_leaf_id()
        result: list[dict[str, Any]] = []
        for entry in self._read_message_lines():
            if entry.get("type") != "message" or not isinstance(entry.get("id"), str):
                continue
            entry_id = str(entry["id"])
            message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
            result.append(
                {
                    "id": entry_id,
                    "parent_id": entry.get("parent_id"),
                    "timestamp": entry.get("timestamp"),
                    "role": message.get("role", "unknown"),
                    "preview": _preview_message(message),
                    "depth": max(0, len(self.get_entry_path(entry_id)) - 1),
                    "is_leaf": entry_id == leaf_id,
                }
            )
        result.sort(key=lambda item: str(item.get("timestamp", "")))
        return result

    def get_leaf_id(self) -> str | None:
        value = (self.read_meta() or {}).get("leaf_id")
        return value if isinstance(value, str) else None

    def get_entry_path(self, entry_id: str) -> list[str]:
        by_id = {
            str(line["id"]): line
            for line in self._read_message_lines()
            if line.get("type") == "message" and isinstance(line.get("id"), str)
        }
        if entry_id not in by_id:
            raise ValueError(f"Entry not found: {entry_id}")
        path: list[str] = []
        current: str | None = entry_id
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            path.append(current)
            parent = by_id[current].get("parent_id")
            current = parent if isinstance(parent, str) else None
        path.reverse()
        return path

    def set_leaf(self, entry_id: str) -> None:
        if entry_id not in set(self.list_entry_ids()):
            raise ValueError(f"Entry not found: {entry_id}")
        self.update_meta({"leaf_id": entry_id})

    def get_session_tree(self) -> list[dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        for entry in self._read_message_lines():
            if entry.get("type") != "message" or not isinstance(entry.get("id"), str):
                continue
            entry_id = str(entry["id"])
            message = entry.get("message") if isinstance(entry.get("message"), dict) else {}
            nodes[entry_id] = {
                "id": entry_id,
                "parent_id": entry.get("parent_id"),
                "timestamp": entry.get("timestamp"),
                "role": message.get("role", "unknown"),
                "preview": _preview_message(message),
                "children": [],
            }
        for node in nodes.values():
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def fork_to(self, new_session_id: str, *, from_entry_id: str | None = None) -> "SessionStore":
        target = SessionStore(self.workspace_dir, new_session_id)
        meta = self.read_meta() or {}
        target.ensure_initialized(
            model_id=str(meta.get("model_id", "")),
            provider=str(meta.get("provider", "")),
            system_prompt=str(meta.get("system_prompt", "")),
        )
        target.rewrite_session_messages(self.load_session_messages(leaf_id=from_entry_id))
        task_state = self.load_task_state()
        if task_state is not None:
            target.save_task_state(task_state)
        target.update_meta({"parent_session_id": self.session_id})
        target.append_event(
            {
                "type": "session_forked",
                "from_session_id": self.session_id,
                "from_entry_id": from_entry_id,
                "to_session_id": new_session_id,
            }
        )
        return target

    def append_event(self, event: dict[str, Any]) -> None:
        self.event_recorder.append(event)
        self.run_store.append_event(event)
        self.touch_updated_at()

    def load_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.event_recorder.load(limit=limit)

    def summarize_events(self) -> dict[str, Any]:
        return self.event_recorder.summarize()

    def append_run_result(self, result: AgentRunResult) -> None:
        self.run_store.append_run_result(result)
        self.touch_updated_at()

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.run_store.load_run_results(limit=limit)

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        self.run_store.write_rollback_metadata(run_id, metadata)
        self.touch_updated_at()

    def load_task_state(self) -> dict[str, Any] | None:
        if not self.task_state_file.exists():
            return None
        from codepilot.sessions.task_state import validate_task_state_payload

        data = json.loads(self.task_state_file.read_text(encoding="utf-8"))
        return validate_task_state_payload(data)

    def save_task_state(self, state: dict[str, Any]) -> None:
        from codepilot.sessions.task_state import validate_task_state_payload

        canonical = validate_task_state_payload(state)
        self.task_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.task_state_file.write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.touch_updated_at()

    def _read_message_lines(self) -> list[dict[str, Any]]:
        if not self.messages_file.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.messages_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
        return rows

    def _write_message_lines(self, lines: list[dict[str, Any]]) -> None:
        self.messages_file.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
        self.messages_file.write_text(
            text + ("\n" if text else ""),
            encoding="utf-8",
            newline="\n",
        )

    def _write_session_state(self, state: dict[str, Any]) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


@dataclass(frozen=True)
class SessionOpenMetadata:
    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SessionOpenMetadata":
        return cls(
            provider=_optional_text(value.get("provider")),
            model_id=_optional_text(value.get("model_id")),
            system_prompt=_optional_text(value.get("system_prompt")),
        )


def load_session_open_metadata(
    workspace_dir: str | Path,
    session_id: str | None,
) -> SessionOpenMetadata | None:
    if not session_id:
        return None
    meta = SessionStore(workspace_dir, session_id).read_meta()
    return SessionOpenMetadata.from_mapping(meta) if meta is not None else None


def _preview_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content[:80]
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)[:80]
    return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["SessionOpenMetadata", "SessionStore", "load_session_open_metadata", "new_session_id"]
