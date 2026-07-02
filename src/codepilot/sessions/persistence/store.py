from __future__ import annotations

"""Slim session persistence store.

Canonical layout:

.codepilot/sessions/<session_id>/
  - session.json      session metadata, leaf pointer, task recovery projection
  - messages.jsonl    canonical transcript tree
  - events.jsonl      lazily-created lightweight session events
  - memory.json       lazily-created session durable memory
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.observability import EventRecorder
from codepilot.protocols import AgentRunResult, Message

from ..layout import SessionLayout
from .run_store import RunStore
from .serde import message_from_dict, message_to_dict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


class SessionStore:
    """Session fact store: metadata, transcript tree, and lightweight events."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.session_dir
        self.session_file = self.layout.session_file
        self.messages_file = self.layout.messages_file
        self.events_file = self.layout.session_events_file
        self.memory_file = self.layout.session_memory_file
        self.event_recorder = EventRecorder(self.events_file)
        self.run_store = RunStore(self.workspace_dir, self.session_id)

    def ensure_initialized(
        self,
        *,
        model_id: str,
        provider: str,
        system_prompt: str,
    ) -> None:
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
                    "task_recovery": None,
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                }
            )
        if not self.messages_file.exists():
            self.messages_file.write_text("", encoding="utf-8", newline="\n")

    def touch_updated_at(self) -> None:
        state = self.read_meta()
        if state is None:
            return
        state["updated_at"] = _utc_now_iso()
        self._write_session_state(state)

    def read_meta(self) -> dict[str, Any] | None:
        if not self.session_file.exists():
            return None
        return json.loads(self.session_file.read_text(encoding="utf-8"))

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

    def load_task_recovery(self) -> dict[str, Any] | None:
        state = self.read_meta() or {}
        projection = state.get("task_recovery")
        return dict(projection) if isinstance(projection, dict) else None

    def save_task_recovery(self, projection: dict[str, Any] | None) -> None:
        self.update_meta({"task_recovery": projection})

    def append_message(self, message: Message) -> str:
        lines = self._read_message_lines()
        state = self.read_meta() or {}
        parent_id = state.get("leaf_id")
        entry_id = self._new_entry_id()
        entry = {
            "type": "message",
            "id": entry_id,
            "parent_id": parent_id if isinstance(parent_id, str) else None,
            "timestamp": _utc_now_iso(),
            "message": message_to_dict(message),
        }
        self._write_message_lines([*lines, entry])
        self.update_meta({"leaf_id": entry_id})
        return entry_id

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

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        self.run_store.write_rollback_metadata(run_id, metadata)
        self.touch_updated_at()

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.run_store.load_run_results(limit=limit)

    def rewrite_session_messages(self, messages: list[Message]) -> None:
        rebuilt: list[dict[str, Any]] = []
        parent_id: str | None = None
        for message in messages:
            entry_id = self._new_entry_id()
            rebuilt.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parent_id": parent_id,
                    "timestamp": _utc_now_iso(),
                    "message": message_to_dict(message),
                }
            )
            parent_id = entry_id
        self._write_message_lines(rebuilt)
        self.update_meta({"leaf_id": parent_id})

    def load_session_messages(self, *, leaf_id: str | None = None) -> list[Message]:
        entries = [
            line for line in self._read_message_lines() if line.get("type") == "message"
        ]
        if not entries:
            return []
        by_id = {str(e.get("id")): e for e in entries if isinstance(e.get("id"), str)}
        state = self.read_meta() or {}
        current = leaf_id or state.get("leaf_id")
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
            msg_data = entry.get("message")
            if isinstance(msg_data, dict):
                messages.append(message_from_dict(msg_data))
        return messages

    def list_entry_ids(self) -> list[str]:
        return [
            str(line.get("id"))
            for line in self._read_message_lines()
            if line.get("type") == "message" and isinstance(line.get("id"), str)
        ]

    def get_leaf_id(self) -> str | None:
        state = self.read_meta() or {}
        leaf = state.get("leaf_id")
        return leaf if isinstance(leaf, str) else None

    def list_entries(self) -> list[dict[str, Any]]:
        entries = [
            line for line in self._read_message_lines() if line.get("type") == "message"
        ]
        leaf_id = self.get_leaf_id()
        result: list[dict[str, Any]] = []
        for entry in entries:
            eid = entry.get("id")
            if not isinstance(eid, str):
                continue
            msg = entry.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else "unknown"
            depth = len(self.get_entry_path(eid)) - 1
            result.append(
                {
                    "id": eid,
                    "parent_id": entry.get("parent_id"),
                    "timestamp": entry.get("timestamp"),
                    "role": role,
                    "preview": self._preview_message(msg if isinstance(msg, dict) else {}),
                    "depth": max(depth, 0),
                    "is_leaf": eid == leaf_id,
                }
            )
        result.sort(key=lambda item: str(item.get("timestamp", "")))
        return result

    def get_entry_path(self, entry_id: str) -> list[str]:
        by_id = {
            str(line.get("id")): line
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
        ids = set(self.list_entry_ids())
        if entry_id not in ids:
            raise ValueError(f"Entry not found: {entry_id}")
        self.update_meta({"leaf_id": entry_id})

    def get_session_tree(self) -> list[dict[str, Any]]:
        entries = [
            line for line in self._read_message_lines() if line.get("type") == "message"
        ]
        node_by_id: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        for entry in entries:
            eid = entry.get("id")
            if not isinstance(eid, str):
                continue
            msg = entry.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else "unknown"
            node_by_id[eid] = {
                "id": eid,
                "parent_id": entry.get("parent_id"),
                "timestamp": entry.get("timestamp"),
                "role": role,
                "preview": self._preview_message(msg if isinstance(msg, dict) else {}),
                "children": [],
            }
        for node in node_by_id.values():
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id in node_by_id:
                node_by_id[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def fork_to(
        self,
        new_session_id: str,
        *,
        from_entry_id: str | None = None,
    ) -> "SessionStore":
        target = SessionStore(self.workspace_dir, new_session_id)
        state = self.read_meta() or {}
        target.ensure_initialized(
            model_id=str(state.get("model_id", "")),
            provider=str(state.get("provider", "")),
            system_prompt=str(state.get("system_prompt", "")),
        )
        target.rewrite_session_messages(self.load_session_messages(leaf_id=from_entry_id))
        if self.memory_file.exists():
            target.memory_file.parent.mkdir(parents=True, exist_ok=True)
            target.memory_file.write_text(
                self.memory_file.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )
            try:
                memory_payload = json.loads(target.memory_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                memory_payload = None
            if isinstance(memory_payload, dict):
                memory_payload["session_id"] = new_session_id
                target.memory_file.write_text(
                    json.dumps(memory_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
        target.update_meta(
            {
                "parent_session_id": self.session_id,
                "task_recovery": state.get("task_recovery"),
            }
        )
        target.append_event(
            {
                "type": "session_forked",
                "from_session_id": self.session_id,
                "from_entry_id": from_entry_id,
                "to_session_id": new_session_id,
            }
        )
        return target

    def _new_entry_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _read_message_lines(self) -> list[dict[str, Any]]:
        if not self.messages_file.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.messages_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                out.append(data)
        return out

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

    @staticmethod
    def _preview_message(message: dict[str, Any]) -> str:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            if isinstance(content, str):
                return content[:80]
            if isinstance(content, list):
                return _text_blocks_preview(content)
        if role == "assistant" and isinstance(content, list):
            return _text_blocks_preview(content)
        if role == "toolResult" and isinstance(content, list):
            return _text_blocks_preview(content)
        return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _text_blocks_preview(content: list[Any]) -> str:
    text = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text += str(block.get("text", ""))
    return text[:80]
