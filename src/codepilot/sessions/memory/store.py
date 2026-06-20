from __future__ import annotations

"""结构化记忆的持久化适配器（会话级和项目级记忆）。"""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .records import (
    MEMORY_SCHEMA_VERSION,
    MemoryRecord,
    MemoryStatus,
    utc_now_iso,
)

if TYPE_CHECKING:
    from ..persistence.store import SessionStore


logger = logging.getLogger("codepilot.sessions.memory")


class MemoryStore:
    """记忆持久化存储：负责读写结构化记忆，不决定应记住什么。"""

    def __init__(self, session_store: "SessionStore") -> None:
        self.session_store = session_store
        self.workspace_dir = session_store.workspace_dir
        self.session_id = session_store.session_id
        self.session_file = session_store.memory_file
        self.project_file = self.workspace_dir / ".codepilot" / "memory" / "project.jsonl"

    def load_session(self) -> list[MemoryRecord]:
        if not self.session_file.exists():
            return []
        try:
            payload = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to load session memory file=%s", self.session_file)
            return []
        records = payload.get("records", []) if isinstance(payload, dict) else []
        return [
            MemoryRecord.from_dict(item)
            for item in records
            if isinstance(item, dict)
        ]

    def save_session(self, records: list[MemoryRecord]) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "records": [record.to_dict() for record in records],
        }
        _atomic_write_json(self.session_file, payload)

    def load_project(self) -> list[MemoryRecord]:
        if not self.project_file.exists():
            return []
        latest: dict[str, MemoryRecord] = {}
        for line in self.project_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("skipping invalid project memory line")
                continue
            if isinstance(raw, dict):
                record = MemoryRecord.from_dict(raw)
                latest[record.id] = record
        return list(latest.values())

    def append_project(self, record: MemoryRecord) -> None:
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def upsert_session(self, record: MemoryRecord) -> MemoryRecord:
        records = self.load_session()
        for index, existing in enumerate(records):
            if existing.id == record.id:
                records[index] = record
                break
        else:
            records.append(record)
        self.save_session(records)
        return record

    def update(self, record: MemoryRecord) -> MemoryRecord:
        record.updated_at = utc_now_iso()
        if record.scope == "project":
            self.append_project(record)
        else:
            self.upsert_session(record)
        return record

    def mark_status(self, memory_id: str, status: MemoryStatus) -> MemoryRecord:
        for record in self.load_session():
            if record.id == memory_id:
                record.status = status
                return self.update(record)
        for record in self.load_project():
            if record.id == memory_id:
                record.status = status
                return self.update(record)
        raise ValueError(f"Memory not found: {memory_id}")

    def get(self, memory_id: str) -> MemoryRecord | None:
        for record in [*self.load_session(), *self.load_project()]:
            if record.id == memory_id:
                return record
        return None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


__all__ = ["MemoryStore"]
