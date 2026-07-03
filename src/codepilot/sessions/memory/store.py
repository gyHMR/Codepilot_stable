from __future__ import annotations

# 新手导读：MemoryStore 负责 session/project memory 的读写和状态维护。
# 关注点：它是记忆事实源，检索和写入策略分别在 retriever/writer。

"""Persistence adapter for Memory v2."""

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .records import MEMORY_SCHEMA_VERSION, MemoryRecord, MemoryStatus, utc_now_iso

if TYPE_CHECKING:
    from ..persistence.store import SessionStore


logger = logging.getLogger("codepilot.sessions.memory")

DEFAULT_MAX_SESSION_MEMORY_RECORDS = 80
DEFAULT_MAX_PROJECT_MEMORY_RECORDS = 200
DEFAULT_PROJECT_COMPACT_AFTER_LINES = 400


class MemoryStore:
    """Read and write durable session/project memory records."""

    def __init__(
        self,
        session_store: "SessionStore",
        *,
        max_session_records: int = DEFAULT_MAX_SESSION_MEMORY_RECORDS,
        max_project_records: int = DEFAULT_MAX_PROJECT_MEMORY_RECORDS,
        project_compact_after_lines: int = DEFAULT_PROJECT_COMPACT_AFTER_LINES,
    ) -> None:
        self.session_store = session_store
        self.workspace_dir = session_store.workspace_dir
        self.session_id = session_store.session_id
        self.session_file = session_store.memory_file
        self.project_file = session_store.layout.project_memory_file
        self.max_session_records = max_session_records
        self.max_project_records = max_project_records
        self.project_compact_after_lines = project_compact_after_lines

    def load_session(self) -> list[MemoryRecord]:
        if not self.session_file.exists():
            return []
        try:
            payload = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("failed to load session memory file=%s", self.session_file)
            return []
        if not isinstance(payload, dict):
            return []
        if payload.get("schema_version") != MEMORY_SCHEMA_VERSION:
            logger.warning("ignoring unsupported session memory schema")
            return []
        records = payload.get("records", [])
        return _records_from_values(records)

    def save_session(self, records: list[MemoryRecord]) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "session_id": self.session_id,
            "records": [
                record.to_dict()
                for record in _prune_records(records, self.max_session_records)
            ],
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
                record = MemoryRecord.from_dict(raw) if isinstance(raw, dict) else None
            except (json.JSONDecodeError, ValueError, TypeError):
                logger.warning("skipping invalid project memory line")
                continue
            if record is not None:
                latest[record.id] = record
        return list(latest.values())

    def append_project(self, record: MemoryRecord) -> None:
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        if _line_count(self.project_file) >= self.project_compact_after_lines:
            self.compact_project()

    def compact_project(self) -> list[MemoryRecord]:
        records = _prune_records(self.load_project(), self.max_project_records)
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return records

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
        for record in self.all_records():
            if record.id == memory_id:
                return record
        return None

    def all_records(self) -> list[MemoryRecord]:
        return [*self.load_session(), *self.load_project()]

    def active_records(self) -> list[MemoryRecord]:
        return [record for record in self.all_records() if record.status == "active"]


def _records_from_values(values: object) -> list[MemoryRecord]:
    if not isinstance(values, list):
        return []
    records: list[MemoryRecord] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        try:
            records.append(MemoryRecord.from_dict(value))
        except (TypeError, ValueError):
            logger.warning("skipping invalid session memory record")
    return records


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


def _prune_records(records: list[MemoryRecord], limit: int) -> list[MemoryRecord]:
    if limit <= 0 or len(records) <= limit:
        return list(records)
    indexed = list(enumerate(records))
    indexed.sort(key=lambda item: _record_rank(item[0], item[1]), reverse=True)
    kept = indexed[:limit]
    kept.sort(key=lambda item: _record_rank(item[0], item[1]), reverse=True)
    return [record for _index, record in kept]


def _record_rank(index: int, record: MemoryRecord) -> tuple[int, int, int, int, str, int]:
    kind_rank = {
        "correction": 4,
        "constraint": 3,
        "decision": 2,
        "experience": 1,
    }
    return (
        1 if record.status == "active" else 0,
        kind_rank.get(record.kind, 0),
        1 if "always" in record.triggers else 0,
        record.occurrences,
        record.updated_at,
        index,
    )


def _line_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for _line in handle)
    except OSError:
        return 0


__all__ = ["MemoryStore"]
