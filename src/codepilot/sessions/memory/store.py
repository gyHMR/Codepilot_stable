from __future__ import annotations

# 新手导读：MemoryStore 是长期记忆唯一文件事实源的读写入口。
# 关注点：长期记忆只在 .codepilot/memory/memories.jsonl，不写 session memory 副本。

"""Canonical Memory v4 persistence."""

import json
from typing import TYPE_CHECKING

from .records import MemoryRecord, MemoryStatus, utc_now_iso

if TYPE_CHECKING:
    from ..storage import SessionStore


class MemoryStore:
    """Read and append canonical durable memory records."""

    def __init__(self, session_store: "SessionStore") -> None:
        self.session_store = session_store
        self.workspace_dir = session_store.workspace_dir
        self.session_id = session_store.session_id
        self.project_file = session_store.layout.project_memory_file

    def all_records(self) -> list[MemoryRecord]:
        if not self.project_file.exists():
            return []
        latest: dict[str, MemoryRecord] = {}
        for line_number, line in enumerate(
            self.project_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid memory JSON on line {line_number}"
                ) from exc
            record = MemoryRecord.from_dict(raw)
            latest[record.id] = record
        return list(latest.values())

    def active_records(self) -> list[MemoryRecord]:
        return [record for record in self.all_records() if record.status == "active"]

    def append(self, record: MemoryRecord) -> MemoryRecord:
        self.project_file.parent.mkdir(parents=True, exist_ok=True)
        with self.project_file.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def update(self, record: MemoryRecord) -> MemoryRecord:
        record.updated_at = utc_now_iso()
        return self.append(record)

    def get(self, memory_id: str) -> MemoryRecord | None:
        for record in self.all_records():
            if record.id == memory_id:
                return record
        return None

    def mark_status(self, memory_id: str, status: MemoryStatus) -> MemoryRecord:
        record = self.get(memory_id)
        if record is None:
            raise ValueError(f"Memory not found: {memory_id}")
        record.status = status
        return self.update(record)

    def supersede(self, old_id: str, new_record: MemoryRecord) -> MemoryRecord:
        old = self.get(old_id)
        if old is None:
            raise ValueError(f"Memory not found: {old_id}")
        old.status = "superseded"
        old.superseded_by = new_record.id
        if old.id not in new_record.supersedes:
            new_record.supersedes.append(old.id)
        self.update(old)
        return self.update(new_record)

    def replace(self, record: MemoryRecord) -> MemoryRecord:
        return self.update(record)


__all__ = ["MemoryStore"]
