"""持久化用户级和项目级 Memory 记录及其生命周期历史。"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path

from .contracts import MemoryRecord, MemoryScope


class _JsonlMemoryRepository:
    """JSONL persistence for exactly one Memory scope."""

    def __init__(self, path: str | Path, *, scope: MemoryScope) -> None:
        self.path = Path(path)
        self.scope = scope

    def all_records(self) -> list[MemoryRecord]:
        latest: dict[str, MemoryRecord] = {}
        for payload in _read_json_rows(self.path, recover_incomplete_tail=True):
            record = MemoryRecord.from_dict(payload)
            if record.scope != self.scope:
                raise ValueError(
                    f"memory scope {record.scope} is stored in {self.scope} repository"
                )
            latest[record.id] = record
        return sorted(latest.values(), key=lambda record: (record.updated_at, record.id))

    def get(self, memory_id: str) -> MemoryRecord | None:
        return next(
            (record for record in self.all_records() if record.id == memory_id),
            None,
        )

    def save(self, record: MemoryRecord) -> None:
        if record.scope != self.scope:
            raise ValueError(
                f"cannot save {record.scope} memory in {self.scope} repository"
            )
        rows = _read_json_rows(self.path, recover_incomplete_tail=True)
        rows.append(record.to_dict())
        _atomic_write_rows(self.path, rows)

    def records_for_key(self, key: str) -> list[MemoryRecord]:
        return [record for record in self.all_records() if record.key == key]

    def purge(self, memory_id: str) -> bool:
        rows = _read_json_rows(self.path, recover_incomplete_tail=True)
        kept = [row for row in rows if str(row.get("id") or "") != memory_id]
        if len(kept) == len(rows):
            return False
        _atomic_write_rows(self.path, kept)
        return True

    def replace_all(self, records: Iterable[MemoryRecord]) -> None:
        rows: list[dict[str, object]] = []
        for record in records:
            if record.scope != self.scope:
                raise ValueError(
                    f"cannot save {record.scope} memory in {self.scope} repository"
                )
            rows.append(record.to_dict())
        _atomic_write_rows(self.path, rows)


class UserMemoryRepository(_JsonlMemoryRepository):
    """持久化用户级 Memory 及其历史记录。"""
    def __init__(self, home_dir: str | Path | None = None) -> None:
        root = Path(home_dir).expanduser() if home_dir is not None else Path.home()
        super().__init__(
            root / ".codepilot" / "memory" / "memories.jsonl",
            scope="user",
        )


class ProjectMemoryRepository(_JsonlMemoryRepository):
    """持久化项目级 Memory 及其历史记录。"""
    def __init__(self, workspace_dir: str | Path) -> None:
        root = Path(workspace_dir)
        super().__init__(
            root / ".codepilot" / "memory" / "memories.jsonl",
            scope="project",
        )


def _read_json_rows(
    path: Path,
    *,
    recover_incomplete_tail: bool,
) -> list[dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            is_incomplete_tail = index == len(lines) - 1 and not text.endswith(("\n", "\r"))
            if recover_incomplete_tail and is_incomplete_tail:
                _atomic_write_rows(path, rows)
                return rows
            raise ValueError(f"invalid memory JSON on line {index + 1}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"invalid memory record on line {index + 1}")
        rows.append(payload)
    return rows


def _atomic_write_rows(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    content = "".join(
        json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    _atomic_write_bytes(path, content)


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


__all__ = ["ProjectMemoryRepository", "UserMemoryRepository"]
