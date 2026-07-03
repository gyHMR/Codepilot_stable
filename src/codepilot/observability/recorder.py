from __future__ import annotations

# 新手导读：EventRecorder 负责把事件写入 JSONL。
# 关注点：调试某次运行时，先确认这里记录了哪些原始事件。

"""Passive JSONL recorder for the slim run event contract."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .events import event_to_record, summarize_events
from .redact import redact_artifact


@dataclass(frozen=True)
class EventRecorder:
    """Append-only event recorder.

    Unknown or low-value legacy events normalize to ``{}`` and are skipped; this
    keeps observability from interrupting normal agent execution.
    """

    path: Path

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        self.ensure_parent()
        record = redact_artifact(event_to_record(event))
        if not isinstance(record, dict) or not record:
            return {}
        with self.path.open("a", encoding="utf-8", newline="\n") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def append_many(self, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [record for event in events if (record := self.append(event))]

    def load(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = [
            line.strip()
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit is not None and limit >= 0:
            lines = lines[-limit:]
        records: list[dict[str, Any]] = []
        for line in lines:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        return records

    def summarize(self) -> dict[str, Any]:
        return summarize_events(self.load())


__all__ = ["EventRecorder"]
