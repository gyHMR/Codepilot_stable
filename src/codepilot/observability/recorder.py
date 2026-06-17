from __future__ import annotations

"""JSONL event recorder for sessions, Web Console, and future eval runners."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .events import event_to_record, summarize_events


@dataclass(frozen=True)
class EventRecorder:
    path: Path

    def ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        self.ensure_parent()
        record = event_to_record(event)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def append_many(self, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.append(event) for event in events]

    def load(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = [line.strip() for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if limit is not None and limit >= 0:
            lines = lines[-limit:]
        events: list[dict[str, Any]] = []
        for line in lines:
            data = json.loads(line)
            if isinstance(data, dict):
                events.append(data)
        return events

    def summarize(self) -> dict[str, Any]:
        return summarize_events(self.load())
