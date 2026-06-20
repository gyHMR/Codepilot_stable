from __future__ import annotations

"""JSONL 事件记录器：用于会话、Web Console 和未来的 eval 运行器。"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .audit import redact_artifact
from .events import event_to_record, summarize_events


@dataclass(frozen=True)
class EventRecorder:
    """JSONL 事件记录器：追加写入事件到文件，支持加载和统计。"""
    path: Path

    def ensure_parent(self) -> None:
        """确保父目录和文件存在。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("", encoding="utf-8")

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """追加一条事件到 JSONL 文件，返回规范化后的记录。"""
        self.ensure_parent()
        record = redact_artifact(event_to_record(event))
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def append_many(self, events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.append(event) for event in events]

    def load(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """加载事件列表（可选限制返回最近 N 条）。"""
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
        """加载全部事件并返回统计摘要。"""
        return summarize_events(self.load())
