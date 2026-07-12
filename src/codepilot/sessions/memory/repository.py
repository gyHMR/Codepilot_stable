from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MemoryRepository:
    """Append-only workspace memory persistence."""

    def __init__(self, workspace_dir: str | Path) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.path = self.workspace_dir / ".codepilot" / "memory" / "memories.jsonl"

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid memory JSON on line {line_number}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"invalid memory record on line {line_number}")
            records.append(payload)
        return records

    def append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["MemoryRepository"]
