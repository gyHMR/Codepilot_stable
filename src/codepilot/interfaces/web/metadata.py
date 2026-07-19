"""持久化仅属于 Web 展示层的 Session 元数据。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WebMetadataStore:
    """Store presentation-only titles without changing Session domain state."""

    def __init__(self, workspace: Path) -> None:
        self.path = Path(workspace).resolve() / ".codepilot" / "web" / "metadata.json"

    def get(self, session_id: str) -> dict[str, str]:
        payload = self._load()
        value = payload.get("sessions", {}).get(session_id, {})
        return {str(key): str(item) for key, item in value.items() if item is not None}

    def ensure(self, session_id: str) -> dict[str, str]:
        existing = self.get(session_id)
        if existing:
            return existing
        now = _now()
        return self._update(session_id, {"created_at": now, "updated_at": now})

    def set_title(self, session_id: str, title: str) -> dict[str, str]:
        return self._update(
            session_id,
            {"title": title.strip(), "updated_at": _now()},
        )

    def touch(self, session_id: str) -> None:
        self._update(session_id, {"updated_at": _now()})

    def delete(self, session_id: str) -> None:
        payload = self._load()
        sessions = payload.setdefault("sessions", {})
        if sessions.pop(session_id, None) is not None:
            self._write(payload)

    def _update(self, session_id: str, values: dict[str, str]) -> dict[str, str]:
        payload = self._load()
        sessions = payload.setdefault("sessions", {})
        entry = sessions.setdefault(session_id, {})
        now = _now()
        entry.setdefault("created_at", now)
        entry.setdefault("updated_at", now)
        entry.update(values)
        self._write(payload)
        return {str(key): str(value) for key, value in entry.items()}

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"version": 1, "sessions": {}}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"version": 1, "sessions": {}}
        if not isinstance(value, dict) or not isinstance(value.get("sessions"), dict):
            return {"version": 1, "sessions": {}}
        return value

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["WebMetadataStore"]
