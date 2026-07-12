from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SessionFileLayout:
    workspace_dir: Path

    def __init__(self, workspace_dir: str | Path) -> None:
        object.__setattr__(self, "workspace_dir", Path(workspace_dir))

    @property
    def codepilot_dir(self) -> Path:
        return self.workspace_dir / ".codepilot"

    def session_dir(self, session_id: str) -> Path:
        return self.codepilot_dir / "sessions" / session_id

    def session_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "session.json"

    def messages_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "messages.jsonl"

    def session_events_file(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "events.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self.codepilot_dir / "runs" / run_id

    def run_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def run_events_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"

    def run_artifacts_dir(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "artifacts"


def read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        _fsync_directory(path.parent)
    finally:
        if temp.exists():
            temp.unlink()


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            is_last = index == len(lines) - 1
            has_trailing_newline = raw.endswith(b"\n") or raw.endswith(b"\r")
            if is_last and not has_trailing_newline:
                break
            raise ValueError(f"Invalid JSONL record on line {index + 1}: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record on line {index + 1} must be an object: {path}")
        rows.append(value)
    return rows


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "SessionFileLayout",
    "append_jsonl",
    "atomic_write_json",
    "read_json_object",
    "read_jsonl",
]
