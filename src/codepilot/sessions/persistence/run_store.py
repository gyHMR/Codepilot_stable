from __future__ import annotations

"""Run-level persistence and freshness checks.

SessionStore keeps the long-lived conversation tree. RunStore keeps one task's
events/result/state under .codepilot/runs/<run_id>/ so runs can be inspected
without rewriting session history.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.observability import EventRecorder
from codepilot.observability.events import normalize_event_value
from codepilot.protocols import AgentRunResult, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path


RUN_ARTIFACT_SCHEMA_VERSION = "1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FreshnessResult:
    status: str
    checked_paths: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    workspace_path: str = ""

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checked_paths": list(self.checked_paths),
            "changed_paths": list(self.changed_paths),
            "missing_paths": list(self.missing_paths),
            "workspace_path": self.workspace_path,
        }


class RunStore:
    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.root = self.workspace_dir / ".codepilot" / "runs"

    def append_event(self, event: dict[str, Any]) -> None:
        run_id = event.get("runId") or event.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        EventRecorder(run_dir / "events.jsonl").append(event)

    def load_events(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return EventRecorder(self._run_dir(run_id) / "events.jsonl").load(limit=limit)

    def append_run_result(self, result: AgentRunResult) -> None:
        run_dir = self._run_dir(result.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = normalize_event_value(result)
        record["schema_version"] = RUN_ARTIFACT_SCHEMA_VERSION
        self._write_json(run_dir / "result.json", record)
        self._write_json(
            run_dir / "state.json",
            {
                "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
                "run_id": result.run_id,
                "session_id": result.session_id or self.session_id,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "workspace_path": str(self.workspace_dir.resolve()),
                "tracked_files": self._extract_tracked_files(result),
                "updated_at": _utc_now_iso(),
            },
        )

    def load_run_result(self, run_id: str) -> dict[str, Any]:
        result_file = self._run_dir(run_id) / "result.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Run result not found: {run_id}")
        return json.loads(result_file.read_text(encoding="utf-8"))

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        records: list[tuple[str, dict[str, Any]]] = []
        for run_dir in sorted(self.root.iterdir()):
            if not run_dir.is_dir():
                continue
            result_file = run_dir / "result.json"
            if not result_file.exists():
                continue
            data = json.loads(result_file.read_text(encoding="utf-8"))
            if data.get("session_id") != self.session_id:
                continue
            state = self._read_json(run_dir / "state.json") or {}
            sort_key = str(state.get("updated_at") or run_dir.name)
            records.append((sort_key, data))
        records.sort(key=lambda item: item[0])
        out = [item[1] for item in records]
        return out[-limit:] if limit is not None else out

    def evaluate_freshness(self) -> FreshnessResult:
        workspace_path = str(self.workspace_dir.resolve())
        tracked = self._latest_tracked_files()
        if not tracked:
            return FreshnessResult(status="valid", workspace_path=workspace_path)

        checked: list[str] = []
        changed: list[str] = []
        missing: list[str] = []
        mismatch = False
        for state in tracked.values():
            if state.get("workspace_path") and state.get("workspace_path") != workspace_path:
                mismatch = True
            path = state.get("path")
            if not isinstance(path, str) or not path:
                continue
            checked.append(path)
            current = self.file_state_for_path(self.workspace_dir, path)
            if not current.get("exists"):
                missing.append(path)
                continue
            content_changed = current.get("sha256") != state.get("sha256")
            timestamp_changed = current.get("mtime_ns") != state.get("mtime_ns")
            if content_changed or timestamp_changed:
                changed.append(path)

        if mismatch:
            status = "mismatch"
        elif changed or missing:
            status = "stale"
        else:
            status = "valid"
        return FreshnessResult(
            status=status,
            checked_paths=sorted(set(checked)),
            changed_paths=sorted(set(changed)),
            missing_paths=sorted(set(missing)),
            workspace_path=workspace_path,
        )

    @staticmethod
    def file_state_for_path(workspace_dir: str | Path, path: str | Path) -> dict[str, Any]:
        return file_state_for_path(workspace_dir, path)

    def _latest_tracked_files(self) -> dict[str, dict[str, Any]]:
        tracked: dict[str, dict[str, Any]] = {}
        if not self.root.exists():
            return tracked
        for run_dir in sorted(self.root.iterdir()):
            state = self._read_json(run_dir / "state.json")
            if not isinstance(state, dict) or state.get("session_id") != self.session_id:
                continue
            for item in state.get("tracked_files", []):
                if not isinstance(item, dict):
                    continue
                path = item.get("path")
                if isinstance(path, str) and path:
                    tracked[path] = item
        return tracked

    def _extract_tracked_files(self, result: AgentRunResult) -> list[dict[str, Any]]:
        tracked: dict[str, dict[str, Any]] = {}
        for message in result.messages:
            if not isinstance(message, ToolResultMessage):
                continue
            state = message.metadata.get("file_state")
            if isinstance(state, dict) and isinstance(state.get("path"), str):
                tracked[str(state["path"])] = dict(state)
        for path in result.affected_paths:
            tracked.setdefault(path, self.file_state_for_path(self.workspace_dir, path))
        return list(tracked.values())

    def _run_dir(self, run_id: str) -> Path:
        return self.root / run_id

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
