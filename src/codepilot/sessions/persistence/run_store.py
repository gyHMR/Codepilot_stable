from __future__ import annotations

"""Run-level persistence with a single canonical run.json file."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from codepilot.observability import (
    EventRecorder,
    build_run_trace,
    redact_artifact,
    write_run_trace,
)
from codepilot.observability.events import normalize_event_value
from codepilot.protocols import AgentRunResult, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path

from ..layout import SessionLayout


RUN_ARTIFACT_SCHEMA_VERSION = "1"
FreshnessStatus = Literal["valid", "stale", "mismatch"]
_FRESHNESS_STATUSES: set[str] = {"valid", "stale", "mismatch"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FreshnessResult:
    status: FreshnessStatus
    checked_paths: list[str] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    missing_paths: list[str] = field(default_factory=list)
    workspace_path: str = ""

    def __post_init__(self) -> None:
        if self.status not in _FRESHNESS_STATUSES:
            raise ValueError(f"Unknown freshness status: {self.status}")

    def should_record_event(self) -> bool:
        return bool(self.checked_paths) or self.status != "valid"

    def requires_steering(self) -> bool:
        return self.status != "valid"

    def to_event_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checked_paths": list(self.checked_paths),
            "changed_paths": list(self.changed_paths),
            "missing_paths": list(self.missing_paths),
            "workspace_path": self.workspace_path,
        }


class RunStore:
    """Run fact store: run state/result/rollback plus run-local events."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.codepilot_dir / "runs"

    def append_event(self, event: dict[str, Any]) -> None:
        run_id = event.get("runId") or event.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        EventRecorder(self.layout.run_events_file(run_id)).append(event)
        self._update_state_from_event(run_id, event)

    def load_events(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return EventRecorder(self.layout.run_events_file(run_id)).load(limit=limit)

    def append_run_result(self, result: AgentRunResult) -> None:
        run_dir = self._run_dir(result.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = self._read_json(self.layout.run_file(result.run_id)) or {}
        record = redact_artifact(normalize_event_value(result))
        record.update(
            {
                "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
                "run_id": result.run_id,
                "session_id": result.session_id or self.session_id,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "model_attempts": result.counters.model_attempts,
                "tool_calls": result.counters.tool_calls,
                "workspace_path": str(self.workspace_dir.resolve()),
                "affected_paths": list(result.affected_paths),
                "workspace_changed": result.workspace_changed,
                "task": redact_artifact(normalize_event_value(result.task)),
                "tracked_files": self._extract_tracked_files(result),
                "rollback": existing.get("rollback"),
                "updated_at": _utc_now_iso(),
            }
        )
        self._write_json(self.layout.run_file(result.run_id), record)
        trace = build_run_trace(
            self.load_events(result.run_id),
            result=record,
        )
        write_run_trace(run_dir / "trace.json", trace)

    def load_run_result(self, run_id: str) -> dict[str, Any]:
        data = self._read_json(self.layout.run_file(run_id))
        if data is None:
            raise FileNotFoundError(f"Run result not found: {run_id}")
        return data

    def load_run_state(self, run_id: str) -> dict[str, Any]:
        data = self._read_json(self.layout.run_file(run_id))
        if data is None:
            raise FileNotFoundError(f"Run state not found: {run_id}")
        return data

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        state = self._read_json(self.layout.run_file(run_id)) or {
            "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "session_id": self.session_id,
            "workspace_path": str(self.workspace_dir.resolve()),
        }
        state["rollback"] = redact_artifact(metadata)
        state["updated_at"] = _utc_now_iso()
        self._write_json(self.layout.run_file(run_id), state)

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        records: list[tuple[str, dict[str, Any]]] = []
        for run_dir in sorted(self.root.iterdir()):
            if not run_dir.is_dir():
                continue
            data = self._read_json(run_dir / "run.json")
            if not isinstance(data, dict) or data.get("session_id") != self.session_id:
                continue
            records.append((str(data.get("updated_at") or run_dir.name), data))
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
            if current.get("sha256") != state.get("sha256") or current.get("mtime_ns") != state.get("mtime_ns"):
                changed.append(path)

        if mismatch:
            status: FreshnessStatus = "mismatch"
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
            state = self._read_json(run_dir / "run.json")
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
        return self.layout.run_dir(run_id)

    def _update_state_from_event(self, run_id: str, event: dict[str, Any]) -> None:
        path = self.layout.run_file(run_id)
        state = self._read_json(path) or {
            "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "session_id": event.get("sessionId") or self.session_id,
            "status": "running",
            "stop_reason": None,
            "model_attempts": 0,
            "tool_calls": 0,
            "workspace_path": str(self.workspace_dir.resolve()),
            "affected_paths": [],
            "workspace_changed": False,
        }
        event_type = event.get("type")
        if event_type == "message_end":
            message = event.get("message")
            role = getattr(message, "role", None)
            if role is None and isinstance(message, dict):
                role = message.get("role")
            if role == "assistant":
                state["model_attempts"] = int(state.get("model_attempts", 0)) + 1
        elif event_type == "tool_execution_end":
            state["tool_calls"] = int(state.get("tool_calls", 0)) + 1
            result = event.get("result")
            if isinstance(result, dict):
                affected = result.get("affected_paths", [])
                changed = result.get("workspace_changed")
            else:
                affected = getattr(result, "affected_paths", [])
                changed = getattr(result, "workspace_changed", None)
            state["affected_paths"] = sorted(
                {
                    *[str(item) for item in state.get("affected_paths", [])],
                    *[str(item) for item in affected or []],
                }
            )
            if changed is True:
                state["workspace_changed"] = True
        elif event_type in {
            "task_plan_created",
            "task_step_updated",
            "task_decision",
            "completion_checked",
        }:
            state["task"] = redact_artifact(normalize_event_value(event))
        elif event_type == "agent_end":
            state["status"] = event.get("status", "completed")
            state["stop_reason"] = event.get("stopReason")
        elif event_type == "error":
            state["last_error"] = redact_artifact(normalize_event_value(event))
        state["updated_at"] = _utc_now_iso()
        self._write_json(path, redact_artifact(state))

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

