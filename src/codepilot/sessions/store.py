from __future__ import annotations

"""Session-owned files: metadata, transcript, events, runs, and artifacts."""

import hashlib
import json
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from codepilot.observability import build_run_trace, redact_artifact, write_run_trace
from codepilot.observability.events import normalize_event_value
from codepilot.protocols import AgentRunResult, Message, ToolResultMessage
from codepilot.sessions.workspace_state import file_state_for_path

from .serde import message_from_dict, message_to_dict


FreshnessStatus = Literal["valid", "stale", "mismatch"]
RUN_ARTIFACT_SCHEMA_VERSION = "1"
_FRESHNESS_STATUSES = frozenset({"valid", "stale", "mismatch"})
_MANIFEST_PROJECT_TYPES = {
    "pyproject.toml": "Python",
    "package.json": "JavaScript/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
}
_TEST_DIR_NAMES = {"test", "tests", "spec", "specs"}
_INSTRUCTION_FILES = ["AGENTS.md", "CLAUDE.md", "COPILOT.md", "INSTRUCTIONS.md"]
_INTERNAL_TOP_LEVEL_NAMES = {".git", ".codepilot", ".pytest_cache", "__pycache__"}


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:12]}"


def new_event_id() -> str:
    return f"event_{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class SessionLayout:
    workspace_dir: Path
    session_id: str

    @classmethod
    def for_workspace(cls, workspace_dir: str | Path, session_id: str) -> "SessionLayout":
        return cls(Path(workspace_dir), session_id)

    @property
    def codepilot_dir(self) -> Path:
        return self.workspace_dir / ".codepilot"

    @property
    def session_dir(self) -> Path:
        return self.codepilot_dir / "sessions" / self.session_id

    @property
    def session_file(self) -> Path:
        return self.session_dir / "session.json"

    @property
    def messages_file(self) -> Path:
        return self.session_dir / "messages.jsonl"

    @property
    def session_events_file(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def plan_state_file(self) -> Path:
        return self.session_dir / "plan_state.json"

    @property
    def context_ledger_file(self) -> Path:
        return self.session_dir / "context_ledger.jsonl"

    @property
    def tool_outputs_dir(self) -> Path:
        return self.session_dir / "artifacts" / "tool_outputs"

    @property
    def project_memory_file(self) -> Path:
        return self.codepilot_dir / "memory" / "memories.jsonl"

    def run_dir(self, run_id: str) -> Path:
        return self.codepilot_dir / "runs" / run_id

    def run_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def run_events_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"


@dataclass(frozen=True)
class GitInfo:
    root: Path
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None


@dataclass(frozen=True)
class RepositoryBootstrap:
    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


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


@dataclass(frozen=True)
class SessionOpenMetadata:
    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SessionOpenMetadata":
        model = value.get("model") if isinstance(value.get("model"), dict) else {}
        return cls(
            provider=_optional_text(model.get("provider")),
            model_id=_optional_text(model.get("model")),
            system_prompt=None,
        )


class RunStore:
    """Store run.json, run events, rollback metadata, and freshness evidence."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.codepilot_dir / "runs"

    def append_event(self, event: dict[str, Any]) -> None:
        run_id = _event_run_id(event)
        if run_id is None:
            return
        self.layout.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        _append_jsonl(self.layout.run_events_file(run_id), _event_payload(event))
        self._merge_event_into_run_state(run_id, event)

    def load_events(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        events = _read_jsonl(self.layout.run_events_file(run_id))
        return events[-limit:] if limit is not None else events

    def append_run_result(self, result: AgentRunResult) -> None:
        run_dir = self.layout.run_dir(result.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = _read_json(self.layout.run_file(result.run_id)) or {}
        events = self.load_events(result.run_id)
        model_attempts = sum(
            1
            for event in events
            if event.get("type") == "message_end"
            and _message_role(event.get("message")) == "assistant"
        )
        tool_calls = sum(
            1
            for event in events
            if event.get("type") in {"tool_completed", "tool_failed", "tool_interrupted"}
        )
        agent_starts = sum(1 for event in events if event.get("type") == "agent_start")
        affected_paths = sorted(
            {
                *(
                    path
                    for path in existing.get("affected_paths", [])
                    if isinstance(path, str)
                ),
                *result.affected_paths,
            }
        )
        tracked_files = {
            str(item["path"]): item
            for item in existing.get("tracked_files", [])
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        tracked_files.update(
            {
                str(item["path"]): item
                for item in self._tracked_files_from_result(result)
                if isinstance(item.get("path"), str)
            }
        )
        rollback = existing.get("rollback")
        if isinstance(rollback, dict):
            rollback = {
                **rollback,
                "affected_paths": affected_paths,
                "workspace_changed": bool(
                    existing.get("workspace_changed") or result.workspace_changed
                ),
            }
        record = redact_artifact(normalize_event_value(result))
        record.update(
            {
                "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
                "run_id": result.run_id,
                "session_id": result.session_id or self.session_id,
                "status": result.status,
                "stop_reason": result.stop_reason,
                "model_attempts": model_attempts or result.counters.model_attempts,
                "tool_calls": tool_calls or result.counters.tool_calls,
                "resume_count": max(0, agent_starts - 1),
                "phase": _run_phase(result.status, result.stop_reason),
                "workspace_path": str(self.workspace_dir.resolve()),
                "affected_paths": affected_paths,
                "workspace_changed": bool(
                    existing.get("workspace_changed") or result.workspace_changed
                ),
                "plan": redact_artifact(normalize_event_value(result.plan)),
                "signals": redact_artifact(normalize_event_value(result.signals)),
                "tracked_files": list(tracked_files.values()),
                "rollback": rollback,
                "updated_at": _utc_now_iso(),
            }
        )
        _write_json(self.layout.run_file(result.run_id), record)
        write_run_trace(
            run_dir / "trace.json",
            build_run_trace(self.load_events(result.run_id), result=record),
        )

    def load_run_result(self, run_id: str) -> dict[str, Any]:
        data = _read_json(self.layout.run_file(run_id))
        if data is None:
            raise FileNotFoundError(f"Run result not found: {run_id}")
        return data

    def load_run_state(self, run_id: str) -> dict[str, Any]:
        return self.load_run_result(run_id)

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        records: list[tuple[str, dict[str, Any]]] = []
        for run_dir in sorted(self.root.iterdir()):
            if not run_dir.is_dir():
                continue
            data = _read_json(run_dir / "run.json")
            if isinstance(data, dict) and data.get("session_id") == self.session_id:
                records.append((str(data.get("updated_at") or run_dir.name), data))
        records.sort(key=lambda item: item[0])
        items = [item[1] for item in records]
        return items[-limit:] if limit is not None else items

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        state = _read_json(self.layout.run_file(run_id)) or {
            "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "session_id": self.session_id,
            "workspace_path": str(self.workspace_dir.resolve()),
        }
        state["rollback"] = redact_artifact(metadata)
        state["updated_at"] = _utc_now_iso()
        _write_json(self.layout.run_file(run_id), state)

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
            elif current.get("sha256") != state.get("sha256") or current.get("mtime_ns") != state.get("mtime_ns"):
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
        for run in self.load_run_results():
            for item in run.get("tracked_files", []):
                if isinstance(item, dict) and isinstance(item.get("path"), str):
                    tracked[str(item["path"])] = dict(item)
        return tracked

    def _tracked_files_from_result(self, result: AgentRunResult) -> list[dict[str, Any]]:
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

    def _merge_event_into_run_state(self, run_id: str, event: dict[str, Any]) -> None:
        state = _read_json(self.layout.run_file(run_id)) or {
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
        if event_type == "message_end" and _message_role(event.get("message")) == "assistant":
            state["model_attempts"] = int(state.get("model_attempts", 0)) + 1
        elif event_type in {"tool_completed", "tool_failed", "tool_interrupted"}:
            state["tool_calls"] = int(state.get("tool_calls", 0)) + 1
            affected, changed = _tool_event_effects(event.get("result"))
            state["affected_paths"] = sorted({*state.get("affected_paths", []), *affected})
            if changed is True:
                state["workspace_changed"] = True
        elif event_type in {
            "plan_proposed",
            "plan_approval_required",
            "plan_approved",
            "plan_rejected",
            "plan_updated",
            "plan_completed",
            "plan_abandoned",
        }:
            state["plan"] = redact_artifact(normalize_event_value(event.get("plan")))
        elif event_type == "run_guard_checked":
            state["signals"] = redact_artifact(normalize_event_value(event.get("signals")))
        elif event_type == "agent_end":
            state["status"] = event.get("status", "completed")
            state["stop_reason"] = event.get("stopReason")
        elif event_type == "error":
            state["last_error"] = redact_artifact(normalize_event_value(event))
        state["updated_at"] = _utc_now_iso()
        _write_json(self.layout.run_file(run_id), redact_artifact(state))


class SessionStore:
    """One readable boundary for all files owned by a session."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.session_dir
        self.session_file = self.layout.session_file
        self.messages_file = self.layout.messages_file
        self.events_file = self.layout.session_events_file
        self.plan_state_file = self.layout.plan_state_file
        self.context_ledger_file = self.layout.context_ledger_file
        self.run_store = RunStore(self.workspace_dir, self.session_id)

    def ensure_initialized(self, *, model_id: str, provider: str, system_prompt: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.session_file.exists():
            _write_json(
                self.session_file,
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "workspace_root": str(self.workspace_dir.resolve()).replace("\\", "/"),
                    "model": {"provider": provider, "model": model_id},
                    "system_prompt_hash": _hash_text(system_prompt),
                    "current_mode": "build",
                    "active_plan_id": None,
                    "leaf_message_id": None,
                    "last_run_id": None,
                    "runtime_checkpoint": None,
                    "context": {
                        "compacted_until_message_id": None,
                        "last_compact_summary": "",
                        "last_context_id": None,
                        "last_compacted_at": None,
                    },
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                },
            )
        for path in (self.messages_file, self.events_file, self.context_ledger_file):
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8", newline="\n")

    def read_meta(self) -> dict[str, Any] | None:
        return _read_json(self.session_file)

    def update_meta(self, updates: dict[str, Any]) -> dict[str, Any]:
        state = self.read_meta() or {
            "schema_version": 1,
            "session_id": self.session_id,
            "created_at": _utc_now_iso(),
        }
        state.update(updates)
        state["session_id"] = self.session_id
        state["updated_at"] = _utc_now_iso()
        _write_json(self.session_file, state)
        return state

    def update_context_meta(self, updates: dict[str, Any]) -> dict[str, Any]:
        state = self.read_meta() or {}
        context = state.get("context") if isinstance(state.get("context"), dict) else {}
        state["context"] = {**context, **updates}
        return self.update_meta(state)

    def set_checkpoint(self, checkpoint: dict[str, Any] | None) -> None:
        if checkpoint is not None:
            checkpoint = dict(checkpoint)
            checkpoint.setdefault("created_at", _utc_now_iso())
        self.update_meta({"runtime_checkpoint": checkpoint})
        event_type = "checkpoint_cleared" if checkpoint is None else "checkpoint_saved"
        self.append_event(
            {
                "type": event_type,
                "sessionId": self.session_id,
                "checkpoint": checkpoint,
            }
        )

    def touch_updated_at(self) -> None:
        state = self.read_meta()
        if state is not None:
            state["updated_at"] = _utc_now_iso()
            _write_json(self.session_file, state)

    def append_message(self, message: Message, *, run_id: str | None = None) -> str:
        parent_id = (self.read_meta() or {}).get("leaf_message_id")
        message_id = new_message_id()
        row = {
            "id": message_id,
            "run_id": run_id,
            "parent_id": parent_id if isinstance(parent_id, str) else None,
            "created_at": _utc_now_iso(),
            **message_to_dict(message),
        }
        _append_jsonl(self.messages_file, row)
        self.update_meta({"leaf_message_id": message_id})
        return message_id

    def append_messages(self, messages: list[Message], *, run_id: str | None = None) -> list[str]:
        return [self.append_message(message, run_id=run_id) for message in messages]

    def rewrite_session_messages(self, messages: list[Message]) -> None:
        parent_id: str | None = None
        rows: list[dict[str, Any]] = []
        for message in messages:
            message_id = new_message_id()
            rows.append(
                {
                    "id": message_id,
                    "run_id": None,
                    "parent_id": parent_id,
                    "created_at": _utc_now_iso(),
                    **message_to_dict(message),
                }
            )
            parent_id = message_id
        _write_jsonl(self.messages_file, rows)
        self.update_meta({"leaf_message_id": parent_id})

    def load_session_messages(self, *, leaf_id: str | None = None) -> list[Message]:
        rows = self._message_chain(leaf_id)
        return [message_from_dict(row) for row in rows]

    def list_entry_ids(self) -> list[str]:
        return [str(row["id"]) for row in _read_jsonl(self.messages_file) if isinstance(row.get("id"), str)]

    def list_entries(self) -> list[dict[str, Any]]:
        leaf_id = self.get_leaf_id()
        result = []
        for row in _read_jsonl(self.messages_file):
            message_id = row.get("id")
            if not isinstance(message_id, str):
                continue
            result.append(
                {
                    "id": message_id,
                    "parent_id": row.get("parent_id"),
                    "timestamp": row.get("created_at"),
                    "role": row.get("role", "unknown"),
                    "preview": _preview_message(row),
                    "depth": max(0, len(self.get_entry_path(message_id)) - 1),
                    "is_leaf": message_id == leaf_id,
                }
            )
        result.sort(key=lambda item: str(item.get("timestamp", "")))
        return result

    def get_leaf_id(self) -> str | None:
        value = (self.read_meta() or {}).get("leaf_message_id")
        return value if isinstance(value, str) else None

    def get_entry_path(self, entry_id: str) -> list[str]:
        by_id = self._messages_by_id()
        if entry_id not in by_id:
            raise ValueError(f"Entry not found: {entry_id}")
        path: list[str] = []
        current: str | None = entry_id
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            path.append(current)
            parent = by_id[current].get("parent_id")
            current = parent if isinstance(parent, str) else None
        path.reverse()
        return path

    def set_leaf(self, entry_id: str) -> None:
        if entry_id not in set(self.list_entry_ids()):
            raise ValueError(f"Entry not found: {entry_id}")
        self.update_meta({"leaf_message_id": entry_id})

    def get_session_tree(self) -> list[dict[str, Any]]:
        nodes: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        for row in _read_jsonl(self.messages_file):
            message_id = row.get("id")
            if not isinstance(message_id, str):
                continue
            nodes[message_id] = {
                "id": message_id,
                "parent_id": row.get("parent_id"),
                "timestamp": row.get("created_at"),
                "role": row.get("role", "unknown"),
                "preview": _preview_message(row),
                "children": [],
            }
        for node in nodes.values():
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id in nodes:
                nodes[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def fork_to(self, new_session_id: str, *, from_entry_id: str | None = None) -> "SessionStore":
        target = SessionStore(self.workspace_dir, new_session_id)
        meta = self.read_meta() or {}
        model = meta.get("model") if isinstance(meta.get("model"), dict) else {}
        target.ensure_initialized(
            model_id=str(model.get("model") or ""),
            provider=str(model.get("provider") or ""),
            system_prompt="",
        )
        target.rewrite_session_messages(self.load_session_messages(leaf_id=from_entry_id))
        plan_state = self.load_plan_state()
        if plan_state is not None:
            target.save_plan_state(plan_state)
        target.update_meta({"parent_session_id": self.session_id})
        target.append_event(
            {
                "type": "session_forked",
                "from_session_id": self.session_id,
                "from_entry_id": from_entry_id,
                "to_session_id": new_session_id,
            }
        )
        return target

    def append_event(self, event: dict[str, Any]) -> None:
        payload = _event_payload(event)
        _append_jsonl(self.events_file, payload)
        self.run_store.append_event(payload)
        self.touch_updated_at()

    def load_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        events = _read_jsonl(self.events_file)
        return events[-limit:] if limit is not None else events

    def summarize_events(self) -> dict[str, Any]:
        summary: dict[str, int] = {}
        for event in self.load_events():
            event_type = str(event.get("type") or "unknown")
            summary[event_type] = summary.get(event_type, 0) + 1
        return summary

    def append_run_result(self, result: AgentRunResult) -> None:
        self.run_store.append_run_result(result)
        self.update_meta({"last_run_id": result.run_id})

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.run_store.load_run_results(limit=limit)

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        self.run_store.write_rollback_metadata(run_id, metadata)
        self.touch_updated_at()

    def load_plan_state(self) -> dict[str, Any] | None:
        if not self.plan_state_file.exists():
            return None
        from codepilot.sessions.plan_state import validate_plan_state_payload

        data = json.loads(self.plan_state_file.read_text(encoding="utf-8"))
        return validate_plan_state_payload(data)

    def save_plan_state(self, state: dict[str, Any]) -> None:
        from codepilot.sessions.plan_state import validate_plan_state_payload

        canonical = validate_plan_state_payload(state)
        self.plan_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.plan_state_file.write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        active_plan_id = (
            canonical.get("plan_id")
            if canonical.get("status") in {"proposed", "active"}
            else None
        )
        self.update_meta({"active_plan_id": active_plan_id})

    def append_context_ledger(self, payload: dict[str, Any]) -> None:
        _append_jsonl(self.context_ledger_file, payload)

    def load_context_ledger(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.context_ledger_file)
        return rows[-limit:] if limit is not None else rows

    def _messages_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            str(row["id"]): row
            for row in _read_jsonl(self.messages_file)
            if isinstance(row.get("id"), str)
        }

    def _message_chain(self, leaf_id: str | None = None) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.messages_file)
        if not rows:
            return []
        by_id = {str(row["id"]): row for row in rows if isinstance(row.get("id"), str)}
        current = leaf_id or self.get_leaf_id()
        if not isinstance(current, str) or current not in by_id:
            current = str(rows[-1].get("id"))
        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            row = by_id[current]
            chain.append(row)
            parent_id = row.get("parent_id")
            current = parent_id if isinstance(parent_id, str) else None
        chain.reverse()
        return chain


def load_session_open_metadata(
    workspace_dir: str | Path,
    session_id: str | None,
) -> SessionOpenMetadata | None:
    if not session_id:
        return None
    meta = SessionStore(workspace_dir, session_id).read_meta()
    return SessionOpenMetadata.from_mapping(meta) if meta is not None else None


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    root = Path(workspace).resolve()
    entries = _top_level_entries(root)
    manifest_files = [name for name in _MANIFEST_PROJECT_TYPES if (root / name).is_file()]
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=_project_type(manifest_files),
        manifest_files=manifest_files,
        top_level_entries=entries,
        test_directories=[
            entry for entry in entries if entry.endswith("/") and entry[:-1] in _TEST_DIR_NAMES
        ],
        instruction_files=[name for name in _INSTRUCTION_FILES if (root / name).is_file()],
        git=_build_git_info(root),
    )


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    lines = [
        "## Repository Context",
        f"- Workspace: {bootstrap.workspace_root}",
        f"- Project type: {bootstrap.project_type or 'unknown'}",
        f"- Manifests: {_joined(bootstrap.manifest_files)}",
        f"- Top-level: {_joined(bootstrap.top_level_entries, empty='(empty)')}",
        f"- Test directories: {_joined(bootstrap.test_directories)}",
        f"- Instruction files: {_joined(bootstrap.instruction_files)}",
    ]
    if bootstrap.git is None:
        lines.append("- Git: not a git repository")
    else:
        lines.extend(
            [
                f"- Git branch: {bootstrap.git.branch or 'detached HEAD'}",
                f"- HEAD: {bootstrap.git.head_sha or 'unknown'}",
                f"- Working tree: {'modified' if bootstrap.git.is_dirty else 'clean'}",
            ]
        )
    return "\n".join(lines)


def _event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.setdefault("eventId", new_event_id())
    payload.setdefault("created_at", _utc_now_iso())
    payload.setdefault("sessionId", payload.get("session_id"))
    return redact_artifact(normalize_event_value(payload))


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8", newline="\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _build_git_info(root: Path) -> GitInfo | None:
    if _git(root, "rev-parse", "--git-dir").returncode != 0:
        return None
    git_root = _stdout(_git(root, "rev-parse", "--show-toplevel"))
    return GitInfo(
        root=Path(git_root) if git_root else root,
        branch=_stdout(_git(root, "branch", "--show-current")) or None,
        head_sha=_stdout(_git(root, "rev-parse", "--short", "HEAD")) or None,
        is_dirty=any(
            not _is_internal_status_line(line)
            for line in _stdout(_git(root, "status", "--porcelain")).splitlines()
            if line.strip()
        ),
        remote_url=_stdout(_git(root, "remote", "get-url", "origin")) or None,
    )


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return subprocess.CompletedProcess(["git", *args], 1, "", "")


def _stdout(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout.strip()


def _top_level_entries(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    items = sorted(
        (item for item in root.iterdir() if item.name not in _INTERNAL_TOP_LEVEL_NAMES),
        key=lambda item: item.name.lower(),
    )
    return [f"{item.name}/" if item.is_dir() else item.name for item in items[:30]]


def _is_internal_status_line(line: str) -> bool:
    if len(line) < 4:
        return False
    path = line[3:].split(" -> ")[-1].replace("\\", "/").strip("/")
    return bool(path and path.split("/", 1)[0] in _INTERNAL_TOP_LEVEL_NAMES)


def _project_type(manifest_files: list[str]) -> str | None:
    for manifest, project_type in _MANIFEST_PROJECT_TYPES.items():
        if manifest in manifest_files:
            return project_type
    return None


def _joined(values: list[str], *, empty: str = "(none)") -> str:
    return ", ".join(values) if values else empty


def _event_run_id(event: dict[str, Any]) -> str | None:
    run_id = event.get("runId") or event.get("run_id")
    return run_id if isinstance(run_id, str) and run_id else None


def _message_role(message: object) -> str | None:
    if isinstance(message, dict):
        value = message.get("role")
        return value if isinstance(value, str) else None
    value = getattr(message, "role", None)
    return value if isinstance(value, str) else None


def _tool_event_effects(result: object) -> tuple[list[str], bool | None]:
    if isinstance(result, dict):
        affected = result.get("affected_paths", [])
        changed = result.get("workspace_changed")
    else:
        affected = getattr(result, "affected_paths", [])
        changed = getattr(result, "workspace_changed", None)
    paths = [str(path) for path in affected or []]
    return paths, changed if isinstance(changed, bool) else None


def _run_phase(status: object, stop_reason: object) -> str:
    if status == "waiting_approval":
        return "tool_approval"
    if status == "waiting_user":
        if stop_reason == "plan_approval_required":
            return "plan_approval"
        if stop_reason == "plan_clarification_required":
            return "plan_clarification"
        return "waiting_user"
    return "terminal"


def _preview_message(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content[:80]
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "".join(parts)[:80]
    return ""


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "FreshnessResult",
    "FreshnessStatus",
    "GitInfo",
    "RepositoryBootstrap",
    "RunStore",
    "SessionLayout",
    "SessionOpenMetadata",
    "SessionStore",
    "build_repository_bootstrap",
    "load_session_open_metadata",
    "new_event_id",
    "new_message_id",
    "new_session_id",
    "render_repository_context",
]
