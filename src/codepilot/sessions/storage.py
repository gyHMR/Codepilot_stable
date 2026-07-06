from __future__ import annotations

# 新手导读：storage.py 集中管理 session/run 文件布局、序列化、会话存储和仓库打开快照。
# 关注点：sessions 里凡是落盘或读取工作区静态事实的代码，都先从这里找入口。

# ---- filesystem layout ----

# 新手导读：SessionLayout 统一定义 session/run/memory/context/artifact 的文件路径。
# 关注点：新增持久化文件前先看这里，避免路径拼接散落各处。

"""Canonical filesystem layout for session-owned state."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionLayout:
    """Resolve all session/run paths from one small contract."""

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
    def task_state_file(self) -> Path:
        return self.session_dir / "task_state.json"

    @property
    def session_memory_file(self) -> Path:
        return self.session_dir / "memory.json"

    @property
    def context_ledger_file(self) -> Path:
        return self.session_dir / "context_ledger.jsonl"

    @property
    def tool_outputs_dir(self) -> Path:
        return self.session_dir / "artifacts" / "tool_outputs"

    @property
    def project_memory_file(self) -> Path:
        return self.codepilot_dir / "memory" / "memories.jsonl"

    @property
    def pinned_memory_file(self) -> Path:
        return self.codepilot_dir / "MEMORY.md"

    def run_dir(self, run_id: str) -> Path:
        return self.codepilot_dir / "runs" / run_id

    def run_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "run.json"

    def run_events_file(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "events.jsonl"


__all__ = ["SessionLayout"]

# ---- message serialization ----

# 新手导读：serde.py 集中处理协议对象和 JSON 之间的序列化/反序列化。
# 关注点：新增持久化字段时优先在这里确认格式转换。

"""
消息序列化/反序列化工具。

目标：
1) 把 ai 层 dataclass 消息安全写入 jsonl；
2) 下次启动时恢复为同等结构，继续参与 Agent 推理。
"""

from typing import Any
from dataclasses import asdict

from codepilot.protocols import (
    AssistantMessage,
    Cost,
    ImageContent,
    LLMErrorInfo,
    Message,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from codepilot.protocols.tools import coerce_tool_result_status


def _user_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    return {"type": "image", "data": block.data, "mime_type": block.mime_type}


def _assistant_block_to_dict(block: TextContent | ThinkingContent | ToolCall) -> dict[str, Any]:
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text, "text_signature": block.text_signature}
    if isinstance(block, ThinkingContent):
        return {
            "type": "thinking",
            "thinking": block.thinking,
            "thinking_signature": block.thinking_signature,
            "redacted": block.redacted,
        }
    return {"type": "toolCall", "id": block.id, "name": block.name, "arguments": block.arguments}


def _tool_result_block_to_dict(block: TextContent | ImageContent) -> dict[str, Any]:
    return _user_block_to_dict(block)


def message_to_dict(message: Message) -> dict[str, Any]:
    """
    将 Message 转成可持久化 dict。
    """

    if isinstance(message, UserMessage):
        content: str | list[dict[str, Any]]
        if isinstance(message.content, str):
            content = message.content
        else:
            content = [_user_block_to_dict(b) for b in message.content]
        return {"role": "user", "content": content, "timestamp": message.timestamp}

    if isinstance(message, AssistantMessage):
        return {
            "role": "assistant",
            "content": [_assistant_block_to_dict(b) for b in message.content],
            "api": message.api,
            "provider": message.provider,
            "model": message.model,
            "usage": {
                "input": message.usage.input,
                "output": message.usage.output,
                "cache_read": message.usage.cache_read,
                "cache_write": message.usage.cache_write,
                "total_tokens": message.usage.total_tokens,
                "cost": {
                    "input": message.usage.cost.input,
                    "output": message.usage.cost.output,
                    "cache_read": message.usage.cost.cache_read,
                    "cache_write": message.usage.cost.cache_write,
                    "total": message.usage.cost.total,
                },
            },
            "stop_reason": message.stop_reason,
            "response_id": message.response_id,
            "error_message": message.error_message,
            "error_info": asdict(message.error_info) if message.error_info else None,
            "timestamp": message.timestamp,
            "metadata": message.metadata,
        }

    if isinstance(message, ToolResultMessage):
        return {
            "role": "toolResult",
            "tool_call_id": message.tool_call_id,
            "tool_name": message.tool_name,
            "content": [_tool_result_block_to_dict(b) for b in message.content],
            "status": message.status,
            "is_error": message.is_error,
            "approved": message.approved,
            "approval_id": message.approval_id,
            "error_code": message.error_code,
            "exit_code": message.exit_code,
            "affected_paths": message.affected_paths,
            "workspace_changed": message.workspace_changed,
            "diff_summary": message.diff_summary,
            "verification": message.verification,
            "details": message.details,
            "timestamp": message.timestamp,
            "metadata": message.metadata,
        }

    raise TypeError(f"Unsupported message type: {type(message)!r}")


def _user_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    if data.get("type") == "image":
        return ImageContent(data=data.get("data", ""), mime_type=data.get("mime_type", "image/png"))
    return TextContent(text=data.get("text", ""), text_signature=data.get("text_signature"))


def _assistant_block_from_dict(data: dict[str, Any]) -> TextContent | ThinkingContent | ToolCall:
    t = data.get("type")
    if t == "thinking":
        return ThinkingContent(
            thinking=data.get("thinking", ""),
            thinking_signature=data.get("thinking_signature"),
            redacted=bool(data.get("redacted", False)),
        )
    if t == "toolCall":
        return ToolCall(
            id=data.get("id", ""),
            name=data.get("name", ""),
            arguments=data.get("arguments", {}) if isinstance(data.get("arguments"), dict) else {},
        )
    return TextContent(text=data.get("text", ""), text_signature=data.get("text_signature"))


def _tool_result_block_from_dict(data: dict[str, Any]) -> TextContent | ImageContent:
    return _user_block_from_dict(data)


def message_from_dict(data: dict[str, Any]) -> Message:
    """
    将持久化 dict 恢复成 Message。
    """

    role = data.get("role")
    if role == "user":
        raw_content = data.get("content", "")
        if isinstance(raw_content, str):
            content: str | list[TextContent | ImageContent] = raw_content
        else:
            content = [_user_block_from_dict(i) for i in raw_content if isinstance(i, dict)]
        return UserMessage(content=content, timestamp=int(data.get("timestamp", 0)))

    if role == "assistant":
        usage_data = data.get("usage", {})
        cost_data = usage_data.get("cost", {}) if isinstance(usage_data, dict) else {}
        usage = Usage(
            input=int(usage_data.get("input", 0)) if isinstance(usage_data, dict) else 0,
            output=int(usage_data.get("output", 0)) if isinstance(usage_data, dict) else 0,
            cache_read=int(usage_data.get("cache_read", 0)) if isinstance(usage_data, dict) else 0,
            cache_write=int(usage_data.get("cache_write", 0)) if isinstance(usage_data, dict) else 0,
            total_tokens=int(usage_data.get("total_tokens", 0)) if isinstance(usage_data, dict) else 0,
            cost=Cost(
                input=float(cost_data.get("input", 0.0)) if isinstance(cost_data, dict) else 0.0,
                output=float(cost_data.get("output", 0.0)) if isinstance(cost_data, dict) else 0.0,
                cache_read=float(cost_data.get("cache_read", 0.0)) if isinstance(cost_data, dict) else 0.0,
                cache_write=float(cost_data.get("cache_write", 0.0)) if isinstance(cost_data, dict) else 0.0,
                total=float(cost_data.get("total", 0.0)) if isinstance(cost_data, dict) else 0.0,
            ),
        )
        error_info_data = data.get("error_info")
        error_info = None
        if isinstance(error_info_data, dict):
            error_info = LLMErrorInfo(
                code=str(error_info_data.get("code", "llm.unknown")),
                message=str(error_info_data.get("message", "")),
                retryable=bool(error_info_data.get("retryable", False)),
                kind=error_info_data.get("kind", "unknown"),
                provider=str(error_info_data.get("provider", "")),
                model=str(error_info_data.get("model", "")),
                status_code=error_info_data.get("status_code"),
                details=error_info_data.get("details", {})
                if isinstance(error_info_data.get("details"), dict)
                else {},
            )
        return AssistantMessage(
            content=[_assistant_block_from_dict(i) for i in data.get("content", []) if isinstance(i, dict)],
            api=data.get("api", ""),
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            usage=usage,
            stop_reason=data.get("stop_reason", "stop"),
            response_id=data.get("response_id"),
            error_message=data.get("error_message"),
            error_info=error_info,
            timestamp=int(data.get("timestamp", 0)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        )

    if role == "toolResult":
        is_error = bool(data.get("is_error", False))
        return ToolResultMessage(
            tool_call_id=data.get("tool_call_id", ""),
            tool_name=data.get("tool_name", ""),
            content=[_tool_result_block_from_dict(i) for i in data.get("content", []) if isinstance(i, dict)],
            status=coerce_tool_result_status(
                data.get("status"),
                default="error" if is_error else "success",
            ),
            is_error=is_error,
            approved=bool(data.get("approved", True)),
            approval_id=(
                data.get("approval_id")
                if isinstance(data.get("approval_id"), str)
                else None
            ),
            error_code=data.get("error_code"),
            exit_code=data.get("exit_code"),
            affected_paths=[
                str(path)
                for path in data.get("affected_paths", [])
                if isinstance(path, str)
            ],
            workspace_changed=(
                data.get("workspace_changed")
                if isinstance(data.get("workspace_changed"), bool)
                else None
            ),
            diff_summary=data.get("diff_summary"),
            verification=(
                data.get("verification")
                if isinstance(data.get("verification"), dict)
                else None
            ),
            details=data.get("details"),
            timestamp=int(data.get("timestamp", 0)),
            metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
        )

    raise ValueError(f"Unknown role: {role!r}")

# ---- run store ----

# 新手导读：RunStore 管理 run.json/events.jsonl，并评估 run 与工作区状态的新鲜度。
# 关注点：它关注一次运行的证据，而不是整个会话的所有消息。

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
from codepilot.sessions.workspace_state import file_state_for_path



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

# ---- session store ----

# 新手导读：SessionStore 是 session.json/messages.jsonl 的事实源封装。
# 关注点：会话恢复、消息追加和 task recovery 当前投影都从这里落盘。

"""Slim session persistence store.

Canonical layout:

.codepilot/sessions/<session_id>/
  - session.json      session metadata and leaf pointer
  - messages.jsonl    canonical transcript tree
  - events.jsonl      lazily-created lightweight session events
  - task_state.json   authoritative current task state
"""

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.observability import EventRecorder
from codepilot.protocols import AgentRunResult, Message



def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


_TASK_STATE_KEYS = {
    "schema_version",
    "task_id",
    "raw_user_request",
    "current_mode",
    "approval_state",
    "goal",
    "proposed_plan",
    "approved_plan",
    "current_step_id",
    "steps",
    "verification_status",
    "evidence_refs",
    "blocked_reason",
    "recovery_summary",
    "source_run_id",
    "created_at",
    "updated_at",
}
_TASK_STEP_STATUSES = {"pending", "in_progress", "completed", "blocked"}
_TASK_STEP_KINDS = {"investigate", "edit", "verify", "summarize", "other"}
_TASK_MODES = {"read", "plan", "build"}


def normalize_task_state_payload(raw: object) -> dict[str, Any] | None:
    """Return canonical task_state.json payload from canonical or legacy data."""

    if not isinstance(raw, dict):
        return None
    state = canonicalize_task_state_for_write(raw)
    progress = raw.get("task_progress")
    if not isinstance(state.get("steps"), list) and isinstance(progress, dict):
        state["steps"] = _legacy_task_progress_steps(progress)
    if "current_mode" not in state:
        state["current_mode"] = _task_mode(raw.get("task_mode"), default="build")
    else:
        state["current_mode"] = _task_mode(state.get("current_mode"), default="build")
    state.setdefault("schema_version", 1)
    state.setdefault("approval_state", "none")
    state.setdefault("proposed_plan", None)
    state.setdefault("approved_plan", None)
    state.setdefault("current_step_id", _first_active_step_id(state.get("steps")))
    state.setdefault("steps", [])
    state.setdefault("verification_status", _legacy_verification_status(progress))
    state.setdefault("evidence_refs", [])
    state.setdefault("blocked_reason", _legacy_blocked_reason(progress))
    state.setdefault("recovery_summary", "")
    return state


def canonicalize_task_state_for_write(state: dict[str, Any]) -> dict[str, Any]:
    """Strip historical task fields before persisting task_state.json."""

    return {
        key: value
        for key, value in state.items()
        if key in _TASK_STATE_KEYS
    }


def _legacy_task_progress_steps(progress: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    details = progress.get("step_details")
    step_details = details if isinstance(details, dict) else {}

    def add(raw: object, status: str) -> None:
        if not isinstance(raw, list):
            return
        for item in raw:
            title = " ".join(str(item).strip().split())
            if not title or title in seen:
                continue
            seen.add(title)
            detail = step_details.get(title)
            detail_map = detail if isinstance(detail, dict) else {}
            steps.append(
                {
                    "id": f"step_{len(steps) + 1}",
                    "title": title[:160],
                    "status": status,
                    "kind": _task_step_kind(detail_map.get("kind")),
                    "acceptance": _optional_task_text(detail_map.get("acceptance")),
                    "verification_hint": _optional_task_text(
                        detail_map.get("verification_hint")
                    ),
                }
            )

    add(progress.get("completed_steps"), "completed")
    add(progress.get("blocked_steps"), "blocked")
    add(progress.get("pending_steps"), "pending")
    return steps


def _first_active_step_id(steps: object) -> str | None:
    if not isinstance(steps, list):
        return None
    for item in steps:
        if not isinstance(item, dict):
            continue
        if item.get("status") in {"in_progress", "pending"}:
            step_id = item.get("id")
            return step_id if isinstance(step_id, str) and step_id else None
    return None


def _legacy_verification_status(progress: object) -> str:
    if not isinstance(progress, dict):
        return "unknown"
    if progress.get("completion_satisfied") is True:
        return "passed"
    if progress.get("blocked_steps"):
        return "revision_needed"
    return "unknown"


def _legacy_blocked_reason(progress: object) -> str | None:
    if not isinstance(progress, dict):
        return None
    reason = progress.get("completion_reason")
    return reason if isinstance(reason, str) and reason else None


def _task_mode(value: object, *, default: str) -> str:
    text = value.strip() if isinstance(value, str) else ""
    return text if text in _TASK_MODES else default


def _task_step_kind(value: object) -> str:
    text = value.strip() if isinstance(value, str) else ""
    return text if text in _TASK_STEP_KINDS else "other"


def _optional_task_text(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text[:240] or None


class SessionStore:
    """Session fact store: metadata, transcript tree, and lightweight events."""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.layout = SessionLayout.for_workspace(self.workspace_dir, self.session_id)
        self.root = self.layout.session_dir
        self.session_file = self.layout.session_file
        self.messages_file = self.layout.messages_file
        self.events_file = self.layout.session_events_file
        self.task_state_file = self.layout.task_state_file
        self.memory_file = self.layout.session_memory_file
        self.event_recorder = EventRecorder(self.events_file)
        self.run_store = RunStore(self.workspace_dir, self.session_id)

    def ensure_initialized(
        self,
        *,
        model_id: str,
        provider: str,
        system_prompt: str,
    ) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.session_file.exists():
            self._write_session_state(
                {
                    "schema_version": 1,
                    "session_id": self.session_id,
                    "parent_session_id": None,
                    "model_id": model_id,
                    "provider": provider,
                    "system_prompt": system_prompt,
                    "system_prompt_hash": _hash_text(system_prompt),
                    "leaf_id": None,
                    "created_at": _utc_now_iso(),
                    "updated_at": _utc_now_iso(),
                }
            )
        if not self.messages_file.exists():
            self.messages_file.write_text("", encoding="utf-8", newline="\n")

    def touch_updated_at(self) -> None:
        state = self.read_meta()
        if state is None:
            return
        state["updated_at"] = _utc_now_iso()
        self._write_session_state(state)

    def read_meta(self) -> dict[str, Any] | None:
        if not self.session_file.exists():
            return None
        return json.loads(self.session_file.read_text(encoding="utf-8"))

    def update_meta(self, updates: dict[str, Any]) -> dict[str, Any]:
        state = self.read_meta() or {
            "schema_version": 1,
            "session_id": self.session_id,
            "created_at": _utc_now_iso(),
        }
        state.update(updates)
        state["session_id"] = self.session_id
        state["updated_at"] = _utc_now_iso()
        self._write_session_state(state)
        return state

    def load_task_recovery(self) -> dict[str, Any] | None:
        return self.load_task_state()

    def save_task_recovery(self, projection: dict[str, Any] | None) -> None:
        if projection is None:
            if self.task_state_file.exists():
                self.task_state_file.unlink()
            self.touch_updated_at()
            return
        self.save_task_state(projection)

    def load_task_state(self) -> dict[str, Any] | None:
        if not self.task_state_file.exists():
            return None
        data = json.loads(self.task_state_file.read_text(encoding="utf-8"))
        return normalize_task_state_payload(data)

    def save_task_state(self, state: dict[str, Any]) -> None:
        canonical = canonicalize_task_state_for_write(state)
        self.task_state_file.parent.mkdir(parents=True, exist_ok=True)
        self.task_state_file.write_text(
            json.dumps(canonical, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.touch_updated_at()

    def append_message(self, message: Message) -> str:
        lines = self._read_message_lines()
        state = self.read_meta() or {}
        parent_id = state.get("leaf_id")
        entry_id = self._new_entry_id()
        entry = {
            "type": "message",
            "id": entry_id,
            "parent_id": parent_id if isinstance(parent_id, str) else None,
            "timestamp": _utc_now_iso(),
            "message": message_to_dict(message),
        }
        self._write_message_lines([*lines, entry])
        self.update_meta({"leaf_id": entry_id})
        return entry_id

    def append_event(self, event: dict[str, Any]) -> None:
        self.event_recorder.append(event)
        self.run_store.append_event(event)
        self.touch_updated_at()

    def load_events(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.event_recorder.load(limit=limit)

    def summarize_events(self) -> dict[str, Any]:
        return self.event_recorder.summarize()

    def append_run_result(self, result: AgentRunResult) -> None:
        self.run_store.append_run_result(result)
        self.touch_updated_at()

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        self.run_store.write_rollback_metadata(run_id, metadata)
        self.touch_updated_at()

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.run_store.load_run_results(limit=limit)

    def rewrite_session_messages(self, messages: list[Message]) -> None:
        rebuilt: list[dict[str, Any]] = []
        parent_id: str | None = None
        for message in messages:
            entry_id = self._new_entry_id()
            rebuilt.append(
                {
                    "type": "message",
                    "id": entry_id,
                    "parent_id": parent_id,
                    "timestamp": _utc_now_iso(),
                    "message": message_to_dict(message),
                }
            )
            parent_id = entry_id
        self._write_message_lines(rebuilt)
        self.update_meta({"leaf_id": parent_id})

    def load_session_messages(self, *, leaf_id: str | None = None) -> list[Message]:
        entries = [
            line for line in self._read_message_lines() if line.get("type") == "message"
        ]
        if not entries:
            return []
        by_id = {str(e.get("id")): e for e in entries if isinstance(e.get("id"), str)}
        state = self.read_meta() or {}
        current = leaf_id or state.get("leaf_id")
        if not isinstance(current, str) or current not in by_id:
            current = str(entries[-1].get("id"))

        chain: list[dict[str, Any]] = []
        seen: set[str] = set()
        while isinstance(current, str) and current in by_id and current not in seen:
            seen.add(current)
            entry = by_id[current]
            chain.append(entry)
            parent_id = entry.get("parent_id")
            current = parent_id if isinstance(parent_id, str) else None

        chain.reverse()
        messages: list[Message] = []
        for entry in chain:
            msg_data = entry.get("message")
            if isinstance(msg_data, dict):
                messages.append(message_from_dict(msg_data))
        return messages

    def list_entry_ids(self) -> list[str]:
        return [
            str(line.get("id"))
            for line in self._read_message_lines()
            if line.get("type") == "message" and isinstance(line.get("id"), str)
        ]

    def get_leaf_id(self) -> str | None:
        state = self.read_meta() or {}
        leaf = state.get("leaf_id")
        return leaf if isinstance(leaf, str) else None

    def list_entries(self) -> list[dict[str, Any]]:
        entries = [
            line for line in self._read_message_lines() if line.get("type") == "message"
        ]
        leaf_id = self.get_leaf_id()
        result: list[dict[str, Any]] = []
        for entry in entries:
            eid = entry.get("id")
            if not isinstance(eid, str):
                continue
            msg = entry.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else "unknown"
            depth = len(self.get_entry_path(eid)) - 1
            result.append(
                {
                    "id": eid,
                    "parent_id": entry.get("parent_id"),
                    "timestamp": entry.get("timestamp"),
                    "role": role,
                    "preview": self._preview_message(msg if isinstance(msg, dict) else {}),
                    "depth": max(depth, 0),
                    "is_leaf": eid == leaf_id,
                }
            )
        result.sort(key=lambda item: str(item.get("timestamp", "")))
        return result

    def get_entry_path(self, entry_id: str) -> list[str]:
        by_id = {
            str(line.get("id")): line
            for line in self._read_message_lines()
            if line.get("type") == "message" and isinstance(line.get("id"), str)
        }
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
        ids = set(self.list_entry_ids())
        if entry_id not in ids:
            raise ValueError(f"Entry not found: {entry_id}")
        self.update_meta({"leaf_id": entry_id})

    def get_session_tree(self) -> list[dict[str, Any]]:
        entries = [
            line for line in self._read_message_lines() if line.get("type") == "message"
        ]
        node_by_id: dict[str, dict[str, Any]] = {}
        roots: list[dict[str, Any]] = []
        for entry in entries:
            eid = entry.get("id")
            if not isinstance(eid, str):
                continue
            msg = entry.get("message", {})
            role = msg.get("role") if isinstance(msg, dict) else "unknown"
            node_by_id[eid] = {
                "id": eid,
                "parent_id": entry.get("parent_id"),
                "timestamp": entry.get("timestamp"),
                "role": role,
                "preview": self._preview_message(msg if isinstance(msg, dict) else {}),
                "children": [],
            }
        for node in node_by_id.values():
            parent_id = node.get("parent_id")
            if isinstance(parent_id, str) and parent_id in node_by_id:
                node_by_id[parent_id]["children"].append(node)
            else:
                roots.append(node)
        return roots

    def fork_to(
        self,
        new_session_id: str,
        *,
        from_entry_id: str | None = None,
    ) -> "SessionStore":
        target = SessionStore(self.workspace_dir, new_session_id)
        state = self.read_meta() or {}
        target.ensure_initialized(
            model_id=str(state.get("model_id", "")),
            provider=str(state.get("provider", "")),
            system_prompt=str(state.get("system_prompt", "")),
        )
        target.rewrite_session_messages(self.load_session_messages(leaf_id=from_entry_id))
        if self.memory_file.exists():
            target.memory_file.parent.mkdir(parents=True, exist_ok=True)
            target.memory_file.write_text(
                self.memory_file.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )
            try:
                memory_payload = json.loads(target.memory_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                memory_payload = None
            if isinstance(memory_payload, dict):
                memory_payload["session_id"] = new_session_id
                target.memory_file.write_text(
                    json.dumps(memory_payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                    newline="\n",
                )
        if self.task_state_file.exists():
            task_state = self.load_task_state()
            if task_state is not None:
                target.save_task_state(task_state)
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

    def _new_entry_id(self) -> str:
        return uuid.uuid4().hex[:8]

    def _read_message_lines(self) -> list[dict[str, Any]]:
        if not self.messages_file.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.messages_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                out.append(data)
        return out

    def _write_message_lines(self, lines: list[dict[str, Any]]) -> None:
        self.messages_file.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines)
        self.messages_file.write_text(
            text + ("\n" if text else ""),
            encoding="utf-8",
            newline="\n",
        )

    def _write_session_state(self, state: dict[str, Any]) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.session_file.write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @staticmethod
    def _preview_message(message: dict[str, Any]) -> str:
        role = message.get("role")
        content = message.get("content")
        if role == "user":
            if isinstance(content, str):
                return content[:80]
            if isinstance(content, list):
                return _text_blocks_preview(content)
        if role == "assistant" and isinstance(content, list):
            return _text_blocks_preview(content)
        if role == "toolResult" and isinstance(content, list):
            return _text_blocks_preview(content)
        return ""


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _text_blocks_preview(content: list[Any]) -> str:
    text = ""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text += str(block.get("text", ""))
    return text[:80]

# ---- session reopen metadata ----

from dataclasses import dataclass
from pathlib import Path
from typing import Any



@dataclass(frozen=True)
class SessionOpenMetadata:
    """Read-only metadata needed when reopening an existing session."""

    provider: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SessionOpenMetadata":
        return cls(
            provider=_optional_text(value.get("provider")),
            model_id=_optional_text(value.get("model_id")),
            system_prompt=_optional_text(value.get("system_prompt")),
        )


def load_session_open_metadata(
    workspace_dir: str | Path,
    session_id: str | None,
) -> SessionOpenMetadata | None:
    """Load the stable metadata snapshot required to reopen a session."""

    if not session_id:
        return None
    raw = SessionStore(workspace_dir=workspace_dir, session_id=session_id).read_meta()
    if raw is None:
        return None
    return SessionOpenMetadata.from_mapping(raw)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


__all__ = ["SessionOpenMetadata", "load_session_open_metadata"]

# ---- repository bootstrap ----

# 新手导读：repository.py 定义会话层可公开消费的仓库引导快照。
# 关注点：Runtime 可以用它生成启动提示词；ContextGovernor 会在每轮 run 前刷新更动态的上下文。

"""Repository bootstrap snapshots owned by the sessions layer."""

import subprocess
from dataclasses import dataclass
from pathlib import Path


_MANIFEST_PROJECT_TYPES = {
    "pyproject.toml": "Python",
    "package.json": "JavaScript/TypeScript",
    "Cargo.toml": "Rust",
    "go.mod": "Go",
    "pom.xml": "Java",
}
_TEST_DIR_NAMES = {"test", "tests", "spec", "specs"}
_TOP_LEVEL_LIMIT = 30
_INSTRUCTION_FILES = ["AGENTS.md", "CLAUDE.md", "COPILOT.md", "INSTRUCTIONS.md"]
_INTERNAL_TOP_LEVEL_NAMES = {".git", ".codepilot", ".pytest_cache", "__pycache__"}


@dataclass(frozen=True)
class GitInfo:
    """Git repository facts captured for a read-only snapshot."""

    root: Path
    branch: str | None = None
    head_sha: str | None = None
    is_dirty: bool = False
    remote_url: str | None = None


@dataclass(frozen=True)
class RepositoryBootstrap:
    """Static repository facts used when opening a session."""

    workspace_root: str
    project_type: str | None
    manifest_files: list[str]
    top_level_entries: list[str]
    test_directories: list[str]
    instruction_files: list[str]
    git: GitInfo | None = None


def build_repository_bootstrap(workspace: Path) -> RepositoryBootstrap:
    """Scan the workspace root and build a stable repository bootstrap view."""

    root = workspace.resolve()
    entries = _top_level_entries(root)
    manifest_files = [name for name in _MANIFEST_PROJECT_TYPES if (root / name).is_file()]
    project_type = _project_type(manifest_files)
    test_directories = [
        entry
        for entry in entries
        if entry.endswith("/") and entry[:-1] in _TEST_DIR_NAMES
    ]
    instruction_files = [name for name in _INSTRUCTION_FILES if (root / name).is_file()]
    git_info = _build_git_info(root)
    return RepositoryBootstrap(
        workspace_root=str(root).replace("\\", "/"),
        project_type=project_type,
        manifest_files=manifest_files,
        top_level_entries=entries,
        test_directories=test_directories,
        instruction_files=instruction_files,
        git=git_info,
    )


def render_repository_context(bootstrap: RepositoryBootstrap) -> str:
    """Render repository bootstrap facts as Markdown for a system prompt."""

    project_type = bootstrap.project_type or "unknown"
    manifests = ", ".join(bootstrap.manifest_files) if bootstrap.manifest_files else "(none)"
    top_level = ", ".join(bootstrap.top_level_entries) if bootstrap.top_level_entries else "(empty)"
    tests = ", ".join(bootstrap.test_directories) if bootstrap.test_directories else "(none)"
    instructions = ", ".join(bootstrap.instruction_files) if bootstrap.instruction_files else "(none)"

    lines = [
        "## Repository Context",
        f"- Workspace: {bootstrap.workspace_root}",
        f"- Project type: {project_type}",
        f"- Manifests: {manifests}",
        f"- Top-level: {top_level}",
        f"- Test directories: {tests}",
        f"- Instruction files: {instructions}",
    ]

    if bootstrap.git:
        branch = bootstrap.git.branch or "detached HEAD"
        dirty = "modified" if bootstrap.git.is_dirty else "clean"
        lines.append(f"- Git branch: {branch}")
        lines.append(f"- HEAD: {bootstrap.git.head_sha or 'unknown'}")
        lines.append(f"- Working tree: {dirty}")
    else:
        lines.append("- Git: not a git repository")

    return "\n".join(lines)


def _build_git_info(root: Path) -> GitInfo | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode != 0:
            return None

        git_root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        git_root = Path(git_root_result.stdout.strip()) if git_root_result.returncode == 0 else root

        branch_result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        branch = branch_result.stdout.strip() or None

        head_result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        head_sha = head_result.stdout.strip() or None

        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        is_dirty = any(
            not _is_internal_status_line(line)
            for line in status_result.stdout.splitlines()
            if line.strip()
        )

        remote_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        remote_url = remote_result.stdout.strip() or None

        return GitInfo(
            root=git_root,
            branch=branch,
            head_sha=head_sha,
            is_dirty=is_dirty,
            remote_url=remote_url,
        )
    except Exception:
        return None


def _top_level_entries(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    items = sorted(
        (
            item
            for item in root.iterdir()
            if item.name not in _INTERNAL_TOP_LEVEL_NAMES
        ),
        key=lambda item: item.name.lower(),
    )
    entries = []
    for item in items[:_TOP_LEVEL_LIMIT]:
        entries.append(f"{item.name}/" if item.is_dir() else item.name)
    return entries


def _is_internal_status_line(line: str) -> bool:
    if len(line) < 4:
        return False
    path = line[3:].split(" -> ")[-1].replace("\\", "/").strip("/")
    if not path:
        return False
    return path.split("/", 1)[0] in _INTERNAL_TOP_LEVEL_NAMES


def _project_type(manifest_files: list[str]) -> str | None:
    for manifest in _MANIFEST_PROJECT_TYPES:
        if manifest in manifest_files:
            return _MANIFEST_PROJECT_TYPES[manifest]
    return None


__all__ = [
    "GitInfo",
    "RepositoryBootstrap",
    "build_repository_bootstrap",
    "render_repository_context",
]

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
    "message_from_dict",
    "message_to_dict",
    "new_session_id",
    "render_repository_context",
]
