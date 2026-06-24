from __future__ import annotations

"""Run 级别的持久化和新鲜度检查。

SessionStore 维护长期对话树；RunStore 将每次任务的事件/结果/状态
保存在 .codepilot/runs/<run_id>/ 下，便于独立检查而无需重写会话历史。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codepilot.observability import (
    EventRecorder,
    build_audit_report,
    redact_artifact,
)
from codepilot.observability.events import normalize_event_value
from codepilot.protocols import AgentRunResult, ToolResultMessage
from codepilot.tools.sandbox import file_state_for_path


RUN_ARTIFACT_SCHEMA_VERSION = "1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class FreshnessResult:
    """文件新鲜度检查结果。"""
    status: str                              # 状态：valid/stale/mismatch
    checked_paths: list[str] = field(default_factory=list)   # 已检查的文件路径
    changed_paths: list[str] = field(default_factory=list)   # 内容已变更的文件
    missing_paths: list[str] = field(default_factory=list)   # 已删除的文件
    workspace_path: str = ""                 # 工作区路径

    def to_event_payload(self) -> dict[str, Any]:
        """转换为事件载荷字典。"""
        return {
            "status": self.status,
            "checked_paths": list(self.checked_paths),
            "changed_paths": list(self.changed_paths),
            "missing_paths": list(self.missing_paths),
            "workspace_path": self.workspace_path,
        }


class RunStore:
    """Run 持久化存储：管理每次任务的事件、结果和文件状态。"""

    def __init__(self, workspace_dir: str | Path, session_id: str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.session_id = session_id
        self.root = self.workspace_dir / ".codepilot" / "runs"

    def append_event(self, event: dict[str, Any]) -> None:
        """追加事件到对应 Run 的 events.jsonl，并增量更新 state.json。

        state.json 是 Run 的实时状态摘要，包含：
        - model_attempts: 模型调用次数（message_end + role=assistant 时递增）
        - tool_calls: 工具调用次数（tool_execution_end 时递增）
        - affected_paths: 所有受影响的文件路径
        - workspace_changed: 工作区是否被修改
        - task: 任务计划信息（task_plan_created/task_step_updated 等事件）
        - status/stop_reason: 最终状态（agent_end 时写入）
        """
        run_id = event.get("runId") or event.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            return
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        EventRecorder(run_dir / "events.jsonl").append(event)
        self._update_state_from_event(run_id, event)

    def load_events(self, run_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
        return EventRecorder(self._run_dir(run_id) / "events.jsonl").load(limit=limit)

    def append_run_result(self, result: AgentRunResult) -> None:
        """持久化 Run 结果：写入 result.json、state.json 和 report.json。

        - result.json: 完整的 AgentRunResult 序列化
        - state.json: 合并后的 Run 状态（含 tracked_files 用于新鲜度检查）
        - report.json: 审计报告（由 observability.build_audit_report 生成）
        """
        run_dir = self._run_dir(result.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        record = redact_artifact(normalize_event_value(result))
        record["schema_version"] = RUN_ARTIFACT_SCHEMA_VERSION
        self._write_json(run_dir / "result.json", record)
        state = {
            **(self._read_json(run_dir / "state.json") or {}),
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
            "updated_at": _utc_now_iso(),
        }
        self._write_json(run_dir / "state.json", state)
        events = self.load_events(result.run_id)
        self._write_json(
            run_dir / "report.json",
            build_audit_report(record, events=events, state=state),
        )

    def load_run_result(self, run_id: str) -> dict[str, Any]:
        result_file = self._run_dir(run_id) / "result.json"
        if not result_file.exists():
            raise FileNotFoundError(f"Run result not found: {run_id}")
        return json.loads(result_file.read_text(encoding="utf-8"))

    def load_run_state(self, run_id: str) -> dict[str, Any]:
        """加载某次 Run 的 state.json。"""
        state = self._read_json(self._run_dir(run_id) / "state.json")
        if state is None:
            raise FileNotFoundError(f"Run state not found: {run_id}")
        return state

    def write_rollback_metadata(self, run_id: str, metadata: dict[str, Any]) -> None:
        """将 Git 回退元数据写入 Run state。"""
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        state = self._read_json(run_dir / "state.json") or {
            "schema_version": RUN_ARTIFACT_SCHEMA_VERSION,
            "run_id": run_id,
            "session_id": self.session_id,
            "workspace_path": str(self.workspace_dir.resolve()),
        }
        state["rollback"] = redact_artifact(metadata)
        state["updated_at"] = _utc_now_iso()
        self._write_json(run_dir / "state.json", state)

    def load_run_results(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """加载当前 session 的所有 Run 结果（按 updated_at 排序）。

        遍历 .codepilot/runs/ 下所有子目录，过滤出属于当前 session 的
        result.json，按 state.json 中的 updated_at 排序后返回。
        """
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
        """评估最近 Run 跟踪的文件相对于当前工作区的新鲜度。"""
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
        """从 Run 结果中提取被跟踪的文件状态快照。

        来源：
        1. ToolResultMessage.metadata["file_state"]（read/write/edit 工具返回）
        2. result.affected_paths（所有受影响路径的当前状态）

        返回的文件状态用于 evaluate_freshness() 检测文件是否被外部修改。
        """
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

    def _update_state_from_event(
        self,
        run_id: str,
        event: dict[str, Any],
    ) -> None:
        run_dir = self._run_dir(run_id)
        path = run_dir / "state.json"
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
                    *[
                        str(item)
                        for item in state.get("affected_paths", [])
                    ],
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
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
