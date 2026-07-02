from __future__ import annotations

"""Session-scoped task recovery state.

This module stores the current task projection separately from durable memory.
It is used only to resume an unfinished task across runs; it is not recalled as
long-term project knowledge.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from codepilot.protocols import AgentRunResult

from ..memory.files import sanitize_memory_text
from ..memory.records import utc_now_iso


class TaskRecoveryStore:
    """Persist the current task projection for one session."""

    def __init__(self, session_store) -> None:
        self.session_store = session_store
        self.path = session_store.task_recovery_file

    def begin_task(self, text: str, *, run_id: str | None = None) -> dict[str, Any]:
        """Start or refresh a task without discarding same-goal progress."""

        goal = sanitize_memory_text(text, limit=1200)
        current = self.load_projection()
        if current and current.get("goal") == goal:
            current["source_run_id"] = run_id
            current["updated_at"] = utc_now_iso()
            self.save_projection(current, run_id=run_id)
            return current
        projection = {
            "goal": goal,
            "task_progress": None,
            "next_action": None,
            "source_run_id": run_id,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
        }
        self.save_projection(projection, run_id=run_id)
        return projection

    def load_projection(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        projection = payload.get("projection") if isinstance(payload, dict) else None
        return dict(projection) if isinstance(projection, dict) else None

    def active_projection(self) -> dict[str, Any] | None:
        projection = self.load_projection()
        if not projection:
            return None
        progress = projection.get("task_progress")
        if isinstance(progress, dict) and progress.get("completion_satisfied") is True:
            return None
        return projection if isinstance(progress, dict) else None

    def save_projection(
        self,
        projection: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        item = dict(projection)
        item["source_run_id"] = run_id or item.get("source_run_id")
        item["updated_at"] = utc_now_iso()
        payload = {
            "schema_version": 1,
            "session_id": self.session_store.session_id,
            "projection": item,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.path, payload)
        return item

    def update_from_result(self, result: AgentRunResult) -> dict[str, Any] | None:
        projection = build_task_recovery_projection(
            result,
            current_projection=self.load_projection() or {},
        )
        if projection is None:
            return self.load_projection()
        return self.save_projection(projection, run_id=result.run_id)


def build_task_recovery_projection(
    result: AgentRunResult,
    *,
    current_projection: dict[str, Any],
) -> dict[str, Any] | None:
    """将结构化 Run 结果映射为会话恢复投影。

    这是任务状态跨 run 恢复的唯一写入映射：TaskController 输出
    TaskSummary，Session 保存该投影，下一次 run 再由 TaskController 恢复。
    """

    summary = result.task
    if summary is None:
        return None
    return {
        "goal": sanitize_memory_text(summary.goal, limit=1200),
        "task_mode": str(summary.control_signal.get("mode") or "edit"),
        "plan_source": str(summary.control_signal.get("plan_source") or "default"),
        "task_progress": {
            "completed_steps": list(summary.completed_steps),
            "pending_steps": list(summary.pending_steps),
            "blocked_steps": list(summary.blocked_steps),
            "completion_satisfied": summary.completion_satisfied,
            "completion_reason": summary.completion_reason,
            "step_details": dict(summary.step_details),
        },
        "next_action": None if summary.completion_satisfied else summary.next_action,
        "source_run_id": result.run_id,
        "created_at": current_projection.get("created_at", utc_now_iso()),
        "updated_at": utc_now_iso(),
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


__all__ = ["TaskRecoveryStore", "build_task_recovery_projection"]
