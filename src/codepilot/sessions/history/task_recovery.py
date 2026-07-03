from __future__ import annotations

# 新手导读：TaskRecoveryStore 保存当前任务恢复投影，支持中断后继续任务控制状态。
# 关注点：恢复的是任务进度和证据，不是重新运行历史工具。

"""Session-scoped task recovery state.

This module stores the current task projection separately from durable memory.
It is used only to resume an unfinished task across runs; it is not recalled as
long-term project knowledge.
"""

from collections.abc import Mapping
from typing import Any

from codepilot.protocols import AgentRunResult

from ..memory.files import sanitize_memory_text
from ..memory.records import utc_now_iso


class TaskRecoveryStore:
    """Persist the current task projection for one session."""

    def __init__(self, session_store) -> None:
        self.session_store = session_store

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
        return self.session_store.load_task_recovery()

    def active_projection(self) -> dict[str, Any] | None:
        projection = self.load_projection()
        if not projection:
            return None
        progress = projection.get("task_progress")
        if isinstance(progress, dict) and progress.get("completion_satisfied") is True:
            return None
        if isinstance(progress, dict):
            return projection
        planning = projection.get("planning")
        if isinstance(planning, dict) and isinstance(planning.get("discovery"), dict):
            return projection
        return None

    def save_projection(
        self,
        projection: dict[str, Any],
        *,
        run_id: str | None = None,
    ) -> dict[str, Any]:
        item = dict(projection)
        item["source_run_id"] = run_id or item.get("source_run_id")
        item["updated_at"] = utc_now_iso()
        self.session_store.save_task_recovery(item)
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
    signal = summary.control_signal
    mode = str(signal.get("mode") or "edit")
    return {
        "goal": sanitize_memory_text(summary.goal, limit=1200),
        "task_mode": mode,
        "planning": _planning_projection(signal, current_projection, mode),
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


def _planning_projection(
    control_signal: dict[str, Any],
    current_projection: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    raw = control_signal.get("planning")
    if not isinstance(raw, Mapping):
        raw = current_projection.get("planning") if isinstance(current_projection.get("planning"), Mapping) else {}
    phase = raw.get("phase") if isinstance(raw.get("phase"), str) else ("execution" if mode == "plan" else "none")
    source = raw.get("source") if isinstance(raw.get("source"), str) else "default"
    return {
        "phase": phase,
        "source": source,
        "budget": _copy_mapping_or_none(raw.get("budget")),
        "discovery": _copy_mapping_or_none(raw.get("discovery")),
        "fallback_reason": raw.get("fallbackReason") or raw.get("fallback_reason"),
    }


def _copy_mapping_or_none(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return dict(value)


__all__ = ["TaskRecoveryStore", "build_task_recovery_projection"]
