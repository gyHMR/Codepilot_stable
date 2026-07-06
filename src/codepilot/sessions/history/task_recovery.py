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
            "schema_version": 1,
            "task_id": None,
            "raw_user_request": goal,
            "current_mode": "build",
            "approval_state": "none",
            "goal": goal,
            "proposed_plan": None,
            "approved_plan": None,
            "current_step_id": None,
            "steps": [],
            "verification_status": "unknown",
            "evidence_refs": [],
            "blocked_reason": None,
            "recovery_summary": "",
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
        steps = projection.get("steps")
        if isinstance(steps, list) and any(
            isinstance(step, dict)
            and step.get("status") in {"pending", "in_progress", "blocked"}
            for step in steps
        ):
            return projection
        if projection.get("approval_state") in {"proposed", "approved"}:
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
    mode = _clean_task_mode(
        signal.get("mode")
        or current_projection.get("current_mode")
        or "build"
    )
    current_step_id = _optional_text(signal.get("current_step_id"), limit=120)
    steps = _canonical_steps(summary, current_step_id=current_step_id)
    if current_step_id is None:
        current_step = next(
            (step for step in steps if step["status"] == "in_progress"),
            None,
        )
        current_step_id = (
            str(current_step["id"])
            if isinstance(current_step, Mapping) and current_step.get("id")
            else None
        )
    recent_error_code = _optional_text(signal.get("recent_error_code"), limit=120)
    verification_status = _verification_status(summary, recent_error_code)
    raw_user_request = sanitize_memory_text(
        current_projection.get("raw_user_request") or summary.goal,
        limit=1200,
    )
    evidence = _collect_evidence_refs(summary, current_projection)
    return {
        "schema_version": 1,
        "task_id": summary.task_id,
        "raw_user_request": raw_user_request,
        "goal": sanitize_memory_text(summary.goal, limit=1200),
        "current_mode": mode,
        "approval_state": _approval_state(current_projection.get("approval_state")),
        "proposed_plan": _copy_mapping_or_none(current_projection.get("proposed_plan")),
        "approved_plan": _copy_mapping_or_none(current_projection.get("approved_plan")),
        "current_step_id": current_step_id,
        "steps": steps,
        "verification_status": verification_status,
        "evidence_refs": evidence,
        "blocked_reason": recent_error_code or ("blocked_steps" if summary.blocked_steps else None),
        "recovery_summary": _optional_text(
            current_projection.get("recovery_summary"),
            limit=2000,
        )
        or "",
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


def _clean_task_mode(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text in {"read", "plan", "build"} else "build"


def _approval_state(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text in {"none", "proposed", "approved", "rejected"} else "none"


def _optional_text(value: object, *, limit: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).strip().split())
    return text[:limit] if text else None


def _canonical_steps(summary, *, current_step_id: str | None) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_titles(titles: list[str], status: str) -> None:
        for title in titles:
            if title in seen:
                continue
            seen.add(title)
            step_id = f"step_{len(steps) + 1}"
            detail = summary.step_details.get(title, {})
            if not isinstance(detail, Mapping):
                detail = {}
            actual_status = (
                "in_progress"
                if status == "pending" and current_step_id == step_id
                else status
            )
            steps.append(
                {
                    "id": step_id,
                    "title": title,
                    "status": actual_status,
                    "kind": _step_kind(detail.get("kind")),
                    "acceptance": _optional_text(detail.get("acceptance"), limit=300),
                    "verification_hint": _optional_text(
                        detail.get("verification_hint"),
                        limit=300,
                    ),
                    "summary": _optional_text(detail.get("summary"), limit=500),
                    "evidence_refs": _clean_text_list(
                        detail.get("evidence_refs"),
                        limit=120,
                    ),
                }
            )

    add_titles(list(summary.completed_steps), "completed")
    add_titles(list(summary.blocked_steps), "blocked")
    add_titles(list(summary.pending_steps), "pending")
    if current_step_id is None:
        for step in steps:
            if step["status"] == "pending":
                step["status"] = "in_progress"
                break
    return steps


def _step_kind(value: object) -> str:
    text = str(value).strip() if value is not None else ""
    return text if text in {"investigate", "edit", "verify", "summarize", "other"} else "other"


def _verification_status(summary, recent_error_code: str | None) -> str:
    if recent_error_code == "verification_failed":
        return "failed"
    if summary.completion_satisfied:
        return "passed"
    return "unknown"


def _collect_evidence_refs(summary, current_projection: Mapping[str, Any]) -> list[str]:
    refs: list[str] = []
    refs.extend(_clean_text_list(current_projection.get("evidence_refs"), limit=120))
    for attempt in summary.attempts:
        if isinstance(attempt, Mapping):
            refs.extend(_clean_text_list(attempt.get("evidence_refs"), limit=120))
    for change_set in summary.change_sets:
        if isinstance(change_set, Mapping):
            refs.extend(_clean_text_list(change_set.get("verification_refs"), limit=120))
            refs.extend(_clean_text_list(change_set.get("affected_paths"), limit=240))
    return _dedupe(refs)


def _clean_text_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [
        text
        for item in value
        if (text := _optional_text(item, limit=limit))
    ]


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


__all__ = ["TaskRecoveryStore", "build_task_recovery_projection"]
