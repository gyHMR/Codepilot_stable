from __future__ import annotations

"""Session-owned semantic actions used by slash commands.

This module keeps command-facing state views close to the session layer without
making ``SessionRuntime`` grow a public command API or a bag of private command
helpers.
"""

from typing import Any

from codepilot.core.task_control import TaskMode, ensure_task_mode
from codepilot.protocols import AssistantMessage

from .history.branching import create_fresh_session as branch_create_fresh_session
from .history.branching import fork_session as branch_fork_session
from .history.branching import switch_to_entry as branch_switch_to_entry
from .history.git_rollback import (
    GitRollbackBaseline,
    GitRollbackPlan,
    GitRollbackResult,
    capture_git_baseline,
    plan_run_rollback,
    revert_run_changes,
)
from .memory import load_global_memory, render_memory


def cumulative_usage(session: Any) -> dict[str, Any]:
    """Return cumulative assistant token usage for command output."""

    total_input = 0
    total_output = 0
    total_tokens = 0
    total_cost = 0.0
    for msg in session.conversation.messages:
        if isinstance(msg, AssistantMessage):
            total_input += msg.usage.input
            total_output += msg.usage.output
            total_tokens += msg.usage.total_tokens
            total_cost += msg.usage.cost.total
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }


def set_task_mode(session: Any, mode: TaskMode | str) -> TaskMode:
    """Set the task mode used by future runs in this session."""

    normalized = ensure_task_mode(mode)
    changed = normalized != session.task_mode
    if changed:
        session.task_mode = normalized
        session.conversation.set_task_mode(normalized)
        session.store.append_event(
            {
                "type": "task_mode_changed",
                "sessionId": session.session_id,
                "taskMode": normalized,
            }
        )
    projection = session.task_recovery.load_projection()
    if projection is not None and projection.get("task_mode") != normalized:
        projection["task_mode"] = normalized
        session.task_recovery.save_projection(projection)
    return normalized


def list_entry_ids(session: Any) -> list[str]:
    return session.store.list_entry_ids()


def list_entries(session: Any) -> list[dict[str, Any]]:
    return session.store.list_entries()


def get_leaf_id(session: Any) -> str | None:
    return session.store.get_leaf_id()


def get_entry_path(session: Any, entry_id: str) -> list[str]:
    return session.store.get_entry_path(entry_id)


def get_session_tree(session: Any) -> list[dict[str, Any]]:
    return session.store.get_session_tree()


def create_fresh_session(session: Any) -> Any:
    """Create a sibling session with the same runtime settings and empty history."""

    return branch_create_fresh_session(session)


def fork_from_entry(session: Any, entry_id: str) -> Any:
    return branch_fork_session(session, from_entry_id=entry_id)


def switch_to_entry(session: Any, entry_id: str) -> None:
    branch_switch_to_entry(session, entry_id)


def memory_summary(session: Any) -> dict[str, int]:
    """Return command-facing memory counts without exposing memory stores."""

    session_records = session.memory_store.load_session()
    project_records = session.memory_store.load_project()
    records = [*session_records, *project_records]
    pinned = load_global_memory(session.workspace_dir)
    return {
        "pinned_chars": len(pinned),
        "session_active": sum(record.status == "active" for record in session_records),
        "project_active": sum(record.status == "active" for record in project_records),
        "superseded": sum(record.status == "superseded" for record in records),
        "deleted": sum(record.status == "deleted" for record in records),
    }


def list_memory_records(session: Any, scope: str) -> list[dict[str, str]]:
    """Return formatted memory records for command output."""

    records = [
        *session.memory_store.load_session(),
        *session.memory_store.load_project(),
    ]
    if scope == "session":
        records = [record for record in records if record.scope == "session"]
    elif scope == "project":
        records = [record for record in records if record.scope == "project"]
    elif scope in {"correction", "constraint", "decision", "experience"}:
        records = [record for record in records if record.kind == scope]
    elif scope in {"deleted", "superseded"}:
        records = [record for record in records if record.status == scope]
    else:
        records = [record for record in records if record.status != "deleted"]
    return [
        {
            "id": record.id,
            "scope": str(record.scope),
            "kind": str(record.kind),
            "status": str(record.status),
            "text": render_memory(record),
        }
        for record in records
    ]


def add_project_memory(session: Any, text: str) -> str:
    """Add a durable project memory record through the session memory writer."""

    record = session.memory_writer.add_project(text)
    session.store.append_event(
        {
            "type": "memory_updated",
            "sessionId": session.session_id,
            "action": "add",
            "memoryId": record.id,
            "kind": record.kind,
            "scope": record.scope,
        }
    )
    return record.id


def promote_memory(session: Any, memory_id: str) -> str:
    """Promote active session experience memory into project memory."""

    record = session.memory_writer.promote(memory_id)
    session.store.append_event(
        {
            "type": "memory_updated",
            "sessionId": session.session_id,
            "action": "promote",
            "memoryId": record.id,
            "sourceMemoryId": memory_id,
        }
    )
    return record.id


def forget_memory(session: Any, memory_id: str) -> str:
    """Mark a memory record as deleted."""

    record = session.memory_store.mark_status(memory_id, "deleted")
    session.store.append_event(
        {
            "type": "memory_updated",
            "sessionId": session.session_id,
            "action": "forget",
            "memoryId": record.id,
            "status": "deleted",
        }
    )
    return record.id


def context_command_view(session: Any, detail: str) -> dict[str, Any]:
    """Return the latest context governance report as a command-facing view."""

    report = session.latest_context_report
    if report is None:
        return {"available": False}
    if detail == "items":
        return {
            "available": True,
            "detail": "items",
            "sections": [
                {
                    "name": section.get("name"),
                    "selected_items": section.get("selected_items", 0),
                    "candidate_items": section.get("candidate_items", 0),
                    "estimated_tokens_after": section.get("estimated_tokens_after", 0),
                    "budget_tokens": section.get("budget_tokens", 0),
                }
                for section in report.get("sections", [])
                if isinstance(section, dict)
            ],
            "dropped_count": len(report.get("dropped_items", [])),
        }
    if detail == "stale":
        return {
            "available": True,
            "detail": "stale",
            "stale_items": list(report.get("stale_items", [])),
        }
    return {
        "available": True,
        "detail": "summary",
        "context_id": report.get("context_id", ""),
        "repository_fingerprint": str(report.get("repository_fingerprint", ""))[:12],
        "total_budget_tokens": report.get("total_budget_tokens", 0),
        "estimated_tokens_before": report.get("estimated_tokens_before", 0),
        "estimated_tokens_after": report.get("estimated_tokens_after", 0),
        "stale_count": len(report.get("stale_items", [])),
        "dropped_count": len(report.get("dropped_items", [])),
    }


def capture_run_rollback_baseline(session: Any) -> GitRollbackBaseline:
    """Capture the workspace rollback baseline for a run-owned transaction."""

    return capture_git_baseline(session.workspace_dir)


def rollback_preview_view(session: Any, run_id: str | None = None) -> dict[str, Any]:
    """Return a rollback preview view for command rendering."""

    plan = preview_run_rollback(session, run_id) if run_id else preview_last_run_rollback(session)
    return _rollback_plan_to_view(plan)


def rollback_apply_view(session: Any, run_id: str | None = None) -> dict[str, Any]:
    """Apply rollback and return a command-facing result view."""

    result = revert_run(session, run_id) if run_id else revert_last_run(session)
    return _rollback_result_to_view(result)


def revert_last_run(session: Any) -> GitRollbackResult:
    """Revert the latest run that supports Git clean-worktree rollback."""

    plan = preview_last_run_rollback(session)
    if not plan.run_id:
        return GitRollbackResult(
            status="not_eligible",
            run_id="",
            reason=plan.reason or "missing_run_id",
        )
    return revert_run(session, plan.run_id)


def preview_last_run_rollback(session: Any) -> GitRollbackPlan:
    """Preview rollback for the latest persisted run."""

    runs = session.store.load_run_results(limit=1)
    if not runs:
        return GitRollbackPlan(
            status="not_eligible",
            run_id="",
            reason="no_run_results",
        )
    run_id = runs[-1].get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return GitRollbackPlan(
            status="not_eligible",
            run_id="",
            reason="missing_run_id",
        )
    return preview_run_rollback(session, run_id)


def preview_run_rollback(session: Any, run_id: str) -> GitRollbackPlan:
    """Preview workspace file rollback for a persisted run id."""

    try:
        state = session.store.run_store.load_run_state(run_id)
    except FileNotFoundError:
        return GitRollbackPlan(
            status="not_eligible",
            run_id=run_id,
            reason="missing_run_state",
        )
    return plan_run_rollback(session.workspace_dir, state)


def revert_run(session: Any, run_id: str) -> GitRollbackResult:
    """Revert workspace file changes recorded by a run id."""

    try:
        state = session.store.run_store.load_run_state(run_id)
    except FileNotFoundError:
        return GitRollbackResult(
            status="not_eligible",
            run_id=run_id,
            reason="missing_run_state",
        )
    result = revert_run_changes(session.workspace_dir, state)
    session.store.append_event(
        {
            "type": "run_reverted",
            "sessionId": session.session_id,
            "targetRunId": run_id,
            "status": result.status,
            "reason": result.reason,
            "restoredPaths": list(result.restored_paths),
            "removedPaths": list(result.removed_paths),
            "conflictedPaths": list(result.conflicted_paths),
        }
    )
    return result


def _rollback_plan_to_view(plan: GitRollbackPlan) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "status": plan.status,
        "reason": plan.reason,
        "actions": [
            {
                "path": action.path,
                "action": action.action,
                "reason": action.reason,
            }
            for action in plan.actions
        ],
        "ignored_paths": list(plan.ignored_paths),
    }


def _rollback_result_to_view(result: GitRollbackResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "status": result.status,
        "reason": result.reason,
        "restored_paths": list(result.restored_paths),
        "removed_paths": list(result.removed_paths),
        "conflicted_paths": list(result.conflicted_paths),
    }


__all__ = [
    "add_project_memory",
    "capture_run_rollback_baseline",
    "context_command_view",
    "create_fresh_session",
    "cumulative_usage",
    "forget_memory",
    "fork_from_entry",
    "get_entry_path",
    "get_leaf_id",
    "get_session_tree",
    "list_entries",
    "list_entry_ids",
    "list_memory_records",
    "memory_summary",
    "preview_last_run_rollback",
    "preview_run_rollback",
    "promote_memory",
    "revert_last_run",
    "revert_run",
    "rollback_apply_view",
    "rollback_preview_view",
    "set_task_mode",
    "switch_to_entry",
]
