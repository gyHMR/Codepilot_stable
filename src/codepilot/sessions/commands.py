from __future__ import annotations

import inspect
import uuid
from typing import Any

from codepilot.protocols.commands import SessionCommandContext, SessionCommandView

from .contracts import SessionCommandIntent, SessionCommandRecord


# ---- session command state helpers ----
"""Session-owned semantic actions used by slash commands.

This module keeps command-facing state views close to the session layer without
making ``SessionRuntime`` grow a public command API or a bag of private command
helpers.
"""

from typing import Any

from codepilot.core.task import TaskMode, ensure_task_mode
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
from .memory import MemoryRecord, load_global_memory, render_memory


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
    if projection is not None and projection.get("current_mode") != normalized:
        projection["current_mode"] = normalized
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
        records = [record for record in records if record.type == scope]
    elif scope in {"deleted", "superseded"}:
        records = [record for record in records if record.status == scope]
    else:
        records = [record for record in records if record.status != "deleted"]
    return [
        {
            "id": record.id,
            "scope": str(record.scope),
            "kind": str(record.type),
            "status": str(record.status),
            "text": render_memory(record),
        }
        for record in records
    ]


def search_memory_records(session: Any, query: str) -> list[dict[str, str]]:
    """Search memory records by id, subject, content, keyword, or path."""

    terms = [part.lower() for part in query.split() if part.strip()]
    if not terms:
        return []
    matches = []
    for record in session.memory_store.all_records():
        haystack = " ".join(
            [
                record.id,
                record.type,
                record.scope,
                record.subject,
                record.predicate,
                record.value,
                record.content,
                " ".join(record.keywords),
                " ".join(record.paths),
            ]
        ).lower()
        if all(term in haystack for term in terms):
            matches.append(record)
    return [
        {
            "id": record.id,
            "scope": str(record.scope),
            "kind": str(record.type),
            "status": str(record.status),
            "text": render_memory(record),
        }
        for record in matches
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
            "kind": record.type,
            "scope": record.scope,
        }
    )
    return record.id


def approve_memory(session: Any, memory_id: str) -> str:
    record = session.memory_store.mark_status(memory_id, "active")
    _record_memory_event(session, "approve", record)
    return record.id


def edit_memory(session: Any, memory_id: str, content: str) -> str:
    record = session.memory_store.get(memory_id)
    if record is None:
        raise ValueError(f"Memory not found: {memory_id}")
    record.content = content
    record.value = content
    record.confidence = "explicit"
    record.source = "user_explicit"
    record.created_by_session_id = record.created_by_session_id or session.session_id
    updated = session.memory_store.update(record)
    _record_memory_event(session, "edit", updated)
    return updated.id


def disable_memory(session: Any, memory_id: str) -> str:
    record = session.memory_store.mark_status(memory_id, "disabled")
    _record_memory_event(session, "disable", record)
    return record.id


def delete_memory(session: Any, memory_id: str) -> str:
    record = session.memory_store.mark_status(memory_id, "deleted")
    _record_memory_event(session, "delete", record)
    return record.id


def supersede_memory(session: Any, memory_id: str, content: str) -> str:
    old = session.memory_store.get(memory_id)
    if old is None:
        raise ValueError(f"Memory not found: {memory_id}")
    old.status = "superseded"
    session.memory_store.update(old)
    new = MemoryRecord(
        id=f"mem_{uuid.uuid4().hex[:12]}",
        type=old.type,
        scope=old.scope,
        subject=old.subject,
        predicate=old.predicate,
        value=content,
        content=content,
        keywords=list(old.keywords),
        paths=list(old.paths),
        status="active",
        source="user_explicit",
        confidence="explicit",
        priority=old.priority,
        created_by_session_id=session.session_id,
        evidence_refs=[f"session:{session.session_id}"],
        supersedes=[old.id],
    )
    created = session.memory_store.update(new)
    _record_memory_event(session, "supersede", created, source_memory_id=old.id)
    return created.id


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


def _record_memory_event(
    session: Any,
    action: str,
    record: MemoryRecord,
    *,
    source_memory_id: str | None = None,
) -> None:
    payload = {
        "type": "memory_updated",
        "sessionId": session.session_id,
        "action": action,
        "memoryId": record.id,
        "kind": record.type,
        "scope": record.scope,
        "status": record.status,
    }
    if source_memory_id:
        payload["sourceMemoryId"] = source_memory_id
    session.store.append_event(payload)


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

# ---- command router ----


async def apply_session_command(
    session_id: str,
    intent: SessionCommandIntent,
    *,
    session: Any | None = None,
    controller: Any | None = None,
) -> SessionCommandRecord:
    text = intent.text
    cmd, _, rest = text.partition(" ")
    arg = rest.strip()

    if session is None:
        raise ValueError("Session command handling requires SessionRuntime")

    if cmd == "/help":
        return _record(
            session_id,
            text,
            output_lines=[_format_help(session)],
        )

    if cmd == "/status":
        model = session.conversation.model
        model_id = f"{model.provider}/{model.id}" if model.provider else model.id
        return _record(
            session_id,
            text,
            output_lines=[
                "=== Status ===",
                f"  Model      : {model_id}",
                f"  Workspace  : {session.workspace_dir}",
                f"  Session    : {session.session_id}",
                f"  Leaf       : {get_leaf_id(session)}",
                f"  Messages   : {len(session.conversation.messages)}",
                f"  Mode       : {session.task_mode}",
            ],
        )

    if cmd == "/mode":
        if not arg:
            return _record(
                session_id,
                text,
                output_lines=[f"task_mode={session.task_mode}"],
                data={"task_mode": session.task_mode},
            )
        try:
            mode = set_task_mode(session, arg)
        except ValueError as exc:
            return _record(
                session_id,
                text,
                output_lines=[str(exc), "usage: /mode read|plan|build"],
            )
        return _record(
            session_id,
            text,
            output_lines=[f"task_mode={mode}"],
            data={"task_mode": mode},
        )

    if cmd == "/session":
        return _record(
            session_id,
            text,
            output_lines=[
                f"session_id={session.session_id} leaf_id={get_leaf_id(session)}"
            ],
            data={
                "session_id": session.session_id,
                "leaf_id": get_leaf_id(session),
            },
        )

    if cmd == "/tree":
        entries = list_entries(session)
        if not entries:
            return _record(
                session_id,
                text,
                output_lines=["(empty)"],
                data={"entries": [], "tree": []},
            )
        lines: list[str] = []
        for item in entries:
            depth = int(item.get("depth", 0))
            prefix = "  " * max(depth, 0)
            leaf_mark = " *" if item.get("is_leaf") else ""
            lines.append(f"{prefix}- {item.get('id')}{leaf_mark}")
        return _record(
            session_id,
            text,
            output_lines=lines,
            data={"entries": entries, "tree": get_session_tree(session)},
        )

    if cmd == "/path":
        if not arg:
            return _record(session_id, text, output_lines=["usage: /path <entry_id>"])
        path = get_entry_path(session, arg)
        return _record(
            session_id,
            text,
            output_lines=[f"path={' -> '.join(path)}"],
            data={"entry_id": arg, "path": path},
        )

    if cmd == "/clear":
        fresh = create_fresh_session(session)
        _stage_derived_controller(controller, fresh)
        return _record(
            session_id,
            text,
            output_lines=[f"context cleared -> new session_id={fresh.session_id}"],
            switched_session_id=fresh.session_id,
            data={"new_session_id": fresh.session_id},
        )

    if cmd in {"/new", "/fork"}:
        from_entry = arg or get_leaf_id(session) or ""
        if not from_entry:
            return _record(session_id, text, output_lines=["cannot resolve source entry"])
        forked = fork_from_entry(session, from_entry)
        _stage_derived_controller(controller, forked)
        return _record(
            session_id,
            text,
            output_lines=[f"forked to session_id={forked.session_id}"],
            switched_session_id=forked.session_id,
            data={
                "from_session_id": session_id,
                "from_entry_id": from_entry,
                "new_session_id": forked.session_id,
            },
        )

    if cmd == "/switch":
        if not arg:
            return _record(session_id, text, output_lines=["usage: /switch <entry_id>"])
        switch_to_entry(session, arg)
        return _record(
            session_id,
            text,
            output_lines=[
                f"switched leaf -> {get_leaf_id(session)}",
                "Note: message history was restored; session memory is not rolled back by entry switching.",
            ],
            data={
                "session_id": session_id,
                "entry_id": arg,
                "path": get_entry_path(session, arg),
            },
        )

    if cmd == "/context":
        return _context_command(session_id, text, session, arg)

    if cmd == "/rollback":
        return _rollback_command(session_id, text, session, arg)

    if cmd == "/memory":
        return _memory_command(session_id, text, session, arg)

    if cmd == "/tools":
        tools = list(intent.tool_catalog)
        if not tools:
            tools = list(getattr(session.conversation, "tools", ()))
        if not tools:
            return _record(session_id, text, output_lines=["(no tools available)"])
        lines: list[str] = ["Available tools:"]
        for tool in tools:
            lines.append(f"  - {_tool_name(tool)}: {_tool_description(tool)[:50]}...")
        return _record(
            session_id,
            text,
            output_lines=lines,
            data={
                "tools": [
                    {"name": _tool_name(tool), "description": _tool_description(tool)}
                    for tool in tools
                ],
            },
        )

    if cmd == "/model":
        model = session.conversation.model
        model_id = f"{model.provider}/{model.id}" if model.provider else model.id
        return _record(
            session_id,
            text,
            output_lines=[
                "=== Model ===",
                f"  ID         : {model_id}",
                f"  Provider   : {model.provider}",
                f"  API        : {model.api}",
                f"  Base URL   : {model.base_url}",
                f"  Reasoning  : {model.reasoning}",
                f"  Vision     : {model.capabilities.vision if model.capabilities else False}",
            ],
            data={
                "model_id": model_id,
                "provider": model.provider,
                "api": model.api,
                "base_url": model.base_url,
                "reasoning": model.reasoning,
                "vision": model.capabilities.vision if model.capabilities else False,
            },
        )

    if cmd == "/usage":
        usage = cumulative_usage(session)
        return _record(
            session_id,
            text,
            output_lines=[
                "=== Usage ===",
                f"  Input tokens  : {usage['input_tokens']:,}",
                f"  Output tokens : {usage['output_tokens']:,}",
                f"  Total tokens  : {usage['total_tokens']:,}",
                f"  Total cost    : ${usage['total_cost']:.4f}",
            ],
            data=usage,
        )

    if cmd == "/exit":
        return _record(session_id, text, output_lines=["Bye."])

    registered = session.extension_commands.get(cmd.strip().lstrip("/"))
    if registered:
        value = registered.handler(
            SessionCommandContext(
                name=registered.name,
                args=[part for part in arg.split(" ") if part],
                raw_text=text,
                session_view=_command_view(session),
            )
        )
        if inspect.isawaitable(value):
            value = await value
        return _record(
            session_id,
            text,
            output_lines=[str(value)] if value else [],
        )

    return SessionCommandRecord(session_id=session_id, command=text, handled=False)


def _context_command(
    session_id: str,
    text: str,
    session: Any,
    arg: str,
) -> SessionCommandRecord:
    view = context_command_view(session, arg)
    if not view.get("available"):
        return _record(
            session_id,
            text,
            output_lines=["No context report yet. Send a prompt first."],
        )
    if view.get("detail") == "items":
        lines = ["=== Context Sections ==="]
        for section in view.get("sections", []):
            lines.append(
                f"  {section.get('name')}: "
                f"{section.get('selected_items', 0)}/{section.get('candidate_items', 0)} items, "
                f"{section.get('estimated_tokens_after', 0)}/"
                f"{section.get('budget_tokens', 0)} tokens"
            )
        lines.append(f"  Dropped items: {view.get('dropped_count', 0)}")
        return _record(session_id, text, output_lines=lines)
    if view.get("detail") == "stale":
        stale = view.get("stale_items", [])
        return _record(
            session_id,
            text,
            output_lines=[
                "=== Stale Context ===",
                *([f"  - {item}" for item in stale] or ["  (none)"]),
            ],
        )
    return _record(
        session_id,
        text,
        output_lines=[
            "=== Context ===",
            f"  Context ID       : {view.get('context_id', '')}",
            f"  Repository       : {view.get('repository_fingerprint', '')}",
            f"  Token budget     : {view.get('total_budget_tokens', 0):,}",
            f"  Estimated before : {view.get('estimated_tokens_before', 0):,}",
            f"  Estimated after  : {view.get('estimated_tokens_after', 0):,}",
            f"  Stale items      : {view.get('stale_count', 0)}",
            f"  Dropped items    : {view.get('dropped_count', 0)}",
            "Use /context items or /context stale for details.",
        ],
    )


def _rollback_command(
    session_id: str,
    text: str,
    session: Any,
    arg: str,
) -> SessionCommandRecord:
    action, _, value = arg.partition(" ")
    action = action.strip()
    value = value.strip()
    if action in {"help", "-h", "--help"}:
        return _record(
            session_id,
            text,
            output_lines=[
                "usage: /rollback [run_id]",
                "       /rollback apply [run_id]",
            ],
        )
    if action == "apply":
        result = rollback_apply_view(session, value or None)
        return _record(session_id, text, output_lines=_format_rollback_result(result))
    plan = rollback_preview_view(session, action or None)
    return _record(session_id, text, output_lines=_format_rollback_plan(plan))


def _memory_command(
    session_id: str,
    text: str,
    session: Any,
    arg: str,
) -> SessionCommandRecord:
    action, _, value = arg.partition(" ")
    action = action.strip()
    value = value.strip()
    if not action:
        summary = memory_summary(session)
        return _record(
            session_id,
            text,
            output_lines=[
                "=== Memory ===",
                f"  Pinned chars     : {summary['pinned_chars']}",
                f"  Session active   : {summary['session_active']}",
                f"  Project active   : {summary['project_active']}",
                f"  Superseded       : {summary['superseded']}",
                f"  Deleted          : {summary['deleted']}",
                "Use /memory list|search|approve|edit|disable|delete|supersede.",
            ],
        )
    if action == "list":
        scope = value or "all"
        records = list_memory_records(session, scope)
        lines = ["=== Memory Records ==="]
        lines.extend(
            f"  {record['id']} [{record['scope']}/{record['kind']}/{record['status']}] "
            f"{record['text'][:160]}"
            for record in records
        )
        if len(lines) == 1:
            lines.append("  (none)")
        return _record(session_id, text, output_lines=lines)
    if action == "search":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory search <query>"])
        records = search_memory_records(session, value)
        lines = ["=== Memory Search ==="]
        lines.extend(
            f"  {record['id']} [{record['scope']}/{record['kind']}/{record['status']}] "
            f"{record['text'][:160]}"
            for record in records
        )
        if len(lines) == 1:
            lines.append("  (none)")
        return _record(session_id, text, output_lines=lines)
    if action == "add":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory add <project knowledge>"])
        memory_id = add_project_memory(session, value)
        return _record(session_id, text, output_lines=[f"project memory added: {memory_id}"])
    if action == "promote":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory promote <memory_id>"])
        promoted_id = promote_memory(session, value)
        return _record(session_id, text, output_lines=[f"memory promoted: {value} -> {promoted_id}"])
    if action == "approve":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory approve <memory_id>"])
        approved_id = approve_memory(session, value)
        return _record(session_id, text, output_lines=[f"memory approved: {approved_id}"])
    if action == "edit":
        memory_id, _, content = value.partition(" ")
        if not memory_id or not content.strip():
            return _record(session_id, text, output_lines=["usage: /memory edit <memory_id> <content>"])
        edited_id = edit_memory(session, memory_id, content.strip())
        return _record(session_id, text, output_lines=[f"memory edited: {edited_id}"])
    if action == "disable":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory disable <memory_id>"])
        disabled_id = disable_memory(session, value)
        return _record(session_id, text, output_lines=[f"memory disabled: {disabled_id}"])
    if action == "delete":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory delete <memory_id>"])
        deleted_id = delete_memory(session, value)
        return _record(session_id, text, output_lines=[f"memory deleted: {deleted_id}"])
    if action == "supersede":
        memory_id, _, content = value.partition(" ")
        if not memory_id or not content.strip():
            return _record(session_id, text, output_lines=["usage: /memory supersede <memory_id> <replacement content>"])
        new_id = supersede_memory(session, memory_id, content.strip())
        return _record(session_id, text, output_lines=[f"memory superseded: {memory_id} -> {new_id}"])
    if action == "forget":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory forget <memory_id>"])
        forgotten_id = forget_memory(session, value)
        return _record(session_id, text, output_lines=[f"memory forgotten: {forgotten_id}"])
    return _record(session_id, text, output_lines=["unknown memory action; use /memory for help"])


def _format_help(session: Any) -> str:
    names = [
        "help",
        "status",
        "mode",
        "session",
        "tree",
        "path",
        "fork",
        "new",
        "switch",
        "clear",
        "context",
        "memory",
        "rollback",
        "tools",
        "model",
        "usage",
        "exit",
    ]
    names.extend(sorted(session.extension_commands))
    return "可用命令：\n" + "\n".join(f"- `/{name}`" for name in names)


def _stage_derived_controller(controller: Any | None, session: Any) -> None:
    if controller is not None and hasattr(controller, "stage_derived_session"):
        controller.stage_derived_session(session)


def _command_view(session: Any) -> SessionCommandView:
    return SessionCommandView(
        session_id=str(session.session_id),
        workspace_dir=str(session.workspace_dir),
        message_count=len(session.conversation.messages),
        task_mode=str(session.task_mode),
        leaf_id=get_leaf_id(session),
    )


def _record(
    session_id: str,
    command: str,
    *,
    output_lines: list[str] | tuple[str, ...],
    switched_session_id: str | None = None,
    data: dict[str, Any] | None = None,
) -> SessionCommandRecord:
    return SessionCommandRecord(
        session_id=session_id,
        command=command,
        handled=True,
        output_lines=tuple(output_lines),
        switched_session_id=switched_session_id,
        data=data or {},
    )


def _tool_name(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if isinstance(tool, dict):
        return str(tool.get("name") or tool.get("id") or "")
    return str(getattr(tool, "name", ""))


def _tool_description(tool: Any) -> str:
    if isinstance(tool, str):
        return tool
    if isinstance(tool, dict):
        return str(tool.get("description") or _tool_name(tool))
    return str(getattr(tool, "description", "") or _tool_name(tool))


def _format_rollback_plan(plan: dict) -> list[str]:
    lines = [
        "=== Rollback preview ===",
        f"  run_id={plan.get('run_id') or '(none)'}",
        f"  status={plan.get('status')}",
    ]
    if plan.get("reason"):
        lines.append(f"  reason={plan.get('reason')}")
    actions = list(plan.get("actions", []))
    if actions:
        lines.append("  actions:")
        for action in actions:
            suffix = f" ({action.get('reason')})" if action.get("reason") else ""
            lines.append(f"    - {action.get('action')} {action.get('path')}{suffix}")
    else:
        lines.append("  actions: (none)")
    ignored_paths = list(plan.get("ignored_paths", []))
    if ignored_paths:
        lines.append("  ignored unrelated changes:")
        lines.extend(f"    - {path}" for path in ignored_paths)
    return lines


def _format_rollback_result(result: dict) -> list[str]:
    lines = [
        "=== Rollback result ===",
        f"  run_id={result.get('run_id') or '(none)'}",
        f"  status={result.get('status')}",
    ]
    if result.get("reason"):
        lines.append(f"  reason={result.get('reason')}")
    restored_paths = list(result.get("restored_paths", []))
    removed_paths = list(result.get("removed_paths", []))
    conflicted_paths = list(result.get("conflicted_paths", []))
    if restored_paths:
        lines.append("  restored:")
        lines.extend(f"    - {path}" for path in restored_paths)
    if removed_paths:
        lines.append("  removed:")
        lines.extend(f"    - {path}" for path in removed_paths)
    if conflicted_paths:
        lines.append("  conflicts:")
        lines.extend(f"    - {path}" for path in conflicted_paths)
    return lines


__all__ = [
    "add_project_memory",
    "apply_session_command",
    "approve_memory",
    "capture_run_rollback_baseline",
    "context_command_view",
    "create_fresh_session",
    "cumulative_usage",
    "delete_memory",
    "disable_memory",
    "edit_memory",
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
    "search_memory_records",
    "supersede_memory",
    "switch_to_entry",
]
