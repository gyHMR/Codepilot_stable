"""实现 Session、Memory、Context 和 rollback 的运行时命令用例。"""

from __future__ import annotations

import inspect
import hashlib
import json
from pathlib import Path
from typing import Any

from codepilot.core.plan import RunMode
from codepilot.protocols import AssistantMessage
from codepilot.protocols.commands import CommandOutcome, SessionCommandContext, SessionCommandView

from codepilot.sessions.contracts import SessionCommandIntent, SessionCommandRecord, SessionOptions
from codepilot.sessions.rollback import (
    GitRollbackBaseline,
    GitRollbackPlan,
    GitRollbackResult,
    capture_git_baseline,
    plan_run_rollback,
    revert_run_changes,
)
from codepilot.sessions.memory import (
    AddMemory,
    ApproveMemory,
    DeleteMemory,
    DisableMemory,
    EditMemory,
    ListMemory,
    MemoryActor,
    MemoryRecord,
)
from codepilot.sessions.service import new_session_id

from .actions import CommandDescriptor


def builtin_commands() -> list[CommandDescriptor]:
    """Return built-in application commands rendered by interfaces."""

    return [
        CommandDescriptor(name="help", description="显示可用命令", source="builtin", group="core"),
        CommandDescriptor(name="status", description="查看模型、工作区、会话、权限和计划摘要", source="builtin", group="core"),
        CommandDescriptor(name="resume", description="列出历史会话或切换到指定会话", source="builtin", usage="/resume [number|session_id]", group="session"),
        CommandDescriptor(name="new", description="创建空白新会话并切换", source="builtin", group="session"),
        CommandDescriptor(name="fork", description="从当前会话复制一份新会话并切换", source="builtin", group="session"),
        CommandDescriptor(name="mode", description="查看或切换运行模式：read/plan/build", source="builtin", usage="/mode [read|plan|build]", group="workflow"),
        CommandDescriptor(name="plan", description="查看、批准、拒绝或清除当前计划", source="builtin", usage="/plan [approve|reject|clear]", group="workflow"),
        CommandDescriptor(name="tools", description="查看当前可用工具", source="builtin", group="workflow"),
        CommandDescriptor(name="model", description="查看当前模型信息", source="builtin", group="system"),
        CommandDescriptor(name="usage", description="查看 token 用量和费用", source="builtin", group="system"),
        CommandDescriptor(name="rollback", description="预览或执行最近一次 run 的 Git 回退", source="builtin", usage="/rollback [apply] [run_id]", group="system"),
        CommandDescriptor(name="memory", description="查看、添加、提升或删除结构化记忆", source="builtin", usage="/memory [list|search|add|approve|edit|disable|delete]", group="system"),
        CommandDescriptor(name="context", description="查看最近一次上下文投影治理报告", source="builtin", group="system"),
        CommandDescriptor(name="exit", description="退出 Codepilot", source="builtin", group="core"),
    ]


def cumulative_usage(session: Any) -> dict[str, Any]:
    total_input = 0
    total_output = 0
    total_tokens = 0
    total_cost = 0.0
    for message in session.conversation.messages:
        if isinstance(message, AssistantMessage):
            total_input += message.usage.input
            total_output += message.usage.output
            total_tokens += message.usage.total_tokens
            total_cost += message.usage.cost.total
    return {
        "input_tokens": total_input,
        "output_tokens": total_output,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
    }


def _set_current_mode(session: Any, mode: RunMode | str) -> RunMode:
    return session.set_current_mode(str(mode))


def list_entry_ids(session: Any) -> list[str]:
    return session.state_service.list_entry_ids(session.session_id)


def list_entries(session: Any) -> list[dict[str, Any]]:
    return session.state_service.list_entries(session.session_id)


def get_leaf_id(session: Any) -> str | None:
    state = session.state_service.get_session(session.session_id)
    return state.leaf_message_id if state is not None else None


def get_entry_path(session: Any, entry_id: str) -> list[str]:
    return session.state_service.get_entry_path(session.session_id, entry_id)


def get_session_tree(session: Any) -> list[dict[str, Any]]:
    return session.state_service.get_session_tree(session.session_id)


def create_fresh_session(session: Any) -> Any:
    return _open_derived_runtime(session, new_session_id())


def fork_from_entry(session: Any, entry_id: str) -> Any:
    target_id = new_session_id()
    session.state_service.fork_session(
        session.session_id,
        target_id,
        from_entry_id=entry_id,
    )
    return _open_derived_runtime(session, target_id)


def switch_to_entry(session: Any, entry_id: str) -> None:
    session.session_state = session.state_service.set_leaf(session.session_id, entry_id)
    session.conversation.set_messages(
        [
            record.message
            for record in session.state_service.load_messages(
                session.session_id,
                leaf_id=entry_id,
            )
        ]
    )
    session.state_service.append_event(
        session.session_id,
        {
            "type": "session_leaf_switched",
            "entry_id": entry_id,
        }
    )


def memory_summary(session: Any) -> dict[str, int]:
    records = session.memory_service.execute(
        ListMemory(),
        _memory_actor(session),
    ).records
    return {
        "active": sum(record.status == "active" for record in records),
        "candidate": sum(record.status == "candidate" for record in records),
        "disabled": sum(record.status == "disabled" for record in records),
        "superseded": sum(record.status == "superseded" for record in records),
        "deleted": sum(record.status == "deleted" for record in records),
    }


def list_memory_records(session: Any, scope: str) -> list[dict[str, str]]:
    records = list(
        session.memory_service.execute(ListMemory(), _memory_actor(session)).records
    )
    if scope in {"project", "user"}:
        records = [record for record in records if record.scope == scope]
    elif scope in {"profile", "feedback", "project", "experience", "reference"}:
        records = [record for record in records if record.type == scope]
    elif scope in {"candidate", "active", "deleted", "superseded", "disabled"}:
        records = [record for record in records if record.status == scope]
    elif scope == "all":
        records = [record for record in records if record.status != "deleted"]
    return [_memory_row(record) for record in records]


def search_memory_records(session: Any, query: str) -> list[dict[str, str]]:
    terms = [part.lower() for part in query.split() if part.strip()]
    if not terms:
        return []
    rows = []
    records = session.memory_service.execute(ListMemory(), _memory_actor(session)).records
    for record in records:
        haystack = " ".join(
            [
                record.id,
                record.type,
                record.scope,
                record.key,
                record.content,
            ]
        ).lower()
        if all(term in haystack for term in terms):
            rows.append(_memory_row(record))
    return rows


def add_project_memory(session: Any, text: str) -> str:
    content = str(text or "").strip()
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    record = session.memory_service.execute(
        AddMemory(
            scope="project",
            type="project",
            key=f"project.note.{digest}",
            content=content,
        ),
        _memory_actor(session),
    ).records[0]
    _record_memory_event(session, "memory_record_created", record)
    return record.id


def approve_memory(session: Any, memory_id: str) -> str:
    record = session.memory_service.execute(
        ApproveMemory(memory_id),
        _memory_actor(session),
    ).records[0]
    _record_memory_event(session, "memory_record_approved", record)
    return record.id


def edit_memory(session: Any, memory_id: str, content: str) -> str:
    record = session.memory_service.execute(
        EditMemory(memory_id, content),
        _memory_actor(session),
    ).records[0]
    _record_memory_event(session, "memory_record_edited", record, source_memory_id=memory_id)
    return record.id


def disable_memory(session: Any, memory_id: str) -> str:
    record = session.memory_service.execute(
        DisableMemory(memory_id),
        _memory_actor(session),
    ).records[0]
    _record_memory_event(session, "memory_record_disabled", record)
    return record.id


def delete_memory(session: Any, memory_id: str) -> str:
    record = session.memory_service.execute(
        DeleteMemory(memory_id),
        _memory_actor(session),
    ).records[0]
    _record_memory_event(session, "memory_record_deleted", record)
    return record.id


def supersede_memory(session: Any, memory_id: str, content: str) -> str:
    record = session.memory_service.execute(
        EditMemory(memory_id, content),
        _memory_actor(session),
    ).records[0]
    _record_memory_event(session, "memory_record_superseded", record, source_memory_id=memory_id)
    return record.id


def context_command_view(session: Any, detail: str) -> dict[str, Any]:
    report = dict(session.context_service.latest_report)
    if not report:
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
    return capture_git_baseline(session.workspace_dir)


def rollback_preview_view(session: Any, run_id: str | None = None) -> dict[str, Any]:
    plan = preview_run_rollback(session, run_id) if run_id else preview_last_run_rollback(session)
    return _rollback_plan_to_view(plan)


def rollback_apply_view(session: Any, run_id: str | None = None) -> dict[str, Any]:
    result = revert_run(session, run_id) if run_id else revert_last_run(session)
    return _rollback_result_to_view(result)


def revert_last_run(session: Any) -> GitRollbackResult:
    plan = preview_last_run_rollback(session)
    if not plan.run_id:
        return GitRollbackResult(
            status="not_eligible",
            run_id="",
            reason=plan.reason or "missing_run_id",
        )
    return revert_run(session, plan.run_id)


def preview_last_run_rollback(session: Any) -> GitRollbackPlan:
    state = session.state_service.load_last_run_view(session.session_id)
    if state is None:
        return GitRollbackPlan(status="not_eligible", run_id="", reason="no_run_results")
    run_id = state.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return GitRollbackPlan(status="not_eligible", run_id="", reason="missing_run_id")
    return preview_run_rollback(session, run_id)


def preview_run_rollback(session: Any, run_id: str) -> GitRollbackPlan:
    try:
        state = session.state_service.load_run_view(session.session_id, run_id)
    except FileNotFoundError:
        return GitRollbackPlan(status="not_eligible", run_id=run_id, reason="missing_run_state")
    return plan_run_rollback(session.workspace_dir, state)


def revert_run(session: Any, run_id: str) -> GitRollbackResult:
    try:
        state = session.state_service.load_run_view(session.session_id, run_id)
    except FileNotFoundError:
        return GitRollbackResult(status="not_eligible", run_id=run_id, reason="missing_run_state")
    result = revert_run_changes(session.workspace_dir, state)
    session.append_event(
        {
            "type": "run_reverted",
            "target_run_id": run_id,
            "status": result.status,
            "reason": result.reason,
            "restored_paths": list(result.restored_paths),
            "removed_paths": list(result.removed_paths),
            "conflicted_paths": list(result.conflicted_paths),
        }
    )
    return result


async def apply_session_command(
    session_id: str,
    intent: SessionCommandIntent,
    *,
    session: Any | None = None,
    controller: Any | None = None,
) -> SessionCommandRecord:
    if session is None:
        raise ValueError("Session command handling requires RuntimeSessionCoordinator")
    text = intent.text
    command, _, rest = text.partition(" ")
    arg = rest.strip()

    if command == "/help":
        return _record(session_id, text, output_lines=[_format_help(session)])
    if command == "/status":
        return _status_record(session_id, text, session)
    if command == "/mode":
        return _mode_record(session_id, text, session, arg)
    if command == "/plan":
        return _plan_record(session_id, text, session, arg)
    if command == "/resume":
        record = _resume_record(session_id, text, session, arg)
        if record.switched_session_id:
            _stage_derived_controller(controller, _open_derived_runtime(session, record.switched_session_id))
        return record
    if command == "/new":
        fresh = create_fresh_session(session)
        _stage_derived_controller(controller, fresh)
        return _record(
            session_id,
            text,
            output_lines=[f"new session -> session_id={fresh.session_id}"],
            switched_session_id=fresh.session_id,
            data={"new_session_id": fresh.session_id},
        )
    if command == "/fork":
        source_entry = get_leaf_id(session) or ""
        if not source_entry:
            return _record(session_id, text, output_lines=["cannot resolve source entry"])
        forked = fork_from_entry(session, source_entry)
        _stage_derived_controller(controller, forked)
        return _record(
            session_id,
            text,
            output_lines=[f"forked session -> session_id={forked.session_id}"],
            switched_session_id=forked.session_id,
            data={"from_session_id": session_id, "from_entry_id": source_entry, "new_session_id": forked.session_id},
        )
    if command == "/context":
        return _context_record(session_id, text, session, arg)
    if command == "/rollback":
        return _rollback_record(session_id, text, session, arg)
    if command == "/memory":
        return _memory_record(session_id, text, session, arg)
    if command == "/tools":
        return _tools_record(session_id, text, session, tuple(intent.tool_catalog))
    if command == "/model":
        return _model_record(session_id, text, session)
    if command == "/usage":
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
    if command == "/exit":
        return _record(session_id, text, output_lines=["Bye."])

    registered = session.extension_commands.get(command.strip().lstrip("/"))
    if registered is not None:
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
        if isinstance(value, CommandOutcome):
            return _record(
                session_id,
                text,
                output_lines=[value.output] if value.output else [],
                data={"prompt": value.prompt} if value.prompt else {},
            )
        return _record(session_id, text, output_lines=[str(value)] if value else [])

    return SessionCommandRecord(session_id=session_id, command=text, handled=False)


def _status_record(session_id: str, text: str, session: Any) -> SessionCommandRecord:
    model = session.conversation.model
    model_id = f"{model.provider}/{model.id}" if model.provider else model.id
    plan = session.plan_summary()
    output_lines = [
        "=== Status ===",
        f"  Model      : {model_id}",
        f"  Workspace  : {session.workspace_dir}",
        f"  Session    : {session.session_id}",
        f"  Leaf       : {get_leaf_id(session)}",
        f"  Messages   : {len(session.conversation.messages)}",
        f"  Mode       : {session.current_mode}",
    ]
    if isinstance(plan, dict):
        output_lines.append(
            f"  Plan       : {plan.get('status')} "
            f"{plan.get('done_items', 0)}/{plan.get('total_items', 0)} "
            f"{plan.get('goal_preview', '')}".rstrip()
        )
    return _record(
        session_id,
        text,
        output_lines=output_lines,
    )


def _mode_record(session_id: str, text: str, session: Any, arg: str) -> SessionCommandRecord:
    if not arg:
        return _record(
            session_id,
            text,
            output_lines=[f"current_mode={session.current_mode}"],
            data={"current_mode": session.current_mode},
        )
    requested_mode = arg.strip().lower()
    plan = session.pending_plan_approval()
    if requested_mode in {"build", "read"} and isinstance(plan, dict):
        return _record(
            session_id,
            text,
            output_lines=[
                f"current_mode={session.current_mode}",
                f"{requested_mode} mode is blocked while a proposed plan is waiting for approval.",
                "Use /plan to review it, /plan approve to execute, or /plan reject to discard it.",
                *_format_plan_lines(plan),
            ],
            data={
                "current_mode": session.current_mode,
                "status": "proposed",
                "blocked": True,
            },
        )
    checkpoint = session.runtime_checkpoint()
    checkpoint_phase = (
        str(checkpoint.get("phase") or "")
        if isinstance(checkpoint, dict)
        else ""
    )
    if checkpoint_phase == "tool_approval":
        return _record(
            session_id,
            text,
            output_lines=[
                f"current_mode={session.current_mode}",
                "Resolve the pending tool approval before switching mode.",
            ],
            data={
                "current_mode": session.current_mode,
                "blocked": True,
                "checkpoint_phase": checkpoint_phase,
            },
        )
    previous_mode = session.current_mode
    try:
        mode = _set_current_mode(session, arg)
    except ValueError as exc:
        return _record(session_id, text, output_lines=[str(exc), "usage: /mode read|plan|build"])
    lines = [f"current_mode={mode}"]
    data: dict[str, Any] = {"current_mode": mode}
    if (
        isinstance(checkpoint, dict)
        and str(checkpoint.get("run_id") or "").strip()
        and mode != previous_mode
    ):
        data.update(
            {
                "continuation_kind": "mode_changed",
                "continuation_run_id": str(checkpoint["run_id"]),
            }
        )
    return _record(
        session_id,
        text,
        output_lines=lines,
        data=data,
    )


def _plan_record(session_id: str, text: str, session: Any, arg: str) -> SessionCommandRecord:
    action, _, _rest = arg.partition(" ")
    action = action.strip().lower()
    if action in {"help", "-h", "--help"}:
        return _record(
            session_id,
            text,
            output_lines=[
                "usage: /plan",
                "       /plan approve",
                "       /plan reject",
                "       /plan clear",
            ],
        )
    if action == "approve":
        before = session.current_plan_state()
        if not isinstance(before, dict):
            return _record(
                session_id,
                text,
                output_lines=["No proposed plan to approve."],
                data={"status": None},
            )
        pending_revision = before.get("pending_revision") is not None
        if before.get("status") == "active" and pending_revision:
            continuation_run_id = session.continuation_run_id()
            state = session.approve_current_plan_revision()
            return _record(
                session_id,
                text,
                output_lines=[
                    "Plan revision approved. Continuing implementation.",
                    *_format_plan_lines(state),
                ],
                data={
                    "status": state.get("status") if isinstance(state, dict) else None,
                    "current_mode": session.current_mode,
                    "continuation_kind": "plan_approved",
                    "continuation_run_id": continuation_run_id,
                },
            )
        if before.get("status") in {"active", "completed"}:
            return _record(
                session_id,
                text,
                output_lines=["Plan already approved.", *_format_plan_lines(before)],
                data={
                    "status": before.get("status"),
                    "current_mode": session.current_mode,
                },
            )
        if before.get("status") != "proposed":
            return _record(
                session_id,
                text,
                output_lines=[
                    f"Cannot approve plan with status={before.get('status')}.",
                    *_format_plan_lines(before),
                ],
                data={"status": before.get("status")},
            )
        continuation_run_id = session.continuation_run_id()
        state = session.approve_current_plan(switch_to_build=True)
        return _record(
            session_id,
            text,
            output_lines=[
                "Plan approved. current_mode=build",
                "Starting implementation from the approved plan.",
                *_format_plan_lines(state),
            ],
            data={
                "status": state.get("status") if isinstance(state, dict) else None,
                "current_mode": session.current_mode,
                "continuation_kind": "plan_approved",
                "continuation_run_id": continuation_run_id,
            },
        )
    if action == "reject":
        before = session.current_plan_state()
        if (
            isinstance(before, dict)
            and before.get("status") == "active"
            and before.get("pending_revision") is not None
        ):
            continuation_run_id = session.continuation_run_id()
            state = session.reject_current_plan_revision()
            return _record(
                session_id,
                text,
                output_lines=[
                    "Plan revision rejected. Continuing the approved plan.",
                    *_format_plan_lines(state),
                ],
                data={
                    "status": state.get("status") if isinstance(state, dict) else None,
                    "current_mode": session.current_mode,
                    "continuation_kind": "plan_approved",
                    "continuation_run_id": continuation_run_id,
                },
            )
        if not isinstance(before, dict) or before.get("status") != "proposed":
            return _record(
                session_id,
                text,
                output_lines=["No proposed plan to reject."],
                data={"status": before.get("status") if isinstance(before, dict) else None},
            )
        state = session.reject_current_plan()
        return _record(
            session_id,
            text,
            output_lines=["Plan rejected.", *_format_plan_lines(state)],
            data={
                "status": state.get("status") if isinstance(state, dict) else None,
                "current_mode": session.current_mode,
            },
        )
    if action in {"clear", "abandon"}:
        before = session.current_plan_state()
        if not isinstance(before, dict):
            return _record(
                session_id,
                text,
                output_lines=["No plan to clear."],
                data={"status": None},
            )
        state = session.abandon_current_plan()
        return _record(
            session_id,
            text,
            output_lines=["Plan cleared.", *_format_plan_lines(state)],
            data={"status": state.get("status") if isinstance(state, dict) else None},
        )
    if action:
        return _record(
            session_id,
            text,
            output_lines=["usage: /plan [approve|reject|clear]"],
        )
    state = session.current_plan_state()
    lines = _format_plan_lines(state)
    if isinstance(state, dict) and state.get("status") == "proposed":
        lines.extend(
            [
                "",
                "Plan is waiting for approval.",
                "Use /plan approve to execute, /plan reject to discard, or type feedback to revise it.",
            ]
        )
    return _record(
        session_id,
        text,
        output_lines=lines,
        data={"status": state.get("status") if isinstance(state, dict) else None},
    )


def _tree_record(session_id: str, text: str, session: Any) -> SessionCommandRecord:
    entries = list_entries(session)
    if not entries:
        return _record(session_id, text, output_lines=["(empty)"], data={"entries": [], "tree": []})
    lines = []
    for item in entries:
        prefix = "  " * max(int(item.get("depth", 0)), 0)
        leaf = " *" if item.get("is_leaf") else ""
        lines.append(f"{prefix}- {item.get('id')}{leaf}")
    return _record(session_id, text, output_lines=lines, data={"entries": entries, "tree": get_session_tree(session)})


def _resume_record(session_id: str, text: str, session: Any, arg: str) -> SessionCommandRecord:
    sessions = _recent_session_summaries(session)
    recent = sessions[:10]
    if not arg:
        lines = ["=== Recent sessions ==="]
        if not recent:
            lines.append("  (none)")
        for index, item in enumerate(recent, start=1):
            marker = "*" if item["session_id"] == session.session_id else " "
            lines.append(
                f"  {index}. {marker} {item['session_id']}  {item.get('updated_at', '')}  "
                f"mode={item.get('mode', 'build')}  plan={item.get('plan') or '-'}  "
                f"{item.get('preview') or '(empty)'}"
            )
        return _record(session_id, text, output_lines=lines, data={"sessions": recent})

    target = _resolve_resume_target(arg, recent)
    if target is None:
        return _record(
            session_id,
            text,
            output_lines=["Session not found. Use /resume to list recent sessions."],
            data={"target": arg},
        )
    return _record(
        session_id,
        text,
        output_lines=[f"resumed session -> session_id={target}"],
        switched_session_id=target,
        data={"session_id": target},
    )


def _context_record(session_id: str, text: str, session: Any, arg: str) -> SessionCommandRecord:
    view = context_command_view(session, arg)
    if not view.get("available"):
        return _record(session_id, text, output_lines=["No context report yet. Send a prompt first."])
    if view.get("detail") == "items":
        lines = ["=== Context Sections ==="]
        for section in view.get("sections", []):
            lines.append(
                f"  {section.get('name')}: {section.get('selected_items', 0)}/"
                f"{section.get('candidate_items', 0)} items, "
                f"{section.get('estimated_tokens_after', 0)}/{section.get('budget_tokens', 0)} tokens"
            )
        lines.append(f"  Dropped items: {view.get('dropped_count', 0)}")
        return _record(session_id, text, output_lines=lines)
    if view.get("detail") == "stale":
        stale = list(view.get("stale_items", []))
        return _record(session_id, text, output_lines=["=== Stale Context ===", *([f"  - {item}" for item in stale] or ["  (none)"])])
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
        ],
    )


def _rollback_record(session_id: str, text: str, session: Any, arg: str) -> SessionCommandRecord:
    action, _, value = arg.partition(" ")
    action = action.strip()
    value = value.strip()
    if action in {"help", "-h", "--help"}:
        return _record(session_id, text, output_lines=["usage: /rollback [run_id]", "       /rollback apply [run_id]"])
    if action == "apply":
        return _record(session_id, text, output_lines=_format_rollback_result(rollback_apply_view(session, value or None)))
    return _record(session_id, text, output_lines=_format_rollback_plan(rollback_preview_view(session, action or None)))


def _memory_record(session_id: str, text: str, session: Any, arg: str) -> SessionCommandRecord:
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
                f"  Active           : {summary['active']}",
                f"  Candidate        : {summary['candidate']}",
                f"  Superseded       : {summary['superseded']}",
                f"  Deleted          : {summary['deleted']}",
                "Use /memory list|search|add|approve|edit|disable|delete|supersede.",
            ],
        )
    if action == "list":
        records = list_memory_records(session, value or "all")
        return _memory_lines(session_id, text, "=== Memory Records ===", records)
    if action == "search":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory search <query>"])
        return _memory_lines(session_id, text, "=== Memory Search ===", search_memory_records(session, value))
    if action == "add":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory add <project knowledge>"])
        memory_id = add_project_memory(session, value)
        return _record(session_id, text, output_lines=[f"project memory added: {memory_id}"])
    if action == "approve":
        return _memory_id_action(session_id, text, value, "approve", lambda mid: approve_memory(session, mid))
    if action == "disable":
        return _memory_id_action(session_id, text, value, "disable", lambda mid: disable_memory(session, mid))
    if action == "delete":
        return _memory_id_action(session_id, text, value, action, lambda mid: delete_memory(session, mid))
    if action in {"edit", "supersede"}:
        memory_id, _, content = value.partition(" ")
        if not memory_id or not content.strip():
            return _record(session_id, text, output_lines=[f"usage: /memory {action} <memory_id> <content>"])
        new_id = edit_memory(session, memory_id, content.strip()) if action == "edit" else supersede_memory(session, memory_id, content.strip())
        verb = "edited" if action == "edit" else "superseded"
        return _record(session_id, text, output_lines=[f"memory {verb}: {new_id}" if action == "edit" else f"memory superseded: {memory_id} -> {new_id}"])
    return _record(session_id, text, output_lines=["unknown memory action; use /memory for help"])


def _tools_record(session_id: str, text: str, session: Any, catalog: tuple[Any, ...]) -> SessionCommandRecord:
    tools = list(catalog) or list(getattr(session.conversation, "tools", ()))
    if not tools:
        return _record(session_id, text, output_lines=["(no tools available)"])
    return _record(
        session_id,
        text,
        output_lines=["Available tools:", *[f"  - {_tool_name(tool)}: {_tool_description(tool)[:50]}..." for tool in tools]],
        data={"tools": [{"name": _tool_name(tool), "description": _tool_description(tool)} for tool in tools]},
    )


def _model_record(session_id: str, text: str, session: Any) -> SessionCommandRecord:
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


def _memory_lines(session_id: str, text: str, title: str, records: list[dict[str, str]]) -> SessionCommandRecord:
    lines = [title]
    lines.extend(
        f"  {record['id']} [{record['scope']}/{record['type']}/{record['status']}] {record['text'][:160]}"
        for record in records
    )
    if len(lines) == 1:
        lines.append("  (none)")
    return _record(session_id, text, output_lines=lines)


def _memory_id_action(
    session_id: str,
    text: str,
    value: str,
    action: str,
    callback: Any,
) -> SessionCommandRecord:
    if not value:
        return _record(session_id, text, output_lines=[f"usage: /memory {action} <memory_id>"])
    memory_id = callback(value)
    verb = "deleted" if action == "delete" else f"{action}d"
    return _record(session_id, text, output_lines=[f"memory {verb}: {memory_id}"])


def _memory_actor(session: Any) -> MemoryActor:
    return MemoryActor(user_id=str(session.session_id))


def _record_memory_event(
    session: Any,
    event_type: str,
    record: MemoryRecord,
    *,
    source_memory_id: str | None = None,
) -> None:
    payload = {
        "type": event_type,
        "memory_id": record.id,
        "memory_type": record.type,
        "scope": record.scope,
        "status": record.status,
    }
    if source_memory_id:
        payload["source_memory_id"] = source_memory_id
    session.append_event(payload)


def _memory_row(record: MemoryRecord) -> dict[str, str]:
    return {
        "id": record.id,
        "scope": str(record.scope),
        "type": str(record.type),
        "status": str(record.status),
        "text": f"{record.key}: {record.content}",
    }


def _rollback_plan_to_view(plan: GitRollbackPlan) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "status": plan.status,
        "reason": plan.reason,
        "actions": [{"path": action.path, "action": action.action, "reason": action.reason} for action in plan.actions],
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


def _format_help(session: Any) -> str:
    commands = [command for command in builtin_commands() if command.visible]
    lines = ["可用命令："]
    for command in commands:
        usage = f" (usage: {command.usage})" if command.usage != f"/{command.name}" else ""
        lines.append(f"- `/{command.name}` - {command.description}{usage}")
    for name in sorted(session.extension_commands):
        lines.append(f"- `/{name}` - extension command")
    return "\n".join(lines)


def _format_plan_lines(state: Any) -> list[str]:
    if not isinstance(state, dict):
        return ["No plan."]
    definition = state.get("definition")
    definition = definition if isinstance(definition, dict) else {}
    lines = [
        "=== Plan ===",
        f"  Plan ID    : {state.get('plan_id', '')}",
        f"  Status     : {state.get('status', '')}",
        f"  Mode       : {state.get('origin', '')}",
        f"  Summary    : {definition.get('summary', '')}",
    ]
    explanation = str(definition.get("explanation") or "").strip()
    if explanation:
        lines.append(f"  Note       : {explanation}")
    for label, key in [
        ("Understanding", "task_understanding"),
        ("Current Impl", "current_implementation"),
        ("Target", "target_design"),
        ("Impact", "impact_scope"),
        ("Verification", "verification_plan"),
    ]:
        value = str(definition.get(key) or "").strip()
        if value:
            lines.append(f"  {label:<11}: {value}")
    risks = definition.get("risks_and_open_questions")
    if isinstance(risks, list) and risks:
        lines.append("  Risks / questions:")
        lines.extend(f"    - {item}" for item in risks if str(item).strip())
    criteria = definition.get("completion_criteria")
    if isinstance(criteria, list) and criteria:
        lines.append("  Completion criteria:")
        lines.extend(f"    - {criterion}" for criterion in criteria if str(criterion).strip())
    items = state.get("steps")
    if isinstance(items, list) and items:
        lines.append("  Items:")
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            lines.append(
                "    "
                f"{index}. [{item.get('status', '')}] {item.get('step', '')}"
            )
    return lines


def _open_derived_runtime(session: Any, session_id: str) -> Any:
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator

    return RuntimeSessionCoordinator(
        SessionOptions(
            model=session.conversation.model,
            workspace_dir=session.workspace_dir,
            system_prompt=session.conversation.system_prompt,
            system_prompt_builder=getattr(session, "_system_prompt_builder", None),
            session_id=session_id,
            thinking_level=session.conversation.thinking_level,
            max_tool_calls_per_turn=session.max_tool_calls_per_turn,
            memory_enabled=session.memory_enabled,
            current_mode=_stored_session_mode(session.workspace_dir, session_id) or session.current_mode,
            planning_budget_profile=session.planning_budget_profile,
            convert_to_llm=session.convert_to_llm,
            get_api_key=session.get_api_key,
            retry_enabled=session.retry_enabled,
            max_retries=session.max_retries,
            retry_base_delay_ms=session.retry_base_delay_ms,
            run_timeout_seconds=session.run_timeout_seconds,
            extension_commands=dict(session.extension_commands),
            before_prompt_hooks=list(session.before_prompt_hooks),
            after_prompt_hooks=list(session.after_prompt_hooks),
            stream_fn=session.stream_fn,
        )
    )


def _recent_session_summaries(session: Any) -> list[dict[str, Any]]:
    root = Path(session.workspace_dir) / ".codepilot" / "sessions"
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for session_dir in root.iterdir():
        if not session_dir.is_dir():
            continue
        meta = _read_json_file(session_dir / "session.json")
        if not isinstance(meta, dict):
            continue
        session_id = str(meta.get("session_id") or session_dir.name)
        items.append(
            {
                "session_id": session_id,
                "updated_at": str(meta.get("updated_at") or ""),
                "mode": str(meta.get("current_mode") or "build"),
                "plan": "",
                "preview": _last_message_preview(session_dir / "messages.jsonl"),
            }
        )
    items.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return items


def _stored_session_mode(workspace_dir: Any, session_id: str) -> str | None:
    meta = _read_json_file(Path(workspace_dir) / ".codepilot" / "sessions" / session_id / "session.json")
    if not isinstance(meta, dict):
        return None
    mode = str(meta.get("current_mode") or "").strip()
    return mode if mode in {"read", "plan", "build"} else None


def _resolve_resume_target(arg: str, sessions: list[dict[str, Any]]) -> str | None:
    value = arg.strip()
    if value.isdigit():
        index = int(value)
        if 1 <= index <= len(sessions):
            return str(sessions[index - 1]["session_id"])
        return None
    for item in sessions:
        session_id = str(item.get("session_id") or "")
        if value == session_id:
            return session_id
    return None


def _plan_summary_text(state: Any) -> str:
    if not isinstance(state, dict):
        return "-"
    status = str(state.get("status") or "").strip()
    if status in {"", "none", "rejected", "abandoned"}:
        return "-"
    items = state.get("items")
    items = items if isinstance(items, list) else []
    total = len([item for item in items if isinstance(item, dict)])
    done = len([
        item
        for item in items
        if isinstance(item, dict) and item.get("status") in {"completed", "done"}
    ])
    return f"{status} {done}/{total}" if total else status


def _last_message_preview(path: Path) -> str:
    if not path.exists():
        return ""
    last: dict[str, Any] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                last = value
    if last is None:
        return ""
    return _short_preview(_message_preview_from_row(last), limit=64)


def _message_preview_from_row(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _read_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _short_preview(value: object, *, limit: int) -> str:
    text = str(value).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _stage_derived_controller(controller: Any | None, session: Any) -> None:
    if controller is not None and hasattr(controller, "stage_derived_session"):
        controller.stage_derived_session(session)


def _command_view(session: Any) -> SessionCommandView:
    return SessionCommandView(
        session_id=str(session.session_id),
        workspace_dir=str(session.workspace_dir),
        message_count=len(session.conversation.messages),
        current_mode=str(session.current_mode),
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


def _format_rollback_plan(plan: dict[str, Any]) -> list[str]:
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
    ignored = list(plan.get("ignored_paths", []))
    if ignored:
        lines.append("  ignored unrelated changes:")
        lines.extend(f"    - {path}" for path in ignored)
    return lines


def _format_rollback_result(result: dict[str, Any]) -> list[str]:
    lines = [
        "=== Rollback result ===",
        f"  run_id={result.get('run_id') or '(none)'}",
        f"  status={result.get('status')}",
    ]
    if result.get("reason"):
        lines.append(f"  reason={result.get('reason')}")
    for label, key in (("restored", "restored_paths"), ("removed", "removed_paths"), ("conflicts", "conflicted_paths")):
        paths = list(result.get(key, []))
        if paths:
            lines.append(f"  {label}:")
            lines.extend(f"    - {path}" for path in paths)
    return lines


__all__ = [
    "add_project_memory",
    "apply_session_command",
    "approve_memory",
    "builtin_commands",
    "capture_run_rollback_baseline",
    "context_command_view",
    "create_fresh_session",
    "cumulative_usage",
    "delete_memory",
    "disable_memory",
    "edit_memory",
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
    "revert_last_run",
    "revert_run",
    "rollback_apply_view",
    "rollback_preview_view",
    "search_memory_records",
    "supersede_memory",
    "switch_to_entry",
]
