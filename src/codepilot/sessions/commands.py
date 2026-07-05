from __future__ import annotations

import inspect
from typing import Any

from codepilot.protocols.commands import SessionCommandContext, SessionCommandView

from . import command_state
from .contracts import SessionCommandIntent, SessionCommandRecord


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
        return _simple_command(session_id, intent)

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
                f"  Leaf       : {command_state.get_leaf_id(session)}",
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
            mode = command_state.set_task_mode(session, arg)
        except ValueError as exc:
            return _record(
                session_id,
                text,
                output_lines=[str(exc), "usage: /mode read|edit|plan"],
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
                f"session_id={session.session_id} leaf_id={command_state.get_leaf_id(session)}"
            ],
            data={
                "session_id": session.session_id,
                "leaf_id": command_state.get_leaf_id(session),
            },
        )

    if cmd == "/tree":
        entries = command_state.list_entries(session)
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
            data={"entries": entries, "tree": command_state.get_session_tree(session)},
        )

    if cmd == "/path":
        if not arg:
            return _record(session_id, text, output_lines=["usage: /path <entry_id>"])
        path = command_state.get_entry_path(session, arg)
        return _record(
            session_id,
            text,
            output_lines=[f"path={' -> '.join(path)}"],
            data={"entry_id": arg, "path": path},
        )

    if cmd == "/clear":
        fresh = command_state.create_fresh_session(session)
        _stage_derived_controller(controller, fresh)
        return _record(
            session_id,
            text,
            output_lines=[f"context cleared -> new session_id={fresh.session_id}"],
            switched_session_id=fresh.session_id,
            data={"new_session_id": fresh.session_id},
        )

    if cmd in {"/new", "/fork"}:
        from_entry = arg or command_state.get_leaf_id(session) or ""
        if not from_entry:
            return _record(session_id, text, output_lines=["cannot resolve source entry"])
        forked = command_state.fork_from_entry(session, from_entry)
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
        command_state.switch_to_entry(session, arg)
        return _record(
            session_id,
            text,
            output_lines=[
                f"switched leaf -> {command_state.get_leaf_id(session)}",
                "Note: message history was restored; session memory is not rolled back by entry switching.",
            ],
            data={
                "session_id": session_id,
                "entry_id": arg,
                "path": command_state.get_entry_path(session, arg),
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
        usage = command_state.cumulative_usage(session)
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


def _simple_command(session_id: str, intent: SessionCommandIntent) -> SessionCommandRecord:
    if intent.text == "/status":
        return _record(
            session_id,
            intent.text,
            output_lines=[f"session {session_id} is ready"],
        )
    return SessionCommandRecord(
        session_id=session_id,
        command=intent.text,
        handled=False,
        output_lines=(f"unknown command: {intent.text}",),
    )


def _context_command(
    session_id: str,
    text: str,
    session: Any,
    arg: str,
) -> SessionCommandRecord:
    view = command_state.context_command_view(session, arg)
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
        result = command_state.rollback_apply_view(session, value or None)
        return _record(session_id, text, output_lines=_format_rollback_result(result))
    plan = command_state.rollback_preview_view(session, action or None)
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
        summary = command_state.memory_summary(session)
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
                "Use /memory list [session|project|correction|experience|deleted], /memory add <text>,",
                "    /memory promote <id>, or /memory forget <id>.",
            ],
        )
    if action == "list":
        scope = value or "all"
        records = command_state.list_memory_records(session, scope)
        lines = ["=== Memory Records ==="]
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
        memory_id = command_state.add_project_memory(session, value)
        return _record(session_id, text, output_lines=[f"project memory added: {memory_id}"])
    if action == "promote":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory promote <memory_id>"])
        promoted_id = command_state.promote_memory(session, value)
        return _record(session_id, text, output_lines=[f"memory promoted: {value} -> {promoted_id}"])
    if action == "forget":
        if not value:
            return _record(session_id, text, output_lines=["usage: /memory forget <memory_id>"])
        forgotten_id = command_state.forget_memory(session, value)
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
        leaf_id=command_state.get_leaf_id(session),
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


__all__ = ["apply_session_command"]
