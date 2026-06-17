from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Literal

from codepilot.extensions.types import ExtensionCommandContext, RegisteredCommand
from codepilot.sessions.branching import create_fresh_session
from codepilot.sessions.session import AgentSession

CommandSource = Literal["builtin", "extension", "skill", "prompt"]


@dataclass
class RuntimeCommand:
    name: str
    description: str
    source: CommandSource


@dataclass
class RuntimeCommandResult:
    handled: bool
    output_lines: list[str] = field(default_factory=list)
    switched_session: AgentSession | None = None


def builtin_commands() -> list[RuntimeCommand]:
    return [
        RuntimeCommand(name="help", description="显示可用命令", source="builtin"),
        RuntimeCommand(name="session", description="查看当前会话与叶子节点", source="builtin"),
        RuntimeCommand(name="tree", description="查看当前会话树", source="builtin"),
        RuntimeCommand(name="fork", description="从指定节点分叉新会话", source="builtin"),
        RuntimeCommand(name="new", description="等价于从当前叶子分叉新会话", source="builtin"),
        RuntimeCommand(name="switch", description="切换到指定叶子节点", source="builtin"),
        RuntimeCommand(name="clear", description="IM 中等价于 /new", source="builtin"),
    ]


def list_runtime_commands(session: AgentSession) -> list[RuntimeCommand]:
    items: dict[str, RuntimeCommand] = {cmd.name: cmd for cmd in builtin_commands()}
    for cmd in session.extension_commands.values():
        items[cmd.name] = RuntimeCommand(
            name=cmd.name,
            description=cmd.description or "扩展命令",
            source=cmd.source,
        )
    return sorted(items.values(), key=lambda x: (x.source, x.name))


def format_commands_for_help(session: AgentSession) -> str:
    lines: list[str] = ["可用命令："]
    for item in list_runtime_commands(session):
        lines.append(f"- `/{item.name}` [{item.source}] {item.description}")
    return "\n".join(lines)


def resolve_registered_command(session: AgentSession, name: str) -> RegisteredCommand | None:
    return session.extension_commands.get(name.strip().lstrip("/"))


async def handle_runtime_command(session: AgentSession, text: str) -> RuntimeCommandResult:
    """Handle a slash command without depending on any interface implementation."""

    cmd, _, rest = text.partition(" ")
    arg = rest.strip()

    if cmd == "/help":
        return RuntimeCommandResult(handled=True, output_lines=[format_commands_for_help(session)])

    if cmd == "/session":
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"session_id={session.session_id} leaf_id={session.get_leaf_id()}"],
        )

    if cmd == "/tree":
        entries = session.list_entries()
        if not entries:
            return RuntimeCommandResult(handled=True, output_lines=["(empty)"])
        lines: list[str] = []
        for item in entries:
            depth = int(item.get("depth", 0))
            prefix = "  " * max(depth, 0)
            leaf_mark = " *" if item.get("is_leaf") else ""
            lines.append(f"{prefix}- {item.get('id')}{leaf_mark}")
        return RuntimeCommandResult(handled=True, output_lines=lines)

    if cmd == "/clear":
        fresh = create_fresh_session(session)
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"context cleared -> new session_id={fresh.session_id}"],
            switched_session=fresh,
        )

    if cmd in {"/new", "/fork"}:
        from_entry = arg or session.get_leaf_id() or ""
        if not from_entry:
            return RuntimeCommandResult(handled=True, output_lines=["cannot resolve source entry"])
        forked = session.fork_from_entry(from_entry)
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"forked to session_id={forked.session_id}"],
            switched_session=forked,
        )

    if cmd == "/switch":
        if not arg:
            return RuntimeCommandResult(handled=True, output_lines=["usage: /switch <entry_id>"])
        session.switch_to_entry(arg)
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"switched leaf -> {session.get_leaf_id()}"],
        )

    reg = resolve_registered_command(session, cmd)
    if reg:
        value = reg.handler(
            ExtensionCommandContext(
                name=reg.name,
                args=[p for p in arg.split(" ") if p],
                raw_text=text,
                session=session,
                message=None,
            )
        )
        if inspect.isawaitable(value):
            value = await value
        return RuntimeCommandResult(
            handled=True,
            output_lines=[str(value)] if value else [],
        )

    return RuntimeCommandResult(handled=False)
