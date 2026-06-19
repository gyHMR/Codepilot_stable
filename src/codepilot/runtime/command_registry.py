from __future__ import annotations

"""
运行时命令注册与处理模块。

管理 CLI/IM 中的斜杠命令（如 /help、/session、/fork 等），
支持内置命令和扩展注册的自定义命令。

内置命令：
- /help: 显示可用命令列表
- /session: 查看当前会话 ID 和叶子节点 ID
- /tree: 查看会话的树形分支结构
- /fork: 从指定节点分叉新会话
- /new: 从当前叶子节点分叉新会话
- /switch: 切换到指定的叶子节点
- /clear: 清空上下文（等价于 /new）
"""

from dataclasses import dataclass, field
import inspect
from typing import Literal

from codepilot.extensions.types import ExtensionCommandContext, RegisteredCommand
from codepilot.sessions.branching import create_fresh_session
from codepilot.sessions.session import AgentSession

# 命令来源类型
CommandSource = Literal["builtin", "extension", "skill", "prompt"]


@dataclass
class RuntimeCommand:
    """运行时命令定义。

    Attributes:
        name: 命令名称（不含 "/" 前缀）。
        description: 命令描述。
        source: 命令来源（builtin/extension/skill/prompt）。
    """

    name: str
    description: str
    source: CommandSource


@dataclass
class RuntimeCommandResult:
    """命令处理结果。

    Attributes:
        handled: 是否已处理（True 表示匹配到命令，False 表示未匹配）。
        output_lines: 命令输出的文本行列表。
        switched_session: 如果命令导致会话切换，此字段包含新的 AgentSession。
    """

    handled: bool
    output_lines: list[str] = field(default_factory=list)
    switched_session: AgentSession | None = None


def builtin_commands() -> list[RuntimeCommand]:
    """返回内置命令列表。"""
    return [
        RuntimeCommand(name="help", description="显示可用命令", source="builtin"),
        RuntimeCommand(name="status", description="查看模型、工作区、会话和权限状态", source="builtin"),
        RuntimeCommand(name="session", description="查看当前会话与叶子节点", source="builtin"),
        RuntimeCommand(name="tree", description="查看当前会话树", source="builtin"),
        RuntimeCommand(name="fork", description="从指定节点分叉新会话", source="builtin"),
        RuntimeCommand(name="new", description="等价于从当前叶子分叉新会话", source="builtin"),
        RuntimeCommand(name="switch", description="切换到指定叶子节点", source="builtin"),
        RuntimeCommand(name="clear", description="清空上下文，创建新会话", source="builtin"),
        RuntimeCommand(name="exit", description="退出 Codepilot", source="builtin"),
    ]


def list_runtime_commands(session: AgentSession) -> list[RuntimeCommand]:
    """列出当前会话的所有可用命令（内置 + 扩展注册）。

    Args:
        session: 当前 AgentSession 实例。

    Returns:
        按来源和名称排序的 RuntimeCommand 列表。
    """
    items: dict[str, RuntimeCommand] = {cmd.name: cmd for cmd in builtin_commands()}
    for cmd in session.extension_commands.values():
        items[cmd.name] = RuntimeCommand(
            name=cmd.name,
            description=cmd.description or "扩展命令",
            source=cmd.source,
        )
    return sorted(items.values(), key=lambda x: (x.source, x.name))


def format_commands_for_help(session: AgentSession) -> str:
    """格式化命令列表为 /help 的输出文本。

    Args:
        session: 当前 AgentSession 实例。

    Returns:
        格式化的命令帮助文本。
    """
    lines: list[str] = ["可用命令："]
    for item in list_runtime_commands(session):
        lines.append(f"- `/{item.name}` [{item.source}] {item.description}")
    return "\n".join(lines)


def resolve_registered_command(session: AgentSession, name: str) -> RegisteredCommand | None:
    """按名称查找扩展注册的命令。

    Args:
        session: 当前 AgentSession 实例。
        name: 命令名称（可带 "/" 前缀，会被自动去除）。

    Returns:
        匹配的 RegisteredCommand；未找到时返回 None。
    """
    return session.extension_commands.get(name.strip().lstrip("/"))


async def handle_runtime_command(session: AgentSession, text: str) -> RuntimeCommandResult:
    """处理斜杠命令（独立于任何接口实现）。

    解析用户输入的斜杠命令并执行对应逻辑。

    Args:
        session: 当前 AgentSession 实例。
        text: 用户输入的完整文本（如 "/fork entry_123"）。

    Returns:
        RuntimeCommandResult，包含是否已处理、输出文本和可能的会话切换。
    """
    cmd, _, rest = text.partition(" ")
    arg = rest.strip()

    # /help: 显示可用命令
    if cmd == "/help":
        return RuntimeCommandResult(handled=True, output_lines=[format_commands_for_help(session)])

    # /status: 显示当前状态（模型、工作区、会话、权限）
    if cmd == "/status":
        model = session.agent.state.model
        model_id = f"{model.provider}/{model.id}" if model.provider else model.id
        workspace = str(session.workspace_dir)
        session_id = session.session_id
        leaf_id = session.get_leaf_id() or "N/A"
        message_count = len(session.messages)

        lines = [
            "=== Status ===",
            f"  Model      : {model_id}",
            f"  Workspace  : {workspace}",
            f"  Session    : {session_id}",
            f"  Leaf       : {leaf_id}",
            f"  Messages   : {message_count}",
            f"  Permission : workspace-write",  # TODO: 从配置读取
        ]
        return RuntimeCommandResult(handled=True, output_lines=lines)

    # /session: 显示当前会话信息
    if cmd == "/session":
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"session_id={session.session_id} leaf_id={session.get_leaf_id()}"],
        )

    # /tree: 显示会话树形结构
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

    # /clear: 清空上下文，创建全新会话
    if cmd == "/clear":
        fresh = create_fresh_session(session)
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"context cleared -> new session_id={fresh.session_id}"],
            switched_session=fresh,
        )

    # /new 或 /fork: 从指定节点分叉新会话
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

    # /switch: 切换到指定的叶子节点
    if cmd == "/switch":
        if not arg:
            return RuntimeCommandResult(handled=True, output_lines=["usage: /switch <entry_id>"])
        session.switch_to_entry(arg)
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"switched leaf -> {session.get_leaf_id()}"],
        )

    # /exit: 退出 CLI（由上层处理实际退出逻辑）
    if cmd == "/exit":
        return RuntimeCommandResult(
            handled=True,
            output_lines=["Bye."],
        )

    # 尝试匹配扩展注册的自定义命令
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

    # 未匹配任何命令：返回明确错误，不发送给模型
    return RuntimeCommandResult(
        handled=True,
        output_lines=[f"Unknown command: {cmd}", "Type /help to see available commands."],
    )
