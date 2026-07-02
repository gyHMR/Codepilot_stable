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
from typing import Literal, cast

from codepilot.extensions.types import ExtensionCommandContext, RegisteredCommand
from codepilot.sessions.history.branching import create_fresh_session
from codepilot.sessions.session import AgentSession
from codepilot.sessions.memory import load_global_memory, render_memory

# 命令来源类型
CommandSource = Literal["builtin", "extension", "skill", "prompt"]
_COMMAND_SOURCES = frozenset({"builtin", "extension", "skill", "prompt"})


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

    def __post_init__(self) -> None:
        self.name = _normalize_command_name(self.name)
        self.description = _require_command_text(
            self.description,
            field_name="description",
        )
        self.source = _normalize_command_source(self.source)

    def to_dict(self) -> dict[str, str]:
        """返回 CLI/Web/RPC 可直接展示的命令元数据。"""

        return {
            "name": self.name,
            "description": self.description,
            "source": self.source,
        }


def _normalize_command_name(value: object) -> str:
    text = _require_command_text(value, field_name="name").lstrip("/")
    if not text:
        raise ValueError("RuntimeCommand.name cannot be empty")
    if any(char.isspace() for char in text):
        raise ValueError("RuntimeCommand.name cannot contain whitespace")
    return text


def _require_command_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"RuntimeCommand.{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"RuntimeCommand.{field_name} cannot be empty")
    return text


def _normalize_command_source(value: object) -> CommandSource:
    source = _require_command_text(value, field_name="source")
    if source not in _COMMAND_SOURCES:
        raise ValueError(f"Unknown RuntimeCommand.source: {value}")
    return cast(CommandSource, source)


@dataclass
class RuntimeCommandResult:
    """命令处理结果。

    Attributes:
        handled: 是否已处理（True 表示匹配到命令，False 表示未匹配）。
        output_lines: 命令输出的文本行列表。
        switched_session: 命令处理器内部创建的新 AgentSession。
        switched_session_id: RuntimeService 注册后的新会话 ID。
    """

    handled: bool
    output_lines: list[str] = field(default_factory=list)
    switched_session: AgentSession | None = None
    switched_session_id: str | None = None


def builtin_commands() -> list[RuntimeCommand]:
    """返回内置命令列表。"""
    return [
        RuntimeCommand(name="help", description="显示可用命令", source="builtin"),
        RuntimeCommand(name="status", description="查看模型、工作区、会话和权限状态", source="builtin"),
        RuntimeCommand(name="mode", description="查看或切换任务模式：read/edit/plan", source="builtin"),
        RuntimeCommand(name="session", description="查看当前会话与叶子节点", source="builtin"),
        RuntimeCommand(name="tree", description="查看当前会话树", source="builtin"),
        RuntimeCommand(name="fork", description="从指定节点分叉新会话", source="builtin"),
        RuntimeCommand(name="new", description="等价于从当前叶子分叉新会话", source="builtin"),
        RuntimeCommand(name="switch", description="切换到指定叶子节点", source="builtin"),
        RuntimeCommand(name="clear", description="清空上下文，创建新会话", source="builtin"),
        RuntimeCommand(name="compact", description="手动触发上下文压缩", source="builtin"),
        RuntimeCommand(name="context", description="查看最近一次上下文编译报告", source="builtin"),
        RuntimeCommand(name="memory", description="查看、添加、提升或删除结构化记忆", source="builtin"),
        RuntimeCommand(name="tools", description="查看当前可用工具", source="builtin"),
        RuntimeCommand(name="model", description="查看当前模型信息", source="builtin"),
        RuntimeCommand(name="usage", description="查看 token 用量和费用", source="builtin"),
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
        command = RuntimeCommand(
            name=cmd.name,
            description=cmd.description or "扩展命令",
            source=cmd.source,
        )
        items[command.name] = command
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
            f"  Mode       : {session.task_mode}",
        ]
        return RuntimeCommandResult(handled=True, output_lines=lines)

    # /mode [read|edit|plan]: 查看或切换任务模式
    if cmd == "/mode":
        if not arg:
            return RuntimeCommandResult(
                handled=True,
                output_lines=[f"task_mode={session.task_mode}"],
            )
        try:
            mode = session.set_task_mode(arg)
        except ValueError as exc:
            return RuntimeCommandResult(
                handled=True,
                output_lines=[str(exc), "usage: /mode read|edit|plan"],
            )
        return RuntimeCommandResult(
            handled=True,
            output_lines=[f"task_mode={mode}"],
        )

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
            output_lines=[
                f"switched leaf -> {session.get_leaf_id()}",
                "Note: message history was restored; Session Memory is not rolled back in V1.",
            ],
        )

    # /compact: legacy command; context is now governed before every model call.
    if cmd == "/compact":
        return RuntimeCommandResult(
            handled=True,
            output_lines=[
                "Manual compaction has been removed.",
                "ContextGovernor now projects a fresh ContextView before every model call.",
            ],
        )

    # /context [items|stale]: 查看最近一次模型调用的上下文治理报告
    if cmd == "/context":
        report = session.latest_context_report
        if report is None:
            return RuntimeCommandResult(
                handled=True,
                output_lines=["No context report yet. Send a prompt first."],
            )
        if arg == "items":
            lines = ["=== Context Sections ==="]
            for section in report.get("sections", []):
                lines.append(
                    f"  {section.get('name')}: "
                    f"{section.get('selected_items', 0)}/{section.get('candidate_items', 0)} items, "
                    f"{section.get('estimated_tokens_after', 0)}/"
                    f"{section.get('budget_tokens', 0)} tokens"
                )
            dropped = report.get("dropped_items", [])
            lines.append(f"  Dropped items: {len(dropped)}")
            return RuntimeCommandResult(handled=True, output_lines=lines)
        if arg == "stale":
            stale = report.get("stale_items", [])
            return RuntimeCommandResult(
                handled=True,
                output_lines=["=== Stale Context ===", *([f"  - {item}" for item in stale] or ["  (none)"])],
            )
        lines = [
            "=== Context ===",
            f"  Context ID       : {report.get('context_id', '')}",
            f"  Repository       : {str(report.get('repository_fingerprint', ''))[:12]}",
            f"  Token budget     : {report.get('total_budget_tokens', 0):,}",
            f"  Estimated before : {report.get('estimated_tokens_before', 0):,}",
            f"  Estimated after  : {report.get('estimated_tokens_after', 0):,}",
            f"  Stale items      : {len(report.get('stale_items', []))}",
            f"  Dropped items    : {len(report.get('dropped_items', []))}",
            "Use /context items or /context stale for details.",
        ]
        return RuntimeCommandResult(handled=True, output_lines=lines)

    # /memory [list|add|promote|forget]
    if cmd == "/memory":
        action, _, value = arg.partition(" ")
        action = action.strip()
        value = value.strip()
        if not action:
            session_records = session.memory_store.load_session()
            project_records = session.memory_store.load_project()
            records = [*session_records, *project_records]
            pinned = load_global_memory(session.workspace_dir)
            return RuntimeCommandResult(
                handled=True,
                output_lines=[
                    "=== Memory ===",
                    f"  Pinned chars     : {len(pinned)}",
                    f"  Session active   : {sum(record.status == 'active' for record in session_records)}",
                    f"  Project active   : {sum(record.status == 'active' for record in project_records)}",
                    f"  Superseded       : {sum(record.status == 'superseded' for record in records)}",
                    f"  Deleted          : {sum(record.status == 'deleted' for record in records)}",
                    "Use /memory list [session|project|correction|experience|deleted], /memory add <text>,",
                    "    /memory promote <id>, or /memory forget <id>.",
                ],
            )
        if action == "list":
            scope = value or "all"
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
            lines = ["=== Memory Records ==="]
            lines.extend(
                f"  {record.id} [{record.scope}/{record.kind}/{record.status}] "
                f"{render_memory(record)[:160]}"
                for record in records
            )
            if len(lines) == 1:
                lines.append("  (none)")
            return RuntimeCommandResult(handled=True, output_lines=lines)
        if action == "add":
            if not value:
                return RuntimeCommandResult(
                    handled=True,
                    output_lines=["usage: /memory add <project knowledge>"],
                )
            record = session.memory_writer.add_project(value)
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
            return RuntimeCommandResult(
                handled=True,
                output_lines=[f"project memory added: {record.id}"],
            )
        if action == "promote":
            if not value:
                return RuntimeCommandResult(
                    handled=True,
                    output_lines=["usage: /memory promote <memory_id>"],
                )
            record = session.memory_writer.promote(value)
            session.store.append_event(
                {
                    "type": "memory_updated",
                    "sessionId": session.session_id,
                    "action": "promote",
                    "memoryId": record.id,
                    "sourceMemoryId": value,
                }
            )
            return RuntimeCommandResult(
                handled=True,
                output_lines=[f"memory promoted: {value} -> {record.id}"],
            )
        if action == "forget":
            if not value:
                return RuntimeCommandResult(
                    handled=True,
                    output_lines=["usage: /memory forget <memory_id>"],
                )
            record = session.memory_store.mark_status(value, "deleted")
            session.store.append_event(
                {
                    "type": "memory_updated",
                    "sessionId": session.session_id,
                    "action": "forget",
                    "memoryId": record.id,
                    "status": "deleted",
                }
            )
            return RuntimeCommandResult(
                handled=True,
                output_lines=[f"memory forgotten: {record.id}"],
            )
        return RuntimeCommandResult(
            handled=True,
            output_lines=["unknown memory action; use /memory for help"],
        )

    # /tools: 查看当前可用工具
    if cmd == "/tools":
        tools = session.agent.state.tools
        if not tools:
            return RuntimeCommandResult(handled=True, output_lines=["(no tools available)"])
        lines: list[str] = ["Available tools:"]
        for tool in tools:
            name = tool.name if hasattr(tool, "name") else str(tool)
            desc = tool.description if hasattr(tool, "description") else ""
            lines.append(f"  - {name}: {desc[:50]}...")
        return RuntimeCommandResult(handled=True, output_lines=lines)

    # /model: 查看当前模型信息
    if cmd == "/model":
        model = session.agent.state.model
        model_id = f"{model.provider}/{model.id}" if model.provider else model.id
        lines = [
            "=== Model ===",
            f"  ID         : {model_id}",
            f"  Provider   : {model.provider}",
            f"  API        : {model.api}",
            f"  Base URL   : {model.base_url}",
            f"  Reasoning  : {model.reasoning}",
            f"  Vision     : {model.capabilities.vision if model.capabilities else False}",
        ]
        return RuntimeCommandResult(handled=True, output_lines=lines)

    # /usage: 查看 token 用量和费用
    if cmd == "/usage":
        usage = session.cumulative_usage
        lines = [
            "=== Usage ===",
            f"  Input tokens  : {usage['input_tokens']:,}",
            f"  Output tokens : {usage['output_tokens']:,}",
            f"  Total tokens  : {usage['total_tokens']:,}",
            f"  Total cost    : ${usage['total_cost']:.4f}",
        ]
        return RuntimeCommandResult(handled=True, output_lines=lines)

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
