from __future__ import annotations

"""
CLI 终端渲染。

本模块只负责把 Runtime/Agent 事件转换为终端输出：
- 启动摘要；
- 流式文本；
- 工具执行状态；
- 审批提示；
- 错误信息。

它不解析命令行参数、不切换运行模式、不直接操作 Session，也不执行工具。
"""

from dataclasses import dataclass, field
import sys
import time
from typing import Any

from codepilot.core import AgentEvent
from codepilot.protocols import AssistantMessage, TextContent
from codepilot.runtime.service import SessionStatus
from codepilot.runtime.types import OutputFn


@dataclass(frozen=True)
class CliStartupState:
    """CLI 启动时需要展示的状态信息。

    Attributes:
        version: Codepilot 版本号。
        model_id: 当前模型 ID（如 deepseek/deepseek-chat）。
        workspace: 工作区目录路径。
        session_id: 会话 ID（新建或恢复）。
        permission_mode: 权限模式（如 workspace-write、read-only）。
        warnings: 启动警告列表（如凭证缺失、只读模式）。
    """

    version: str
    model_id: str
    workspace: str
    session_id: str
    permission_mode: str = "workspace-write"
    warnings: list[str] = field(default_factory=list)


def build_startup_state(
    status: SessionStatus,
    warnings: list[str] | None = None,
) -> CliStartupState:
    """从 SessionStatus 构建启动状态信息。"""

    return CliStartupState(
        version="0.3",
        model_id=status.model_id,
        workspace=status.workspace,
        session_id=status.session_id,
        permission_mode=status.permission_mode,
        warnings=warnings if warnings is not None else list(status.warnings or []),
    )


class TerminalRenderer:
    """将 AgentEvent 转换为终端输出（使用 rich 库美化）。

    职责：
    - 渲染启动摘要面板
    - 实时输出流式文本（text_delta）
    - 显示工具执行状态（紧凑格式）
    - 渲染错误信息
    - 处理 verbose 模式下的调试信息

    不负责：
    - 修改 Session 状态
    - 执行工具
    - 解析配置
    """

    def __init__(
        self,
        *,
        output: OutputFn | None = None,
        verbose: bool = False,
        use_rich: bool = True,
    ) -> None:
        self.verbose = verbose
        self.use_rich = use_rich
        self._stream_started = False
        self._current_tool: str | None = None
        self._tool_start_time: float = 0

        if use_rich:
            from rich.console import Console
            from rich.theme import Theme

            theme = Theme({
                "info": "cyan",
                "warning": "yellow",
                "error": "bold red",
                "success": "green",
                "tool": "blue",
                "model": "magenta",
                "path": "cyan",
            })
            self._console = Console(theme=theme)
            self._output = None
        else:
            self._console = None
            self._output = output or print

    @property
    def has_rich_console(self) -> bool:
        """当前渲染器是否可以使用 rich console。"""

        return self._console is not None

    def print(self, text: str = "", **kwargs: Any) -> None:
        """输出一行文本，供运行模式层复用。"""

        self._print(text, **kwargs)

    def input(self, prompt: str) -> str:
        """通过 rich console 读取输入。"""

        if self._console is None:
            raise RuntimeError("rich console is not available")
        return self._console.input(prompt)

    def _print(self, text: str = "", **kwargs: Any) -> None:
        """统一输出方法。"""

        if self._console:
            self._console.print(text, **kwargs)
        else:
            self._output(text)

    def render_startup(self, state: CliStartupState) -> None:
        """渲染启动摘要面板。"""

        if self._console:
            self._render_rich_startup(state)
        else:
            self._render_plain_startup(state)

    def _render_rich_startup(self, state: CliStartupState) -> None:
        """使用 rich 渲染启动面板。"""

        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", width=12)
        table.add_column()

        workspace = state.workspace
        if len(workspace) > 45:
            workspace = "..." + workspace[-42:]

        session_display = state.session_id
        if len(session_display) > 12:
            session_display = session_display[:10] + ".."

        table.add_row("Model", Text(state.model_id, style="magenta"))
        table.add_row("Workspace", Text(workspace, style="path"))
        table.add_row("Session", Text(session_display, style="green"))
        table.add_row("Permission", Text(state.permission_mode, style="yellow"))

        panel = Panel(
            table,
            title=f"[bold]Codepilot {state.version}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
        self._console.print(panel)

        for warning in state.warnings:
            self._console.print(f"⚠️  [warning]{warning}[/warning]")

        if state.warnings:
            self._console.print()

        self._console.print("💡 输入 [bold]/help[/bold] 查看命令，[bold]Ctrl+C[/bold] 取消当前任务")
        self._console.print()

    def _render_plain_startup(self, state: CliStartupState) -> None:
        """使用纯文本渲染启动面板。"""

        workspace = state.workspace
        if len(workspace) > 40:
            workspace = "..." + workspace[-37:]

        model_display = state.model_id
        if len(model_display) > 35:
            model_display = model_display[:32] + "..."

        session_display = state.session_id
        if len(session_display) > 10:
            session_display = session_display[:8] + ".."

        self._print()
        self._print(f"╭─ Codepilot {state.version} ─────────────────────────────╮")
        self._print(f"│ Model       {model_display:<35} │")
        self._print(f"│ Workspace   {workspace:<35} │")
        self._print(f"│ Session     {session_display:<35} │")
        self._print(f"│ Permission  {state.permission_mode:<35} │")
        self._print("╰─────────────────────────────────────────────╯")
        self._print()

        for warning in state.warnings:
            self._print(f"Warning: {warning}")
        if state.warnings:
            self._print()

        self._print("Tip: 输入 /help 查看命令，Ctrl+C 取消当前任务")

    def handle_event(self, event: AgentEvent) -> None:
        """处理 Agent 事件，转换为终端输出。"""

        event_type = event.get("type")

        if event_type == "message_update":
            self._handle_text_delta(event)
            return

        if event_type == "tool_execution_start":
            self._handle_tool_start(event)
            return

        if event_type == "tool_execution_end":
            self._handle_tool_end(event)
            return

        if event_type == "tool_approval_required":
            self._handle_approval_required(event)
            return

        if event_type == "error":
            self._handle_error(event)
            return

        if event_type == "model_retry_start" and self.verbose:
            attempt = event.get("attempt", 0)
            max_attempts = event.get("maxAttempts", 0)
            self._print(f"🔄 [info]Retry attempt {attempt}/{max_attempts}[/info]")

        if self.verbose and event_type not in {"turn_start", "turn_end", "agent_start", "agent_end"}:
            self._print(f"[dim]event: {event_type}[/dim]")

    def _handle_text_delta(self, event: AgentEvent) -> None:
        """处理流式文本更新事件。"""

        assistant_event = event.get("assistantMessageEvent") or {}
        if assistant_event.get("type") != "text_delta":
            return

        delta = str(assistant_event.get("delta", ""))
        if not delta:
            return

        if not self._stream_started:
            if self._console:
                self._console.print()
            else:
                self._print()
            self._stream_started = True

        if self._console:
            self._console.print(delta, end="")
        else:
            sys.stdout.write(delta)
            sys.stdout.flush()

    def _handle_tool_start(self, event: AgentEvent) -> None:
        """处理工具执行开始事件。"""

        tool_name = event.get("toolName", "unknown")
        args = event.get("args", {})
        target = self._extract_tool_target(tool_name, args)

        if self._stream_started:
            self._print()
            self._stream_started = False

        if self._console:
            from rich.text import Text

            text = Text()
            text.append("● ", style="bold blue")
            text.append(tool_name, style="bold")
            if target:
                text.append(f" {target}", style="dim")
            self._console.print(text)
        else:
            self._print(f"● {tool_name} {target}")

        self._current_tool = tool_name
        self._tool_start_time = time.time()

    def _handle_tool_end(self, event: AgentEvent) -> None:
        """处理工具执行结束事件。"""

        is_error = event.get("isError", False)
        error_reason = event.get("errorReason")

        elapsed = time.time() - self._tool_start_time if self._tool_start_time else 0
        elapsed_str = f"{elapsed:.1f}s" if elapsed >= 1 else f"{elapsed * 1000:.0f}ms"

        if self._console:
            from rich.text import Text

            text = Text()
            text.append("  ")
            if is_error:
                text.append("✗ ", style="bold red")
                text.append(error_reason or "failed", style="red")
            else:
                text.append("✓ ", style="bold green")
                text.append(f"Completed in {elapsed_str}", style="dim")
            self._console.print(text)
        else:
            if is_error:
                self._print(f"  × {error_reason or 'failed'}")
            else:
                self._print(f"  Completed in {elapsed_str}")

        self._current_tool = None
        self._tool_start_time = 0

    def _handle_approval_required(self, event: AgentEvent) -> None:
        """处理工具审批请求事件。"""

        tool_name = event.get("toolName", "unknown")
        args = event.get("args", {})
        risk_level = event.get("riskLevel", "medium")

        if self._stream_started:
            self._print()
            self._stream_started = False

        target = self._extract_tool_target(tool_name, args)

        if self._console:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            content.append("Tool requests permission:\n", style="bold yellow")
            content.append(f"  {tool_name}", style="bold")
            if target:
                content.append(f" {target}", style="dim")
            content.append(f"\n  Risk level: {risk_level}", style="dim")

            panel = Panel(
                content,
                title="[bold yellow]Approval Required[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
            self._console.print(panel)
        else:
            self._print()
            self._print(f"⚠️  Tool requests permission: {tool_name} {target}")
            self._print(f"   Risk level: {risk_level}")

    def _handle_error(self, event: AgentEvent) -> None:
        """处理错误事件。"""

        error = event.get("error", "unknown error")
        message = event.get("message", "")
        provider = event.get("provider", "")
        model = event.get("model", "")

        if self._stream_started:
            self._print()
            self._stream_started = False

        if self._console:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            content.append(error, style="bold red")
            if message:
                content.append(f"\n{message}")
            if provider:
                content.append(f"\nProvider: {provider}", style="dim")
            if model:
                content.append(f"\nModel: {model}", style="dim")

            panel = Panel(
                content,
                title="[bold red]Error[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
            self._console.print(panel)
        else:
            self._print()
            self._print(f"Error: {error}")
            if message:
                self._print(f"  {message}")
            if provider:
                self._print(f"  Provider: {provider}")
            if model:
                self._print(f"  Model: {model}")

    def _extract_tool_target(self, tool_name: str, args: dict[str, Any]) -> str:
        """提取工具调用的目标信息。"""

        if tool_name in {"Read", "Write", "Edit"}:
            return str(args.get("file_path", ""))
        if tool_name == "Grep":
            pattern = args.get("pattern", "")
            path = args.get("path", "")
            return f'"{pattern}" {path}' if path else f'"{pattern}"'
        if tool_name == "Glob":
            return str(args.get("pattern", ""))
        if tool_name == "Bash":
            cmd = args.get("command", "")
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            return cmd
        return ""

    def render_final(self, final_assistant: AssistantMessage | None) -> None:
        """渲染最终结果。"""

        if not self._stream_started and final_assistant is not None:
            text = self._extract_assistant_text(final_assistant)
            if text:
                self._print(text)
            elif self.verbose:
                self._print("[dim](empty response)[/dim]")

        if self._stream_started:
            self._print()
            self._stream_started = False

        if self.verbose and final_assistant is not None:
            if final_assistant.stop_reason:
                self._print(f"[dim]stop_reason: {final_assistant.stop_reason}[/dim]")
            if final_assistant.error_message:
                self._print(f"[dim]error: {final_assistant.error_message}[/dim]")

    def _extract_assistant_text(self, message: AssistantMessage) -> str:
        """从 AssistantMessage 中提取纯文本内容。"""

        return "".join(
            block.text for block in message.content if isinstance(block, TextContent)
        ).strip()

    def reset(self) -> None:
        """重置渲染器状态。"""

        self._stream_started = False
        self._current_tool = None
        self._tool_start_time = 0


class SimpleRenderer:
    """简单的终端渲染器，用于 print 模式。

    不使用 rich，输出纯文本，适合管道消费。
    """

    def __init__(self, output: OutputFn = print) -> None:
        self.output = output
        self._stream_started = False

    def handle_event(self, event: AgentEvent) -> None:
        """处理事件，只收集文本 delta。"""

        event_type = event.get("type")
        if event_type == "message_update":
            assistant_event = event.get("assistantMessageEvent") or {}
            if assistant_event.get("type") == "text_delta":
                delta = str(assistant_event.get("delta", ""))
                if delta:
                    self.output(delta, end="")
                    self._stream_started = True

    def render_final(self, final_assistant: AssistantMessage | None) -> None:
        """渲染最终结果。"""

        if self._stream_started:
            self.output()
        elif final_assistant is not None:
            text = "".join(
                block.text for block in final_assistant.content
                if isinstance(block, TextContent)
            ).strip()
            if text:
                self.output(text)

    def reset(self) -> None:
        """重置状态。"""

        self._stream_started = False


__all__ = [
    "CliStartupState",
    "SimpleRenderer",
    "TerminalRenderer",
    "build_startup_state",
]
