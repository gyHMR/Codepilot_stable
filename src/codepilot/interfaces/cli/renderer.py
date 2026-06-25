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

from html import escape
import sys
import time
from typing import Any

from codepilot.protocols import AgentEvent, AssistantMessage, TextContent

from .startup import CliStartupState
from .types import OutputFn


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
        self._tool_start_times: dict[str, float] = {}

        if use_rich:
            from rich.console import Console
            from rich.theme import Theme

            theme = Theme({
                "brand": "bold bright_cyan",
                "info": "cyan",
                "muted": "grey62",
                "warning": "yellow",
                "error": "bold red",
                "success": "green",
                "cancelled": "yellow",
                "tool": "bright_cyan",
                "model": "bright_magenta",
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
        """使用 Rich 渲染紧凑启动摘要。"""

        from rich.text import Text

        workspace = self._shorten_tail(state.workspace, 54)
        session_display = self._short_session(state.session_id)

        title = Text("  Codepilot", style="brand")
        title.append(f" {state.version}", style="muted")
        title.append("  Local coding agent", style="muted")
        self._console.print()
        self._console.print(title)

        primary = Text("  ")
        primary.append(self._shorten_tail(state.model_id, 40), style="model")
        primary.append("  ·  ", style="muted")
        primary.append(workspace, style="path")
        self._console.print(primary)

        secondary = Text("  ")
        secondary.append(state.permission_mode, style=self._permission_style(state.permission_mode))
        secondary.append("  ·  ", style="muted")
        secondary.append(session_display, style="muted")
        self._console.print(secondary)

        for warning in state.warnings:
            self.render_status(warning, kind="warning")

        hint = Text("  /help", style="muted")
        hint.append(" commands  ·  Ctrl+C exit / cancel run  ·  Alt+Enter newline", style="muted")
        self._console.print(hint)
        self._console.print()

    def _render_plain_startup(self, state: CliStartupState) -> None:
        """使用纯文本渲染紧凑启动摘要。"""

        workspace = self._shorten_tail(state.workspace, 54)
        model_display = self._shorten_tail(state.model_id, 40)
        session_display = self._short_session(state.session_id)

        self._print()
        self._print(f"  Codepilot {state.version}  Local coding agent")
        self._print(f"  {model_display}  ·  {workspace}")
        self._print(f"  {state.permission_mode}  ·  {session_display}")

        for warning in state.warnings:
            self.render_status(warning, kind="warning")

        self._print("  /help commands  ·  Ctrl+C exit / cancel run  ·  Alt+Enter newline")
        self._print()

    def build_toolbar(self, state: CliStartupState) -> str:
        """构建 prompt_toolkit 底栏，保持内容短且可扫读。"""

        model = escape(self._shorten_tail(state.model_id, 34))
        permission = escape(state.permission_mode)
        return (
            f"<b>{model}</b>  ·  {permission}"
            "  ·  <b>/help</b> commands  ·  <b>Ctrl+C</b> exit / cancel run"
        )

    def render_status(self, message: str, *, kind: str = "info") -> None:
        """渲染统一的单行状态反馈。"""

        symbols = {
            "info": "•",
            "success": "✓",
            "warning": "!",
            "error": "×",
            "cancelled": "■",
        }
        symbol = symbols.get(kind, symbols["info"])
        if self._console:
            from rich.text import Text

            text = Text(f"{symbol} ", style=kind if kind in symbols else "info")
            text.append(message)
            self._console.print(text)
            return
        self._print(f"{symbol} {message}")

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
        target = self._shorten_tail(self._extract_tool_target(tool_name, args), 68)

        if self._stream_started:
            self._print()
            self._stream_started = False

        if self._console:
            from rich.text import Text

            text = Text()
            text.append("◆ ", style="tool")
            text.append(tool_name, style="bold")
            if target:
                text.append(f"  {target}", style="muted")
            self._console.print(text)
        else:
            self._print(f"◆ {tool_name}" + (f"  {target}" if target else ""))

        self._current_tool = tool_name
        self._tool_start_time = time.time()
        tool_call_id = str(event.get("toolCallId", ""))
        if tool_call_id:
            self._tool_start_times[tool_call_id] = self._tool_start_time

    def _handle_tool_end(self, event: AgentEvent) -> None:
        """处理工具执行结束事件。"""

        is_error = event.get("isError", False)
        error_reason = event.get("errorReason")
        status = event.get("status", "error" if is_error else "success")

        tool_call_id = str(event.get("toolCallId", ""))
        started_at = self._tool_start_times.pop(tool_call_id, 0) if tool_call_id else self._tool_start_time
        elapsed = time.time() - started_at if started_at else 0
        elapsed_str = f"{elapsed:.1f}s" if elapsed >= 1 else f"{elapsed * 1000:.0f}ms"

        if self._console:
            from rich.text import Text

            text = Text()
            text.append("  └ ", style="muted")
            if status == "cancelled":
                text.append("cancelled", style="warning")
            elif is_error:
                text.append("failed", style="error")
                if error_reason:
                    text.append(f"  {error_reason}", style="muted")
            else:
                text.append("done", style="success")
                text.append(f"  {elapsed_str}", style="muted")
            self._console.print(text)
        else:
            if status == "cancelled":
                self._print(f"  └ cancelled  {elapsed_str}")
            elif is_error:
                suffix = f"  {error_reason}" if error_reason else ""
                self._print(f"  └ failed{suffix}")
            else:
                self._print(f"  └ done  {elapsed_str}")

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

        target = self._shorten_tail(self._extract_tool_target(tool_name, args), 64)

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
                title="[bold yellow]Permission required[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
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
        error_info = event.get("errorInfo")
        details = getattr(error_info, "details", None)
        if details is None and isinstance(error_info, dict):
            details = error_info.get("details")
        provider_response = details.get("response_text", "") if isinstance(details, dict) else ""

        if self._stream_started:
            self._print()
            self._stream_started = False

        if self._console:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            if message:
                content.append(message)
            else:
                content.append(error, style="error")
            if provider_response:
                content.append(f"\nProvider response: {provider_response}", style="dim")
            if provider:
                content.append(f"\nProvider: {provider}", style="dim")
            if model:
                content.append(f"\nModel: {model}", style="dim")

            title = Text("Error · ", style="error")
            title.append(str(error), style="error")
            panel = Panel(
                content,
                title=title,
                border_style="red",
                padding=(0, 1),
            )
            self._console.print(panel)
        else:
            self._print()
            self._print(f"Error: {error}")
            if message:
                self._print(f"  {message}")
            if provider_response:
                self._print(f"  Provider response: {provider_response}")
            if provider:
                self._print(f"  Provider: {provider}")
            if model:
                self._print(f"  Model: {model}")

    def _extract_tool_target(self, tool_name: str, args: dict[str, Any]) -> str:
        """提取工具调用的目标信息。"""

        normalized_name = tool_name.lower()
        if normalized_name in {"read", "write", "edit"}:
            path = str(args.get("path") or args.get("file_path") or "")
            if normalized_name == "read" and path and args.get("offset") is not None:
                offset = int(args["offset"])
                limit = args.get("limit")
                if limit is not None:
                    return f"{path}:{offset}-{offset + int(limit) - 1}"
                return f"{path}:{offset}"
            return path
        if normalized_name == "ls":
            return str(args.get("path") or ".")
        if normalized_name == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", "")
            return f'"{pattern}" {path}' if path else f'"{pattern}"'
        if normalized_name in {"glob", "find"}:
            pattern = str(args.get("pattern", ""))
            path = str(args.get("path", ""))
            return f"{pattern} {path}".strip()
        if normalized_name == "bash":
            cmd = args.get("command", "")
            if len(cmd) > 50:
                cmd = cmd[:47] + "..."
            return cmd
        return ""

    @staticmethod
    def _shorten_tail(value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return "…" + value[-(max_length - 1):]

    @staticmethod
    def _short_session(session_id: str) -> str:
        return session_id if len(session_id) <= 11 else session_id[:9] + ".."

    @staticmethod
    def _permission_style(permission_mode: str) -> str:
        if permission_mode == "read-only":
            return "warning"
        return "success"

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
        self._tool_start_times.clear()


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
    "SimpleRenderer",
    "TerminalRenderer",
]
