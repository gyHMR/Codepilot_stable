from __future__ import annotations

# 新手导读：render.py 把 CLI 状态和 Agent 事件渲染成人类可读的终端输出。
# 关注点：这里关心显示体验，不改变运行结果。

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
from html import escape
import os
import sys
import time
from typing import Any, Callable, Iterable

from codepilot.protocols import AgentEvent, AssistantMessage, TextContent

OutputFn = Callable[[str], None]


CP_CIRCUIT_MARK = (
    "╭─ C P ─╮\n"
    "│ ╭╮ ╭─ │\n"
    "│ ╰╯ ╰─ │\n"
    "╰─╼╾─╼╾╯"
)

PLAIN_CP_MARK = (
    "+- C P -+\n"
    "| () <- |\n"
    "| [] -> |\n"
    "+-------+"
)

_CLI_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_CLI_TASK_MODES = frozenset({"read", "plan", "build"})


@dataclass(frozen=True)
class CliStartupState:
    """CLI 启动时需要展示的状态信息。"""

    version: str
    model_id: str
    workspace: str
    session_id: str
    permission_mode: str = "workspace-write"
    task_mode: str = "build"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "version",
            _require_cli_startup_text(self.version, field_name="version"),
        )
        object.__setattr__(
            self,
            "model_id",
            _require_cli_startup_text(self.model_id, field_name="model_id"),
        )
        object.__setattr__(
            self,
            "workspace",
            _require_cli_startup_text(self.workspace, field_name="workspace"),
        )
        object.__setattr__(
            self,
            "session_id",
            _require_cli_startup_text(self.session_id, field_name="session_id"),
        )
        object.__setattr__(
            self,
            "permission_mode",
            _ensure_cli_permission_mode(self.permission_mode),
        )
        object.__setattr__(self, "task_mode", _ensure_cli_task_mode(self.task_mode))
        object.__setattr__(self, "warnings", _normalize_warning_texts(self.warnings))


def build_startup_state(
    status: Any,
    warnings: list[str] | None = None,
) -> CliStartupState:
    """Build the CLI startup view model from runtime session status."""

    return CliStartupState(
        version="0.3",
        model_id=status.model_id,
        workspace=status.workspace,
        session_id=status.session_id,
        permission_mode=status.permission_mode,
        task_mode=status.task_mode,
        warnings=tuple(warnings) if warnings is not None else tuple(status.warnings or ()),
    )


def no_color_requested() -> bool:
    """Return True when the environment asks for plain terminal output."""

    return bool(os.getenv("NO_COLOR"))


def create_console(*, no_color: bool | None = None):
    """Create a Rich console with Codepilot's cyber-terminal theme."""

    from rich.console import Console

    return Console(
        theme=create_theme(),
        no_color=no_color_requested() if no_color is None else no_color,
    )


def create_theme():
    """Build the shared Rich theme without importing Rich at module import time."""

    from rich.theme import Theme

    return Theme({
        "brand": "bold #22d3ee",
        "brand.hot": "bold #f0abfc",
        "brand.dim": "#0891b2",
        "panel.border": "#155e75",
        "panel.title": "bold #67e8f9",
        "label": "#94a3b8",
        "value": "#e5e7eb",
        "muted": "#64748b",
        "muted2": "#94a3b8",
        "info": "#7dd3fc",
        "warning": "#fbbf24",
        "error": "bold #fb7185",
        "success": "#86efac",
        "cancelled": "#fbbf24",
        "tool": "#67e8f9",
        "model": "#c084fc",
        "path": "#93c5fd",
    })


def cyber_title(text: str) -> str:
    return f"CP // {text.upper()}"


def render_key_value_panel(
    title: str,
    rows: Iterable[tuple[str, object]],
    *,
    console=None,
    border_style: str = "panel.border",
) -> None:
    """Render a compact two-column panel for human CLI output."""

    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    target = console or create_console()
    table = Table.grid(padding=(0, 2))
    table.add_column(style="label", no_wrap=True)
    table.add_column(style="value")
    for key, value in rows:
        table.add_row(str(key), str(value))
    target.print(
        Panel(
            table,
            title=f"[panel.title]{cyber_title(title)}[/panel.title]",
            border_style=border_style,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def format_plain_panel(title: str, rows: Iterable[tuple[str, object]]) -> list[str]:
    lines = [f"+-- {cyber_title(title)} " + "-" * 28]
    for key, value in rows:
        lines.append(f"| {key:<12} {value}")
    lines.append("+" + "-" * 48)
    return lines


def format_help_text(*, prog: str = "codepilot", no_color: bool | None = None) -> str:
    """Return a branded argparse-compatible help screen."""

    plain = no_color_requested() if no_color is None else no_color
    mark = PLAIN_CP_MARK if plain else CP_CIRCUIT_MARK
    return f"""{mark}

{cyber_title("Command Deck")}

Usage:
  {prog} [options]
  {prog} -p "explain this function"
  {prog} config <init|show|check|explain> [key]
  {prog} rpc

Options:
  -p, --prompt TEXT          Single prompt mode; prints only the assistant reply
  --cwd, --workspace PATH    Workspace directory (default: current directory)
  --resume SESSION_ID        Resume an existing session
  --model PROVIDER/MODEL     Override model for this run
  --permission-mode MODE     read-only | workspace-write | ask
  --task-mode MODE           read | plan | build
  --planning-budget PROFILE  conservative | balanced | wide
  --verbose                  Show debug events and config sources
  --no-color                 Disable colored terminal UI
  --version                  Show version and exit

Commands:
  config                     Manage local model/runtime configuration
  rpc                        Start JSONL RPC mode; no human UI is emitted
"""


def format_config_help_text(*, prog: str = "codepilot config") -> str:
    mark = PLAIN_CP_MARK if no_color_requested() else CP_CIRCUIT_MARK
    return f"""{mark}

{cyber_title("Config Deck")}

Usage:
  {prog} init
  {prog} show
  {prog} check
  {prog} explain <key>

Actions:
  init       Create .codepilot/model.local.json
  show       Display sanitized model and settings
  check      Validate model configuration and credentials
  explain    Show where a runtime config value came from
"""


def format_error_text(message: str, *, usage: str | None = None) -> str:
    lines = [cyber_title("Argument Error"), f"Error: {message}"]
    if usage:
        lines.append("")
        lines.append(usage.strip())
    return "\n".join(lines) + "\n"


def _require_cli_startup_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CliStartupState.{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"CliStartupState.{field_name} cannot be empty")
    return text


def _ensure_cli_permission_mode(value: object) -> str:
    text = _require_cli_startup_text(value, field_name="permission_mode")
    if text not in _CLI_PERMISSION_MODES:
        raise ValueError(f"Unknown CLI permission_mode: {value}")
    return text


def _ensure_cli_task_mode(value: object) -> str:
    text = _require_cli_startup_text(value, field_name="task_mode")
    if text not in _CLI_TASK_MODES:
        raise ValueError(f"Unknown CLI task_mode: {value}")
    return text


def _normalize_warning_texts(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("CliStartupState.warnings must be a sequence of strings")
    warnings: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("CliStartupState.warnings must contain strings")
        text = item.strip()
        if text:
            warnings.append(text)
    return tuple(warnings)


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
            self._console = create_console()
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

    def render_command_output(self, lines: list[str]) -> None:
        """Render slash-command output without changing command semantics."""

        expanded: list[str] = []
        for line in lines:
            expanded.extend(str(line).splitlines() or [""])

        if not self._console:
            for line in expanded:
                self._print(line)
            return

        from rich.text import Text

        for line in expanded:
            stripped = line.strip()
            if stripped.startswith("===") and stripped.endswith("==="):
                title = stripped.strip("= ").strip() or "Status"
                self._console.rule(
                    Text(cyber_title(title), style="bold #67e8f9"),
                    style="#155e75",
                )
            elif stripped.startswith("- `"):
                self._console.print(Text(line, style="#e5e7eb"))
            elif stripped.startswith("Use "):
                self._console.print(Text(line, style="#94a3b8"))
            else:
                self._console.print(line)

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

        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        workspace = self._shorten_tail(state.workspace, 54)
        session_display = self._short_session(state.session_id)

        title = Text(" Codepilot", style="brand")
        title.append(f" {state.version}", style="muted")
        title.append("  cyber engineering console", style="muted")

        mark = Text(CP_CIRCUIT_MARK, style="brand")

        identity = Table.grid(expand=True)
        identity.add_column(ratio=1)
        identity.add_row(Text("Neural workspace online", style="bold #e5e7eb"))
        identity.add_row(Text("Local coding agent // execution deck", style="brand.dim"))

        status = Table.grid(padding=(0, 2))
        status.add_column(style="#94a3b8", no_wrap=True)
        status.add_column()
        status.add_row("Model", Text(self._shorten_tail(state.model_id, 40), style="model"))
        status.add_row("Workspace", Text(workspace, style="path"))
        status.add_row(
            "Permission",
            Text(
                state.permission_mode,
                style="warning" if state.permission_mode == "read-only" else "success",
            ),
        )
        status.add_row("Mode", Text(state.task_mode, style="tool"))
        status.add_row("Session", Text(session_display, style="muted2"))

        quickstart = Table.grid(expand=True)
        quickstart.add_column(ratio=1)
        quickstart.add_row(Text("Command uplink", style="warning"))
        quickstart.add_row(Text("/help       command deck", style="value"))
        quickstart.add_row(Text("/status     telemetry", style="muted2"))
        quickstart.add_row(Text("Ctrl+C      exit / cancel run", style="muted2"))
        quickstart.add_row(Text("Alt+Enter   newline", style="muted2"))

        body = Table.grid(expand=True)
        body.add_column(ratio=2)
        body.add_column(ratio=4)
        body.add_column(ratio=4)
        body.add_row(mark, identity, quickstart)
        body.add_row("", status, "")

        self._console.print()
        self._console.print(
            Panel(
                body,
                title=title,
                border_style="#155e75",
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )

        for warning in state.warnings:
            self.render_status(warning, kind="warning")
        self._console.print()

    def _render_plain_startup(self, state: CliStartupState) -> None:
        """使用纯文本渲染紧凑启动摘要。"""

        workspace = self._shorten_tail(state.workspace, 54)
        model_display = self._shorten_tail(state.model_id, 40)
        session_display = self._short_session(state.session_id)

        self._print()
        for line in PLAIN_CP_MARK.splitlines():
            self._print(f"| {line}")
        self._print(f"+-- Codepilot {state.version} - cyber engineering console " + "-" * 18)
        self._print("| Neural workspace online")
        self._print(f"| Model      {model_display}")
        self._print(f"| Workspace  {workspace}")
        self._print(f"| Permission {state.permission_mode}")
        self._print(f"| Mode       {state.task_mode}")
        self._print(f"| Session    {session_display}")
        self._print("|")
        self._print("| /help command deck   /status telemetry   Ctrl+C exit/cancel")
        self._print("+" + "-" * 72)

        for warning in state.warnings:
            self.render_status(warning, kind="warning")
        self._print()

    def build_toolbar(self, state: CliStartupState) -> str:
        """构建 prompt_toolkit 底栏，保持内容短且可扫读。"""

        model = escape(self._shorten_tail(state.model_id, 28))
        permission = escape(state.permission_mode)
        task_mode = escape(state.task_mode)
        session = escape(self._short_session(state.session_id))
        return (
            f"<b>CP</b>  <b>{model}</b>  |  {permission}  |  {task_mode}  |  {session}"
            "  |  <b>/help</b> deck  |  <b>Ctrl+C</b> cancel  |  <b>Alt+Enter</b> newline"
        )

    def render_status(self, message: str, *, kind: str = "info") -> None:
        """渲染统一的单行状态反馈。"""

        symbols = {
            "info": "◇",
            "success": "◆",
            "warning": "▲",
            "error": "✕",
            "cancelled": "■",
        }
        symbol = symbols.get(kind, symbols["info"])
        if self._console:
            from rich.text import Text

            styles = {
                "info": "info",
                "success": "success",
                "warning": "warning",
                "error": "error",
                "cancelled": "cancelled",
            }
            text = Text(f"{symbol} ", style=styles.get(kind, styles["info"]))
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
        target = self._shorten_tail(self._extract_tool_target(tool_name, args), 64)

        if self._stream_started:
            self._print()
            self._stream_started = False

        if self._console:
            from rich.text import Text

            text = Text()
            text.append("↯ tool ", style="tool")
            text.append(tool_name, style="bold #e5e7eb")
            if target:
                text.append(f"  {target}", style="muted")
            self._console.print(text)
        else:
            self._print(f"[tool] {tool_name}" + (f"  {target}" if target else ""))

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
            text.append("  ", style="muted")
            if status == "cancelled":
                text.append("■ cancelled", style="cancelled")
            elif is_error:
                text.append("✕ error", style="error")
                if error_reason:
                    text.append(f"  {error_reason}", style="muted")
            else:
                text.append("◆ ok", style="success")
                text.append(f"  {elapsed_str}", style="muted")
            self._console.print(text)
        else:
            if status == "cancelled":
                self._print(f"  [cancelled] {elapsed_str}")
            elif is_error:
                suffix = f"  {error_reason}" if error_reason else ""
                self._print(f"  [error]{suffix}")
            else:
                self._print(f"  [ok] {elapsed_str}")

        self._current_tool = None
        self._tool_start_time = 0

    def _handle_approval_required(self, event: AgentEvent) -> None:
        """处理工具审批请求事件。"""

        tool_name = event.get("toolName", "unknown")
        args = event.get("args", {})
        risk_level = event.get("riskLevel", "medium")
        approval_id = str(event.get("approvalId") or "")

        if self._stream_started:
            self._print()
            self._stream_started = False

        target = self._shorten_tail(self._extract_tool_target(tool_name, args), 64)

        if self._console:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            content.append("Tool requests permission:\n", style="warning")
            content.append(f"  {tool_name}", style="bold")
            if target:
                content.append(f" {target}", style="dim")
            content.append(f"\n  Risk level: {risk_level}", style="dim")
            if approval_id:
                content.append(f"\n  Approval: {approval_id}", style="dim")
                content.append(
                    f"\n  /approve {approval_id}    /deny {approval_id}",
                    style="muted",
                )

            panel = Panel(
                content,
                title=f"[warning]{cyber_title('Permission Required')}[/warning]",
                border_style="#fbbf24",
                padding=(0, 1),
            )
            self._console.print(panel)
        else:
            self._print()
            self._print(f"[approval] Tool requests permission: {tool_name} {target}")
            self._print(f"   Risk level: {risk_level}")
            if approval_id:
                self._print(f"   Approval: {approval_id}")
                self._print(f"   /approve {approval_id} or /deny {approval_id}")

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
                content.append(error, style="bold #f87171")
            if provider_response:
                content.append(f"\nProvider response: {provider_response}", style="dim")
            if provider:
                content.append(f"\nProvider: {provider}", style="dim")
            if model:
                content.append(f"\nModel: {model}", style="dim")

            title = Text(cyber_title("Error") + " · ", style="error")
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
        elif event_type == "tool_approval_required":
            approval_id = str(event.get("approvalId") or "")
            tool_name = str(event.get("toolName") or "unknown")
            risk_level = str(event.get("riskLevel") or "unknown")
            self.output(
                f"Approval required for tool {tool_name} "
                f"(risk={risk_level}, approval_id={approval_id})"
            )

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
    "CP_CIRCUIT_MARK",
    "CliStartupState",
    "OutputFn",
    "PLAIN_CP_MARK",
    "SimpleRenderer",
    "TerminalRenderer",
    "build_startup_state",
    "create_console",
    "create_theme",
    "cyber_title",
    "format_config_help_text",
    "format_error_text",
    "format_help_text",
    "format_plain_panel",
    "no_color_requested",
    "render_key_value_panel",
]
