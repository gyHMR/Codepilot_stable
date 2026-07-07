from __future__ import annotations

"""Terminal rendering helpers for the CLI adapter."""

from dataclasses import dataclass, field
from html import escape
import os
import sys
import time
from typing import Any, Callable, Iterable

from codepilot.protocols import AgentEvent, AssistantMessage, TextContent

OutputFn = Callable[..., None]


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
    """Small view model for the startup panel."""

    version: str
    model_id: str
    workspace: str
    session_id: str
    permission_mode: str = "workspace-write"
    task_mode: str = "build"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _require_text(self.version, "version"))
        object.__setattr__(self, "model_id", _require_text(self.model_id, "model_id"))
        object.__setattr__(self, "workspace", _require_text(self.workspace, "workspace"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "permission_mode", _ensure_permission_mode(self.permission_mode))
        object.__setattr__(self, "task_mode", _ensure_task_mode(self.task_mode))
        object.__setattr__(self, "warnings", _normalize_warnings(self.warnings))


def build_startup_state(status: Any, warnings: list[str] | None = None) -> CliStartupState:
    """Build the CLI startup view model from a runtime status view."""

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


def create_console(*, no_color: bool | None = None):
    """Create a Rich console with the Codepilot theme."""

    from rich.console import Console

    return Console(
        theme=create_theme(),
        no_color=no_color_requested() if no_color is None else no_color,
    )


def cyber_title(text: str) -> str:
    return f"CP // {text.upper()}"


def render_key_value_panel(
    title: str,
    rows: Iterable[tuple[str, object]],
    *,
    console=None,
    border_style: str = "panel.border",
) -> None:
    """Render a compact two-column panel."""

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


class TerminalRenderer:
    """Render runtime progress and results for the human terminal."""

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
        self._activity_started = False
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
        return self._console is not None

    def input(self, prompt: str) -> str:
        if self._console is None:
            raise RuntimeError("rich console is not available")
        return self._console.input(prompt)

    def print(self, text: str = "", **kwargs: Any) -> None:
        self._print(text, **kwargs)

    def render_startup(self, state: CliStartupState) -> None:
        if self._console:
            self._render_rich_startup(state)
        else:
            self._render_plain_startup(state)

    def render_status(self, message: str, *, kind: str = "info") -> None:
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

    def render_activity(self, message: str = "thinking") -> None:
        if self._activity_started or self._stream_started:
            return
        self._activity_started = True
        if self._console:
            from rich.text import Text

            text = Text("◇ ", style="info")
            text.append(message, style="muted2")
            text.append(" ...", style="muted")
            self._console.print(text)
            return
        self._print(f"◇ {message} ...")

    def render_progress_event(self, event: AgentEvent) -> None:
        event_type = event.get("type")
        if event_type == "message_update":
            self._render_text_delta(event)
            return
        if event_type == "tool_execution_start":
            self._render_tool_start(event)
            return
        if event_type == "tool_execution_end":
            self._render_tool_end(event)
            return
        if event_type == "error":
            self._render_error_event(event)
            return
        if event_type == "model_retry_start" and self.verbose:
            attempt = event.get("attempt", 0)
            max_attempts = event.get("maxAttempts", 0)
            self._print(f"retry attempt {attempt}/{max_attempts}")
        elif self.verbose and event_type not in {"turn_start", "turn_end", "agent_start", "agent_end"}:
            self._print(f"event: {event_type}")

    def render_approval_required(self, frame: Any) -> None:
        approval = frame.approval
        risk = getattr(approval, "risk", None)
        self._render_approval(
            tool_name=str(getattr(approval, "tool_name", "unknown")),
            args=dict(getattr(approval, "arguments", {}) or {}),
            risk_level=str(getattr(risk, "level", "unknown")),
            approval_id=str(getattr(approval, "approval_id", "")),
        )

    def render_command_output(self, lines: Iterable[str]) -> None:
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

    def render_final(self, record: Any | None) -> None:
        message = _final_message_from_record(record)
        if not self._stream_started and message is not None:
            text = _assistant_text(message)
            if text:
                self._activity_started = False
                if self._console:
                    from rich.text import Text

                    self._console.print()
                    self._console.print(Text("CP // ASSISTANT", style="brand.dim"))
                else:
                    self._print()
                    self._print("CP // ASSISTANT")
                self._print(text)
            elif self.verbose:
                self._print("(empty response)")

        if self._stream_started:
            self._print()
            self._stream_started = False
        self._activity_started = False

        if self.verbose and message is not None:
            stop_reason = getattr(message, "stop_reason", None)
            error_message = getattr(message, "error_message", None)
            if stop_reason:
                self._print(f"stop_reason: {stop_reason}")
            if error_message:
                self._print(f"error: {error_message}")

    def reset(self) -> None:
        self._stream_started = False
        self._activity_started = False
        self._current_tool = None
        self._tool_start_time = 0
        self._tool_start_times.clear()

    def build_toolbar(self, state: CliStartupState) -> str:
        model = escape(self._shorten_tail(state.model_id, 28))
        permission = escape(state.permission_mode)
        task_mode = escape(state.task_mode)
        session = escape(self._short_session(state.session_id))
        return (
            f"<b>CP</b>  <b>{model}</b>  |  {permission}  |  {task_mode}  |  {session}"
            "  |  <b>/help</b> deck  |  <b>Ctrl+C</b> cancel  |  <b>Alt+Enter</b> newline"
        )

    def build_shell_prompt(self) -> str:
        return "<prompt>╭─ YOU</prompt>\n<prompt>╰─› </prompt>"

    def build_console_prompt(self) -> str:
        return "[bold bright_cyan]╭─ YOU[/bold bright_cyan]\n[bold bright_cyan]╰─›[/bold bright_cyan] "

    def build_plain_prompt(self) -> str:
        return "╭─ YOU\n╰─› "

    def _render_rich_startup(self, state: CliStartupState) -> None:
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
        self._print()
        for line in PLAIN_CP_MARK.splitlines():
            self._print(f"| {line}")
        self._print(f"+-- Codepilot {state.version} - cyber engineering console " + "-" * 18)
        self._print("| Neural workspace online")
        self._print(f"| Model      {self._shorten_tail(state.model_id, 40)}")
        self._print(f"| Workspace  {self._shorten_tail(state.workspace, 54)}")
        self._print(f"| Permission {state.permission_mode}")
        self._print(f"| Mode       {state.task_mode}")
        self._print(f"| Session    {self._short_session(state.session_id)}")
        self._print("|")
        self._print("| /help command deck   /status telemetry   Ctrl+C exit/cancel")
        self._print("+" + "-" * 72)
        for warning in state.warnings:
            self.render_status(warning, kind="warning")
        self._print()

    def _render_text_delta(self, event: AgentEvent) -> None:
        assistant_event = event.get("assistantMessageEvent") or {}
        delta = str(assistant_event.get("delta", event.get("delta", "")))
        if not delta:
            return
        if not self._stream_started:
            self._activity_started = False
            if self._console:
                from rich.text import Text

                self._console.print()
                self._console.print(Text("CP // ASSISTANT", style="brand.dim"))
            else:
                self._print()
                self._print("CP // ASSISTANT")
            self._stream_started = True
        if self._console:
            self._console.print(delta, end="")
        else:
            sys.stdout.write(delta)
            sys.stdout.flush()

    def _render_tool_start(self, event: AgentEvent) -> None:
        self._activity_started = False
        tool_name = str(event.get("toolName", "unknown"))
        args = dict(event.get("args", {}) or {})
        target = self._shorten_tail(self._extract_tool_target(tool_name, args), 64)
        action = self._tool_action_label(tool_name)
        if self._stream_started:
            self._print()
            self._stream_started = False
        if self._console:
            from rich.text import Text

            text = Text()
            text.append("↯ ", style="tool")
            text.append(action, style="tool")
            text.append(" ")
            text.append(tool_name, style="bold #e5e7eb")
            if target:
                text.append(f"  {target}", style="muted")
            self._console.print(text)
        else:
            self._print(f"[tool] {action} {tool_name}" + (f"  {target}" if target else ""))
        self._current_tool = tool_name
        self._tool_start_time = time.time()
        tool_call_id = str(event.get("toolCallId", ""))
        if tool_call_id:
            self._tool_start_times[tool_call_id] = self._tool_start_time

    def _render_tool_end(self, event: AgentEvent) -> None:
        status = event.get("status", "error" if event.get("isError", False) else "success")
        if status == "approval_required":
            self._clear_current_tool(event)
            return

        is_error = event.get("isError", False)
        error_reason = event.get("errorReason")
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
        elif status == "cancelled":
            self._print(f"  [cancelled] {elapsed_str}")
        elif is_error:
            suffix = f"  {error_reason}" if error_reason else ""
            self._print(f"  [error]{suffix}")
        else:
            self._print(f"  [ok] {elapsed_str}")
        self._current_tool = None
        self._tool_start_time = 0

    def _render_approval(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        risk_level: str,
        approval_id: str,
    ) -> None:
        self._activity_started = False
        if self._stream_started:
            self._print()
            self._stream_started = False

        target = self._shorten_tail(self._extract_tool_target(tool_name, args), 64)
        if self._console:
            from rich import box
            from rich.columns import Columns
            from rich.panel import Panel
            from rich.text import Text

            summary = Text()
            summary.append("Tool\n", style="label")
            summary.append(tool_name, style="bold value")
            if target:
                summary.append(f"\n{target}", style="muted")
            summary.append("\n\nRisk\n", style="label")
            summary.append(risk_level, style=self._risk_style(risk_level))

            command = Text()
            if approval_id:
                command.append("Approval id\n", style="label")
                command.append(approval_id, style="muted2")
                command.append("\n\n")
                command.append(f"/approve {approval_id}", style="success")
                command.append("    ")
                command.append(f"/deny {approval_id}", style="error")
            else:
                command.append("Approval id missing", style="error")

            self._console.print(
                Panel(
                    Columns(
                        [
                            Panel(summary, border_style="#fbbf24", box=box.ROUNDED),
                            Panel(command, border_style="#155e75", box=box.ROUNDED),
                        ],
                        equal=True,
                        expand=False,
                    ),
                    title=f"[warning]{cyber_title('Permission Required')}[/warning]",
                    border_style="#fbbf24",
                    padding=(0, 1),
                    expand=False,
                )
            )
            return

        self._print()
        self._print("+-- APPROVAL REQUIRED " + "-" * 20)
        self._print(f"| Tool  {tool_name}" + (f"  {target}" if target else ""))
        self._print(f"| Risk  {risk_level}")
        if approval_id:
            self._print(f"| Id    {approval_id}")
            self._print("|")
            self._print(f"| /approve {approval_id}")
            self._print(f"| /deny    {approval_id}")
        self._print("+" + "-" * 40)

    def _render_error_event(self, event: AgentEvent) -> None:
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
        self._activity_started = False
        if self._console:
            from rich.panel import Panel
            from rich.text import Text

            content = Text()
            content.append(message or str(error), style=None if message else "bold #f87171")
            if provider_response:
                content.append(f"\nProvider response: {provider_response}", style="dim")
            if provider:
                content.append(f"\nProvider: {provider}", style="dim")
            if model:
                content.append(f"\nModel: {model}", style="dim")
            title = Text(cyber_title("Error") + " · ", style="error")
            title.append(str(error), style="error")
            self._console.print(Panel(content, title=title, border_style="red", padding=(0, 1)))
            return

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

    def _clear_current_tool(self, event: AgentEvent) -> None:
        self._current_tool = None
        self._tool_start_time = 0
        tool_call_id = str(event.get("toolCallId", ""))
        if tool_call_id:
            self._tool_start_times.pop(tool_call_id, None)

    def _print(self, text: str = "", **kwargs: Any) -> None:
        if self._console:
            self._console.print(text, **kwargs)
        else:
            self._output(text)

    @staticmethod
    def _extract_tool_target(tool_name: str, args: dict[str, Any]) -> str:
        normalized = tool_name.lower()
        if normalized in {"read", "write", "edit"}:
            path = str(args.get("path") or args.get("file_path") or "")
            if normalized == "read" and path and args.get("offset") is not None:
                offset = int(args["offset"])
                limit = args.get("limit")
                return f"{path}:{offset}-{offset + int(limit) - 1}" if limit is not None else f"{path}:{offset}"
            return path
        if normalized == "ls":
            return str(args.get("path") or ".")
        if normalized == "grep":
            pattern = args.get("pattern", "")
            path = args.get("path", "")
            return f'"{pattern}" {path}' if path else f'"{pattern}"'
        if normalized in {"glob", "find"}:
            return f"{args.get('pattern', '')} {args.get('path', '')}".strip()
        if normalized == "bash":
            command = str(args.get("command", ""))
            return command[:47] + "..." if len(command) > 50 else command
        return ""

    @staticmethod
    def _tool_action_label(tool_name: str) -> str:
        normalized = tool_name.lower()
        if normalized in {"write", "edit"}:
            return "writing"
        if normalized in {"bash", "shell"}:
            return "running"
        if normalized in {"read", "ls", "grep", "glob", "find"}:
            return "reading"
        return "using"

    @staticmethod
    def _risk_style(risk_level: str) -> str:
        normalized = risk_level.lower()
        if normalized in {"high", "critical"}:
            return "error"
        if normalized == "medium":
            return "warning"
        return "success"

    @staticmethod
    def _shorten_tail(value: str, max_length: int) -> str:
        return value if len(value) <= max_length else "…" + value[-(max_length - 1):]

    @staticmethod
    def _short_session(session_id: str) -> str:
        return session_id if len(session_id) <= 11 else session_id[:9] + ".."


class SimpleRenderer:
    """Plain renderer for single prompt mode."""

    def __init__(self, output: OutputFn = print) -> None:
        self.output = output
        self._stream_started = False

    def render_activity(self, message: str = "thinking") -> None:
        return None

    def render_progress_event(self, event: AgentEvent) -> None:
        if event.get("type") != "message_update":
            return
        assistant_event = event.get("assistantMessageEvent") or {}
        delta = str(assistant_event.get("delta", event.get("delta", "")))
        if delta:
            self.output(delta, end="")
            self._stream_started = True

    def render_approval_required(self, frame: Any) -> None:
        approval = frame.approval
        risk = getattr(getattr(approval, "risk", None), "level", "unknown")
        self.output(
            "Approval required for tool "
            f"{getattr(approval, 'tool_name', 'unknown')} "
            f"(risk={risk}, approval_id={getattr(approval, 'approval_id', '')})"
        )

    def render_final(self, record: Any | None) -> None:
        message = _final_message_from_record(record)
        if self._stream_started:
            self.output()
            return
        if message is None:
            return
        text = _assistant_text(message)
        if text:
            self.output(text)

    def render_status(self, message: str, *, kind: str = "info") -> None:
        self.output(message)

    def reset(self) -> None:
        self._stream_started = False


def _final_message_from_record(record: Any | None) -> AssistantMessage | None:
    if record is None:
        return None
    if isinstance(record, AssistantMessage):
        return record
    outcome = getattr(record, "outcome", None)
    if outcome is not None:
        message = getattr(outcome, "final_message", None)
        if message is not None:
            return message
    return getattr(record, "final_message", None)


def _assistant_text(message: AssistantMessage) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextContent)).strip()


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"CliStartupState.{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"CliStartupState.{field_name} cannot be empty")
    return text


def _ensure_permission_mode(value: object) -> str:
    text = _require_text(value, "permission_mode")
    if text not in _CLI_PERMISSION_MODES:
        raise ValueError(f"Unknown CLI permission_mode: {value}")
    return text


def _ensure_task_mode(value: object) -> str:
    text = _require_text(value, "task_mode")
    if text not in _CLI_TASK_MODES:
        raise ValueError(f"Unknown CLI task_mode: {value}")
    return text


def _normalize_warnings(value: object) -> tuple[str, ...]:
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
