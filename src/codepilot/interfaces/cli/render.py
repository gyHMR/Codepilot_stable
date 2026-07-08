from __future__ import annotations

"""CLI 终端渲染工具。

本文件只负责“怎么显示”，不负责“业务怎么运行”：

- ``CliStartupState`` 是启动面板使用的轻量 view model。
- ``TerminalRenderer`` 面向交互式 REPL，负责启动面板、流式模型输出、工具状态、审批框等。
- ``SimpleRenderer`` 面向 ``codepilot -p`` 单次模式，只输出模型文本和必要审批提示。
- 顶层 ``format_*`` 函数提供 argparse/config 等场景使用的纯文本或 Rich 面板。

runtime 层会产出 frame/event，CLI renderer 只把这些对象转换成终端文本。
"""

from dataclasses import dataclass, field
from html import escape
import os
import sys
import time
from typing import Any, Callable, Iterable

from codepilot.protocols import AgentEvent, AssistantMessage, TextContent

OutputFn = Callable[..., None]
"""CLI 输出函数类型。

默认是 ``print``；测试时可替换成列表收集器。使用 ``Callable[..., None]`` 是为了兼容
``print(text, end="")`` 这类带额外关键字参数的调用。
"""


CP_CIRCUIT_MARK = (
    "╭─ C P ─╮\n"
    "│ ╭╮ ╭─ │\n"
    "│ ╰╯ ╰─ │\n"
    "╰─╼╾─╼╾╯"
)
"""Rich/彩色终端下展示的 Codepilot 标识。"""

PLAIN_CP_MARK = (
    "+- C P -+\n"
    "| () <- |\n"
    "| [] -> |\n"
    "+-------+"
)
"""纯文本终端下展示的 Codepilot 标识。"""

_CLI_PERMISSION_MODES = frozenset({"read-only", "workspace-write", "ask"})
_CLI_RUN_MODES = frozenset({"read", "plan", "build"})


@dataclass(frozen=True)
class CliStartupState:
    """启动面板使用的只读 view model。

    runtime.describe(session_id).status 中包含很多运行时状态，本类只保留 CLI 首页真正需要
    展示的字段。这样 renderer 不需要直接依赖 runtime status 的完整内部结构。
    """

    # 当前 CLI 版本号，用于启动面板标题。
    version: str
    # 实际使用的模型 ID，通常来自 runtime 配置解析结果。
    model_id: str
    # 当前 workspace 路径，用于提醒用户 agent 正在哪个项目中工作。
    workspace: str
    # 当前会话 ID，便于用户确认 resume/switch 后所在会话。
    session_id: str
    # 工具权限模式：read-only / workspace-write / ask。
    permission_mode: str = "workspace-write"
    # 运行模式：read / plan / build。
    current_mode: str = "build"
    # 当前计划摘要；无计划时为 None，避免底部栏和启动面板产生噪音。
    plan_summary: dict[str, object] | None = None
    # 启动时需要提示用户的警告，例如配置缺失或降级信息。
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """校验并规范化启动面板字段。

        frozen dataclass 不能直接赋值，因此使用 ``object.__setattr__`` 写回规范化结果。
        """
        object.__setattr__(self, "version", _require_text(self.version, "version"))
        object.__setattr__(self, "model_id", _require_text(self.model_id, "model_id"))
        object.__setattr__(self, "workspace", _require_text(self.workspace, "workspace"))
        object.__setattr__(self, "session_id", _require_text(self.session_id, "session_id"))
        object.__setattr__(self, "permission_mode", _ensure_permission_mode(self.permission_mode))
        object.__setattr__(self, "current_mode", _ensure_run_mode(self.current_mode))
        object.__setattr__(self, "plan_summary", _normalize_plan_summary(self.plan_summary))
        object.__setattr__(self, "warnings", _normalize_warnings(self.warnings))


def build_startup_state(status: Any, warnings: list[str] | None = None) -> CliStartupState:
    """从 runtime status view 构造 CLI 启动面板状态。

    Args:
        status: ``runtime.describe(session_id).status`` 返回的状态对象。
        warnings: 可选覆盖警告列表；未传时使用 status 自带 warnings。

    Returns:
        ``CliStartupState``，供 ``TerminalRenderer.render_startup`` 使用。
    """

    return CliStartupState(
        version="0.3",
        model_id=status.model_id,
        workspace=status.workspace,
        session_id=status.session_id,
        permission_mode=status.permission_mode,
        current_mode=status.current_mode,
        plan_summary=getattr(status, "plan_summary", None),
        warnings=tuple(warnings) if warnings is not None else tuple(status.warnings or ()),
    )


def no_color_requested() -> bool:
    """判断环境变量是否要求禁用颜色。

    Returns:
        当存在 ``NO_COLOR`` 环境变量时返回 ``True``。
    """

    return bool(os.getenv("NO_COLOR"))


def create_theme():
    """创建 CLI 共享 Rich 主题。

    Returns:
        ``rich.theme.Theme`` 实例。

    Rich 是可选显示依赖，因此函数内部再导入 Rich，避免模块导入阶段过早失败。
    """

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
    """创建带 Codepilot 主题的 Rich Console。

    Args:
        no_color: 显式控制是否禁用颜色；为 ``None`` 时遵循 ``NO_COLOR`` 环境变量。

    Returns:
        ``rich.console.Console`` 实例。
    """

    from rich.console import Console

    return Console(
        theme=create_theme(),
        no_color=no_color_requested() if no_color is None else no_color,
    )


def cyber_title(text: str) -> str:
    """生成统一的面板标题文本。"""
    return f"CP // {text.upper()}"


def render_key_value_panel(
    title: str,
    rows: Iterable[tuple[str, object]],
    *,
    console=None,
    border_style: str = "panel.border",
) -> None:
    """渲染紧凑的两列表格面板。

    Args:
        title: 面板标题，会自动加上 ``CP //`` 前缀。
        rows: ``(key, value)`` 二元组序列。
        console: 可选 Rich Console；未传时创建默认 Console。
        border_style: Rich 边框样式名称。
    """

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
    """生成纯文本两列面板。

    Args:
        title: 面板标题。
        rows: ``(key, value)`` 二元组序列。

    Returns:
        已按行拆分的纯文本面板，供无 Rich 环境输出。
    """
    lines = [f"+-- {cyber_title(title)} " + "-" * 28]
    for key, value in rows:
        lines.append(f"| {key:<12} {value}")
    lines.append("+" + "-" * 48)
    return lines


def format_help_text(*, prog: str = "codepilot", no_color: bool | None = None) -> str:
    """生成主命令帮助文本。

    Args:
        prog: argparse 注入的程序名。
        no_color: 是否使用纯文本标识；未传时读取 ``NO_COLOR``。

    Returns:
        可直接交给 argparse 输出的帮助字符串。
    """

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
  --mode MODE           read | plan | build
  --planning-budget PROFILE  conservative | balanced | wide
  --verbose                  Show debug events and config sources
  --no-color                 Disable colored terminal UI
  --version                  Show version and exit

Commands:
  config                     Manage local model/runtime configuration
  rpc                        Start JSONL RPC mode; no human UI is emitted
"""


def format_config_help_text(*, prog: str = "codepilot config") -> str:
    """生成 ``codepilot config`` 子命令帮助文本。

    Args:
        prog: argparse 注入的子命令名。

    Returns:
        子命令帮助字符串。
    """
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
    """生成统一的参数错误文本。

    Args:
        message: 错误消息。
        usage: 可选 usage 文本；传入时会追加到错误信息后。
    """
    lines = [cyber_title("Argument Error"), f"Error: {message}"]
    if usage:
        lines.append("")
        lines.append(usage.strip())
    return "\n".join(lines) + "\n"


class TerminalRenderer:
    """交互式终端渲染器。

    ``interactive.py`` 把 runtime frame 交给这个类，本类负责将其显示成人类可读界面：
    启动面板、普通状态、模型增量文本、工具开始/结束、审批提示和最终回答。
    """

    def __init__(
        self,
        *,
        output: OutputFn | None = None,
        verbose: bool = False,
        use_rich: bool = True,
    ) -> None:
        """创建终端渲染器。

        Args:
            output: 非 Rich 模式下的输出函数，默认是 ``print``。
            verbose: 是否显示调试事件、stop_reason、error_message 等详细信息。
            use_rich: 是否使用 Rich Console。为 ``False`` 时走纯文本输出。
        """
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
        """当前 renderer 是否持有 Rich Console。"""
        return self._console is not None

    def input(self, prompt: str) -> str:
        """通过 Rich Console 读取用户输入。

        Args:
            prompt: Rich markup 格式的提示符。

        Raises:
            RuntimeError: 当前 renderer 没有 Rich Console 时抛出。
        """
        if self._console is None:
            raise RuntimeError("rich console is not available")
        return self._console.input(prompt)

    def print(self, text: str = "", **kwargs: Any) -> None:
        """对外暴露的通用打印方法。"""
        self._print(text, **kwargs)

    def render_startup(self, state: CliStartupState) -> None:
        """渲染 CLI 启动面板。

        Args:
            state: 启动面板所需的模型、workspace、权限、session 等信息。
        """
        if self._console:
            self._render_rich_startup(state)
        else:
            self._render_plain_startup(state)

    def render_status(self, message: str, *, kind: str = "info") -> None:
        """渲染一条状态消息。

        Args:
            message: 要展示的人类文本。
            kind: 消息类型，影响图标和颜色。常见值：info/success/warning/error/cancelled。
        """
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
        """渲染“正在工作”的轻量提示。

        Args:
            message: 活动描述，例如 ``thinking`` 或 ``resuming``。

        该提示会在模型开始流式输出或工具状态出现后自动让位，避免用户误以为程序卡住。
        """
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
        """根据 runtime 过程事件选择具体渲染方式。

        Args:
            event: runtime/core 发出的 ``AgentEvent`` 字典。CLI 只读取标准字段，
                不修改事件对象。

        主要事件：
        - ``message_update``：模型文本增量。
        - ``tool_started`` / ``tool_completed`` / ``tool_failed`` / ``tool_interrupted``：工具执行状态。
        - ``error``：模型或运行时错误。
        """
        event_type = event.get("type")
        if event_type == "message_update":
            self._render_text_delta(event)
            return
        if event_type == "tool_started":
            self._render_tool_start(event)
            return
        if event_type in {"tool_completed", "tool_failed", "tool_interrupted"}:
            self._render_tool_end(event)
            return
        if event_type == "plan_approval_required":
            self._render_plan_approval(event)
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
        """渲染工具权限审批提示。

        Args:
            frame: ``ApprovalRequiredFrame``，其中包含 approval id、工具名、参数和风险等级。

        审批提示表示当前 agent loop 已暂停，用户需要输入 ``/approve`` 或 ``/deny``。
        """
        approval = frame.approval
        risk = getattr(approval, "risk", None)
        self._render_approval(
            tool_name=str(getattr(approval, "tool_name", "unknown")),
            args=dict(getattr(approval, "arguments", {}) or {}),
            risk_level=str(getattr(risk, "level", "unknown")),
            approval_id=str(getattr(approval, "approval_id", "")),
        )

    def render_command_output(self, lines: Iterable[str]) -> None:
        """渲染斜杠命令的人类输出。

        Args:
            lines: runtime 命令系统返回的输出行。每个元素可包含换行，会在这里展开。
        """
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
        """渲染一次 run 的最终结果。

        Args:
            record: ``RunFinishedFrame.record``。可能是 run record，也可能直接是
                ``AssistantMessage``，因此通过 ``_final_message_from_record`` 统一提取。

        如果模型已经通过增量事件流式输出，这里只补一个换行；如果没有流式输出，
        则从最终记录中提取完整助手文本并显示。
        """
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
        """重置一次用户输入相关的渲染状态。

        新的 prompt、审批恢复或命令执行开始前调用，避免上一轮的流式输出和工具计时影响本轮。
        """
        self._stream_started = False
        self._activity_started = False
        self._current_tool = None
        self._tool_start_time = 0
        self._tool_start_times.clear()

    def build_toolbar(self, state: CliStartupState) -> str:
        """构造 prompt_toolkit 底部工具栏文本。

        Args:
            state: 当前 session 状态。

        Returns:
            prompt_toolkit HTML 片段，显示模型、权限模式、运行模式和 session。
        """
        model = escape(self._shorten_tail(state.model_id, 28))
        permission = escape(state.permission_mode)
        current_mode = escape(state.current_mode)
        session = escape(self._short_session(state.session_id))
        plan = _format_plan_summary(state.plan_summary)
        plan_part = f"  |  {escape(plan)}" if plan else ""
        return (
            f"<b>CP</b>  <b>{model}</b>  |  {permission}  |  {current_mode}{plan_part}  |  {session}"
            "  |  <b>/help</b> deck  |  <b>Ctrl+C</b> cancel  |  <b>Alt+Enter</b> newline"
        )

    def build_shell_prompt(self) -> str:
        """构造 prompt_toolkit 输入框提示符。"""
        return "<prompt>╭─ YOU</prompt>\n<prompt>╰─› </prompt>"

    def build_console_prompt(self) -> str:
        """构造 Rich Console fallback 输入提示符。"""
        return "[bold bright_cyan]╭─ YOU[/bold bright_cyan]\n[bold bright_cyan]╰─›[/bold bright_cyan] "

    def build_plain_prompt(self) -> str:
        """构造纯文本输入提示符。"""
        return "╭─ YOU\n╰─› "

    def _render_rich_startup(self, state: CliStartupState) -> None:
        """使用 Rich 组件渲染启动面板。"""
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
        status.add_row("Mode", Text(state.current_mode, style="tool"))
        plan = _format_plan_summary(state.plan_summary)
        if plan:
            status.add_row("Plan", Text(plan, style="warning"))
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
        """在无 Rich 或禁用颜色时渲染纯文本启动面板。"""
        self._print()
        for line in PLAIN_CP_MARK.splitlines():
            self._print(f"| {line}")
        self._print(f"+-- Codepilot {state.version} - cyber engineering console " + "-" * 18)
        self._print("| Neural workspace online")
        self._print(f"| Model      {self._shorten_tail(state.model_id, 40)}")
        self._print(f"| Workspace  {self._shorten_tail(state.workspace, 54)}")
        self._print(f"| Permission {state.permission_mode}")
        self._print(f"| Mode       {state.current_mode}")
        plan = _format_plan_summary(state.plan_summary)
        if plan:
            self._print(f"| Plan       {plan}")
        self._print(f"| Session    {self._short_session(state.session_id)}")
        self._print("|")
        self._print("| /help command deck   /status telemetry   Ctrl+C exit/cancel")
        self._print("+" + "-" * 72)
        for warning in state.warnings:
            self.render_status(warning, kind="warning")
        self._print()

    def _render_text_delta(self, event: AgentEvent) -> None:
        """渲染模型文本增量。

        Args:
            event: ``message_update`` 事件。文本通常在
                ``assistantMessageEvent.delta``，旧格式可能在顶层 ``delta``。
        """
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
        """渲染工具开始执行的提示。

        Args:
            event: ``tool_started`` 事件，包含工具名、参数和可选 tool_call_id。

        这里会记录工具开始时间，供 ``_render_tool_end`` 计算耗时。
        """
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
        """渲染工具执行结束状态。

        Args:
            event: 工具完成/失败/中断事件，包含 status、isError、errorReason 等字段。

        ``approval_required`` 表示工具执行被审批中断，真正的审批框由
        ``ApprovalRequiredFrame`` 渲染，因此这里仅清理当前工具状态。
        """
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
        """渲染工具权限审批框。

        Args:
            tool_name: 请求审批的工具名，例如 ``bash``、``write``。
            args: 工具参数，用于提取命令或文件路径等目标信息。
            risk_level: 权限风险等级，影响显示颜色。
            approval_id: 用户需要在 ``/approve`` 或 ``/deny`` 后输入的审批 ID。
        """
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
            command.append("Approve\n", style="label")
            command.append("/approve", style="success")
            command.append("  or  yes", style="muted2")
            command.append("\n\nDeny\n", style="label")
            command.append("/deny", style="error")
            command.append("     or  no", style="muted2")

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
        self._print("|")
        self._print("| /approve   yes")
        self._print("| /deny      no")
        self._print("+" + "-" * 40)

    def _render_error_event(self, event: AgentEvent) -> None:
        """渲染 runtime 或模型 provider 的错误事件。

        Args:
            event: ``error`` 事件，可能包含 provider、model、errorInfo.details.response_text。
        """
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

    def _render_plan_approval(self, event: AgentEvent) -> None:
        """Render a proposed plan and the user actions that can resolve it."""
        self._activity_started = False
        if self._stream_started:
            self._print()
            self._stream_started = False
        self.render_command_output(
            [
                "=== Plan Approval Required ===",
                *_format_plan_payload_lines(event.get("plan")),
                "",
                "Use /plan approve to execute this plan.",
                "Use /plan reject to discard it.",
                "Type feedback to revise the plan.",
            ]
        )
        self._stream_started = True

    def _clear_current_tool(self, event: AgentEvent) -> None:
        """清理当前工具执行状态和计时缓存。"""
        self._current_tool = None
        self._tool_start_time = 0
        tool_call_id = str(event.get("toolCallId", ""))
        if tool_call_id:
            self._tool_start_times.pop(tool_call_id, None)

    def _print(self, text: str = "", **kwargs: Any) -> None:
        """根据当前模式选择 Rich Console 或普通输出函数。"""
        if self._console:
            self._console.print(text, **kwargs)
        else:
            self._output(text)

    @staticmethod
    def _extract_tool_target(tool_name: str, args: dict[str, Any]) -> str:
        """从工具参数中提取适合显示的目标摘要。

        Args:
            tool_name: 工具名。
            args: 工具调用参数。

        Returns:
            文件路径、搜索模式或 shell 命令摘要；无法识别时返回空字符串。
        """
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
        """把工具名转换成用户可感知的动作词。"""
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
        """把风险等级映射成 Rich 样式名。"""
        normalized = risk_level.lower()
        if normalized in {"high", "critical"}:
            return "error"
        if normalized == "medium":
            return "warning"
        return "success"

    @staticmethod
    def _shorten_tail(value: str, max_length: int) -> str:
        """保留字符串尾部并截断过长路径或命令。"""
        return value if len(value) <= max_length else "…" + value[-(max_length - 1):]

    @staticmethod
    def _short_session(session_id: str) -> str:
        """缩短 session id 以适配工具栏和启动面板。"""
        return session_id if len(session_id) <= 11 else session_id[:9] + ".."


class SimpleRenderer:
    """单次 prompt 模式的简化渲染器。

    ``codepilot -p`` 通常被脚本或用户用于快速提问，因此这里只输出助手文本和必要的
    审批提示，不显示启动面板、工具装饰和完整交互 UI。
    """

    def __init__(self, output: OutputFn = print) -> None:
        """创建简化渲染器。

        Args:
            output: 输出函数，默认 ``print``；测试时可替换为收集函数。
        """
        self.output = output
        self._stream_started = False

    def render_activity(self, message: str = "thinking") -> None:
        """单次模式不显示活动提示，避免污染脚本输出。"""
        return None

    def render_progress_event(self, event: AgentEvent) -> None:
        """渲染模型文本增量。

        Args:
            event: runtime 过程事件；只有 ``message_update`` 会被输出。
        """
        if event.get("type") == "plan_approval_required":
            for line in [
                "Plan approval required:",
                *_format_plan_payload_lines(event.get("plan")),
                "Use /plan approve, /plan reject, or type feedback to revise it.",
            ]:
                self.output(line)
            self._stream_started = True
            return
        if event.get("type") != "message_update":
            return
        assistant_event = event.get("assistantMessageEvent") or {}
        delta = str(assistant_event.get("delta", event.get("delta", "")))
        if delta:
            self.output(delta, end="")
            self._stream_started = True

    def render_approval_required(self, frame: Any) -> None:
        """在单次模式下输出最小审批提示。

        Args:
            frame: ``ApprovalRequiredFrame``，包含 approval id 和工具风险信息。
        """
        approval = frame.approval
        risk = getattr(getattr(approval, "risk", None), "level", "unknown")
        self.output(
            "Approval required for tool "
            f"{getattr(approval, 'tool_name', 'unknown')} "
            f"(risk={risk}, approval_id={getattr(approval, 'approval_id', '')})"
        )

    def render_final(self, record: Any | None) -> None:
        """在没有流式输出时输出最终助手文本。

        Args:
            record: runtime 完成记录，可能包含 final_message。
        """
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
        """输出一条普通状态消息。``kind`` 在简化模式下不影响样式。"""
        self.output(message)

    def reset(self) -> None:
        """重置流式输出标记。"""
        self._stream_started = False


def _final_message_from_record(record: Any | None) -> AssistantMessage | None:
    """从不同形态的 run record 中提取最终助手消息。

    Args:
        record: runtime 返回的完成记录。兼容直接传入 ``AssistantMessage``、
            ``record.outcome.final_message`` 和 ``record.final_message`` 三种形态。

    Returns:
        ``AssistantMessage`` 或 ``None``。
    """
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
    """把 ``AssistantMessage`` 中的文本块拼接成纯文本。"""
    return "".join(block.text for block in message.content if isinstance(block, TextContent)).strip()


def _require_text(value: object, field_name: str) -> str:
    """校验 CliStartupState 的必填字符串字段。"""
    if not isinstance(value, str):
        raise TypeError(f"CliStartupState.{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"CliStartupState.{field_name} cannot be empty")
    return text


def _ensure_permission_mode(value: object) -> str:
    """校验权限模式字段。"""
    text = _require_text(value, "permission_mode")
    if text not in _CLI_PERMISSION_MODES:
        raise ValueError(f"Unknown CLI permission_mode: {value}")
    return text


def _ensure_run_mode(value: object) -> str:
    """校验运行模式字段。"""
    text = _require_text(value, "current_mode")
    if text not in _CLI_RUN_MODES:
        raise ValueError(f"Unknown CLI current_mode: {value}")
    return text


def _normalize_plan_summary(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("CliStartupState.plan_summary must be a dict")
    return dict(value)


def _format_plan_summary(value: dict[str, object] | None) -> str:
    if not value:
        return ""
    status = str(value.get("status") or "").strip()
    if not status:
        return ""
    done = value.get("done_items")
    total = value.get("total_items")
    progress = ""
    if isinstance(done, int) and isinstance(total, int) and total > 0:
        progress = f" {done}/{total}"
    objective = str(value.get("objective_preview") or "").strip()
    objective = f" {objective}" if objective else ""
    return f"plan {status}{progress}{objective}".strip()


def _format_plan_payload_lines(value: object) -> list[str]:
    if not isinstance(value, dict):
        return ["No plan."]
    lines = [
        "=== Plan ===",
        f"  Plan ID    : {value.get('plan_id', '')}",
        f"  Status     : {value.get('status', '')}",
        f"  Approval   : {value.get('approval_state', '')}",
        f"  Mode       : {value.get('origin_mode', '')}",
        f"  Objective  : {value.get('objective', '')}",
    ]
    explanation = str(value.get("explanation") or "").strip()
    if explanation:
        lines.append(f"  Note       : {explanation}")
    items = value.get("items")
    if isinstance(items, list) and items:
        lines.append("  Items:")
        for index, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            lines.append(f"    {index}. [{item.get('status', '')}] {item.get('step', '')}")
    return lines


def _normalize_warnings(value: object) -> tuple[str, ...]:
    """把启动警告列表规范化为非空字符串元组。"""
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
