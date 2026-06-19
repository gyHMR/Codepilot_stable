from __future__ import annotations

"""
运行模式入口与终端渲染。

负责将 AgentEvent 转换为人类可读的终端输出，支持三种运行模式：
- print: 单次问答，输出文本与工具事件
- interactive: 交互式 REPL
- rpc: 极简 JSON-RPC 模式，供外部程序调用

设计原则：
- TerminalRenderer 只负责渲染，不修改 Session 或执行工具
- CLI 通过 RuntimeService 操作 Session，不直接访问 Session 内部
- 流式文本实时输出，不收集后打印
- 默认隐藏内部调试字段，--verbose 下显示
- print/rpc 模式不被人类界面输出污染
- 使用 rich 库美化交互模式的输出
"""

from dataclasses import dataclass, field
from dataclasses import asdict, is_dataclass
import json
import sys
import time
from typing import Any, Callable

from codepilot.protocols import AssistantMessage, TextContent
from codepilot.core import AgentEvent

from codepilot.runtime.service import RuntimeService, SessionStatus, UserInput
from codepilot.runtime.types import InputFn, OutputFn, RunMode


# ── 启动状态数据结构 ──────────────────────────────────────────────

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


def build_startup_state(status: SessionStatus, warnings: list[str] | None = None) -> CliStartupState:
    """从 SessionStatus 构建启动状态信息。

    Args:
        status: RuntimeService 返回的会话状态。
        warnings: 可选的警告列表。

    Returns:
        CliStartupState 实例。
    """
    return CliStartupState(
        version="0.3",
        model_id=status.model_id,
        workspace=status.workspace,
        session_id=status.session_id,
        permission_mode=status.permission_mode,
        warnings=warnings or [],
    )


# ── 终端渲染器（使用 rich） ──────────────────────────────────────

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

            # 自定义主题
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
            self._output = None  # 使用 rich console
        else:
            self._console = None
            self._output = output or print

    def _print(self, text: str = "", **kwargs: Any) -> None:
        """统一输出方法。"""
        if self._console:
            self._console.print(text, **kwargs)
        else:
            self._output(text)

    def render_startup(self, state: CliStartupState) -> None:
        """渲染启动摘要面板。

        使用 rich Panel 展示当前运行环境。
        """
        if self._console:
            self._render_rich_startup(state)
        else:
            self._render_plain_startup(state)

    def _render_rich_startup(self, state: CliStartupState) -> None:
        """使用 rich 渲染启动面板。"""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # 创建信息表格
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold cyan", width=12)
        table.add_column()

        # 截断过长的路径
        workspace = state.workspace
        if len(workspace) > 45:
            workspace = "..." + workspace[-42:]

        # 截断 session_id
        session_display = state.session_id
        if len(session_display) > 12:
            session_display = session_display[:10] + ".."

        table.add_row("Model", Text(state.model_id, style="magenta"))
        table.add_row("Workspace", Text(workspace, style="path"))
        table.add_row("Session", Text(session_display, style="green"))
        table.add_row("Permission", Text(state.permission_mode, style="yellow"))

        # 创建面板
        panel = Panel(
            table,
            title=f"[bold]Codepilot {state.version}[/bold]",
            border_style="blue",
            padding=(1, 2),
        )
        self._console.print(panel)

        # 显示警告
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

        # 流式文本更新
        if event_type == "message_update":
            self._handle_text_delta(event)
            return

        # 工具执行开始
        if event_type == "tool_execution_start":
            self._handle_tool_start(event)
            return

        # 工具执行结束
        if event_type == "tool_execution_end":
            self._handle_tool_end(event)
            return

        # 工具审批请求
        if event_type == "tool_approval_required":
            self._handle_approval_required(event)
            return

        # 错误事件
        if event_type == "error":
            self._handle_error(event)
            return

        # 模型重试
        if event_type == "model_retry_start" and self.verbose:
            attempt = event.get("attempt", 0)
            max_attempts = event.get("maxAttempts", 0)
            self._print(f"🔄 [info]Retry attempt {attempt}/{max_attempts}[/info]")

        # verbose 模式下显示其他事件
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

        # 首次输出时添加换行
        if not self._stream_started:
            if self._console:
                self._console.print()
            else:
                self._print()
            self._stream_started = True

        # 实时输出文本增量
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

        # 结束之前的流式文本
        if self._stream_started:
            self._print()
            self._stream_started = False

        # 显示工具开始
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

        # 计算耗时
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
        """处理工具审批请求事件。

        显示工具调用信息，等待用户确认。
        """
        tool_name = event.get("toolName", "unknown")
        args = event.get("args", {})
        approval_id = event.get("approvalId", "")
        risk_level = event.get("riskLevel", "medium")

        # 结束之前的流式文本
        if self._stream_started:
            self._print()
            self._stream_started = False

        # 提取目标信息
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

        # 注意：实际的审批流程需要通过 Runtime 的 ApprovalProvider 处理
        # 这里只是显示信息，审批决定通过 Runtime API 返回

    def _handle_error(self, event: AgentEvent) -> None:
        """处理错误事件。"""
        error = event.get("error", "unknown error")
        message = event.get("message", "")
        provider = event.get("provider", "")
        model = event.get("model", "")

        # 结束之前的流式文本
        if self._stream_started:
            self._print()
            self._stream_started = False

        if self._console:
            from rich.panel import Panel
            from rich.text import Text

            # 构建错误内容
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
        # 如果没有流式输出，提取并显示文本
        if not self._stream_started and final_assistant is not None:
            text = self._extract_assistant_text(final_assistant)
            if text:
                self._print(text)
            elif self.verbose:
                self._print("[dim](empty response)[/dim]")

        # 确保流式输出后有换行
        if self._stream_started:
            self._print()
            self._stream_started = False

        # verbose 模式下显示元信息
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


# ── 简单渲染器（用于 print 模式） ────────────────────────────────

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


# ── 运行模式实现 ──────────────────────────────────────────────────

@dataclass
class RunOptions:
    """运行配置选项。"""
    mode: RunMode
    session_id: str
    runtime: RuntimeService
    prompt: str | None = None
    output: OutputFn = print
    input_fn: InputFn = input
    verbose: bool = False
    no_color: bool = False
    exit_commands: tuple[str, ...] = field(default_factory=lambda: ("exit", "quit", ":q"))


async def run_print(
    runtime: RuntimeService,
    session_id: str,
    prompt: str,
    *,
    output: OutputFn = print,
) -> None:
    """单次问答模式。

    通过 RuntimeService 发送消息，消费事件流。

    流程：
    1. 创建渲染器
    2. 通过 RuntimeService.send_message() 消费事件
    3. 实时渲染流式输出
    4. 渲染最终结果
    """
    # print 模式使用简单渲染器，不依赖 rich
    renderer = SimpleRenderer(output=output)

    # 通过 RuntimeService 发送消息并消费事件
    async for event in runtime.send_message(session_id, UserInput(text=prompt)):
        renderer.handle_event(event)

    # 获取最终的 AssistantMessage
    renderer.render_final(runtime.get_latest_assistant_message(session_id))


async def run_interactive(
    runtime: RuntimeService,
    session_id: str,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    verbose: bool = False,
    no_color: bool = False,
    exit_commands: tuple[str, ...] = ("exit", "quit", ":q"),
) -> None:
    """交互式 REPL 模式。

    通过 RuntimeService 操作 Session，不直接访问 Session 内部。

    流程：
    1. 从 RuntimeService 获取会话状态
    2. 渲染启动摘要
    3. 创建 InteractiveShell（支持历史、补全、快捷键）
    4. 循环读取用户输入
    5. "/" 开头 → 通过 RuntimeService 执行命令
    6. 普通文本 → 通过 RuntimeService 发送消息
    7. 捕获 KeyboardInterrupt，通过 RuntimeService 取消任务
    """
    from .shell import create_shell

    # 交互模式使用 rich 渲染器
    renderer = TerminalRenderer(verbose=verbose, use_rich=not no_color)

    # 从 RuntimeService 获取会话状态
    status = runtime.get_session_status(session_id)
    startup_state = build_startup_state(status)
    renderer.render_startup(state=startup_state)

    # 创建 InteractiveShell（支持历史、补全）
    workspace = runtime.get_workspace(session_id)
    shell = create_shell(
        history_dir=workspace / ".codepilot",
        no_color=no_color,
    )

    current_session_id = session_id

    while True:
        # 获取用户输入
        try:
            if shell:
                # 使用 prompt_toolkit（异步版本）
                text = await shell.prompt(
                    prompt_text="> ",
                    bottom_toolbar="<b>Ctrl+C</b> cancel | <b>/help</b> commands",
                )
            elif renderer._console:
                # 使用 rich 的 prompt
                text = renderer._console.input("[bold green]>[/bold green] ")
            else:
                text = input_fn("> ")
            text = text.strip()
        except EOFError:
            # Ctrl+D 退出
            renderer._print("\nBye.")
            return
        except KeyboardInterrupt:
            # Ctrl+C 在输入阶段：清空输入或退出
            renderer._print("\nBye.")
            return

        bare = text.lstrip("/")

        # 检查退出命令
        if bare in exit_commands or text == "/exit":
            renderer._print("Bye.")
            return

        # 空输入跳过
        if not text:
            continue

        # "/" 开头的命令通过 RuntimeService 执行
        if text.startswith("/"):
            try:
                command_result = await runtime.execute_command(current_session_id, text)
                for line in command_result.output_lines:
                    renderer._print(line)
                # 如果命令导致会话切换（如 /fork, /clear）
                if command_result.switched_session_id is not None:
                    # 关闭旧 Session
                    runtime.close_session(current_session_id)
                    # 更新当前会话 ID
                    current_session_id = command_result.switched_session_id
                    # 更新状态显示
                    status = runtime.get_session_status(current_session_id)
                    renderer._print(f"[info]Switched to session: {current_session_id[:8]}..[/info]")
                if command_result.handled:
                    continue
            except Exception as exc:
                renderer._print(f"[error]Command error: {exc}[/error]")
                continue

        # 普通文本 → 通过 RuntimeService 发送消息
        try:
            renderer.reset()
            # 通过 RuntimeService 发送消息并消费事件流
            async for event in runtime.send_message(
                current_session_id,
                UserInput(text=text),
            ):
                renderer.handle_event(event)

            # 获取最终的 AssistantMessage
            renderer.render_final(
                runtime.get_latest_assistant_message(current_session_id)
            )
        except KeyboardInterrupt:
            # Ctrl+C 取消当前运行，不退出 CLI
            renderer._print("\n[yellow][cancelled][/yellow]")
            # 通过 RuntimeService 取消任务
            await runtime.cancel_run(current_session_id)
            continue
        except Exception as exc:
            renderer._print(f"\n[error]Error: {exc}[/error]")
            if verbose:
                import traceback
                traceback.print_exc()
            continue


async def run(options: RunOptions) -> None:
    """统一运行入口。"""
    if options.mode == "print":
        if not options.prompt:
            raise ValueError("print mode requires prompt")
        await run_print(
            options.runtime,
            options.session_id,
            options.prompt,
            output=options.output,
        )
        return

    if options.mode == "rpc":
        await run_rpc(
            options.runtime,
            options.session_id,
            output=options.output,
        )
        return

    # interactive 模式（默认）
    await run_interactive(
        options.runtime,
        options.session_id,
        input_fn=options.input_fn,
        output=options.output,
        verbose=options.verbose,
        no_color=options.no_color,
        exit_commands=options.exit_commands,
    )


async def run_rpc(
    runtime: RuntimeService,
    session_id: str,
    *,
    output: OutputFn = print,
) -> None:
    """极简 RPC 模式（JSONL 协议）。

    通过 stdin/stdout 以 JSON 行格式与外部程序通信。
    不输出任何人类界面内容，只输出严格 JSONL。
    """

    def _json_default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, set):
            return list(value)
        return str(value)

    def _emit(obj: dict[str, Any]) -> None:
        output(json.dumps(obj, ensure_ascii=False, default=_json_default))

    def _emit_error(*, req_id: Any, command: Any, code: str, message: str) -> None:
        _emit({
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "error",
            "error": {"code": code, "message": message},
        })

    def _emit_ok(*, req_id: Any, command: str, data: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "ok",
        }
        if data is not None:
            payload["data"] = data
        _emit(payload)

    # 发送就绪信号
    _emit({"type": "rpc_ready", "session_id": session_id, "protocol_version": "1.2"})

    for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue

            try:
                req = json.loads(line)
            except Exception as exc:
                _emit_error(req_id=None, command=None, code="invalid_json", message=f"Invalid JSON: {exc}")
                continue

            if not isinstance(req, dict):
                _emit_error(req_id=None, command=None, code="invalid_request", message="Request must be object")
                continue

            cmd = req.get("type")
            req_id = req.get("id")

            try:
                if cmd == "prompt":
                    text = str(req.get("text", ""))
                    async for event in runtime.send_message(
                        session_id,
                        UserInput(text=text),
                    ):
                        _emit({"type": "event", "event": event})
                    _emit_ok(req_id=req_id, command="prompt")

                elif cmd == "continue":
                    async for event in runtime.continue_session(session_id):
                        _emit({"type": "event", "event": event})
                    _emit_ok(req_id=req_id, command="continue")

                elif cmd == "state":
                    state = runtime.get_session_state(session_id)
                    _emit_ok(
                        req_id=req_id,
                        command="state",
                        data=state,
                    )

                elif cmd == "list_entries":
                    state = runtime.get_session_state(session_id)
                    _emit_ok(
                        req_id=req_id,
                        command="list_entries",
                        data={
                            "session_id": session_id,
                            "entry_ids": state["entry_ids"],
                            "entries": runtime.list_session_entries(session_id),
                            "leaf_id": state["leaf_id"],
                        },
                    )

                elif cmd == "show_tree":
                    state = runtime.get_session_state(session_id)
                    _emit_ok(
                        req_id=req_id,
                        command="show_tree",
                        data={
                            "session_id": session_id,
                            "tree": runtime.get_session_tree(session_id),
                            "leaf_id": state["leaf_id"],
                        },
                    )

                elif cmd == "entry_path":
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("entry_path requires entry_id")
                    _emit_ok(
                        req_id=req_id,
                        command="entry_path",
                        data={
                            "session_id": session_id,
                            "entry_id": entry_id,
                            "path": runtime.get_entry_path(session_id, entry_id),
                        },
                    )

                elif cmd == "fork_entry":
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("fork_entry requires entry_id")
                    new_session_id = runtime.fork_session(session_id, entry_id)
                    _emit_ok(
                        req_id=req_id,
                        command="fork_entry",
                        data={
                            "from_session_id": session_id,
                            "from_entry_id": entry_id,
                            "new_session_id": new_session_id,
                        },
                    )

                elif cmd == "switch_entry":
                    entry_id = str(req.get("entry_id", ""))
                    if not entry_id:
                        raise ValueError("switch_entry requires entry_id")
                    runtime.switch_entry(session_id, entry_id)
                    _emit_ok(
                        req_id=req_id,
                        command="switch_entry",
                        data={
                            "session_id": session_id,
                            "entry_id": entry_id,
                            "path": runtime.get_entry_path(session_id, entry_id),
                        },
                    )

                elif cmd == "get_commands":
                    _emit_ok(
                        req_id=req_id,
                        command="get_commands",
                        data={
                            "session_id": session_id,
                            "commands": runtime.list_commands(session_id),
                        },
                    )

                elif cmd == "shutdown":
                    _emit_ok(req_id=req_id, command="shutdown")
                    return

                else:
                    _emit_error(req_id=req_id, command=cmd, code="unknown_command", message="Unknown command")

            except Exception as exc:
                _emit_error(req_id=req_id, command=cmd, code="execution_error", message=str(exc))
