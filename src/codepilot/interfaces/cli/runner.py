from __future__ import annotations

# 新手导读：runner.py 分发 print/interactive/rpc 三种运行模式。
# 关注点：它只调用 RuntimeService，不直接改 core 或 session 内部状态。

"""
运行模式入口。

负责调度三种 CLI 运行模式：
- print: 单次问答，输出文本与工具事件
- interactive: 交互式 REPL
- rpc: 极简 JSON-RPC 模式，供外部程序调用

设计原则：
- CLI 通过 RuntimeService 操作 Session，不直接访问 Session 内部
- 默认隐藏内部调试字段，--verbose 下显示
- print/rpc 模式不被人类界面输出污染
"""

from dataclasses import dataclass, field
import json
import sys
from typing import Any

from codepilot.runtime.service import RuntimeService
from codepilot.runtime.contracts import UserInput

from .commands import handle_cli_command, list_runtime_commands
from .renderer import SimpleRenderer, TerminalRenderer
from .rpc_protocol import (
    RpcEmit,
    emit_rpc_ready,
    emit_rpc_error,
    emit_rpc_ok,
    rpc_error_from_exception,
    rpc_json_default,
)
from .startup import build_startup_state
from .types import InputFn, OutputFn, RunMode


__all__ = [
    "RunOptions",
    "run",
    "run_interactive",
    "run_print",
    "run_rpc",
]


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

    def __post_init__(self) -> None:
        self.mode = _ensure_run_mode(self.mode)
        self.session_id = _require_cli_text(
            self.session_id,
            field_name="session_id",
        )
        if not callable(self.output):
            raise TypeError("RunOptions.output must be callable")
        if not callable(self.input_fn):
            raise TypeError("RunOptions.input_fn must be callable")
        self.exit_commands = _normalize_exit_commands(self.exit_commands)


def _ensure_run_mode(value: object) -> RunMode:
    if isinstance(value, str):
        value = value.strip()
    if value not in {"print", "interactive", "rpc"}:
        raise ValueError(f"Unknown CLI run mode: {value}")
    return value  # type: ignore[return-value]


def _require_cli_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"RunOptions.{field_name} must be a string")
    text = value.strip()
    if not text:
        raise ValueError(f"RunOptions.{field_name} is required")
    return text


def _normalize_exit_commands(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError("RunOptions.exit_commands must be a sequence of strings")
    commands: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("RunOptions.exit_commands must contain strings")
        command = item.strip().lstrip("/")
        if command:
            commands.append(command)
    return tuple(commands)


async def _render_prompt_run(
    runtime: RuntimeService,
    session_id: str,
    prompt: str,
    renderer: Any,
) -> None:
    """发送普通用户输入，并把运行事件渲染到 CLI renderer。"""

    async for event in runtime.send_message(session_id, UserInput(text=prompt)):
        renderer.handle_event(event)
    renderer.render_final(runtime.get_latest_assistant_message(session_id))


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

    await _render_prompt_run(runtime, session_id, prompt, renderer)


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
    renderer = TerminalRenderer(
        output=output,
        verbose=verbose,
        use_rich=not no_color,
    )

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
                status = runtime.get_session_status(current_session_id)
                toolbar = renderer.build_toolbar(build_startup_state(status))
                text = await shell.prompt(
                    prompt_text="› ",
                    bottom_toolbar=toolbar,
                )
            elif renderer.has_rich_console:
                # 使用 rich 的 prompt
                text = renderer.input("[bold bright_cyan]›[/bold bright_cyan] ")
            else:
                text = input_fn("› ")
            text = text.strip()
        except EOFError:
            # Ctrl+D 退出
            renderer.render_status("Bye.", kind="info")
            return
        except KeyboardInterrupt:
            # Ctrl+C 在输入阶段：退出 CLI。
            renderer.render_status("Bye.", kind="info")
            return

        bare = text.lstrip("/")

        # 检查退出命令
        if bare in exit_commands or text == "/exit":
            renderer.render_status("Bye.", kind="info")
            return

        # 空输入跳过
        if not text:
            continue

        # "/" 开头的命令通过 RuntimeService 执行
        if text.startswith("/"):
            try:
                command_result = await handle_cli_command(runtime, current_session_id, text)
                renderer.render_command_output(command_result.output_lines)
                # 如果命令导致会话切换（如 /fork, /clear）
                if command_result.switched_session_id is not None:
                    # 关闭旧 Session
                    await runtime.aclose_session(current_session_id)
                    # 更新当前会话 ID
                    current_session_id = command_result.switched_session_id
                    # 更新状态显示
                    status = runtime.get_session_status(current_session_id)
                    renderer.render_status(
                        f"Switched to session {current_session_id[:8]}..",
                        kind="success",
                    )
                if command_result.handled:
                    continue
                # "/" 输入未匹配任何命令时仍停留在命令系统，不转发给模型。
                unknown_cmd = text.partition(" ")[0]
                renderer.render_command_output(
                    [
                        f"Unknown command: {unknown_cmd}",
                        "Type /help to see available commands.",
                    ]
                )
                continue
            except Exception as exc:
                renderer.render_status(f"Command error: {exc}", kind="error")
                continue

        # 普通文本 → 通过 RuntimeService 发送消息
        try:
            renderer.reset()
            await _render_prompt_run(runtime, current_session_id, text, renderer)
        except KeyboardInterrupt:
            # Ctrl+C 取消当前运行，不退出 CLI
            renderer.render_status("Cancelled", kind="cancelled")
            # 通过 RuntimeService 取消任务
            await runtime.cancel_run(current_session_id)
            continue
        except Exception as exc:
            renderer.render_status(f"Error: {exc}", kind="error")
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


async def _handle_rpc_request(
    runtime: RuntimeService,
    session_id: str,
    req: Any,
    emit: RpcEmit,
) -> bool:
    """Handle one JSONL RPC request.

    Returns True when the caller should stop reading stdin.
    """

    if not isinstance(req, dict):
        emit_rpc_error(
            emit,
            req_id=None,
            command=None,
            code="invalid_request",
            message="Request must be object",
        )
        return False

    cmd = req.get("type")
    req_id = req.get("id")

    try:
        if cmd == "prompt":
            text = str(req.get("text", ""))
            task_mode = req.get("task_mode")
            if task_mode is not None and not isinstance(task_mode, str):
                raise ValueError("task_mode must be a string")
            async for event in runtime.send_message(
                session_id,
                UserInput(text=text, task_mode=task_mode),
            ):
                emit({"type": "event", "event": event})
            emit_rpc_ok(emit, req_id=req_id, command="prompt")

        elif cmd == "continue":
            async for event in runtime.continue_session(session_id):
                emit({"type": "event", "event": event})
            emit_rpc_ok(emit, req_id=req_id, command="continue")

        elif cmd == "state":
            state = runtime.get_session_state(session_id)
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="state",
                data=state,
            )

        elif cmd == "set_task_mode":
            mode = req.get("task_mode")
            if not isinstance(mode, str):
                raise ValueError("set_task_mode requires task_mode")
            current = runtime.set_task_mode(session_id, mode)
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="set_task_mode",
                data={"session_id": session_id, "task_mode": current},
            )

        elif cmd == "list_entries":
            state = runtime.get_session_state(session_id)
            emit_rpc_ok(
                emit,
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
            emit_rpc_ok(
                emit,
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
            emit_rpc_ok(
                emit,
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
            emit_rpc_ok(
                emit,
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
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="switch_entry",
                data={
                    "session_id": session_id,
                    "entry_id": entry_id,
                    "path": runtime.get_entry_path(session_id, entry_id),
                },
            )

        elif cmd == "get_commands":
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="get_commands",
                data={
                    "session_id": session_id,
                    "commands": [
                        command.to_dict()
                        for command in list_runtime_commands(runtime.get_session(session_id))
                    ],
                },
            )

        elif cmd == "shutdown":
            emit_rpc_ok(emit, req_id=req_id, command="shutdown")
            return True

        else:
            emit_rpc_error(
                emit,
                req_id=req_id,
                command=cmd,
                code="unknown_command",
                message="Unknown command",
            )

    except Exception as exc:
        error = rpc_error_from_exception(exc)
        emit_rpc_error(
            emit,
            req_id=req_id,
            command=cmd,
            code=error.code,
            message=error.message,
        )
    return False


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

    def emit(obj: dict[str, Any]) -> None:
        output(json.dumps(obj, ensure_ascii=False, default=rpc_json_default))

    emit_rpc_ready(emit, session_id=session_id)

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as exc:
            emit_rpc_error(
                emit,
                req_id=None,
                command=None,
                code="invalid_json",
                message=f"Invalid JSON: {exc}",
            )
            continue

        should_shutdown = await _handle_rpc_request(runtime, session_id, req, emit)
        if should_shutdown:
            return
