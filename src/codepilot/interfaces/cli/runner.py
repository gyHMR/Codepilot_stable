from __future__ import annotations

# 新手导读：runner.py 分发 print/interactive/rpc 三种运行模式。
# 关注点：它只调用 RuntimeGateway 的应用动作和快照，不直接改 core 或 session 内部状态。

"""
运行模式入口。

负责调度三种 CLI 运行模式：
- print: 单次问答，输出文本与工具事件
- interactive: 交互式 REPL
- rpc: 极简 JSON-RPC 模式，供外部程序调用

设计原则：
- CLI 通过 RuntimeGateway 操作 Session，不直接访问 Session 内部
- 默认隐藏内部调试字段，--verbose 下显示
- print/rpc 模式不被人类界面输出污染
"""

from dataclasses import asdict, dataclass, field, is_dataclass
import json
import sys
from pathlib import Path
from typing import Any, Callable, Literal

from codepilot.runtime.actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    FailedFrame,
    ProgressFrame,
    PromptSubmitted,
    RunFinishedFrame,
    RunCancelled,
)

from .commands import handle_cli_command
from .render import (
    OutputFn,
    SimpleRenderer,
    TerminalRenderer,
    build_startup_state,
)


RunMode = Literal["print", "interactive", "rpc"]
InputFn = Callable[[str], str]
RpcEmit = Callable[[dict[str, Any]], None]
RPC_PROTOCOL_VERSION = "1.2"


__all__ = [
    "RunOptions",
    "RPC_PROTOCOL_VERSION",
    "RpcEmit",
    "RpcError",
    "InputFn",
    "OutputFn",
    "RunMode",
    "emit_rpc_error",
    "emit_rpc_ok",
    "emit_rpc_ready",
    "rpc_error_from_exception",
    "rpc_json_default",
    "run",
    "run_interactive",
    "run_print",
    "run_rpc",
]


# ── 运行模式实现 ──────────────────────────────────────────────────

@dataclass(frozen=True)
class RpcError:
    """Normalized JSONL RPC error payload."""

    code: str
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _require_rpc_text(self.code, field_name="code"))
        object.__setattr__(
            self,
            "message",
            _require_rpc_text(self.message, field_name="message"),
        )


def rpc_json_default(value: Any) -> Any:
    """Serialize dataclasses and sets while preserving readable fallback text."""

    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return list(value)
    return str(value)


def rpc_error_from_exception(exc: Exception) -> RpcError:
    """Map Python exceptions to the stable JSONL RPC error contract."""

    raw_code = getattr(exc, "code", None)
    code = raw_code.strip() if isinstance(raw_code, str) and raw_code.strip() else "execution_error"
    message = str(exc).strip() or type(exc).__name__
    return RpcError(code=code, message=message)


def emit_rpc_error(
    emit: RpcEmit,
    *,
    req_id: Any,
    command: Any,
    code: str,
    message: str,
) -> None:
    """Emit one JSONL RPC error response."""

    error = RpcError(code=code, message=message)
    emit(
        {
            "type": "response",
            "id": req_id,
            "command": command,
            "status": "error",
            "error": {"code": error.code, "message": error.message},
        }
    )


def emit_rpc_ready(
    emit: RpcEmit,
    *,
    session_id: str,
) -> None:
    """Emit the initial JSONL RPC handshake message."""

    normalized_session_id = _require_rpc_text(session_id, field_name="session_id")
    emit(
        {
            "type": "rpc_ready",
            "session_id": normalized_session_id,
            "protocol_version": RPC_PROTOCOL_VERSION,
        }
    )


def emit_rpc_ok(
    emit: RpcEmit,
    *,
    req_id: Any,
    command: str,
    data: dict[str, Any] | None = None,
) -> None:
    """Emit one JSONL RPC success response."""

    normalized_command = _require_rpc_text(command, field_name="command")
    payload: dict[str, Any] = {
        "type": "response",
        "id": req_id,
        "command": normalized_command,
        "status": "ok",
    }
    if data is not None:
        payload["data"] = data
    emit(payload)


def _require_rpc_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"RPC error {field_name} must be str")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"RPC error {field_name} cannot be empty")
    return normalized

@dataclass
class RunOptions:
    """运行配置选项。"""
    mode: RunMode
    session_id: str
    runtime: Any
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


def _session_view(runtime: Any, session_id: str) -> Any:
    return runtime.describe(session_id)


def _session_state(runtime: Any, session_id: str) -> dict[str, Any]:
    return dict(_session_view(runtime, session_id).state or {})


def _session_status(runtime: Any, session_id: str):
    return _session_view(runtime, session_id).status


def _session_commands(runtime: Any, session_id: str) -> tuple[Any, ...]:
    return tuple(getattr(_session_view(runtime, session_id), "commands", ()) or ())


async def _render_prompt_run(
    runtime: Any,
    session_id: str,
    prompt: str,
    renderer: Any,
) -> None:
    """发送普通用户输入，并把运行事件渲染到 CLI renderer。"""

    await _render_runtime_frames(
        runtime.dispatch(session_id, PromptSubmitted(text=prompt)),
        renderer,
    )


async def _render_approval_decision_run(
    runtime: Any,
    session_id: str,
    *,
    approval_id: str,
    decision: str,
    reason: str = "",
    renderer: Any,
) -> None:
    """提交审批决定，并把恢复后的运行事件渲染到 CLI renderer。"""

    await _render_runtime_frames(
        runtime.dispatch(
            session_id,
            ApprovalDecided(
                approval_id=approval_id,
                decision=decision,  # type: ignore[arg-type]
                reason=reason,
            ),
        ),
        renderer,
    )


async def _render_runtime_frames(frames: Any, renderer: Any) -> None:
    """Consume runtime frames and render progress, approval, final, or failure."""

    final_message = None
    async for frame in frames:
        if isinstance(frame, ProgressFrame):
            renderer.handle_event(frame.event)
        elif isinstance(frame, ApprovalRequiredFrame):
            _render_approval_required(frame, renderer)
        elif isinstance(frame, RunFinishedFrame):
            final_message = _final_message_from_record(frame.record)
        elif isinstance(frame, FailedFrame):
            raise _runtime_error_from_frame(frame.error)
    renderer.render_final(final_message)


def _final_message_from_record(record: Any) -> Any:
    outcome = getattr(record, "outcome", None)
    if outcome is not None:
        message = getattr(outcome, "final_message", None)
        if message is not None:
            return message
    return getattr(record, "final_message", None)


def _render_approval_required(frame: ApprovalRequiredFrame, renderer: Any) -> None:
    event = _approval_event_from_frame(frame)
    renderer.handle_event(event)
    render_status = getattr(renderer, "render_status", None)
    if callable(render_status):
        approval_id = event["approvalId"]
        render_status(
            f"Approval pending: /approve {approval_id} or /deny {approval_id}",
            kind="warning",
        )


def _approval_event_from_frame(frame: ApprovalRequiredFrame) -> dict[str, Any]:
    approval = frame.approval
    risk = getattr(approval, "risk", None)
    return {
        "type": "tool_approval_required",
        "toolName": str(getattr(approval, "tool_name", "")),
        "toolCallId": str(getattr(approval, "tool_call_id", "")),
        "approvalId": str(getattr(approval, "approval_id", "")),
        "runId": str(getattr(approval, "run_id", "")),
        "args": dict(getattr(approval, "arguments", {}) or {}),
        "riskLevel": str(getattr(risk, "level", "unknown")),
        "reason": str(getattr(approval, "reason", "")),
        "status": "approval_required",
    }


def _frame_error_message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Runtime failed")
    return str(error)


class _RuntimeFrameError(RuntimeError):
    code = "runtime.dispatch_failed"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


def _runtime_error_from_frame(error: Any) -> _RuntimeFrameError:
    if isinstance(error, dict):
        return _RuntimeFrameError(
            _frame_error_message(error),
            code=str(error.get("code") or "runtime.dispatch_failed"),
        )
    return _RuntimeFrameError(str(error))


async def run_print(
    runtime: Any,
    session_id: str,
    prompt: str,
    *,
    output: OutputFn = print,
) -> None:
    """单次问答模式。

    通过 RuntimeGateway 发送消息，消费事件流。

    流程：
    1. 创建渲染器
    2. 通过 Runtime dispatch() 消费事件和最终结果
    3. 实时渲染流式输出
    4. 渲染最终结果
    """
    # print 模式使用简单渲染器，不依赖 rich
    renderer = SimpleRenderer(output=output)

    await _render_prompt_run(runtime, session_id, prompt, renderer)


async def run_interactive(
    runtime: Any,
    session_id: str,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    verbose: bool = False,
    no_color: bool = False,
    exit_commands: tuple[str, ...] = ("exit", "quit", ":q"),
) -> None:
    """交互式 REPL 模式。

    通过 RuntimeGateway 操作 Session，不直接访问 Session 内部。

    流程：
    1. 从 RuntimeGateway 获取会话状态
    2. 渲染启动摘要
    3. 创建 InteractiveShell（支持历史、补全、快捷键）
    4. 循环读取用户输入
    5. "/" 开头 → 通过 RuntimeGateway 执行命令
    6. 普通文本 → 通过 RuntimeGateway 发送消息
    7. 捕获 KeyboardInterrupt，通过 RuntimeGateway 取消任务
    """
    from .shell import create_shell

    # 交互模式使用 rich 渲染器
    renderer = TerminalRenderer(
        output=output,
        verbose=verbose,
        use_rich=not no_color,
    )

    # 从 RuntimeGateway 获取会话快照
    status = _session_status(runtime, session_id)
    startup_state = build_startup_state(status)
    renderer.render_startup(state=startup_state)

    # 创建 InteractiveShell（支持历史、补全）
    workspace = Path(status.workspace)
    shell = create_shell(
        history_dir=workspace / ".codepilot",
        no_color=no_color,
        commands=_session_commands(runtime, session_id),
    )

    current_session_id = session_id

    while True:
        # 获取用户输入
        try:
            if shell:
                # 使用 prompt_toolkit（异步版本）
                view = _session_view(runtime, current_session_id)
                if hasattr(shell, "set_commands"):
                    shell.set_commands(tuple(getattr(view, "commands", ()) or ()))
                toolbar = renderer.build_toolbar(build_startup_state(view.status))
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

        # "/" 开头的命令通过 RuntimeGateway 执行
        if text.startswith("/"):
            approval_command = _parse_approval_command(text)
            if approval_command is not None:
                decision, approval_id, reason = approval_command
                if not approval_id:
                    renderer.render_status(
                        f"Usage: /{decision} <approval_id>",
                        kind="error",
                    )
                    continue
                try:
                    renderer.reset()
                    await _render_approval_decision_run(
                        runtime,
                        current_session_id,
                        approval_id=approval_id,
                        decision=decision,
                        reason=reason,
                        renderer=renderer,
                    )
                except Exception as exc:
                    renderer.render_status(f"Approval error: {exc}", kind="error")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                continue
            try:
                command_result = await handle_cli_command(runtime, current_session_id, text)
                renderer.render_command_output(command_result.output_lines)
                # 如果命令导致会话切换（如 /fork, /clear）
                if command_result.switched_session_id is not None:
                    # 关闭旧 Session
                    runtime.close(current_session_id)
                    # 更新当前会话 ID
                    current_session_id = command_result.switched_session_id
                    # 更新状态显示
                    status = _session_status(runtime, current_session_id)
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

        # 普通文本 → 通过 RuntimeGateway 发送消息
        try:
            renderer.reset()
            await _render_prompt_run(runtime, current_session_id, text, renderer)
        except KeyboardInterrupt:
            # Ctrl+C 取消当前运行，不退出 CLI
            renderer.render_status("Cancelled", kind="cancelled")
            async for _frame in runtime.dispatch(
                current_session_id,
                RunCancelled(reason="keyboard_interrupt"),
            ):
                pass
            continue
        except Exception as exc:
            renderer.render_status(f"Error: {exc}", kind="error")
            if verbose:
                import traceback
                traceback.print_exc()
            continue


def _parse_approval_command(text: str) -> tuple[str, str, str] | None:
    parts = text.strip().lstrip("/").split(maxsplit=2)
    if not parts:
        return None
    command = parts[0].lower()
    if command not in {"approve", "deny"}:
        return None
    approval_id = parts[1].strip() if len(parts) >= 2 else ""
    reason = parts[2].strip() if len(parts) >= 3 else ""
    return command, approval_id, reason


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
    runtime: Any,
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
            result_data: dict[str, Any] | None = None
            async for frame in runtime.dispatch(
                session_id,
                PromptSubmitted(text=text, mode_hint=task_mode),
            ):
                if isinstance(frame, ProgressFrame):
                    emit({"type": "event", "event": frame.event})
                elif isinstance(frame, ApprovalRequiredFrame):
                    approval = _approval_rpc_data(frame)
                    emit({"type": "approval_required", "approval": approval})
                    result_data = {
                        "status": "waiting_approval",
                        "approval_id": approval["approval_id"],
                    }
                elif isinstance(frame, RunFinishedFrame):
                    result_data = _run_record_rpc_data(frame.record)
                elif isinstance(frame, FailedFrame):
                    raise _runtime_error_from_frame(frame.error)
            emit_rpc_ok(emit, req_id=req_id, command="prompt", data=result_data)

        elif cmd in {"approve", "deny"}:
            approval_id = str(req.get("approval_id", "")).strip()
            if not approval_id:
                raise ValueError(f"{cmd} requires approval_id")
            reason = str(req.get("reason", "")).strip()
            result_data: dict[str, Any] | None = None
            async for frame in runtime.dispatch(
                session_id,
                ApprovalDecided(
                    approval_id=approval_id,
                    decision=cmd,  # type: ignore[arg-type]
                    reason=reason,
                ),
            ):
                if isinstance(frame, ProgressFrame):
                    emit({"type": "event", "event": frame.event})
                elif isinstance(frame, ApprovalRequiredFrame):
                    approval = _approval_rpc_data(frame)
                    emit({"type": "approval_required", "approval": approval})
                    result_data = {
                        "status": "waiting_approval",
                        "approval_id": approval["approval_id"],
                    }
                elif isinstance(frame, RunFinishedFrame):
                    result_data = _run_record_rpc_data(frame.record)
                elif isinstance(frame, FailedFrame):
                    raise _runtime_error_from_frame(frame.error)
            emit_rpc_ok(emit, req_id=req_id, command=cmd, data=result_data)

        elif cmd == "state":
            state = _session_state(runtime, session_id)
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
            result = await handle_cli_command(
                runtime,
                session_id,
                f"/mode {mode}",
            )
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="set_task_mode",
                data={
                    "session_id": session_id,
                    "task_mode": result.data.get("task_mode"),
                },
            )

        elif cmd == "list_entries":
            state = _session_state(runtime, session_id)
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="list_entries",
                data={
                    "session_id": session_id,
                    "entry_ids": state["entry_ids"],
                    "entries": state.get("entries", []),
                    "leaf_id": state["leaf_id"],
                },
            )

        elif cmd == "show_tree":
            state = _session_state(runtime, session_id)
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="show_tree",
                data={
                    "session_id": session_id,
                    "tree": state.get("tree", []),
                    "leaf_id": state["leaf_id"],
                },
            )

        elif cmd == "entry_path":
            entry_id = str(req.get("entry_id", ""))
            if not entry_id:
                raise ValueError("entry_path requires entry_id")
            result = await handle_cli_command(
                runtime,
                session_id,
                f"/path {entry_id}",
            )
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="entry_path",
                data={
                    "session_id": session_id,
                    "entry_id": entry_id,
                    "path": result.data.get("path", []),
                },
            )

        elif cmd == "fork_entry":
            entry_id = str(req.get("entry_id", ""))
            if not entry_id:
                raise ValueError("fork_entry requires entry_id")
            result = await handle_cli_command(
                runtime,
                session_id,
                f"/fork {entry_id}",
            )
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="fork_entry",
                data=dict(result.data),
            )

        elif cmd == "switch_entry":
            entry_id = str(req.get("entry_id", ""))
            if not entry_id:
                raise ValueError("switch_entry requires entry_id")
            result = await handle_cli_command(
                runtime,
                session_id,
                f"/switch {entry_id}",
            )
            emit_rpc_ok(
                emit,
                req_id=req_id,
                command="switch_entry",
                data=dict(result.data),
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
                        for command in runtime.describe(session_id).commands
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


def _run_record_rpc_data(record: Any) -> dict[str, Any]:
    """Return the stable RPC summary for a completed prompt run."""

    data: dict[str, Any] = {}
    for name in ("run_id", "session_id", "status", "stop_reason", "final_text"):
        value = getattr(record, name, None)
        if value is not None:
            data[name] = value
    return data


def _approval_rpc_data(frame: ApprovalRequiredFrame) -> dict[str, Any]:
    approval = frame.approval
    risk = getattr(approval, "risk", None)
    return {
        "approval_id": str(getattr(approval, "approval_id", "")),
        "run_id": str(getattr(approval, "run_id", "")),
        "tool_call_id": str(getattr(approval, "tool_call_id", "")),
        "tool_name": str(getattr(approval, "tool_name", "")),
        "arguments": dict(getattr(approval, "arguments", {}) or {}),
        "reason": str(getattr(approval, "reason", "")),
        "risk_level": str(getattr(risk, "level", "unknown")),
    }


async def run_rpc(
    runtime: Any,
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
