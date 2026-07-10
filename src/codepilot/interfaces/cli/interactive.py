from __future__ import annotations

"""人类 CLI 交互主流程：读取输入、派发 runtime action、渲染 frame。

本文件描述 CLI 模式下“一次用户输入”的完整分流：

1. ``read_user_text`` 从 prompt_toolkit / Rich / stdin 读取一段文本。
2. ``run_repl`` 判断文本是退出命令、审批命令、斜杠命令还是普通 prompt。
3. 普通 prompt 通过 ``PromptSubmitted`` 派发给 ``RuntimeGateway``。
4. 审批通过 ``ApprovalDecided`` 恢复此前被工具权限中断的 run。
5. 斜杠命令通过 ``CommandSubmitted`` 交给 runtime 的命令系统。
6. ``render_dispatch`` 消费 runtime 产生的 frame，并交给 renderer 显示。

CLI 层不直接调用模型、工具或 session 内部对象；所有跨层动作都经过 runtime action。
"""

from pathlib import Path
from typing import Any, Callable, Iterable, Literal

from codepilot.runtime.actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    CommandFinishedFrame,
    CommandSubmitted,
    FailedFrame,
    ProgressFrame,
    PromptSubmitted,
    RunCancelled,
    RunFinishedFrame,
    RunPausedFrame,
)
from codepilot.runtime.gateway import RuntimeGateway

from .render import OutputFn, SimpleRenderer, TerminalRenderer, build_startup_state


InputFn = Callable[[str], str]
"""纯文本输入函数类型。

测试可传入 fake input；无 prompt_toolkit 时默认使用 Python 内置 ``input``。
参数是提示符字符串，返回用户输入的一行文本。
"""

ApprovalDecision = Literal["approve", "deny"]
"""审批命令允许的决策值。"""

DEFAULT_EXIT_COMMANDS = ("exit", "quit", ":q")
"""交互 REPL 中的默认退出命令。用户也可以输入 ``/exit``。"""


async def run_once(
    runtime: RuntimeGateway,
    session_id: str,
    prompt: str,
    *,
    output: OutputFn = print,
) -> None:
    """运行单次 prompt 并输出助手回复。

    Args:
        runtime: CLI 进入 runtime 层的网关。
        session_id: 已打开的会话 ID。
        prompt: 用户传入的单次问题，通常来自 ``codepilot -p``。
        output: 输出函数，默认是 ``print``；测试时可替换为收集函数。

    单次模式使用 ``SimpleRenderer``，只关心面向用户的模型文本，不展示完整启动面板。
    """

    renderer = SimpleRenderer(output=output)
    renderer.render_activity("thinking")
    await render_dispatch(
        runtime.dispatch(session_id, PromptSubmitted(text=prompt)),
        renderer,
    )


async def run_repl(
    runtime: RuntimeGateway,
    session_id: str,
    *,
    input_fn: InputFn = input,
    output: OutputFn = print,
    verbose: bool = False,
    no_color: bool = False,
    exit_commands: tuple[str, ...] = DEFAULT_EXIT_COMMANDS,
) -> None:
    """运行交互式终端循环。

    Args:
        runtime: runtime 网关，负责接收 Prompt/Command/Approval action。
        session_id: 初始会话 ID。``/switch`` 等命令可能让当前会话发生切换。
        input_fn: plain stdin 模式下的输入函数。
        output: plain 输出函数。Rich 模式下 renderer 会使用自己的 Console。
        verbose: 是否显示调试事件、错误堆栈等详细信息。
        no_color: 是否禁用 Rich/prompt_toolkit 彩色界面。
        exit_commands: 退出命令集合，便于测试或嵌入场景覆盖默认值。

    主循环只负责“识别用户输入类型并派发 action”，不直接执行命令或工具。
    """

    from . import shell as shell_module

    renderer = TerminalRenderer(output=output, verbose=verbose, use_rich=not no_color)
    current_session_id = session_id
    view = runtime.describe(current_session_id)
    renderer.render_startup(build_startup_state(view.status))

    shell = shell_module.create_shell(
        history_dir=Path(view.status.workspace) / ".codepilot",
        no_color=no_color,
        commands=tuple(getattr(view, "commands", ()) or ()),
    )

    while True:
        # 每轮循环只读取一段用户输入；输入为空时不触发 runtime。
        try:
            text = await read_user_text(
                runtime,
                current_session_id,
                renderer,
                shell=shell,
                input_fn=input_fn,
            )
        except (EOFError, KeyboardInterrupt):
            renderer.render_status("Bye.", kind="info")
            return

        if not text:
            continue
        if is_exit_text(text, exit_commands=exit_commands):
            renderer.render_status("Bye.", kind="info")
            return

        view = runtime.describe(current_session_id)

        # 审批命令优先于普通斜杠命令，因为 /approve、yes/no 会恢复被暂停的 run。
        try:
            approval_action = approval_action_from_text(
                text,
                pending_approvals=getattr(view, "pending_approvals", ()),
            )
        except ValueError as exc:
            renderer.render_status(str(exc), kind="error")
            continue
        if approval_action is not None:
            await run_approval(runtime, current_session_id, approval_action, renderer, verbose=verbose)
            continue

        # 其他 /xxx 交给 runtime 命令系统，例如 /help、/status、/fork、/switch。
        if text.startswith("/"):
            switched_session_id = await run_command(runtime, current_session_id, text, renderer)
            if switched_session_id is not None:
                runtime.close(current_session_id)
                current_session_id = switched_session_id
                renderer.render_status(
                    f"Switched to session {current_session_id[:8]}..",
                    kind="success",
                )
            continue

        # 剩余文本视为普通用户消息，进入 agent loop。
        await run_prompt(runtime, current_session_id, text, renderer, verbose=verbose)


async def read_user_text(
    runtime: RuntimeGateway,
    session_id: str,
    renderer: TerminalRenderer,
    *,
    shell: Any,
    input_fn: InputFn,
) -> str:
    """读取一段用户输入。

    Args:
        runtime: 用于读取最新 session view，刷新命令补全和底部工具栏。
        session_id: 当前会话 ID。
        renderer: 负责构造提示符、toolbar 或 Rich prompt。
        shell: ``InteractiveShell`` 实例；为 ``None`` 时退回 Rich console 或普通 stdin。
        input_fn: 最朴素的输入函数，仅在没有 prompt_toolkit/Rich shell 时使用。

    Returns:
        去掉首尾空白后的用户输入文本。

    输入层只收集文本，不解析业务含义；解析由 ``run_repl`` 的分流逻辑完成。
    """

    if shell is not None:
        view = runtime.describe(session_id)
        if hasattr(shell, "set_commands"):
            shell.set_commands(tuple(getattr(view, "commands", ()) or ()))
        return (
            await shell.prompt(
                prompt_text=renderer.build_shell_prompt(),
                bottom_toolbar=renderer.build_toolbar(build_startup_state(view.status)),
            )
        ).strip()

    if renderer.has_rich_console:
        return renderer.input(renderer.build_console_prompt()).strip()
    return input_fn(renderer.build_plain_prompt()).strip()


async def run_prompt(
    runtime: RuntimeGateway,
    session_id: str,
    text: str,
    renderer: TerminalRenderer,
    *,
    verbose: bool,
) -> None:
    """派发普通用户消息并渲染 runtime frame。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        text: 用户输入的普通消息。
        renderer: 终端渲染器。
        verbose: 发生异常时是否打印 traceback。

    Ctrl+C 会被转换成 ``RunCancelled``，让 runtime 有机会清理当前 run。
    """

    try:
        renderer.reset()
        renderer.render_activity("thinking")
        await render_dispatch(
            runtime.dispatch(session_id, PromptSubmitted(text=text)),
            renderer,
        )
    except KeyboardInterrupt:
        renderer.render_status("Cancelled", kind="cancelled")
        async for _frame in runtime.dispatch(
            session_id,
            RunCancelled(reason="keyboard_interrupt"),
        ):
            pass
    except Exception as exc:
        renderer.render_status(f"Error: {exc}", kind="error")
        if verbose:
            print_traceback()


async def run_approval(
    runtime: RuntimeGateway,
    session_id: str,
    action: ApprovalDecided,
    renderer: TerminalRenderer,
    *,
    verbose: bool,
) -> None:
    """派发审批结果并渲染恢复后的 run。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        action: ``ApprovalDecided``，包含 approval_id、approve/deny 和可选原因。
        renderer: 终端渲染器。
        verbose: 发生异常时是否打印 traceback。

    审批不是普通 prompt；它恢复此前因工具权限暂停的 agent loop。
    """

    try:
        renderer.reset()
        renderer.render_activity("resuming")
        await render_dispatch(runtime.dispatch(session_id, action), renderer)
    except Exception as exc:
        renderer.render_status(f"Approval error: {exc}", kind="error")
        if verbose:
            print_traceback()


async def run_command(
    runtime: RuntimeGateway,
    session_id: str,
    text: str,
    renderer: TerminalRenderer,
) -> str | None:
    """派发斜杠命令并渲染人类可读输出。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        text: 用户输入的完整命令文本，例如 ``/help`` 或 ``/switch xxx``。
        renderer: 终端渲染器。

    Returns:
        如果命令切换了会话，返回新的 session id；否则返回 ``None``。
    """

    try:
        result = None
        final_record = None
        run_finished = False
        async for frame in runtime.dispatch(session_id, CommandSubmitted(text=text)):
            if isinstance(frame, CommandFinishedFrame):
                result = frame.record
                renderer.render_command_output(getattr(result, "output_lines", ()) or ())
                continue
            if isinstance(frame, ProgressFrame):
                renderer.render_progress_event(frame.event)
                continue
            if isinstance(frame, ApprovalRequiredFrame):
                renderer.render_approval_required(frame)
                continue
            if isinstance(frame, RunFinishedFrame):
                final_record = frame.record
                run_finished = True
                continue
            if isinstance(frame, RunPausedFrame):
                final_record = frame.record
                continue
            if isinstance(frame, FailedFrame):
                raise runtime_error_from_frame(frame.error)
    except Exception as exc:
        renderer.render_status(f"Command error: {exc}", kind="error")
        return None

    if result is None:
        renderer.render_status("Command error: runtime command finished without a result", kind="error")
        return None
    renderer.render_final(final_record)
    if run_finished:
        render_input_ready = getattr(renderer, "render_input_ready", None)
        if callable(render_input_ready):
            render_input_ready()
    switched_session_id = getattr(result, "switched_session_id", None)
    if switched_session_id is not None:
        return str(switched_session_id)
    if getattr(result, "handled", False):
        return None

    unknown = text.partition(" ")[0]
    renderer.render_command_output(
        [
            f"Unknown command: {unknown}",
            "Type /help to see available commands.",
        ]
    )
    return None


async def dispatch_command(runtime: RuntimeGateway, session_id: str, text: str) -> Any:
    """通过 runtime 边界提交斜杠命令并返回命令记录。

    Args:
        runtime: runtime 网关。
        session_id: 当前会话 ID。
        text: 完整命令文本。

    Returns:
        ``CommandFinishedFrame.record``，其中包含 handled、output_lines、data 等信息。

    Raises:
        RuntimeFrameError: runtime 返回 ``FailedFrame`` 或命令流没有正常结束。
    """

    async for frame in runtime.dispatch(session_id, CommandSubmitted(text=text)):
        if isinstance(frame, CommandFinishedFrame):
            return frame.record
        if isinstance(frame, FailedFrame):
            raise runtime_error_from_frame(frame.error)
    raise RuntimeFrameError("Runtime command finished without a result")


async def render_dispatch(frames: Any, renderer: Any) -> None:
    """消费一次 runtime dispatch 的 frame 流并交给 renderer。

    Args:
        frames: ``RuntimeGateway.dispatch`` 返回的异步 frame 迭代器。
        renderer: ``TerminalRenderer`` 或 ``SimpleRenderer``。只要求它实现本函数调用的方法。

    Runtime frame 是 CLI 层和 runtime 层之间的显示协议：
    - ``ProgressFrame``：模型增量、工具开始/结束等过程事件。
    - ``ApprovalRequiredFrame``：工具需要权限审批，CLI 显示审批提示并停止本轮。
    - ``RunPausedFrame``：run 暂停等待用户输入或审批，保留同一个 run。
    - ``RunFinishedFrame``：run 完成，保存最终记录。
    - ``FailedFrame``：runtime 出错，转换成 CLI 可处理异常。
    """

    final_record = None
    run_finished = False
    async for frame in frames:
        if isinstance(frame, ProgressFrame):
            renderer.render_progress_event(frame.event)
        elif isinstance(frame, CommandFinishedFrame):
            renderer.render_command_output(getattr(frame.record, "output_lines", ()) or ())
        elif isinstance(frame, ApprovalRequiredFrame):
            renderer.render_approval_required(frame)
        elif isinstance(frame, RunFinishedFrame):
            final_record = frame.record
            run_finished = True
        elif isinstance(frame, RunPausedFrame):
            final_record = frame.record
        elif isinstance(frame, FailedFrame):
            raise runtime_error_from_frame(frame.error)
    renderer.render_final(final_record)
    if run_finished:
        render_input_ready = getattr(renderer, "render_input_ready", None)
        if callable(render_input_ready):
            render_input_ready()


def is_exit_text(
    text: str,
    *,
    exit_commands: Iterable[str] = DEFAULT_EXIT_COMMANDS,
) -> bool:
    """判断用户输入是否是退出命令。

    Args:
        text: 用户输入文本。
        exit_commands: 允许的退出命令集合，不要求带 ``/`` 前缀。

    Returns:
        是退出命令时返回 ``True``。
    """
    normalized = text.strip()
    if normalized == "/exit":
        return True
    bare = normalized.lstrip("/")
    return bare in {command.strip().lstrip("/") for command in exit_commands if command.strip()}


def approval_action_from_text(
    text: str,
    *,
    pending_approvals: tuple[Any, ...] | list[Any] = (),
) -> ApprovalDecided | None:
    """把 ``/approve`` 和 ``/deny`` 文本解析成 runtime action。

    Args:
        text: 用户输入文本，格式为 ``/approve <approval_id> [reason]`` 或
            ``/deny <approval_id> [reason]``。

    Returns:
        可直接派发给 runtime 的 ``ApprovalDecided``；如果不是审批命令则返回 ``None``。
    """

    raw = text.strip()
    parts = raw.lstrip("/").split(maxsplit=2)
    if not parts:
        return None
    decision = _approval_decision_alias(parts[0])
    if decision is None:
        return None
    approval_token = parts[1].strip() if len(parts) >= 2 else ""
    reason = parts[2].strip() if len(parts) >= 3 else ""
    approval_id = _resolve_approval_id(approval_token, tuple(pending_approvals))
    if approval_id is None:
        if raw.startswith("/"):
            return None
        return None
    return ApprovalDecided(
        approval_id=approval_id,
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
    )


def _approval_decision_alias(value: str) -> str | None:
    text = value.strip().lower().lstrip("/")
    if text in {"approve", "yes", "y", "ok", "同意", "批准", "可以", "确认"}:
        return "approve"
    if text in {"deny", "no", "n", "拒绝", "不同意", "不行"}:
        return "deny"
    return None


def _resolve_approval_id(token: str, pending: tuple[Any, ...]) -> str | None:
    approvals = [_approval_id(item) for item in pending]
    approvals = [item for item in approvals if item]
    if token:
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(approvals):
                return approvals[index - 1]
        return token
    if len(approvals) == 1:
        return approvals[0]
    if len(approvals) > 1:
        raise ValueError("Multiple approvals are pending. Use /approve <number> or /deny <number>.")
    return None


def _approval_id(item: Any) -> str | None:
    if isinstance(item, dict):
        value = item.get("approval_id")
    else:
        value = getattr(item, "approval_id", None)
    text = str(value).strip() if value is not None else ""
    return text or None


def frame_error_message(error: Any) -> str:
    """从 runtime error payload 中提取适合展示的人类消息。"""
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Runtime failed")
    return str(error)


class RuntimeFrameError(RuntimeError):
    """消费 runtime frame 流时浮出的失败。

    ``code`` 字段用于保留 runtime 的错误类型，CLI 可以在 verbose 或 RPC 模式下继续传递。
    """

    code = "runtime.dispatch_failed"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


def runtime_error_from_frame(error: Any) -> RuntimeFrameError:
    """把 ``FailedFrame.error`` 转换成 CLI 内部异常。"""
    if isinstance(error, dict):
        return RuntimeFrameError(
            frame_error_message(error),
            code=str(error.get("code") or "runtime.dispatch_failed"),
        )
    return RuntimeFrameError(str(error))


def print_traceback() -> None:
    """在 verbose 模式下打印当前异常堆栈。"""
    import traceback

    traceback.print_exc()


__all__ = [
    "DEFAULT_EXIT_COMMANDS",
    "InputFn",
    "RuntimeFrameError",
    "approval_action_from_text",
    "dispatch_command",
    "frame_error_message",
    "is_exit_text",
    "read_user_text",
    "render_dispatch",
    "run_command",
    "run_once",
    "run_prompt",
    "run_repl",
    "runtime_error_from_frame",
]
