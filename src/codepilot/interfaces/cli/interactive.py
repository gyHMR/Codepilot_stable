from __future__ import annotations

"""Human CLI flow: read text, dispatch runtime actions, render frames."""

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
)
from codepilot.runtime.gateway import RuntimeGateway

from .render import OutputFn, SimpleRenderer, TerminalRenderer, build_startup_state


InputFn = Callable[[str], str]
ApprovalDecision = Literal["approve", "deny"]
DEFAULT_EXIT_COMMANDS = ("exit", "quit", ":q")


async def run_once(
    runtime: RuntimeGateway,
    session_id: str,
    prompt: str,
    *,
    output: OutputFn = print,
) -> None:
    """Run one prompt and print only the assistant-facing stream."""

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
    """Run the interactive terminal loop."""

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

        approval_action = approval_action_from_text(text)
        if approval_action is not None:
            await run_approval(runtime, current_session_id, approval_action, renderer, verbose=verbose)
            continue

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

        await run_prompt(runtime, current_session_id, text, renderer, verbose=verbose)


async def read_user_text(
    runtime: RuntimeGateway,
    session_id: str,
    renderer: TerminalRenderer,
    *,
    shell: Any,
    input_fn: InputFn,
) -> str:
    """Read one input from prompt_toolkit, Rich, or plain stdin."""

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
    """Dispatch a normal user prompt and render its runtime frames."""

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
    """Dispatch an approval decision and render the resumed run."""

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
    """Dispatch a slash command and render its human output."""

    try:
        result = await dispatch_command(runtime, session_id, text)
    except Exception as exc:
        renderer.render_status(f"Command error: {exc}", kind="error")
        return None

    renderer.render_command_output(getattr(result, "output_lines", ()) or ())
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
    """Submit a slash command through the runtime boundary and return its record."""

    async for frame in runtime.dispatch(session_id, CommandSubmitted(text=text)):
        if isinstance(frame, CommandFinishedFrame):
            return frame.record
        if isinstance(frame, FailedFrame):
            raise runtime_error_from_frame(frame.error)
    raise RuntimeFrameError("Runtime command finished without a result")


async def render_dispatch(frames: Any, renderer: Any) -> None:
    """Render the frame stream from one runtime dispatch."""

    final_record = None
    async for frame in frames:
        if isinstance(frame, ProgressFrame):
            renderer.render_progress_event(frame.event)
        elif isinstance(frame, ApprovalRequiredFrame):
            renderer.render_approval_required(frame)
        elif isinstance(frame, RunFinishedFrame):
            final_record = frame.record
        elif isinstance(frame, FailedFrame):
            raise runtime_error_from_frame(frame.error)
    renderer.render_final(final_record)


def is_exit_text(
    text: str,
    *,
    exit_commands: Iterable[str] = DEFAULT_EXIT_COMMANDS,
) -> bool:
    normalized = text.strip()
    if normalized == "/exit":
        return True
    bare = normalized.lstrip("/")
    return bare in {command.strip().lstrip("/") for command in exit_commands if command.strip()}


def approval_action_from_text(text: str) -> ApprovalDecided | None:
    """Parse ``/approve`` and ``/deny`` into runtime actions."""

    parts = text.strip().lstrip("/").split(maxsplit=2)
    if not parts:
        return None
    decision = parts[0].lower()
    if decision not in {"approve", "deny"}:
        return None
    approval_id = parts[1].strip() if len(parts) >= 2 else ""
    reason = parts[2].strip() if len(parts) >= 3 else ""
    return ApprovalDecided(
        approval_id=approval_id,
        decision=decision,  # type: ignore[arg-type]
        reason=reason,
    )


def frame_error_message(error: Any) -> str:
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or "Runtime failed")
    return str(error)


class RuntimeFrameError(RuntimeError):
    """Runtime failure surfaced while consuming a frame stream."""

    code = "runtime.dispatch_failed"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        if code:
            self.code = code


def runtime_error_from_frame(error: Any) -> RuntimeFrameError:
    if isinstance(error, dict):
        return RuntimeFrameError(
            frame_error_message(error),
            code=str(error.get("code") or "runtime.dispatch_failed"),
        )
    return RuntimeFrameError(str(error))


def print_traceback() -> None:
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
