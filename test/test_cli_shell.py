from __future__ import annotations

import asyncio
import inspect
from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from codepilot.interfaces.cli.shell import CODEPILOT_STYLE, InteractiveShell
from codepilot.runtime.command_registry import builtin_commands
from codepilot.runtime.types import SessionStatus


def test_shell_uses_coding_agent_prompt_and_visual_styles() -> None:
    prompt_default = inspect.signature(InteractiveShell.prompt).parameters["prompt_text"].default
    style_names = {name for name, _value in CODEPILOT_STYLE.style_rules}

    assert prompt_default == "› "
    assert "prompt" in style_names
    assert "bottom-toolbar" in style_names
    assert "completion-menu.completion" in style_names
    assert "completion-menu.completion.current" in style_names


def test_shell_command_completion_uses_runtime_command_registry(
    monkeypatch,
    tmp_path,
) -> None:
    from codepilot.interfaces.cli import shell as shell_module

    captured: dict[str, object] = {}

    class FakePromptSession:
        def __init__(self, **kwargs):
            captured["completer"] = kwargs["completer"]

    monkeypatch.setattr(shell_module, "PromptSession", FakePromptSession)
    InteractiveShell(history_dir=tmp_path)

    completer = captured["completer"]
    completions = list(completer.get_completions(Document("/mem"), None))
    runtime_memory = next(command for command in builtin_commands() if command.name == "memory")

    assert [completion.text for completion in completions] == ["memory"]
    assert completions[0].display_meta_text == runtime_memory.description


def test_shell_ctrl_c_exits_prompt_and_keyboard_interrupt_is_not_swallowed(
    monkeypatch,
    tmp_path,
) -> None:
    from codepilot.interfaces.cli import shell as shell_module

    captured: dict[str, object] = {}

    class FakePromptSession:
        def __init__(self, **kwargs):
            captured["bindings"] = kwargs["key_bindings"]

        async def prompt_async(self, *_args, **_kwargs):
            raise KeyboardInterrupt

    monkeypatch.setattr(shell_module, "PromptSession", FakePromptSession)
    shell = InteractiveShell(history_dir=tmp_path)

    bindings = captured["bindings"]
    control_c = next(
        binding
        for binding in bindings.bindings
        if any("ControlC" in str(key) for key in binding.keys)
    )

    class FakeApp:
        def exit(self, *, exception):
            captured["exit_exception"] = exception

    class FakeEvent:
        app = FakeApp()

    control_c.handler(FakeEvent())

    assert captured["exit_exception"] is KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        asyncio.run(shell.prompt())


def test_interactive_runner_passes_dynamic_toolbar_and_prompt(monkeypatch) -> None:
    from codepilot.interfaces.cli import runner

    calls: dict[str, object] = {}

    class FakeShell:
        async def prompt(self, **kwargs):
            calls["prompt"] = kwargs
            return "/exit"

    class FakeRenderer:
        def __init__(self, **kwargs):
            calls["renderer_init"] = kwargs
            self.has_rich_console = False

        def render_startup(self, state):
            calls["startup"] = state

        def build_toolbar(self, state):
            calls["toolbar_state"] = state
            return "<b>model</b> · permission"

        def render_status(self, message, *, kind="info"):
            calls["status"] = (message, kind)

    class FakeRuntime:
        def get_session_status(self, session_id):
            return SessionStatus(
                session_id=session_id,
                model_id="deepseek/deepseek-chat",
                workspace="E:/workspace",
                permission_mode="workspace-write",
                message_count=0,
                leaf_id="leaf",
            )

        def get_workspace(self, _session_id):
            return Path("E:/workspace")

    monkeypatch.setattr(runner, "TerminalRenderer", FakeRenderer)
    monkeypatch.setattr("codepilot.interfaces.cli.shell.create_shell", lambda **_kwargs: FakeShell())

    asyncio.run(runner.run_interactive(FakeRuntime(), "session-123"))

    assert calls["renderer_init"]["output"] is print
    assert calls["prompt"] == {
        "prompt_text": "› ",
        "bottom_toolbar": "<b>model</b> · permission",
    }
    assert calls["status"] == ("Bye.", "info")


def test_interactive_runner_exits_when_shell_raises_keyboard_interrupt(monkeypatch) -> None:
    from codepilot.interfaces.cli import runner

    calls: dict[str, object] = {}

    class FakeShell:
        async def prompt(self, **_kwargs):
            raise KeyboardInterrupt

    class FakeRenderer:
        def __init__(self, **_kwargs):
            self.has_rich_console = False

        def render_startup(self, state):
            return None

        def build_toolbar(self, _state):
            return "toolbar"

        def render_status(self, message, *, kind="info"):
            calls["status"] = (message, kind)

    class FakeRuntime:
        def get_session_status(self, session_id):
            return SessionStatus(
                session_id=session_id,
                model_id="deepseek/deepseek-chat",
                workspace="E:/workspace",
                permission_mode="workspace-write",
                message_count=0,
                leaf_id="leaf",
            )

        def get_workspace(self, _session_id):
            return Path("E:/workspace")

    monkeypatch.setattr(runner, "TerminalRenderer", FakeRenderer)
    monkeypatch.setattr("codepilot.interfaces.cli.shell.create_shell", lambda **_kwargs: FakeShell())

    asyncio.run(runner.run_interactive(FakeRuntime(), "session-123"))

    assert calls["status"] == ("Bye.", "info")
