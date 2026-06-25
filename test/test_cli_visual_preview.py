from __future__ import annotations

from io import StringIO

from rich.console import Console

from codepilot.interfaces.cli.renderer import TerminalRenderer
from codepilot.interfaces.cli.startup import CliStartupState
from codepilot.protocols import LLMErrorInfo


def test_rich_cli_preview_has_compact_coding_agent_hierarchy(monkeypatch) -> None:
    output = StringIO()
    renderer = TerminalRenderer(use_rich=True)
    renderer._console = Console(
        file=output,
        width=88,
        color_system=None,
        force_terminal=False,
    )
    timestamps = iter([10.0, 10.614])
    monkeypatch.setattr(
        "codepilot.interfaces.cli.renderer.time.time",
        lambda: next(timestamps),
    )

    renderer.render_startup(
        CliStartupState(
            version="0.3",
            model_id="deepseek/deepseek-chat",
            workspace="E:/Project_python/agent/Codepilot",
            session_id="session-123456789",
            permission_mode="workspace-write",
        )
    )
    renderer.handle_event({
        "type": "tool_execution_start",
        "toolCallId": "read-1",
        "toolName": "read",
        "args": {"path": "src/codepilot/core/agent_loop.py"},
    })
    renderer.handle_event({
        "type": "tool_execution_end",
        "toolCallId": "read-1",
        "toolName": "read",
        "status": "success",
        "isError": False,
    })
    renderer.handle_event({
        "type": "tool_approval_required",
        "toolName": "bash",
        "args": {"command": "git status --short"},
        "riskLevel": "medium",
    })
    info = LLMErrorInfo(
        code="llm.provider_response",
        message="400 Bad Request",
        retryable=False,
        kind="provider_response",
        provider="deepseek",
        model="deepseek-chat",
        status_code=400,
        details={"response_text": '{"error":{"message":"Invalid request"}}'},
    )
    renderer.handle_event({
        "type": "error",
        "error": info.code,
        "message": info.message,
        "provider": info.provider,
        "model": info.model,
        "errorInfo": info,
    })

    preview = output.getvalue()

    assert "Codepilot 0.3  Local coding agent" in preview
    assert "deepseek/deepseek-chat" in preview
    assert "◆ read  src/codepilot/core/agent_loop.py" in preview
    assert "└ done  614ms" in preview
    assert "Permission required" in preview
    assert "bash git status --short" in preview
    assert "Error · llm.provider_response" in preview
    assert 'Provider response: {"error":{"message":"Invalid request"}}' in preview
    assert "│ Model       " not in preview


def test_rich_error_title_treats_error_code_as_plain_text() -> None:
    output = StringIO()
    renderer = TerminalRenderer(use_rich=True)
    renderer._console = Console(
        file=output,
        width=72,
        color_system=None,
        force_terminal=False,
    )

    renderer.handle_event({
        "type": "error",
        "error": "provider[invalid]",
        "message": "Request failed",
    })

    assert "Error · provider[invalid]" in output.getvalue()
