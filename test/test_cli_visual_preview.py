from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

from rich.console import Console

from codepilot.interfaces.cli.render import TerminalRenderer
from codepilot.interfaces.cli.render import CliStartupState
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
        "codepilot.interfaces.cli.render.time.time",
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
    renderer.render_progress_event({
        "type": "tool_started",
        "toolCallId": "read-1",
        "toolName": "read",
        "args": {"path": "src/codepilot/core/loop.py"},
    })
    renderer.render_progress_event({
        "type": "tool_completed",
        "toolCallId": "read-1",
        "toolName": "read",
        "status": "success",
        "isError": False,
    })
    renderer.render_approval_required(
        SimpleNamespace(
            approval=SimpleNamespace(
                tool_name="bash",
                arguments={"command": "git status --short"},
                risk=SimpleNamespace(level="medium"),
                approval_id="approval_1",
            )
        )
    )
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
    renderer.render_progress_event({
        "type": "error",
        "error": info.code,
        "message": info.message,
        "provider": info.provider,
        "model": info.model,
        "errorInfo": info,
    })

    preview = output.getvalue()

    assert "Codepilot 0.3  cyber engineering console" in preview
    assert "C P" in preview
    assert "deepseek/deepseek-chat" in preview
    assert "↯ reading read  src/codepilot/core/loop.py" in preview
    assert "◆ ok  614ms" in preview
    assert "CP // PERMISSION REQUIRED" in preview
    assert "bash" in preview
    assert "git status --short" in preview
    assert "CP // ERROR · llm.provider_response" in preview
    assert 'Provider response: {"error":{"message":"Invalid request"}}' in preview
    assert "Neural workspace online" in preview
    assert "Command uplink" in preview


def test_rich_error_title_treats_error_code_as_plain_text() -> None:
    output = StringIO()
    renderer = TerminalRenderer(use_rich=True)
    renderer._console = Console(
        file=output,
        width=72,
        color_system=None,
        force_terminal=False,
    )

    renderer.render_progress_event({
        "type": "error",
        "error": "provider[invalid]",
        "message": "Request failed",
    })

    assert "CP // ERROR · provider[invalid]" in output.getvalue()
