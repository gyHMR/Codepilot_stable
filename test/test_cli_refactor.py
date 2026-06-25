"""CLI 重构后的测试。

覆盖：
- TerminalRenderer 渲染
- 启动状态构建
- 配置脱敏
- 新 CLI 参数
- Session 切换
"""

import io
import json
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from codepilot.interfaces.cli.renderer import (
    TerminalRenderer,
    SimpleRenderer,
)
from codepilot.interfaces.cli.startup import CliStartupState, build_startup_state
from codepilot.runtime.types import SessionStatus
from codepilot.protocols import AssistantMessage, LLMErrorInfo, TextContent, Usage, Cost


# ── TerminalRenderer 测试 ─────────────────────────────────────────

class TestTerminalRenderer:
    """测试 TerminalRenderer 的渲染逻辑。"""

    def test_init_with_rich(self):
        """测试使用 rich 初始化。"""
        renderer = TerminalRenderer(use_rich=True)
        assert renderer.use_rich is True
        assert renderer._console is not None

    def test_init_without_rich(self):
        """测试不使用 rich 初始化。"""
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
        assert renderer.use_rich is False
        assert renderer._console is None
        assert renderer._output == output

    def test_reset(self):
        """测试重置状态。"""
        renderer = TerminalRenderer(use_rich=False, output=MagicMock())
        renderer._stream_started = True
        renderer._current_tool = "Read"
        renderer._tool_start_time = 100.0

        renderer.reset()

        assert renderer._stream_started is False
        assert renderer._current_tool is None
        assert renderer._tool_start_time == 0

    def test_handle_text_delta(self):
        """测试处理流式文本更新。"""
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        event = {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "text_delta",
                "delta": "Hello",
            },
        }

        renderer.handle_event(event)
        assert renderer._stream_started is True

    def test_handle_tool_start(self):
        """测试处理工具开始事件。"""
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        event = {
            "type": "tool_execution_start",
            "toolName": "Read",
            "args": {"file_path": "/test/file.py"},
        }

        renderer.handle_event(event)
        assert renderer._current_tool == "Read"
        assert renderer._tool_start_time > 0

    def test_handle_lowercase_read_and_ls_show_targets(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.handle_event({
            "type": "tool_execution_start",
            "toolCallId": "read-1",
            "toolName": "read",
            "args": {"path": "src/codepilot/core/agent_loop.py", "offset": 10, "limit": 20},
        })
        renderer.handle_event({
            "type": "tool_execution_start",
            "toolCallId": "ls-1",
            "toolName": "ls",
            "args": {"path": "src/codepilot"},
        })

        rendered = [call.args[0] for call in output.call_args_list]
        assert "[tool] read  src/codepilot/core/agent_loop.py:10-29" in rendered
        assert "[tool] ls  src/codepilot" in rendered

    def test_tool_target_is_shortened_for_narrow_terminal_readability(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
        long_path = "src/" + "/".join(["very_long_directory"] * 8) + "/module.py"

        renderer.handle_event({
            "type": "tool_execution_start",
            "toolCallId": "read-long",
            "toolName": "read",
            "args": {"path": long_path},
        })

        rendered = output.call_args.args[0]
        assert rendered.startswith("[tool] read  …")
        assert rendered.endswith("/module.py")
        assert len(rendered) <= 78

    def test_parallel_tool_timings_are_tracked_by_call_id(self, monkeypatch):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
        timestamps = iter([10.0, 11.0, 12.0, 13.0])
        monkeypatch.setattr(
            "codepilot.interfaces.cli.renderer.time.time",
            lambda: next(timestamps),
        )

        renderer.handle_event({
            "type": "tool_execution_start",
            "toolCallId": "read-1",
            "toolName": "read",
            "args": {"path": "a.py"},
        })
        renderer.handle_event({
            "type": "tool_execution_start",
            "toolCallId": "ls-1",
            "toolName": "ls",
            "args": {"path": "src"},
        })
        renderer.handle_event({
            "type": "tool_execution_end",
            "toolCallId": "read-1",
            "toolName": "read",
            "isError": False,
        })
        renderer.handle_event({
            "type": "tool_execution_end",
            "toolCallId": "ls-1",
            "toolName": "ls",
            "isError": False,
        })

        rendered = [call.args[0] for call in output.call_args_list]
        assert rendered[-2:] == ["  [ok] 2.0s", "  [ok] 2.0s"]

    def test_plain_startup_is_compact_and_has_no_box_table(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
        state = CliStartupState(
            version="0.3",
            model_id="deepseek/deepseek-chat",
            workspace="E:/Project_python/agent/Codepilot",
            session_id="session-123456789",
            permission_mode="workspace-write",
        )

        renderer.render_startup(state)

        rendered = "\n".join(call.args[0] for call in output.call_args_list)
        assert "Codepilot 0.3" in rendered
        assert "local coding agent" in rendered
        assert "deepseek/deepseek-chat" in rendered
        assert "workspace-write" in rendered
        assert "session-1.." in rendered
        assert "╭─" not in rendered
        assert "│ Model" not in rendered

    def test_toolbar_contains_current_model_permission_and_shortcuts(self):
        renderer = TerminalRenderer(use_rich=False, output=MagicMock())
        state = CliStartupState(
            version="0.3",
            model_id="deepseek/deepseek-chat",
            workspace="E:/Project_python/agent/Codepilot",
            session_id="session-123456789",
            permission_mode="workspace-write",
        )

        toolbar = renderer.build_toolbar(state)

        assert "<b>deepseek/deepseek-chat</b>" in toolbar
        assert "workspace-write" in toolbar
        assert "/help" in toolbar
        assert "Ctrl+C" in toolbar

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("info", "• Working"),
            ("success", "✓ Working"),
            ("warning", "! Working"),
            ("error", "× Working"),
            ("cancelled", "■ Working"),
        ],
    )
    def test_plain_status_messages_use_consistent_symbols(self, kind, expected):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.render_status("Working", kind=kind)

        output.assert_called_once_with(expected)

    def test_handle_error(self):
        """测试处理错误事件。"""
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        event = {
            "type": "error",
            "error": "Test error",
            "message": "Error details",
            "provider": "deepseek",
            "model": "deepseek-chat",
        }

        renderer.handle_event(event)
        output.assert_called()

    def test_handle_error_shows_provider_response_body(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
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

        rendered = [call.args[0] for call in output.call_args_list]
        assert '  Provider response: {"error":{"message":"Invalid request"}}' in rendered


# ── SimpleRenderer 测试 ──────────────────────────────────────────

class TestSimpleRenderer:
    """测试 SimpleRenderer 的渲染逻辑。"""

    def test_handle_text_delta(self):
        """测试处理流式文本更新。"""
        output = MagicMock()
        renderer = SimpleRenderer(output=output)

        event = {
            "type": "message_update",
            "assistantMessageEvent": {
                "type": "text_delta",
                "delta": "Hello",
            },
        }

        renderer.handle_event(event)
        assert renderer._stream_started is True

    def test_render_final_with_text(self):
        """测试渲染最终结果（有文本）。"""
        output = MagicMock()
        renderer = SimpleRenderer(output=output)
        renderer._stream_started = True

        renderer.render_final(None)
        output.assert_called()

    def test_render_final_without_text(self):
        """测试渲染最终结果（无文本）。"""
        output = MagicMock()
        renderer = SimpleRenderer(output=output)
        renderer._stream_started = False

        # 创建 mock 的 AssistantMessage
        message = MagicMock(spec=AssistantMessage)
        message.content = [TextContent(text="Hello")]

        renderer.render_final(message)
        output.assert_called_with("Hello")


def test_render_prompt_run_forwards_events_and_final_message():
    from codepilot.interfaces.cli.runner import _render_prompt_run

    class FakeRuntime:
        def __init__(self):
            self.sent = None
            self.final_message = AssistantMessage(content=[TextContent(text="done")])

        async def send_message(self, session_id, message):
            self.sent = (session_id, message.text)
            yield {"type": "message_update", "delta": "hello"}

        def get_latest_assistant_message(self, session_id):
            assert session_id == "session_1"
            return self.final_message

    class FakeRenderer:
        def __init__(self):
            self.events = []
            self.final = None

        def handle_event(self, event):
            self.events.append(event)

        def render_final(self, message):
            self.final = message

    runtime = FakeRuntime()
    renderer = FakeRenderer()

    asyncio.run(_render_prompt_run(runtime, "session_1", "hello", renderer))

    assert runtime.sent == ("session_1", "hello")
    assert renderer.events == [{"type": "message_update", "delta": "hello"}]
    assert renderer.final is runtime.final_message


def test_run_rpc_emits_jsonl_contract_for_state_prompt_errors_and_shutdown(monkeypatch):
    from codepilot.interfaces.cli.runner import run_rpc
    from codepilot.runtime.service import SessionBusyError

    class FakeRuntime:
        def __init__(self):
            self.prompt_calls = 0

        async def send_message(self, session_id, message):
            assert session_id == "session_1"
            self.prompt_calls += 1
            if self.prompt_calls == 1:
                assert message.text == "hello"
            else:
                assert message.text == "busy"
                raise SessionBusyError("Session is already running")
            yield {"type": "message_update", "delta": "hi"}

        def get_session_state(self, session_id):
            assert session_id == "session_1"
            return {
                "session_id": session_id,
                "message_count": 2,
                "entry_ids": ["entry_1"],
                "leaf_id": "entry_1",
            }

    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"type": "state", "id": "state_1"}),
                json.dumps({"type": "prompt", "id": "prompt_1", "text": "hello"}),
                json.dumps({"type": "prompt", "id": "prompt_2", "text": "busy"}),
                "{not-json",
                json.dumps({"type": "shutdown", "id": "shutdown_1"}),
            ]
        )
        + "\n"
    )
    output: list[str] = []
    monkeypatch.setattr("sys.stdin", stdin)

    asyncio.run(run_rpc(FakeRuntime(), "session_1", output=output.append))

    messages = [json.loads(line) for line in output]
    assert messages[0] == {
        "type": "rpc_ready",
        "session_id": "session_1",
        "protocol_version": "1.2",
    }
    assert messages[1] == {
        "type": "response",
        "id": "state_1",
        "command": "state",
        "status": "ok",
        "data": {
            "session_id": "session_1",
            "message_count": 2,
            "entry_ids": ["entry_1"],
            "leaf_id": "entry_1",
        },
    }
    assert messages[2] == {
        "type": "event",
        "event": {"type": "message_update", "delta": "hi"},
    }
    assert messages[3]["command"] == "prompt"
    assert messages[3]["status"] == "ok"
    assert messages[4]["status"] == "error"
    assert messages[4]["command"] == "prompt"
    assert messages[4]["error"]["code"] == "runtime.session_busy"
    assert messages[5]["status"] == "error"
    assert messages[5]["error"]["code"] == "invalid_json"
    assert messages[6]["command"] == "shutdown"
    assert messages[6]["status"] == "ok"


def test_rpc_ready_signal_uses_named_protocol_version() -> None:
    from codepilot.interfaces.cli.rpc_protocol import (
        RPC_PROTOCOL_VERSION,
        emit_rpc_ready,
    )

    emitted: list[dict] = []
    emit_rpc_ready(emitted.append, session_id=" session_1 ")

    assert RPC_PROTOCOL_VERSION == "1.2"
    assert emitted == [
        {
            "type": "rpc_ready",
            "session_id": "session_1",
            "protocol_version": "1.2",
        }
    ]

    with pytest.raises(ValueError, match="session_id"):
        emit_rpc_ready(emitted.append, session_id=" ")


def test_rpc_ok_response_requires_command_name() -> None:
    from codepilot.interfaces.cli.rpc_protocol import emit_rpc_ok

    emitted: list[dict] = []
    emit_rpc_ok(emitted.append, req_id="request_1", command=" state ")

    assert emitted == [
        {
            "type": "response",
            "id": "request_1",
            "command": "state",
            "status": "ok",
        }
    ]

    with pytest.raises(ValueError, match="command"):
        emit_rpc_ok(emitted.append, req_id="request_2", command=" ")


def test_rpc_error_mapping_uses_runtime_error_codes() -> None:
    from codepilot.interfaces.cli.rpc_protocol import rpc_error_from_exception
    from codepilot.runtime.service import SessionBusyError

    runtime_error = SessionBusyError("Session is already running")
    generic_error = ValueError("missing field")

    assert rpc_error_from_exception(runtime_error).code == "runtime.session_busy"
    assert rpc_error_from_exception(runtime_error).message == "Session is already running"
    assert rpc_error_from_exception(generic_error).code == "execution_error"
    assert rpc_error_from_exception(generic_error).message == "missing field"


def test_rpc_error_requires_non_empty_code_and_message() -> None:
    from codepilot.interfaces.cli.rpc_protocol import RpcError, rpc_error_from_exception

    with pytest.raises(ValueError, match="code"):
        RpcError(code="  ", message="Something failed")

    with pytest.raises(ValueError, match="message"):
        RpcError(code="execution_error", message="")

    mapped = rpc_error_from_exception(ValueError())

    assert mapped.code == "execution_error"
    assert mapped.message == "ValueError"


def test_run_options_normalizes_cli_entry_contract() -> None:
    from codepilot.interfaces.cli.runner import RunOptions

    runtime = object()
    output = lambda _text: None
    input_fn = lambda _prompt: "hello"

    options = RunOptions(
        mode=" print ",  # type: ignore[arg-type]
        session_id=" session_1 ",
        runtime=runtime,  # type: ignore[arg-type]
        output=output,
        input_fn=input_fn,
        exit_commands=[" /exit ", " ", "q"],  # type: ignore[arg-type]
    )

    assert options.mode == "print"
    assert options.session_id == "session_1"
    assert options.exit_commands == ("exit", "q")

    with pytest.raises(ValueError, match="mode"):
        RunOptions(mode="daemon", session_id="session_1", runtime=runtime)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="session_id"):
        RunOptions(mode="print", session_id=" ", runtime=runtime)

    with pytest.raises(TypeError, match="output"):
        RunOptions(
            mode="print",
            session_id="session_1",
            runtime=runtime,  # type: ignore[arg-type]
            output="stdout",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="input_fn"):
        RunOptions(
            mode="interactive",
            session_id="session_1",
            runtime=runtime,  # type: ignore[arg-type]
            input_fn="stdin",  # type: ignore[arg-type]
        )


# ── CliStartupState 测试 ─────────────────────────────────────────

class TestCliStartupState:
    """测试 CliStartupState 数据结构。"""

    def test_build_startup_state(self):
        """测试从 SessionStatus 构建启动状态。"""
        status = SessionStatus(
            session_id="test_session_123",
            model_id="deepseek/deepseek-chat",
            workspace="/path/to/workspace",
            permission_mode="read-only",
            message_count=10,
            leaf_id="leaf_123",
        )

        state = build_startup_state(status, warnings=["Test warning"])

        assert state.version == "0.3"
        assert state.model_id == "deepseek/deepseek-chat"
        assert state.workspace == "/path/to/workspace"
        assert state.session_id == "test_session_123"
        assert state.permission_mode == "read-only"
        assert state.warnings == ("Test warning",)

    def test_build_startup_state_defaults(self):
        """默认使用 Runtime 状态中的警告。"""
        status = SessionStatus(
            session_id="test_session",
            model_id="test/model",
            workspace="/workspace",
            permission_mode="workspace-write",
            message_count=0,
            leaf_id="leaf",
            warnings=["Runtime warning"],
        )

        state = build_startup_state(status)

        assert state.warnings == ("Runtime warning",)

    def test_build_startup_state_explicit_warnings_override_runtime_status(self):
        status = SessionStatus(
            session_id="test_session",
            model_id="test/model",
            workspace="/workspace",
            permission_mode="workspace-write",
            message_count=0,
            leaf_id="leaf",
            warnings=["Runtime warning"],
        )

        state = build_startup_state(status, warnings=[])

        assert state.warnings == ()

    def test_startup_state_normalizes_cli_display_snapshot(self):
        warnings = [" Runtime warning ", " "]
        state = CliStartupState(
            version=" 0.3 ",
            model_id=" model ",
            workspace=" /workspace ",
            session_id=" session_1 ",
            permission_mode=" read-only ",
            warnings=warnings,
        )
        warnings.append("late warning")

        assert state.version == "0.3"
        assert state.model_id == "model"
        assert state.workspace == "/workspace"
        assert state.session_id == "session_1"
        assert state.permission_mode == "read-only"
        assert state.warnings == ("Runtime warning",)

        with pytest.raises(ValueError, match="model_id"):
            CliStartupState(
                version="0.3",
                model_id=" ",
                workspace="/workspace",
                session_id="session_1",
            )

        with pytest.raises(ValueError, match="permission_mode"):
            CliStartupState(
                version="0.3",
                model_id="model",
                workspace="/workspace",
                session_id="session_1",
                permission_mode="admin",
            )

        with pytest.raises(TypeError, match="warnings"):
            CliStartupState(
                version="0.3",
                model_id="model",
                workspace="/workspace",
                session_id="session_1",
                warnings="warning",  # type: ignore[arg-type]
            )


# ── SessionStatus 测试 ───────────────────────────────────────────

class TestSessionStatus:
    """测试 SessionStatus 数据结构。"""

    def test_session_status_fields(self):
        """测试字段正确性。"""
        status = SessionStatus(
            session_id="session_123",
            model_id="deepseek/deepseek-chat",
            workspace="/workspace",
            permission_mode="read-only",
            message_count=42,
            leaf_id="leaf_456",
            is_running=True,
        )

        assert status.session_id == "session_123"
        assert status.model_id == "deepseek/deepseek-chat"
        assert status.workspace == "/workspace"
        assert status.permission_mode == "read-only"
        assert status.message_count == 42
        assert status.leaf_id == "leaf_456"
        assert status.is_running is True

    def test_session_status_defaults(self):
        """测试默认值。"""
        status = SessionStatus(
            session_id="session",
            model_id="model",
            workspace="/workspace",
            permission_mode="workspace-write",
            message_count=0,
            leaf_id="leaf",
        )

        assert status.is_running is False


# ── 配置脱敏测试 ─────────────────────────────────────────────────

class TestConfigSanitization:
    """测试配置脱敏逻辑。"""

    def test_permission_mode_read_only(self):
        """测试只读模式显示。"""
        status = SessionStatus(
            session_id="session",
            model_id="model",
            workspace="/workspace",
            permission_mode="read-only",
            message_count=0,
            leaf_id="leaf",
        )

        state = build_startup_state(status)
        assert state.permission_mode == "read-only"

    def test_permission_mode_workspace_write(self):
        """测试工作区写入模式显示。"""
        status = SessionStatus(
            session_id="session",
            model_id="model",
            workspace="/workspace",
            permission_mode="workspace-write",
            message_count=0,
            leaf_id="leaf",
        )

        state = build_startup_state(status)
        assert state.permission_mode == "workspace-write"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
