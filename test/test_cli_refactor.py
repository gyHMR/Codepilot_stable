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
from types import SimpleNamespace

from codepilot.interfaces.cli.render import (
    TerminalRenderer,
    SimpleRenderer,
)
from codepilot.interfaces.cli.render import CliStartupState, build_startup_state
from codepilot.runtime.views import SessionStatus
from codepilot.sessions.contracts import SessionCommandRecord
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

        renderer.render_progress_event(event)
        assert renderer._stream_started is True

    def test_handle_tool_start(self):
        """测试处理工具开始事件。"""
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        event = {
            "type": "tool_started",
            "toolName": "Read",
            "args": {"file_path": "/test/file.py"},
        }

        renderer.render_progress_event(event)
        assert renderer._current_tool == "Read"
        assert renderer._tool_start_time > 0

    def test_handle_lowercase_read_and_ls_show_targets(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.render_progress_event({
            "type": "tool_started",
            "toolCallId": "read-1",
            "toolName": "read",
            "args": {"path": "src/codepilot/core/loop.py", "offset": 10, "limit": 20},
        })
        renderer.render_progress_event({
            "type": "tool_started",
            "toolCallId": "ls-1",
            "toolName": "ls",
            "args": {"path": "src/codepilot"},
        })

        rendered = [call.args[0] for call in output.call_args_list]
        assert "[tool] reading read  src/codepilot/core/loop.py:10-29" in rendered
        assert "[tool] reading ls  src/codepilot" in rendered

    def test_tool_target_is_shortened_for_narrow_terminal_readability(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
        long_path = "src/" + "/".join(["very_long_directory"] * 8) + "/module.py"

        renderer.render_progress_event({
            "type": "tool_started",
            "toolCallId": "read-long",
            "toolName": "read",
            "args": {"path": long_path},
        })

        rendered = output.call_args.args[0]
        assert rendered.startswith("[tool] reading read  …")
        assert rendered.endswith("/module.py")
        assert len(rendered) <= 86

    def test_parallel_tool_timings_are_tracked_by_call_id(self, monkeypatch):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)
        timestamps = iter([10.0, 11.0, 12.0, 13.0])
        monkeypatch.setattr(
            "codepilot.interfaces.cli.render.time.time",
            lambda: next(timestamps),
        )

        renderer.render_progress_event({
            "type": "tool_started",
            "toolCallId": "read-1",
            "toolName": "read",
            "args": {"path": "a.py"},
        })
        renderer.render_progress_event({
            "type": "tool_started",
            "toolCallId": "ls-1",
            "toolName": "ls",
            "args": {"path": "src"},
        })
        renderer.render_progress_event({
            "type": "tool_completed",
            "toolCallId": "read-1",
            "toolName": "read",
            "isError": False,
        })
        renderer.render_progress_event({
            "type": "tool_completed",
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
        assert "cyber engineering console" in rendered
        assert "+- C P -+" in rendered
        assert "deepseek/deepseek-chat" in rendered
        assert "workspace-write" in rendered
        assert "build" in rendered
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
        assert "build" in toolbar
        assert "/help" in toolbar
        assert "Ctrl+C" in toolbar

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            ("info", "◇ Working"),
            ("success", "◆ Working"),
            ("warning", "▲ Working"),
            ("error", "✕ Working"),
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

        renderer.render_progress_event(event)
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

        renderer.render_progress_event({
            "type": "error",
            "error": info.code,
            "message": info.message,
            "provider": info.provider,
            "model": info.model,
            "errorInfo": info,
        })

        rendered = [call.args[0] for call in output.call_args_list]
        assert '  Provider response: {"error":{"message":"Invalid request"}}' in rendered

    def test_plain_activity_prevents_silent_wait(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.render_activity("thinking")
        renderer.render_activity("thinking")

        rendered = [call.args[0] for call in output.call_args_list]
        assert rendered == ["◇ thinking ..."]

    def test_prompt_builders_frame_live_user_input(self):
        renderer = TerminalRenderer(use_rich=False, output=MagicMock())

        assert "╭─ YOU" in renderer.build_shell_prompt()
        assert "╰─›" in renderer.build_shell_prompt()
        assert "╭─ YOU" in renderer.build_plain_prompt()
        assert "╰─›" in renderer.build_plain_prompt()

    def test_plain_write_and_bash_tools_use_action_labels(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.render_progress_event({
            "type": "tool_started",
            "toolName": "write",
            "args": {"path": "demo.txt"},
        })
        renderer.render_progress_event({
            "type": "tool_started",
            "toolName": "bash",
            "args": {"command": "python register.py --demo"},
        })

        rendered = [call.args[0] for call in output.call_args_list]
        assert "[tool] writing write  demo.txt" in rendered
        assert "[tool] running bash  python register.py --demo" in rendered

    def test_approval_required_tool_end_is_not_rendered_as_error(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.render_progress_event({
            "type": "tool_started",
            "toolCallId": "bash-1",
            "toolName": "bash",
            "args": {"command": "head -5 agent-test/chatbot.py"},
        })
        renderer.render_progress_event({
            "type": "tool_interrupted",
            "toolCallId": "bash-1",
            "toolName": "bash",
            "status": "approval_required",
            "isError": True,
            "errorReason": "approval_required",
        })

        rendered = [call.args[0] for call in output.call_args_list]
        assert rendered == ["[tool] running bash  head -5 agent-test/chatbot.py"]

    def test_plain_approval_prompt_is_structured(self):
        output = MagicMock()
        renderer = TerminalRenderer(use_rich=False, output=output)

        renderer.render_approval_required(
            SimpleNamespace(
                approval=SimpleNamespace(
                    tool_name="bash",
                    arguments={"command": "python register.py --demo"},
                    risk=SimpleNamespace(level="medium"),
                    approval_id="approval_1",
                )
            )
        )

        rendered = [call.args[0] for call in output.call_args_list]
        assert "+-- APPROVAL REQUIRED " in rendered[1]
        assert "| Tool  bash  python register.py --demo" in rendered
        assert "| /approve approval_1" in rendered
        assert "| /deny    approval_1" in rendered


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

        renderer.render_progress_event(event)
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


def test_run_once_dispatches_prompt_and_renders_final_message():
    from codepilot.interfaces.cli.interactive import run_once
    from codepilot.runtime.actions import ProgressFrame, PromptSubmitted, RunFinishedFrame

    final_message = AssistantMessage(content=[TextContent(text="done")])

    class FakeRuntime:
        def __init__(self):
            self.sent = None

        async def dispatch(self, session_id, action):
            assert isinstance(action, PromptSubmitted)
            self.sent = (session_id, action.text)
            yield ProgressFrame(event={"type": "message_update", "delta": "hello"})
            yield RunFinishedFrame(
                record=SimpleNamespace(
                    outcome=SimpleNamespace(final_message=final_message)
                )
            )

    output: list[str] = []

    def write(text: str = "", **_kwargs):
        output.append(text)

    runtime = FakeRuntime()

    asyncio.run(run_once(runtime, "session_1", "hello", output=write))

    assert runtime.sent == ("session_1", "hello")
    assert output == ["hello", ""]


def test_render_dispatch_surfaces_approval_required_frame_directly():
    from codepilot.interfaces.cli.interactive import render_dispatch
    from codepilot.runtime.actions import ApprovalRequiredFrame
    from codepilot.tools.contracts import ToolInterruption, ToolRiskView

    interruption = ToolInterruption(
        approval_id="approval_1",
        run_id="run_1",
        tool_call_id="call_1",
        tool_name="write",
        arguments={"path": "demo.txt"},
        reason="workspace_write",
        risk=ToolRiskView(level="high", summary="write file"),
    )

    async def frames():
        yield ApprovalRequiredFrame(approval=interruption)

    class FakeRenderer:
        def __init__(self):
            self.approvals = []
            self.final = "not-called"

        def render_approval_required(self, frame):
            self.approvals.append(frame.approval)

        def render_final(self, record):
            self.final = record

    renderer = FakeRenderer()

    asyncio.run(render_dispatch(frames(), renderer))

    assert renderer.approvals == [interruption]
    assert renderer.final is None


def test_cli_approval_text_builds_runtime_decision():
    from codepilot.interfaces.cli.interactive import approval_action_from_text

    action = approval_action_from_text("/approve approval_1 ok")

    assert action is not None
    assert action.approval_id == "approval_1"
    assert action.decision == "approve"
    assert action.reason == "ok"
    assert approval_action_from_text("/memory status") is None


def test_run_rpc_emits_jsonl_contract_for_state_prompt_errors_and_shutdown(monkeypatch):
    from codepilot.interfaces.cli.rpc import run_rpc
    from codepilot.runtime.actions import (
        CommandFinishedFrame,
        CommandSubmitted,
        FailedFrame,
        ProgressFrame,
        PromptSubmitted,
        RunFinishedFrame,
    )

    class FakeRuntime:
        def __init__(self):
            self.prompt_calls = 0
            self.current_mode = "build"

        async def dispatch(self, session_id, action):
            assert session_id == "session_1"
            if isinstance(action, PromptSubmitted):
                self.prompt_calls += 1
                if self.prompt_calls == 1:
                    assert action.text == "hello"
                    assert action.mode_hint == "plan"
                    yield ProgressFrame(event={"type": "message_update", "delta": "hi"})
                    yield RunFinishedFrame(
                        record=type(
                            "Record",
                            (),
                            {
                                "run_id": "run_prompt_1",
                                "session_id": "session_1",
                                "status": "completed",
                                "stop_reason": "final_answer",
                                "final_text": "done from frame",
                            },
                        )()
                    )
                    return
                assert action.text == "busy"
                yield FailedFrame(
                    error={
                        "code": "runtime.session_busy",
                        "message": "Session is already running",
                    }
                )
                return
            if isinstance(action, CommandSubmitted):
                assert action.text == "/mode read"
                self.current_mode = "read"
                yield CommandFinishedFrame(
                    record=SessionCommandRecord(
                        session_id="session_1",
                        command="/mode read",
                        handled=True,
                        output_lines=["current_mode=read"],
                        data={"current_mode": "read"},
                    )
                )
                return
            raise AssertionError(f"unexpected action: {action!r}")

        def describe(self, session_id):
            assert session_id == "session_1"
            return SimpleNamespace(
                status=SessionStatus(
                    session_id=session_id,
                    model_id="test/model",
                    workspace=str(Path.cwd()),
                    permission_mode="workspace-write",
                    message_count=2,
                    leaf_id="entry_1",
                    current_mode=self.current_mode,
                ),
                state={
                    "session_id": session_id,
                    "message_count": 2,
                    "entry_ids": ["entry_1"],
                    "entries": [{"id": "entry_1"}],
                    "tree": [{"id": "entry_1"}],
                    "leaf_id": "entry_1",
                    "current_mode": self.current_mode,
                },
            )

    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"type": "state", "id": "state_1"}),
                json.dumps({"type": "set_mode", "id": "mode_1", "mode": "read"}),
                json.dumps({"type": "prompt", "id": "prompt_1", "text": "hello", "mode": "plan"}),
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
        "protocol_version": "2.0",
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
                "entries": [{"id": "entry_1"}],
                "tree": [{"id": "entry_1"}],
                "leaf_id": "entry_1",
                "current_mode": "build",
            },
        }
    assert messages[2] == {
        "type": "response",
        "id": "mode_1",
        "command": "set_mode",
        "status": "ok",
        "data": {
            "session_id": "session_1",
            "mode": "read",
        },
    }
    assert messages[3] == {
        "type": "event",
        "event": {"type": "message_update", "delta": "hi"},
    }
    assert messages[4]["command"] == "prompt"
    assert messages[4]["status"] == "ok"
    assert messages[4]["data"] == {
        "run_id": "run_prompt_1",
        "session_id": "session_1",
        "status": "completed",
        "stop_reason": "final_answer",
        "final_text": "done from frame",
    }
    assert messages[5]["status"] == "error"
    assert messages[5]["command"] == "prompt"
    assert messages[5]["error"]["code"] == "runtime.session_busy"
    assert messages[6]["status"] == "error"
    assert messages[6]["error"]["code"] == "invalid_json"
    assert messages[7]["command"] == "shutdown"
    assert messages[7]["status"] == "ok"


def test_run_rpc_surfaces_and_resumes_approval(monkeypatch):
    from codepilot.interfaces.cli.rpc import run_rpc
    from codepilot.runtime.actions import (
        ApprovalDecided,
        ApprovalRequiredFrame,
        PromptSubmitted,
        RunFinishedFrame,
    )
    from codepilot.tools.contracts import ToolInterruption, ToolRiskView

    final_message = AssistantMessage(content=[TextContent(text="approved done")])
    interruption = ToolInterruption(
        approval_id="approval_1",
        run_id="run_1",
        tool_call_id="call_1",
        tool_name="write",
        arguments={"path": "demo.txt"},
        reason="workspace_write",
        risk=ToolRiskView(level="high", summary="write file"),
    )

    class FakeRuntime:
        async def dispatch(self, session_id, action):
            assert session_id == "session_1"
            if isinstance(action, PromptSubmitted):
                yield ApprovalRequiredFrame(approval=interruption)
                return
            if isinstance(action, ApprovalDecided):
                assert action.approval_id == "approval_1"
                assert action.decision == "approve"
                yield RunFinishedFrame(
                    record=type(
                        "Record",
                        (),
                        {
                            "run_id": "run_approved",
                            "session_id": "session_1",
                            "status": "completed",
                            "stop_reason": "final_answer",
                            "final_text": "approved done",
                            "outcome": type(
                                "Outcome",
                                (),
                                {"final_message": final_message},
                            )(),
                        },
                    )()
                )
                return
            raise AssertionError(f"unexpected action: {action!r}")

    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"type": "prompt", "id": "prompt_1", "text": "write"}),
                json.dumps(
                    {
                        "type": "approve",
                        "id": "approve_1",
                        "approval_id": "approval_1",
                        "reason": "ok",
                    }
                ),
                json.dumps({"type": "shutdown", "id": "shutdown_1"}),
            ]
        )
        + "\n"
    )
    output: list[str] = []
    monkeypatch.setattr("sys.stdin", stdin)

    asyncio.run(run_rpc(FakeRuntime(), "session_1", output=output.append))

    messages = [json.loads(line) for line in output]
    assert messages[1] == {
        "type": "approval_required",
        "approval": {
            "approval_id": "approval_1",
            "run_id": "run_1",
            "tool_call_id": "call_1",
            "tool_name": "write",
            "arguments": {"path": "demo.txt"},
            "reason": "workspace_write",
            "risk_level": "high",
        },
    }
    assert messages[2] == {
        "type": "response",
        "id": "prompt_1",
        "command": "prompt",
        "status": "ok",
        "data": {
            "status": "waiting_approval",
            "approval_id": "approval_1",
        },
    }
    assert messages[3]["command"] == "approve"
    assert messages[3]["status"] == "ok"
    assert messages[3]["data"]["run_id"] == "run_approved"


def test_rpc_ready_signal_uses_named_protocol_version() -> None:
    from codepilot.interfaces.cli.rpc import (
        RPC_PROTOCOL_VERSION,
        emit_rpc_ready,
    )

    emitted: list[dict] = []
    emit_rpc_ready(emitted.append, session_id=" session_1 ")

    assert RPC_PROTOCOL_VERSION == "2.0"
    assert emitted == [
        {
            "type": "rpc_ready",
            "session_id": "session_1",
            "protocol_version": "2.0",
        }
    ]

    with pytest.raises(ValueError, match="session_id"):
        emit_rpc_ready(emitted.append, session_id=" ")


def test_rpc_ok_response_requires_command_name() -> None:
    from codepilot.interfaces.cli.rpc import emit_rpc_ok

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


def test_rpc_error_mapping_uses_coded_exception_payloads() -> None:
    from codepilot.interfaces.cli.rpc import rpc_error_from_exception

    class SessionBusyError(Exception):
        code = "runtime.session_busy"

    runtime_error = SessionBusyError("Session is already running")
    generic_error = ValueError("missing field")

    assert rpc_error_from_exception(runtime_error).code == "runtime.session_busy"
    assert rpc_error_from_exception(runtime_error).message == "Session is already running"
    assert rpc_error_from_exception(generic_error).code == "execution_error"
    assert rpc_error_from_exception(generic_error).message == "missing field"


def test_rpc_error_requires_non_empty_code_and_message() -> None:
    from codepilot.interfaces.cli.rpc import RpcError, rpc_error_from_exception

    with pytest.raises(ValueError, match="code"):
        RpcError(code="  ", message="Something failed")

    with pytest.raises(ValueError, match="message"):
        RpcError(code="execution_error", message="")

    mapped = rpc_error_from_exception(ValueError())

    assert mapped.code == "execution_error"
    assert mapped.message == "ValueError"


def test_repl_text_helpers_stay_local_to_interactive_flow() -> None:
    from codepilot.interfaces.cli.interactive import (
        approval_action_from_text,
        is_exit_text,
    )

    assert is_exit_text("/exit")
    assert is_exit_text("quit")
    assert not is_exit_text("/memory status")
    assert approval_action_from_text("hello") is None


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
        assert state.current_mode == "build"
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
        assert state.current_mode == "build"
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

        with pytest.raises(ValueError, match="current_mode"):
            CliStartupState(
                version="0.3",
                model_id="model",
                workspace="/workspace",
                session_id="session_1",
                current_mode="auto",
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
        assert status.current_mode == "build"
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
        assert status.current_mode == "build"


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
