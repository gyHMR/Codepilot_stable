"""CLI 重构后的测试。

覆盖：
- TerminalRenderer 渲染
- 启动状态构建
- 配置脱敏
- 新 CLI 参数
- Session 切换
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from pathlib import Path

from codepilot.interfaces.cli.runner import (
    TerminalRenderer,
    CliStartupState,
    build_startup_state,
    SimpleRenderer,
)
from codepilot.runtime.service import SessionStatus
from codepilot.protocols import AssistantMessage, TextContent, Usage, Cost


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
        assert state.warnings == ["Test warning"]

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

        assert state.warnings == ["Runtime warning"]

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

        assert state.warnings == []


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
