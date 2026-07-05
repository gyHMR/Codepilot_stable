from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_dingtalk_contract_keeps_remote_interface_only() -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
        DingTalkOutboundMessage,
        describe_dingtalk_contract,
    )

    contract = describe_dingtalk_contract()
    config = DingTalkBridgeConfig(
        workspace_dir="  E:/workspace  ",
        allowed_users=(" user_1 ", "user_1"),
        allow_dirty=True,
    )
    message = DingTalkInboundMessage(
        message_id=" msg_1 ",
        sender_id=" user_1 ",
        text="  cp hello  ",
    )

    assert contract["transport"] == ["dingtalk-stream"]
    assert contract["entrypoint"] == "codepilot.interfaces.dingtalk"
    assert "codepilot.runtime" in contract["delegates_to"]
    assert "filesystem_mutation" in contract["non_responsibilities"]
    assert config.workspace_dir == "E:/workspace"
    assert config.allowed_users == ("user_1",)
    assert message.text == "cp hello"

    outbound = DingTalkOutboundMessage(
        receiver_id=" user_1 ",
        text=" **done** ",
        format="markdown",
        title=" Run Result ",
    )
    assert outbound.receiver_id == "user_1"
    assert outbound.format == "markdown"
    assert outbound.title == "Run Result"

    with pytest.raises(ValueError, match="format"):
        DingTalkOutboundMessage(receiver_id="user_1", text="done", format="html")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="allowed_users"):
        DingTalkBridgeConfig(workspace_dir=".", allowed_users=())


def test_dingtalk_command_parser_supports_prompt_and_approval() -> None:
    from codepilot.interfaces.dingtalk.commands import parse_dingtalk_command

    prompt = parse_dingtalk_command("cp 修复 failing test")
    multiline = parse_dingtalk_command("cp\n修复手机换行输入")
    approve = parse_dingtalk_command("approve approval_123")
    deny = parse_dingtalk_command("deny approval_123")
    status = parse_dingtalk_command("status")
    unknown = parse_dingtalk_command("hello")

    assert prompt.action == "prompt"
    assert prompt.prompt == "修复 failing test"
    assert multiline.action == "prompt"
    assert multiline.prompt == "修复手机换行输入"
    assert approve.action == "approve"
    assert approve.approval_id == "approval_123"
    assert deny.action == "deny"
    assert deny.approval_id == "approval_123"
    assert status.action == "status"
    assert unknown.action == "unknown"


def test_dingtalk_bridge_runs_prompt_through_runtime_with_ask_permissions(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="cp fix bug",
            )
        )
    )

    assert runtime.created_options is not None
    assert runtime.created_options.tool_permission_mode == "ask"
    assert runtime.sent_message == ("session_1", "fix bug")
    assert "accepted" in replies[0].text.lower()
    assert replies[0].format == "markdown"
    assert any("run_1" in reply.text for reply in replies)

    audit = _read_dingtalk_audit(tmp_path)
    assert [record["event"] for record in audit] == [
        "message_received",
        "run_accepted",
        "run_finished",
    ]
    assert audit[0]["sender_id"].startswith("sha256:")
    assert audit[0]["sender_id"] != "user_1"
    assert audit[1]["session_id"] == "session_1"
    assert audit[2]["run_id"] == "run_1"


def test_dingtalk_audit_redacts_and_omits_full_prompt(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_secret",
                sender_id="user_1",
                conversation_id="conv_1",
                text="cp fix bug with api_key=sk-very-secret-token and password=hunter2",
            )
        )
    )

    raw = _dingtalk_audit_path(tmp_path).read_text(encoding="utf-8")
    records = _read_dingtalk_audit(tmp_path)

    assert "sk-very-secret-token" not in raw
    assert "hunter2" not in raw
    assert "fix bug with" not in raw
    assert records[0]["conversation_id"] == "conv_1"
    assert all(
        set(record) == {
            "event",
            "timestamp_ms",
            "message_id",
            "sender_id",
            "conversation_id",
            "session_id",
            "run_id",
            "approval_id",
            "command",
            "status",
            "reason",
            "workspace_state",
        }
        for record in records
    )


def test_dingtalk_bridge_streams_prompt_acceptance_before_run_finishes(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = BlockingRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    async def run_case() -> list[str]:
        replies: list[str] = []

        async def collect() -> None:
            async for reply in bridge.iter_replies(
                DingTalkInboundMessage(
                    message_id="msg_1",
                    sender_id="user_1",
                    text="cp fix bug",
                )
            ):
                replies.append(reply.text)

        task = asyncio.create_task(collect())
        for _ in range(20):
            if replies:
                break
            await asyncio.sleep(0.01)
        assert replies and "accepted" in replies[0].lower()
        assert not task.done()
        runtime.release.set()
        await task
        return replies

    texts = asyncio.run(run_case())

    assert any("run_1" in text for text in texts)


def test_dingtalk_approval_required_markdown_contains_commands() -> None:
    from codepilot.interfaces.dingtalk.renderer import render_event

    replies = render_event(
        {
            "type": "tool_execution_end",
            "toolName": "write_file",
            "status": "approval_required",
            "riskLevel": "high",
            "args": {"path": "src/app.py", "content": "secret"},
            "approvalId": "approval_1",
            "errorReason": "mutating tool requires approval",
        }
    )

    assert len(replies) == 1
    assert "approval_1" in replies[0]
    assert "write_file" in replies[0]
    assert "approve approval_1" in replies[0]
    assert "deny approval_1" in replies[0]


def test_dingtalk_bridge_restores_configured_session_before_prompt(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            session_id="session_existing",
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="cp fix bug",
            )
        )
    )

    assert runtime.created_options is not None
    assert runtime.created_options.session_id == "session_existing"
    assert runtime.sent_message == ("session_existing", "fix bug")
    assert any("run_1" in reply.text for reply in replies)


def test_dingtalk_bridge_rejects_unauthorized_and_duplicate_messages(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    rejected = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_2",
                text="cp fix bug",
            )
        )
    )
    first = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_2",
                sender_id="user_1",
                text="status",
            )
        )
    )
    duplicate = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_2",
                sender_id="user_1",
                text="status",
            )
        )
    )

    assert "not authorized" in rejected[0].text
    assert first
    assert duplicate == []

    audit = _read_dingtalk_audit(tmp_path)
    assert audit[0]["event"] == "message_received"
    assert audit[1]["event"] == "message_rejected"
    assert audit[1]["reason"] == "unauthorized_sender"
    assert audit[2]["event"] == "message_received"
    assert audit[3]["event"] == "status_requested"


def test_dingtalk_bridge_allows_retry_after_failed_message(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FailingOnceRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=runtime,
    )
    inbound = DingTalkInboundMessage(
        message_id="msg_1",
        sender_id="user_1",
        text="cp fix bug",
    )

    first = asyncio.run(bridge.handle_message(inbound))
    second = asyncio.run(bridge.handle_message(inbound))

    assert "failed" in first[0].text.lower()
    assert runtime.create_attempts == 2
    assert any("run_1" in reply.text for reply in second)

    audit = _read_dingtalk_audit(tmp_path)
    assert audit[1]["event"] == "bridge_error"
    assert audit[1]["status"] == "error"
    assert audit[1]["reason"] == "temporary failure"


def test_dingtalk_bridge_submits_tool_approval(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            session_id="session_1",
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="approve approval_1",
            )
        )
    )

    assert runtime.approval == ("approval_1", "approve", "session_1")
    assert "received" in replies[0].text.lower()
    assert replies[0].format == "markdown"
    assert any("approved" in reply.text for reply in replies)
    assert any("affected_paths" in reply.text for reply in replies)

    audit = _read_dingtalk_audit(tmp_path)
    assert [record["event"] for record in audit] == [
        "message_received",
        "approval_received",
        "approval_finished",
    ]
    assert audit[1]["approval_id"] == "approval_1"
    assert audit[2]["run_id"] == "run_approved"


def test_dingtalk_bridge_reports_missing_approval_reason(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = ApprovalNotFoundRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            session_id="session_1",
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="approve approval_missing",
            )
        )
    )

    assert any("not found" in reply.text.lower() for reply in replies)
    assert any("approval_missing" in reply.text for reply in replies)
    audit = _read_dingtalk_audit(tmp_path)
    assert audit[-1]["event"] == "approval_finished"
    assert audit[-1]["status"] == "error"


def test_dingtalk_bridge_reports_follow_up_pending_approval(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = ChainedApprovalRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            session_id="session_1",
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="approve approval_1",
            )
        )
    )

    assert any("waiting_approval" in reply.text for reply in replies)
    assert any("approval_2" in reply.text for reply in replies)
    assert any("approve approval_2" in reply.text for reply in replies)
    assert any("deny approval_2" in reply.text for reply in replies)


def test_dingtalk_status_reports_pending_approvals_and_workspace_state(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    with tempfile.TemporaryDirectory() as workspace:
        runtime = FakeRuntime()
        runtime.pending_approvals = [
            {
                "approval_id": "approval_1",
                "tool_name": "bash",
                "run_id": "run_waiting",
                "reason": "high risk shell command",
            }
        ]
        bridge = DingTalkBridge(
            config=DingTalkBridgeConfig(
                workspace_dir=workspace,
                allowed_users=("user_1",),
                session_id="session_1",
                allow_dirty=True,
            ),
            runtime=runtime,
        )

        replies = asyncio.run(
            bridge.handle_message(
                DingTalkInboundMessage(
                    message_id="msg_1",
                    sender_id="user_1",
                    text="status",
                )
            )
        )

    assert replies[0].format == "markdown"
    assert "pending_approvals: `1`" in replies[0].text
    assert "workspace_state" in replies[0].text
    assert "non_git" in replies[0].text
    assert "bash" in replies[0].text
    assert "run_waiting" in replies[0].text
    assert "high risk shell command" in replies[0].text


def test_dingtalk_status_restores_configured_session(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            session_id="session_existing",
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="status",
            )
        )
    )

    assert runtime.created_options is not None
    assert runtime.created_options.session_id == "session_existing"
    assert "session_existing" in replies[0].text


def test_dingtalk_help_is_grouped_markdown(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=FakeRuntime(),
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="help",
            )
        )
    )

    assert replies[0].format == "markdown"
    assert "任务" in replies[0].text
    assert "审批" in replies[0].text
    assert "状态" in replies[0].text
    assert "取消" in replies[0].text


def test_dingtalk_bridge_rejects_dirty_workspace_by_default(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    _init_clean_git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    runtime = FakeRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
        ),
        runtime=runtime,
    )

    replies = asyncio.run(
        bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_1",
                sender_id="user_1",
                text="cp fix bug",
            )
        )
    )

    assert runtime.created_options is None
    assert "dirty" in replies[0].text.lower()

    audit = _read_dingtalk_audit(tmp_path)
    assert audit[-1]["event"] == "message_rejected"
    assert audit[-1]["reason"] == "workspace_dirty"


def test_dingtalk_bridge_rejects_non_git_workspace_by_default(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    with tempfile.TemporaryDirectory() as workspace:
        runtime = FakeRuntime()
        bridge = DingTalkBridge(
            config=DingTalkBridgeConfig(
                workspace_dir=workspace,
                allowed_users=("user_1",),
            ),
            runtime=runtime,
        )

        replies = asyncio.run(
            bridge.handle_message(
                DingTalkInboundMessage(
                    message_id="msg_1",
                    sender_id="user_1",
                    text="cp fix bug",
                )
            )
        )

        assert runtime.created_options is None
        assert "git" in replies[0].text.lower()
        audit = _read_dingtalk_audit(Path(workspace))
        assert audit[-1]["event"] == "message_rejected"
        assert audit[-1]["reason"] == "workspace_non_git"


def test_dingtalk_bridge_reports_busy_without_queueing(tmp_path: Path) -> None:
    from codepilot.interfaces.dingtalk import (
        DingTalkBridge,
        DingTalkBridgeConfig,
        DingTalkInboundMessage,
    )

    runtime = BlockingRuntime()
    bridge = DingTalkBridge(
        config=DingTalkBridgeConfig(
            workspace_dir=str(tmp_path),
            allowed_users=("user_1",),
            allow_dirty=True,
        ),
        runtime=runtime,
    )

    async def run_case() -> list[str]:
        first = asyncio.create_task(
            bridge.handle_message(
                DingTalkInboundMessage(
                    message_id="msg_1",
                    sender_id="user_1",
                    text="cp first",
                )
            )
        )
        await runtime.started.wait()
        second = await bridge.handle_message(
            DingTalkInboundMessage(
                message_id="msg_2",
                sender_id="user_1",
                text="cp second",
            )
        )
        runtime.release.set()
        await first
        return [reply.text for reply in second]

    assert any("busy" in text.lower() for text in asyncio.run(run_case()))

    audit = _read_dingtalk_audit(tmp_path)
    assert any(
        record["event"] == "message_rejected" and record["reason"] == "busy"
        for record in audit
    )


def test_stream_transport_missing_optional_dependency_has_install_hint() -> None:
    from codepilot.interfaces.dingtalk.transport import create_stream_transport

    def missing_importer(_name: str):
        raise ModuleNotFoundError("dingtalk_stream")

    with pytest.raises(RuntimeError, match="codepilot\\[dingtalk\\]"):
        create_stream_transport(
            client_id="client",
            client_secret="secret",
            importer=missing_importer,
        )


def test_stream_transport_acknowledges_before_run_finishes() -> None:
    from codepilot.interfaces.dingtalk.schemas import DingTalkOutboundMessage
    from codepilot.interfaces.dingtalk.transport import create_stream_transport

    async def run_case() -> FakeDingTalkStreamSdk:
        gate = asyncio.Event()

        class SlowBridge:
            async def handle_message(self, inbound):
                await gate.wait()
                return [
                    DingTalkOutboundMessage(
                        receiver_id=inbound.sender_id,
                        conversation_id=inbound.conversation_id,
                        text="done",
                    )
                ]

        sdk = FakeDingTalkStreamSdk()
        transport = create_stream_transport(
            client_id="client",
            client_secret="secret",
            importer=lambda _name: sdk,
        )
        await asyncio.wait_for(transport.start(SlowBridge()), timeout=0.2)
        assert sdk.client is not None
        assert sdk.client.ack == ("OK", "OK")
        assert sdk.replies == []
        gate.set()
        for _ in range(20):
            if sdk.replies:
                break
            await asyncio.sleep(0.01)
        return sdk

    sdk = asyncio.run(run_case())

    assert sdk.replies == ["done"]


def test_stream_transport_uses_markdown_reply_when_available() -> None:
    from codepilot.interfaces.dingtalk.schemas import DingTalkOutboundMessage
    from codepilot.interfaces.dingtalk.transport import create_stream_transport

    async def run_case() -> FakeDingTalkMarkdownStreamSdk:
        class MarkdownBridge:
            async def handle_message(self, inbound):
                return [
                    DingTalkOutboundMessage(
                        receiver_id=inbound.sender_id,
                        conversation_id=inbound.conversation_id,
                        text="**done**",
                        format="markdown",
                        title="Run Result",
                    )
                ]

        sdk = FakeDingTalkMarkdownStreamSdk()
        transport = create_stream_transport(
            client_id="client",
            client_secret="secret",
            importer=lambda _name: sdk,
        )
        await asyncio.wait_for(transport.start(MarkdownBridge()), timeout=0.2)
        for _ in range(20):
            if sdk.markdown_replies:
                break
            await asyncio.sleep(0.01)
        return sdk

    sdk = asyncio.run(run_case())

    assert sdk.markdown_replies == [("Run Result", "**done**")]
    assert sdk.replies == []


def test_stream_transport_falls_back_to_text_without_markdown_method() -> None:
    from codepilot.interfaces.dingtalk.schemas import DingTalkOutboundMessage
    from codepilot.interfaces.dingtalk.transport import create_stream_transport

    async def run_case() -> FakeDingTalkStreamSdk:
        class MarkdownBridge:
            async def handle_message(self, inbound):
                return [
                    DingTalkOutboundMessage(
                        receiver_id=inbound.sender_id,
                        conversation_id=inbound.conversation_id,
                        text="**done**",
                        format="markdown",
                        title="Run Result",
                    )
                ]

        sdk = FakeDingTalkStreamSdk()
        transport = create_stream_transport(
            client_id="client",
            client_secret="secret",
            importer=lambda _name: sdk,
        )
        await asyncio.wait_for(transport.start(MarkdownBridge()), timeout=0.2)
        for _ in range(20):
            if sdk.replies:
                break
            await asyncio.sleep(0.01)
        return sdk

    sdk = asyncio.run(run_case())

    assert sdk.replies == ["**done**"]


class FakeRuntime:
    def __init__(self) -> None:
        self.created_options = None
        self.sent_message = None
        self.approval = None
        self.cancelled = None
        self.pending_approvals = []

    def open_session(self, options):
        self.created_options = options
        return SimpleNamespace(session_id=options.session_id or "session_1")

    def describe(self, session_id):
        return SimpleNamespace(pending_approvals=tuple(self.pending_approvals))

    async def dispatch(self, session_id, action):
        from codepilot.protocols import AgentRunCounters, AgentRunResult, AssistantMessage, TextContent
        from codepilot.runtime.actions import (
            ApprovalDecided,
            CancelledFrame,
            ProgressFrame,
            PromptSubmitted,
            RunCancelled,
            RunFinishedFrame,
        )

        if isinstance(action, ApprovalDecided):
            self.approval = (action.approval_id, action.decision, session_id)
            yield RunFinishedFrame(
                record=SimpleNamespace(
                    run_id="run_approved",
                    status="completed",
                    affected_paths=["src/example.py"],
                    workspace_changed=True,
                )
            )
            return
        if isinstance(action, RunCancelled):
            self.cancelled = session_id
            yield CancelledFrame(session_id=session_id, cancelled=True, reason=action.reason)
            return
        if not isinstance(action, PromptSubmitted):
            return

        self.sent_message = (session_id, action.text)
        event = {
            "type": "agent_end",
            "runId": "run_1",
            "sessionId": session_id,
            "status": "completed",
            "result": {
                "run_id": "run_1",
                "status": "completed",
                "affected_paths": ["src/example.py"],
                "workspace_changed": True,
            },
        }
        final = AssistantMessage(content=[TextContent(text="done")])
        result = AgentRunResult(
            run_id="run_1",
            session_id=session_id,
            status="completed",
            stop_reason="final_answer",
            counters=AgentRunCounters(),
            messages=[final],
            final_message=final,
            affected_paths=["src/example.py"],
            workspace_changed=True,
        )
        yield ProgressFrame(event=event)
        yield RunFinishedFrame(record=result)


class ApprovalNotFoundRuntime(FakeRuntime):
    async def dispatch(self, session_id, action):
        from codepilot.runtime.actions import ApprovalDecided, FailedFrame

        if isinstance(action, ApprovalDecided):
            self.approval = (action.approval_id, action.decision, session_id)
            yield FailedFrame(
                error={
                    "code": "approval.not_found",
                    "message": f"Approval not found: {action.approval_id}",
                }
            )
            return
        async for frame in super().dispatch(session_id, action):
            yield frame


class ChainedApprovalRuntime(FakeRuntime):
    async def dispatch(self, session_id, action):
        from codepilot.runtime.actions import ApprovalDecided, RunFinishedFrame

        if not isinstance(action, ApprovalDecided):
            async for frame in super().dispatch(session_id, action):
                yield frame
            return
        self.approval = (action.approval_id, action.decision, session_id)
        self.pending_approvals = [
            {
                "approval_id": "approval_2",
                "session_id": session_id,
                "run_id": "run_waiting",
                "tool_call_id": "tool_2",
                "tool_name": "bash",
                "reason": "next approval required",
            }
        ]
        yield RunFinishedFrame(
            record=SimpleNamespace(
                run_id="run_waiting",
                status="waiting_approval",
                affected_paths=[],
                workspace_changed=False,
            ),
        )


class FailingOnceRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.create_attempts = 0

    def open_session(self, options):
        self.create_attempts += 1
        if self.create_attempts == 1:
            raise RuntimeError("temporary failure")
        return super().open_session(options)


class BlockingRuntime(FakeRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def dispatch(self, session_id, action):
        from codepilot.protocols import AgentRunCounters, AgentRunResult, AssistantMessage, TextContent
        from codepilot.runtime.actions import ProgressFrame, PromptSubmitted, RunFinishedFrame

        if not isinstance(action, PromptSubmitted):
            async for frame in super().dispatch(session_id, action):
                yield frame
            return
        self.sent_message = (session_id, action.text)
        self.started.set()
        await self.release.wait()
        event = {
            "type": "agent_end",
            "runId": "run_1",
            "sessionId": session_id,
            "status": "completed",
            "result": {"run_id": "run_1", "status": "completed"},
        }
        final = AssistantMessage(content=[TextContent(text="done")])
        result = AgentRunResult(
            run_id="run_1",
            session_id=session_id,
            status="completed",
            stop_reason="final_answer",
            counters=AgentRunCounters(),
            messages=[final],
            final_message=final,
        )
        yield ProgressFrame(event=event)
        yield RunFinishedFrame(record=result)


class FakeDingTalkStreamSdk:
    def __init__(self) -> None:
        self.client = None
        self.replies: list[str] = []

    class AckMessage:
        STATUS_OK = "OK"

    class Credential:
        def __init__(self, client_id: str, client_secret: str) -> None:
            self.client_id = client_id
            self.client_secret = client_secret

    class ChatbotMessage:
        TOPIC = "/v1.0/im/bot/messages/get"

        @classmethod
        def from_dict(cls, data):
            return SimpleNamespace(
                message_id=data["msgId"],
                sender_staff_id=data["senderStaffId"],
                conversation_id=data["conversationId"],
                text=SimpleNamespace(content=data["text"]["content"]),
            )

    class ChatbotHandler:
        def reply_text(self, text, _incoming_message):
            self._sdk.replies.append(text)

    class DingTalkStreamClient:
        def __init__(self, _credential) -> None:
            self.handler = None
            self.ack = None

        def register_callback_handler(self, _topic, handler) -> None:
            handler._sdk = self._sdk
            self.handler = handler

        async def start(self) -> None:
            self.ack = await self.handler.process(
                SimpleNamespace(
                    data={
                        "msgId": "msg_1",
                        "senderStaffId": "user_1",
                        "conversationId": "conversation_1",
                        "text": {"content": "cp fix bug"},
                    }
                )
            )

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        if name == "DingTalkStreamClient":
            sdk = self

            class BoundClient(value):
                def __init__(self, credential) -> None:
                    self._sdk = sdk
                    super().__init__(credential)
                    sdk.client = self

            return BoundClient
        return value


class FakeDingTalkMarkdownStreamSdk(FakeDingTalkStreamSdk):
    def __init__(self) -> None:
        super().__init__()
        self.markdown_replies: list[tuple[str, str]] = []

    class ChatbotHandler:
        def reply_text(self, text, _incoming_message):
            self._sdk.replies.append(text)

        def reply_markdown(self, title, text, _incoming_message):
            self._sdk.markdown_replies.append((title, text))


def _init_clean_git_repo(path: Path) -> None:
    _git(path, "init")
    (path / "tracked.txt").write_text("initial\n", encoding="utf-8")
    _git(path, "add", "tracked.txt")
    _git(path, "-c", "user.email=test@example.com", "-c", "user.name=Test", "commit", "-m", "init")


def _git(path: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def _dingtalk_audit_path(workspace: Path) -> Path:
    return workspace / ".codepilot" / "dingtalk" / "audit.jsonl"


def _read_dingtalk_audit(workspace: Path) -> list[dict[str, object]]:
    path = _dingtalk_audit_path(workspace)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
