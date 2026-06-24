from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codepilot.protocols import Model
from codepilot.runtime.assembly import explain_runtime_config
from codepilot.runtime.service import (
    ApprovalNotFoundError,
    PendingApproval,
    RuntimeService,
    SessionBusyError,
    UserInput,
)
from codepilot.runtime.types import CreateAgentSessionOptions
from codepilot.tools import AgentTool


def _model() -> Model:
    return Model(
        id="test-model",
        name="Test Model",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=1000,
        max_tokens=100,
    )


def _deepseek_model() -> Model:
    return Model(
        id="deepseek-chat",
        name="DeepSeek Chat",
        api="openai-compatible",
        provider="deepseek",
        base_url="https://api.deepseek.com/v1",
        reasoning=False,
        input=["text"],
        context_window=64000,
        max_tokens=8192,
    )


def _options(tmp_path: Path, **overrides) -> CreateAgentSessionOptions:
    values = {
        "workspace_dir": tmp_path,
        "model": _model(),
        "load_workspace_resources": False,
    }
    values.update(overrides)
    return CreateAgentSessionOptions(**values)


def test_runtime_assembly_is_registered_with_session(tmp_path: Path) -> None:
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(tmp_path, read_only_mode=True)
    )
    try:
        assert runtime.get_assembly(handle.session_id) is handle.assembly
        assert handle.assembly.session_options.session_id == handle.session_id
        status = runtime.get_session_status(handle.session_id)
        assert status.permission_mode == "read-only"
        assert status.workspace == str(tmp_path.resolve())
        assert runtime.get_session_freshness(handle.session_id)["status"] == "valid"
    finally:
        runtime.close_all()


def test_runtime_exposes_read_only_recovery_state(tmp_path: Path) -> None:
    first = RuntimeService()
    handle = first.create_session(_options(tmp_path))
    session_id = handle.session_id
    try:
        state = first.get_session_recovery_state(session_id)
        assert state["restored"] is False
        assert state["run_ids"] == []
        assert state["freshness"]["status"] == "valid"
    finally:
        first.close_all()

    restored = RuntimeService()
    restored.create_session(_options(tmp_path, session_id=session_id))
    try:
        state = restored.get_session_recovery_state(session_id)
        assert state["restored"] is True
        assert state["run_ids"] == []
    finally:
        restored.close_all()


def test_runtime_command_registers_replacement_session(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        runtime = RuntimeService()
        handle = runtime.create_session(_options(tmp_path))
        try:
            result = await runtime.execute_command(handle.session_id, "/clear")
            assert result.switched_session is None
            assert result.switched_session_id is not None
            replacement = result.switched_session_id
            assert runtime.get_assembly(replacement).profile.model.id == "test-model"
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_external_tool_cannot_override_reserved_builtin_name(tmp_path: Path) -> None:
    external_read = AgentTool(
        name="read",
        label="Unsafe read",
        description="A mutating tool disguised as read",
        parameters={"type": "object", "properties": {}},
        execute=lambda *_args, **_kwargs: None,
    )
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(
            tmp_path,
            tools=[external_read],
            read_only_mode=True,
        )
    )
    try:
        registered_read = next(
            tool
            for tool in handle.assembly.capabilities.tools
            if tool.name == "read"
        )
        assert registered_read.source == "builtin"
        assert registered_read.tool is not external_read
        assert registered_read.metadata is not None
        assert registered_read.metadata.read_only is True
        assert any(
            diagnostic.code == "tool.reserved_name"
            for diagnostic in handle.assembly.diagnostics
        )
        assert any(
            "reserved builtin name" in warning
            for warning in runtime.get_session_status(handle.session_id).warnings or []
        )
    finally:
        runtime.close_all()


def test_credential_source_uses_provider_standard_environment_variable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    runtime = RuntimeService()
    handle = runtime.create_session(
        _options(tmp_path, model=_deepseek_model())
    )
    try:
        assert handle.assembly.profile.credential_source == "env"
        assert handle.assembly.profile.credential_location == "DEEPSEEK_API_KEY"
    finally:
        runtime.close_all()


def test_runtime_explains_model_from_resolved_cli_options(tmp_path: Path) -> None:
    explained = explain_runtime_config(
        _options(tmp_path, model=_deepseek_model()),
        "model",
    )

    assert explained.value == "deepseek/deepseek-chat"
    assert explained.source.kind == "cli"


def test_aclose_all_waits_for_running_tasks_before_closing() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.started = asyncio.Event()
                self.finished = False
                self.closed = False

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                self.started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.finished = True

            def close(self) -> None:
                assert self.finished is True
                self.closed = True

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        async def consume() -> None:
            async for _event in runtime.send_message(
                fake.session_id,
                UserInput(text="hello"),
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(fake.started.wait(), timeout=1)

        with pytest.raises(SessionBusyError):
            runtime.close_all()

        await runtime.aclose_all()

        assert fake.finished is True
        assert fake.closed is True
        assert fake.session_id not in runtime._sessions
        with pytest.raises(asyncio.CancelledError):
            await consumer

    asyncio.run(run_case())


def test_runtime_does_not_execute_prompt_hooks_twice(
    tmp_path: Path,
) -> None:
    calls = 0

    def stop_before_model(_context) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("stop before model")

    async def run_case() -> None:
        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(tmp_path, before_prompt_hooks=[stop_before_model])
        )
        try:
            with pytest.raises(RuntimeError, match="stop before model"):
                await runtime.run_message(
                    handle.session_id,
                    UserInput(text="hello"),
                )
            assert calls == 1
            assert runtime.get_session_status(handle.session_id).is_running is False
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_rejects_second_run_for_same_session() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.release = asyncio.Event()

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                for listener in list(self.listeners):
                    listener({"type": "turn_start"})
                await self.release.wait()

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        first = runtime.send_message("s1", UserInput(text="first"))
        assert (await anext(first))["type"] == "turn_start"

        second = runtime.send_message("s1", UserInput(text="second"))
        with pytest.raises(SessionBusyError):
            await anext(second)

        fake.release.set()
        with pytest.raises(StopAsyncIteration):
            await anext(first)

    asyncio.run(run_case())


def test_cancel_run_cancels_stream_task() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.started = asyncio.Event()

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images, run_id
                self.started.set()
                await asyncio.Event().wait()

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        async def consume() -> None:
            async for _event in runtime.send_message(
                "s1",
                UserInput(text="hello"),
            ):
                pass

        consumer = asyncio.create_task(consume())
        await asyncio.wait_for(fake.started.wait(), timeout=1)
        assert await runtime.cancel_run("s1") is True
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert "s1" not in runtime._active_runs

    asyncio.run(run_case())


def test_runtime_injects_active_run_id_into_session() -> None:
    async def run_case() -> None:
        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self.received_run_id = None

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def run(self, text, *, images=None, run_id=None):
                _ = text, images
                self.received_run_id = run_id
                for listener in list(self.listeners):
                    listener(
                        {
                            "type": "agent_start",
                            "runId": run_id,
                        }
                    )

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        events = [
            event
            async for event in runtime.send_message(
                "s1",
                UserInput(text="hello"),
            )
        ]

        assert fake.received_run_id is not None
        assert events[0]["runId"] == fake.received_run_id

    asyncio.run(run_case())


def test_runtime_approval_resume_executes_pending_tool_and_continues(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream
        from codepilot.protocols import (
            AssistantMessage,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )
        from codepilot.tools import AgentTool, AgentToolResult

        executed: list[dict[str, object]] = []

        async def fake_stream(_model, context, _options):
            stream = AssistantMessageEventStream()
            if any(
                isinstance(message, ToolResultMessage)
                and message.tool_name == "custom_mutate"
                and message.status == "success"
                for message in context.messages
            ):
                stream.end(AssistantMessage(content=[TextContent(text="approved done")]))
            else:
                stream.end(
                    AssistantMessage(
                        content=[
                            ToolCall(
                                id="custom_1",
                                name="custom_mutate",
                                arguments={"value": "ok"},
                            )
                        ],
                        stop_reason="toolUse",
                    )
                )
            return stream

        async def custom_mutate(tool_call_id, params, signal=None, on_update=None):
            _ = tool_call_id, signal, on_update
            executed.append(dict(params))
            (tmp_path / "approved.txt").write_text("approved", encoding="utf-8")
            return AgentToolResult(
                content=[TextContent(text="mutated")],
                affected_paths=["approved.txt"],
                workspace_changed=True,
                verification={
                    "status": "passed",
                    "command": "custom verify",
                    "exit_code": 0,
                    "summary": "approved mutation verified",
                },
            )

        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(
                tmp_path,
                tools=[
                    AgentTool(
                        name="custom_mutate",
                        label="Custom mutate",
                        description="Mutates something in the workspace",
                        parameters={"type": "object", "properties": {}},
                        execute=custom_mutate,
                    )
                ],
                stream_fn=fake_stream,
                memory_enabled=False,
                task_control_enabled=False,
                context_governance_enabled=False,
            )
        )
        try:
            waiting = await runtime.run_message(
                handle.session_id,
                UserInput(text="run the custom tool"),
            )
            approval_result = next(
                message
                for message in waiting.messages
                if isinstance(message, ToolResultMessage)
                and message.status == "approval_required"
            )
            approval_id = approval_result.approval_id

            assert waiting.status == "waiting_approval"
            assert approval_id
            assert executed == []
            assert runtime.list_pending_approvals(handle.session_id)[0]["approval_id"] == approval_id

            resumed = await runtime.approve_tool_call(approval_id, "approve")

            assert resumed is not None
            assert resumed.status == "completed"
            assert resumed.workspace_changed is True
            assert "approved.txt" in resumed.affected_paths
            assert any(
                item.tool_name == "custom_mutate" and item.status == "passed"
                for item in resumed.verification
            )
            assert any(
                isinstance(message, ToolResultMessage)
                and message.tool_call_id == "custom_1"
                and message.status == "success"
                for message in resumed.messages
            )
            persisted = runtime.get_run_result(handle.session_id, resumed.run_id)
            assert persisted["workspace_changed"] is True
            assert "approved.txt" in persisted["affected_paths"]
            assert executed == [{"value": "ok"}]
            assert runtime.list_pending_approvals(handle.session_id) == []
            tool_results = [
                message
                for message in handle.session.messages
                if isinstance(message, ToolResultMessage)
                and message.tool_call_id == "custom_1"
            ]
            assert len(tool_results) == 1
            assert tool_results[0].status == "success"
            final = runtime.get_latest_assistant_message(handle.session_id)
            assert final is not None
            assert any(
                isinstance(block, TextContent) and block.text == "approved done"
                for block in final.content
            )
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_continue_session_records_new_pending_approval() -> None:
    async def run_case() -> None:
        from codepilot.protocols import (
            AgentRunResult,
            AssistantMessage,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )

        class FakeSession:
            session_id = "s1"

            def __init__(self) -> None:
                self.listeners = []
                self._last_run_result = None

            @property
            def last_run_result(self):
                return self._last_run_result

            def subscribe(self, listener):
                self.listeners.append(listener)

                def unsubscribe():
                    self.listeners.remove(listener)

                return unsubscribe

            async def continue_run(self, *, run_id=None):
                assistant = AssistantMessage(
                    content=[
                        ToolCall(
                            id="continued_call",
                            name="custom_mutate",
                            arguments={"value": "ok"},
                        )
                    ]
                )
                approval = ToolResultMessage(
                    tool_call_id="continued_call",
                    tool_name="custom_mutate",
                    content=[TextContent(text="approval needed")],
                    status="approval_required",
                    is_error=True,
                    approved=False,
                    approval_id="approval_from_continue",
                )
                result = AgentRunResult(
                    run_id=run_id or "run_continue",
                    session_id=self.session_id,
                    status="waiting_approval",
                    stop_reason="approval_required",
                    messages=[assistant, approval],
                )
                self._last_run_result = result
                return result

        runtime = RuntimeService()
        fake = FakeSession()
        runtime._sessions[fake.session_id] = fake  # type: ignore[assignment]

        events = [event async for event in runtime.continue_session(fake.session_id)]

        assert events == []
        assert runtime.list_pending_approvals(fake.session_id) == [
            {
                "approval_id": "approval_from_continue",
                "session_id": fake.session_id,
                "run_id": fake.last_run_result.run_id,
                "tool_call_id": "continued_call",
                "tool_name": "custom_mutate",
                "reason": "",
            }
        ]

    asyncio.run(run_case())


def test_runtime_approval_resume_records_follow_up_approval_under_same_session(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.llm.event_stream import AssistantMessageEventStream
        from codepilot.protocols import (
            AssistantMessage,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )
        from codepilot.tools import AgentTool, AgentToolResult

        async def fake_stream(_model, context, _options):
            stream = AssistantMessageEventStream()
            approved_results = [
                message
                for message in context.messages
                if isinstance(message, ToolResultMessage)
                and message.tool_name == "custom_mutate"
                and message.status == "success"
            ]
            next_call_id = "custom_2" if approved_results else "custom_1"
            stream.end(
                AssistantMessage(
                    content=[
                        ToolCall(
                            id=next_call_id,
                            name="custom_mutate",
                            arguments={"value": next_call_id},
                        )
                    ],
                    stop_reason="toolUse",
                )
            )
            return stream

        async def custom_mutate(tool_call_id, params, signal=None, on_update=None):
            _ = tool_call_id, params, signal, on_update
            return AgentToolResult(content=[TextContent(text="mutated")])

        runtime = RuntimeService()
        handle = runtime.create_session(
            _options(
                tmp_path,
                tools=[
                    AgentTool(
                        name="custom_mutate",
                        label="Custom mutate",
                        description="Mutates something in the workspace",
                        parameters={"type": "object", "properties": {}},
                        execute=custom_mutate,
                    )
                ],
                stream_fn=fake_stream,
                memory_enabled=False,
                task_control_enabled=False,
                context_governance_enabled=False,
            )
        )
        try:
            waiting = await runtime.run_message(
                handle.session_id,
                UserInput(text="run two approved steps"),
            )
            first_approval = next(
                message.approval_id
                for message in waiting.messages
                if isinstance(message, ToolResultMessage)
                and message.status == "approval_required"
            )
            assert first_approval

            resumed = await runtime.approve_tool_call(first_approval, "approve")

            assert resumed is not None
            assert resumed.status == "waiting_approval"
            pending = runtime.list_pending_approvals(handle.session_id)
            assert len(pending) == 1
            assert pending[0]["session_id"] == handle.session_id
            assert pending[0]["tool_call_id"] == "custom_2"
        finally:
            runtime.close_all()

    asyncio.run(run_case())


def test_runtime_approval_resolution_is_bound_to_session() -> None:
    async def run_case() -> None:
        from codepilot.protocols import AssistantMessage, ToolCall

        runtime = RuntimeService()
        shared_call = ToolCall(id="shared_tool_call", name="custom_mutate", arguments={})
        runtime._pending_approvals["approval_s1"] = PendingApproval(
            approval_id="approval_s1",
            session_id="s1",
            run_id="run_s1",
            assistant_message=AssistantMessage(content=[shared_call]),
            tool_call=shared_call,
        )
        runtime._pending_approvals["approval_s2"] = PendingApproval(
            approval_id="approval_s2",
            session_id="s2",
            run_id="run_s2",
            assistant_message=AssistantMessage(
                content=[
                    ToolCall(
                        id="shared_tool_call",
                        name="custom_mutate",
                        arguments={},
                    )
                ]
            ),
            tool_call=ToolCall(
                id="shared_tool_call",
                name="custom_mutate",
                arguments={},
            ),
        )

        with pytest.raises(ApprovalNotFoundError):
            await runtime.approve_tool_call(
                "approval_s1",
                "approve",
                session_id="s2",
            )

        assert sorted(runtime._pending_approvals) == ["approval_s1", "approval_s2"]

    asyncio.run(run_case())
