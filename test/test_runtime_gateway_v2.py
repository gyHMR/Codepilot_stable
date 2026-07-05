from __future__ import annotations

import asyncio


class _EchoModelPort:
    async def stream(self, _request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent

        yield LLMCompleted(
            message=AssistantMessage(content=[TextContent(text="echo done")])
        )


def _unit_model():
    from codepilot.protocols import Model

    return Model(
        id="runtime-v2",
        name="Runtime V2",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def _open_test_session(gateway, workspace):
    from codepilot.runtime import SessionOpenIntent

    return gateway.open_session(
        SessionOpenIntent(
            workspace_dir=workspace,
            model=_unit_model(),
            memory_enabled=False,
            task_control_enabled=False,
        )
    )


def test_runtime_gateway_dispatch_prompt_streams_progress_and_finished_frames(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.actions import ProgressFrame, PromptSubmitted, RunFinishedFrame
        from codepilot.runtime.gateway import RuntimeGateway

        gateway = RuntimeGateway(model_port=_EchoModelPort())
        ref = _open_test_session(gateway, tmp_path)

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello"))
        ]

        assert any(isinstance(frame, ProgressFrame) for frame in frames)
        finished = [frame for frame in frames if isinstance(frame, RunFinishedFrame)]
        assert finished
        assert finished[-1].record.final_text

    asyncio.run(run_case())


def test_runtime_gateway_dispatch_command_and_cancel_as_frames(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.actions import (
            CancelledFrame,
            CommandFinishedFrame,
            CommandSubmitted,
            RunCancelled,
        )
        from codepilot.runtime.gateway import RuntimeGateway

        gateway = RuntimeGateway(model_port=_EchoModelPort())
        ref = _open_test_session(gateway, tmp_path)

        command_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, CommandSubmitted(text="/status"))
        ]
        cancel_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, RunCancelled(reason="user"))
        ]

        assert isinstance(command_frames[-1], CommandFinishedFrame)
        assert command_frames[-1].record.handled is True
        assert isinstance(cancel_frames[-1], CancelledFrame)
        assert cancel_frames[-1].cancelled is False

    asyncio.run(run_case())


def test_runtime_gateway_default_open_session_uses_real_assembly(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.runtime.actions import PromptSubmitted, RunFinishedFrame
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="real done")]))
            return stream

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
                stream_fn=fake_stream,
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello"))
        ]
        finished = [frame for frame in frames if isinstance(frame, RunFinishedFrame)]

        assert finished
        assert finished[-1].record.final_text == "real done"
        assert gateway.describe(ref.session_id).session.last_run_id == finished[-1].record.run_id

    asyncio.run(run_case())


def test_runtime_gateway_real_command_flow_updates_session_mode(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.runtime.actions import CommandFinishedFrame, CommandSubmitted
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, CommandSubmitted(text="/mode plan"))
        ]

        assert isinstance(frames[-1], CommandFinishedFrame)
        assert frames[-1].record.handled is True
        assert frames[-1].record.data["task_mode"] == "plan"
        assert gateway.describe(ref.session_id).session.task_mode == "plan"

    asyncio.run(run_case())


def test_runtime_gateway_tools_command_renders_tool_port_catalog(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.runtime.actions import CommandFinishedFrame, CommandSubmitted
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, CommandSubmitted(text="/tools"))
        ]

        assert isinstance(frames[-1], CommandFinishedFrame)
        names = {item["name"] for item in frames[-1].record.data["tools"]}
        assert "read" in names
        assert "write" in names

    asyncio.run(run_case())


def test_runtime_gateway_command_can_register_derived_session(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.runtime.actions import CommandFinishedFrame, CommandSubmitted, PromptSubmitted
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
            return stream

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
                stream_fn=fake_stream,
            )
        )
        _ = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="seed"))
        ]

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, CommandSubmitted(text="/new"))
        ]
        finished = [frame for frame in frames if isinstance(frame, CommandFinishedFrame)]

        assert finished
        new_session_id = finished[-1].record.switched_session_id
        assert new_session_id
        assert gateway.describe(new_session_id).session.session_id == new_session_id

    asyncio.run(run_case())


def test_runtime_gateway_prompt_failure_returns_failed_frame(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import Model
        from codepilot.runtime.actions import FailedFrame, PromptSubmitted
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        async def broken_stream(_model, _context, _options):
            _ = AssistantMessageEventStream
            raise RuntimeError("model unavailable")

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
                stream_fn=broken_stream,
                retry_enabled=False,
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello"))
        ]

        assert isinstance(frames[-1], FailedFrame)
        assert frames[-1].error["code"] == "runtime.dispatch_failed"
        cancel_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello again"))
        ]
        assert isinstance(cancel_frames[-1], FailedFrame)

    asyncio.run(run_case())


def test_runtime_gateway_real_prompt_flow_retries_model_failure(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.runtime.actions import ProgressFrame, PromptSubmitted, RunFinishedFrame
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        calls = 0

        async def flaky_stream(_model, _context, _options):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("temporary model outage")
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="recovered")]))
            return stream

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
                stream_fn=flaky_stream,
                retry_enabled=True,
                max_retries=1,
                retry_base_delay_ms=0,
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello"))
        ]

        assert calls == 2
        assert any(
            isinstance(frame, ProgressFrame)
            and frame.event["type"] == "model_retry_start"
            for frame in frames
        )
        finished = [frame for frame in frames if isinstance(frame, RunFinishedFrame)]
        assert finished
        assert finished[-1].record.final_text == "recovered"

    asyncio.run(run_case())


def test_runtime_gateway_approval_decision_resumes_through_v2_tool_port(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall
        from codepilot.runtime.actions import (
            ApprovalDecided,
            ApprovalRequiredFrame,
            PromptSubmitted,
            RunFinishedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.ports import (
            ToolInterruption,
            ToolObservation,
            ToolRiskView,
        )

        class FakeModel:
            def __init__(self) -> None:
                self.requests = []

            async def stream(self, request):
                self.requests.append(request)
                if len(self.requests) == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="call1",
                                    name="shell",
                                    arguments={"cmd": "touch created.txt"},
                                )
                            ]
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="approved done")])
                )

        class FakeTools:
            def __init__(self) -> None:
                self.resume_decisions = []

            def catalog(self):
                return {"tools": ["shell"]}

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                        arguments=invocation.arguments,
                        reason="writes workspace",
                        risk=ToolRiskView(level="high"),
                    ),
                )

            async def resume(self, decision):
                assert gateway.describe(ref.session_id).status.is_running is True
                self.resume_decisions.append(decision)
                return ToolObservation(
                    tool_call_id="call1",
                    name="shell",
                    status="success",
                    content=(TextContent(text="created"),),
                    affected_paths=("created.txt",),
                    workspace_changed=True,
                    metadata={"approval_id": decision.approval_id},
                )

        model = FakeModel()
        tools = FakeTools()
        gateway = RuntimeGateway(model_port=model, tool_port=tools)
        ref = _open_test_session(gateway, tmp_path)

        prompt_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="change file"))
        ]
        approval_frames = [
            frame for frame in prompt_frames if isinstance(frame, ApprovalRequiredFrame)
        ]
        assert approval_frames
        assert approval_frames[-1].approval.approval_id == "approval1"

        resume_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                ApprovalDecided(approval_id="approval1", decision="approve", reason="ok"),
            )
        ]
        finished = [frame for frame in resume_frames if isinstance(frame, RunFinishedFrame)]

        assert [decision.approval_id for decision in tools.resume_decisions] == ["approval1"]
        assert finished
        assert finished[-1].record.final_text == "approved done"

    asyncio.run(run_case())


def test_runtime_gateway_approval_resume_failure_returns_failed_frame(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted, LLMFailed
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall
        from codepilot.runtime.actions import (
            ApprovalDecided,
            ApprovalRequiredFrame,
            FailedFrame,
            PromptSubmitted,
            RunFinishedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.ports import ToolInterruption, ToolObservation

        class FakeModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[ToolCall(id="call1", name="shell", arguments={})]
                        )
                    )
                    return
                yield LLMFailed(error={"code": "llm.failed_after_approval"})

        class FakeTools:
            def catalog(self):
                return []

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                    ),
                )

            async def resume(self, decision):
                return ToolObservation(
                    tool_call_id="call1",
                    name="shell",
                    status="success",
                    content=(TextContent(text=f"decision={decision.decision}"),),
                    metadata={"approval_id": decision.approval_id},
                )

        gateway = RuntimeGateway(model_port=FakeModel(), tool_port=FakeTools())
        ref = _open_test_session(gateway, tmp_path)

        prompt_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="change file"))
        ]
        assert any(isinstance(frame, ApprovalRequiredFrame) for frame in prompt_frames)

        resume_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                ApprovalDecided(approval_id="approval1", decision="approve"),
            )
        ]

        assert isinstance(resume_frames[-1], FailedFrame)
        assert not any(isinstance(frame, RunFinishedFrame) for frame in resume_frames)

    asyncio.run(run_case())


def test_runtime_gateway_approval_resume_exception_returns_failed_frame(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.runtime.actions import (
            ApprovalDecided,
            ApprovalRequiredFrame,
            FailedFrame,
            PromptSubmitted,
        )
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.ports import ToolInterruption, ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="shell", arguments={})]
                    )
                )

        class FakeTools:
            def catalog(self):
                return []

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                    ),
                )

            async def resume(self, _decision):
                raise RuntimeError("approval resume adapter failed")

        gateway = RuntimeGateway(model_port=FakeModel(), tool_port=FakeTools())
        ref = _open_test_session(gateway, tmp_path)

        prompt_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="change file"))
        ]
        assert any(isinstance(frame, ApprovalRequiredFrame) for frame in prompt_frames)

        resume_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                ApprovalDecided(approval_id="approval1", decision="approve"),
            )
        ]

        assert isinstance(resume_frames[-1], FailedFrame)
        assert resume_frames[-1].error["code"] == "runtime.dispatch_failed"

    asyncio.run(run_case())


def test_runtime_gateway_approval_decision_is_bound_to_origin_session(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.runtime.actions import (
            ApprovalDecided,
            ApprovalRequiredFrame,
            FailedFrame,
            PromptSubmitted,
        )
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.ports import ToolInterruption, ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="write", arguments={})]
                    )
                )

        class FakeTools:
            def catalog(self):
                return []

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                    ),
                )

            async def resume(self, _decision):  # pragma: no cover - must not be reached
                raise AssertionError("wrong session must not resume approval")

        gateway = RuntimeGateway(
            model_port=FakeModel(),
            tool_port=FakeTools(),
        )
        first = _open_test_session(gateway, tmp_path)
        second = _open_test_session(gateway, tmp_path)

        prompt_frames = [
            frame
            async for frame in gateway.dispatch(
                first.session_id,
                PromptSubmitted(text="needs approval"),
            )
        ]
        assert any(isinstance(frame, ApprovalRequiredFrame) for frame in prompt_frames)

        resume_frames = [
            frame
            async for frame in gateway.dispatch(
                second.session_id,
                ApprovalDecided(approval_id="approval1", decision="approve"),
            )
        ]

        assert isinstance(resume_frames[-1], FailedFrame)
        assert resume_frames[-1].error["code"] == "runtime.approval_session_mismatch"

    asyncio.run(run_case())


def test_runtime_gateway_real_prompt_flow_uses_v2_core_and_tool_ports(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, ToolCall
        from codepilot.runtime.actions import ApprovalRequiredFrame, PromptSubmitted
        from codepilot.runtime import RuntimeGateway, SessionOpenIntent

        async def fake_stream(_model, context, _options):
            stream = AssistantMessageEventStream()
            assert context.tools
            stream.end(
                AssistantMessage(
                    content=[
                        ToolCall(
                            id="call_write",
                            name="write",
                            arguments={
                                "path": "created.txt",
                                "content": "hello",
                                "overwrite": True,
                            },
                        )
                    ]
                )
            )
            return stream

        gateway = RuntimeGateway()
        ref = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=Model(
                    id="runtime-v2",
                    name="Runtime V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                memory_enabled=False,
                task_control_enabled=False,
                tool_permission_mode="ask",
                stream_fn=fake_stream,
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="write file"))
        ]
        approval = [frame for frame in frames if isinstance(frame, ApprovalRequiredFrame)]

        assert approval
        assert approval[-1].approval.tool_name == "write"
        assert approval[-1].approval.tool_call_id == "call_write"
        assert not (tmp_path / "created.txt").exists()

    asyncio.run(run_case())


def test_runtime_gateway_close_all_closes_sessions_and_pending_approvals(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, ToolCall
        from codepilot.runtime.actions import ApprovalRequiredFrame, PromptSubmitted
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.ports import ToolInterruption, ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="write", arguments={})]
                    )
                )

        class FakeTools:
            def catalog(self):
                return []

            async def execute(self, invocation):
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval1",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                    ),
                )

            async def resume(self, _decision):
                raise AssertionError("not used")

        gateway = RuntimeGateway(
            model_port=FakeModel(),
            tool_port=FakeTools(),
        )
        ref = _open_test_session(gateway, tmp_path)
        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello"))
        ]
        assert any(isinstance(frame, ApprovalRequiredFrame) for frame in frames)
        assert [
            approval.approval_id
            for approval in gateway.describe(ref.session_id).pending_approvals
        ] == ["approval1"]

        await gateway.close_all()

        try:
            gateway.describe(ref.session_id)
        except KeyError:
            pass
        else:  # pragma: no cover
            raise AssertionError("close_all must remove the session")

    asyncio.run(run_case())


def test_runtime_gateway_close_clears_active_run_marker_for_session_id(tmp_path) -> None:
    from codepilot.protocols import Model
    from codepilot.runtime import RuntimeGateway, SessionOpenIntent

    gateway = RuntimeGateway()
    ref = gateway.open_session(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            session_id="session_reopen",
            model=Model(
                id="runtime-v2",
                name="Runtime V2",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            memory_enabled=False,
        )
    )

    gateway._active_runs.start(ref.session_id, "run_orphaned")  # noqa: SLF001
    assert gateway.describe(ref.session_id).status.is_running is True

    gateway.close(ref.session_id)
    reopened = gateway.open_session(
        SessionOpenIntent(
            workspace_dir=tmp_path,
            session_id=ref.session_id,
            model=Model(
                id="runtime-v2",
                name="Runtime V2",
                api="unit-test",
                provider="unit-test",
                base_url="",
                reasoning=False,
                input=["text"],
                context_window=4000,
                max_tokens=500,
            ),
            memory_enabled=False,
        )
    )

    assert reopened.session_id == ref.session_id
    assert gateway.describe(reopened.session_id).status.is_running is False
