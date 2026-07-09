from __future__ import annotations

import asyncio


async def _collect_frames(iterator):
    return [frame async for frame in iterator]


class _EchoModelPort:
    async def stream(self, _request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent

        yield LLMCompleted(
            message=AssistantMessage(content=[TextContent(text="echo done")])
        )


class _PlanLifecycleModelPort:
    def __init__(self) -> None:
        self.calls = 0
        self.requests = []

    async def stream(self, request):
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent, ToolCall

        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            yield LLMCompleted(
                message=AssistantMessage(
                    content=[
                        ToolCall(
                            id="plan_done",
                            name="update_plan",
                            arguments={
                                "summary": "先确认注册边界，再实施并验证。",
                                "explanation": "执行完成",
                                "plan": [
                                    {
                                        "step": "阅读实现",
                                        "details": "确认注册逻辑和调用入口。",
                                        "verification": "列出受影响文件和行为。",
                                        "status": "completed",
                                    },
                                    {
                                        "step": "修改登录逻辑",
                                        "details": "按现有风格实施修改。",
                                        "verification": "运行注册相关测试。",
                                        "status": "in_progress",
                                    },
                                ],
                            },
                        )
                    ],
                    stop_reason="toolUse",
                )
            )
            return
        if self.calls == 2:
            yield LLMCompleted(
                message=AssistantMessage(
                    content=[TextContent(text="计划已整理，请审批。")]
                )
            )
            return
        yield LLMCompleted(
            message=AssistantMessage(content=[TextContent(text="实现和验证已完成。")])
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
        )
    )


def _persistent_session(gateway, session_id: str):
    controller = gateway._require_session(session_id)  # noqa: SLF001
    session = getattr(controller, "_session", None)
    assert session is not None
    return session


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


def test_runtime_gateway_plan_approval_resumes_same_task_run(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.actions import (
            CommandFinishedFrame,
            CommandSubmitted,
            PromptSubmitted,
            RunFinishedFrame,
            RunPausedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway

        model = _PlanLifecycleModelPort()
        gateway = RuntimeGateway(model_port=model)
        ref = _open_test_session(gateway, tmp_path)
        session = _persistent_session(gateway, ref.session_id)
        session.set_current_mode("plan")

        plan_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                PromptSubmitted(text="优化登录逻辑"),
            )
        ]
        paused = next(frame for frame in plan_frames if isinstance(frame, RunPausedFrame))
        run_id = paused.record.run_id
        assert not any(isinstance(frame, RunFinishedFrame) for frame in plan_frames)
        assert session.store.read_meta()["runtime_checkpoint"]["run_id"] == run_id

        frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                CommandSubmitted(text="/plan approve"),
            )
        ]

        command = next(frame for frame in frames if isinstance(frame, CommandFinishedFrame))
        finished = [frame for frame in frames if isinstance(frame, RunFinishedFrame)]
        current_plan = session.plan_state.current()

        assert "followup_prompt" not in command.record.data
        assert finished
        assert finished[-1].record.run_id == run_id
        assert finished[-1].record.status == "completed"
        assert session.current_mode == "build"
        assert current_plan["status"] == "completed"
        assert current_plan["approval_state"] == "approved"
        assert session.store.read_meta()["active_plan_id"] is None
        assert session.store.read_meta()["runtime_checkpoint"] is None
        assert len(list((tmp_path / ".codepilot" / "runs").iterdir())) == 1
        stored_user_texts = [
            message.content
            for message in session.store.load_session_messages()
            if getattr(message, "role", "") == "user"
        ]
        assert stored_user_texts == ["优化登录逻辑"]
        assert model.requests[-1].correlation.run_id == run_id
        assert "## Mode Policy" in model.requests[-1].system_prompt
        assert "Approved Execution Contract" in model.requests[-1].system_prompt
        assert "重新制定" in model.requests[-1].system_prompt
        assert not any(
            getattr(message, "metadata", {}).get("message_kind") == "plan_summary"
            for message in model.requests[-1].messages
        )

        events = session.store.run_store.load_events(run_id)
        event_ids = [event["eventId"] for event in events if "eventId" in event]
        turn_ids = [
            event["turnId"]
            for event in events
            if isinstance(event.get("turnId"), int)
        ]
        stored_run = session.store.run_store.load_run_result(run_id)
        assert len(event_ids) == len(set(event_ids))
        assert turn_ids == sorted(turn_ids)
        assert stored_run["resume_count"] == 1
        assert stored_run["model_attempts"] == 3
        assert stored_run["tool_calls"] == 1
        assert stored_run["phase"] == "terminal"

    asyncio.run(run_case())


def test_runtime_gateway_natural_language_is_plan_feedback_not_approval(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.actions import PromptSubmitted, RunPausedFrame
        from codepilot.runtime.gateway import RuntimeGateway

        class FeedbackModel(_PlanLifecycleModelPort):
            async def stream(self, request):
                from codepilot.llm.ports import LLMCompleted
                from codepilot.protocols import AssistantMessage, TextContent, ToolCall

                self.calls += 1
                self.requests.append(request)
                if self.calls in {1, 3}:
                    suffix = "并先补测试" if self.calls == 3 else ""
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id=f"plan_{self.calls}",
                                    name="update_plan",
                                    arguments={
                                        "summary": f"优化登录逻辑{suffix}。",
                                        "plan": [
                                            {
                                                "step": "修改登录逻辑",
                                                "details": f"实施目标内调整{suffix}。",
                                                "verification": "运行注册测试。",
                                                "status": "pending",
                                            }
                                        ],
                                    },
                                )
                            ],
                            stop_reason="toolUse",
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[TextContent(text="计划已整理，请审批。")]
                    )
                )

        gateway = RuntimeGateway(model_port=FeedbackModel())
        ref = _open_test_session(gateway, tmp_path)
        session = _persistent_session(gateway, ref.session_id)
        session.set_current_mode("plan")

        first_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                PromptSubmitted(text="优化登录逻辑"),
            )
        ]
        first_pause = next(
            frame for frame in first_frames if isinstance(frame, RunPausedFrame)
        )

        frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                PromptSubmitted(text="同意"),
            )
        ]

        second_pause = next(frame for frame in frames if isinstance(frame, RunPausedFrame))
        current_plan = session.plan_state.current()

        assert second_pause.record.run_id == first_pause.record.run_id
        assert current_plan["status"] == "proposed"
        assert current_plan["approval_state"] == "pending"
        assert current_plan["revision"] == 2
        assert session.current_mode == "plan"

    asyncio.run(run_case())


def test_runtime_gateway_plan_reject_waits_for_feedback_in_same_run(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.actions import (
            CommandFinishedFrame,
            CommandSubmitted,
            PromptSubmitted,
            RunPausedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway

        gateway = RuntimeGateway(model_port=_PlanLifecycleModelPort())
        ref = _open_test_session(gateway, tmp_path)
        session = _persistent_session(gateway, ref.session_id)
        session.set_current_mode("plan")

        plan_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                PromptSubmitted(text="优化登录逻辑"),
            )
        ]
        first_pause = next(
            frame for frame in plan_frames if isinstance(frame, RunPausedFrame)
        )

        reject_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                CommandSubmitted(text="/reject"),
            )
        ]
        command = next(
            frame for frame in reject_frames if isinstance(frame, CommandFinishedFrame)
        )
        second_pause = next(
            frame for frame in reject_frames if isinstance(frame, RunPausedFrame)
        )
        plan = session.plan_state.current()

        assert command.record.command == "/plan reject"
        assert second_pause.record.run_id == first_pause.record.run_id
        assert second_pause.record.stop_reason == "plan_clarification_required"
        assert plan["status"] == "rejected"
        assert plan["approval_state"] == "rejected"
        assert session.store.read_meta()["runtime_checkpoint"]["phase"] == "plan_clarification"

    asyncio.run(run_case())


def test_runtime_gateway_mode_switch_continues_paused_run(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent
        from codepilot.runtime.actions import (
            CommandFinishedFrame,
            CommandSubmitted,
            PromptSubmitted,
            RunFinishedFrame,
            RunPausedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway

        class ClarifyThenBuildModel:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                text = (
                    "是否保留旧命令兼容？"
                    if self.calls == 1
                    else "已按 build 模式完成任务。"
                )
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text=text)])
                )

        gateway = RuntimeGateway(model_port=ClarifyThenBuildModel())
        ref = _open_test_session(gateway, tmp_path)
        session = _persistent_session(gateway, ref.session_id)
        session.set_current_mode("plan")

        plan_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                PromptSubmitted(text="处理兼容性重构"),
            )
        ]
        paused = next(frame for frame in plan_frames if isinstance(frame, RunPausedFrame))

        mode_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                CommandSubmitted(text="/mode build"),
            )
        ]
        command = next(
            frame for frame in mode_frames if isinstance(frame, CommandFinishedFrame)
        )
        finished = next(
            frame for frame in mode_frames if isinstance(frame, RunFinishedFrame)
        )

        assert command.record.data["current_mode"] == "build"
        assert finished.record.run_id == paused.record.run_id
        assert session.current_mode == "build"
        assert len(list((tmp_path / ".codepilot" / "runs").iterdir())) == 1

    asyncio.run(run_case())


def test_runtime_gateway_rejects_mode_switch_while_run_is_executing(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent
        from codepilot.runtime.actions import (
            CommandSubmitted,
            FailedFrame,
            PromptSubmitted,
        )
        from codepilot.runtime.gateway import RuntimeGateway

        started = asyncio.Event()
        release = asyncio.Event()

        class SlowModel:
            async def stream(self, _request):
                started.set()
                await release.wait()
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        gateway = RuntimeGateway(model_port=SlowModel())
        ref = _open_test_session(gateway, tmp_path)
        session = _persistent_session(gateway, ref.session_id)

        prompt_task = asyncio.create_task(
            _collect_frames(
                gateway.dispatch(
                    ref.session_id,
                    PromptSubmitted(text="执行长任务"),
                )
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)

        mode_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                CommandSubmitted(text="/mode plan"),
            )
        ]
        release.set()
        await prompt_task

        assert isinstance(mode_frames[-1], FailedFrame)
        assert mode_frames[-1].error["code"] == "runtime.run_active"
        assert session.current_mode == "build"

    asyncio.run(run_case())


def test_runtime_gateway_cancel_stops_active_task_and_records_aborted_run(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, TextContent
        from codepilot.runtime.actions import (
            CancelledFrame,
            PromptSubmitted,
            RunCancelled,
            RunFinishedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway

        started = asyncio.Event()
        cancelled = asyncio.Event()
        release = asyncio.Event()

        class SlowModel:
            async def stream(self, _request):
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="too late")])
                )

        gateway = RuntimeGateway(model_port=SlowModel())
        ref = _open_test_session(gateway, tmp_path)

        async def collect_prompt_frames():
            return [
                frame
                async for frame in gateway.dispatch(
                    ref.session_id,
                    PromptSubmitted(text="slow request"),
                )
            ]

        prompt_task = asyncio.create_task(collect_prompt_frames())
        await asyncio.wait_for(started.wait(), timeout=1)

        cancel_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                RunCancelled(reason="user"),
            )
        ]

        try:
            await asyncio.wait_for(cancelled.wait(), timeout=1)
        finally:
            release.set()
            prompt_frames = await asyncio.wait_for(prompt_task, timeout=1)

        session = _persistent_session(gateway, ref.session_id)
        runs = session.store.load_run_results(limit=1)

        assert isinstance(cancel_frames[-1], CancelledFrame)
        assert cancel_frames[-1].cancelled is True
        assert any(isinstance(frame, RunFinishedFrame) for frame in prompt_frames)
        assert runs
        assert runs[-1]["status"] == "aborted"
        assert runs[-1]["stop_reason"] == "aborted"
        assert runs[-1]["signals"]["cancelled"] is True

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
        assert frames[-1].record.data["current_mode"] == "plan"
        assert gateway.describe(ref.session_id).session.current_mode == "plan"

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


def test_runtime_gateway_tools_command_respects_current_mode(tmp_path) -> None:
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
                current_mode="read",
            )
        )

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, CommandSubmitted(text="/tools"))
        ]

        assert isinstance(frames[-1], CommandFinishedFrame)
        names = {item["name"] for item in frames[-1].record.data["tools"]}
        assert "read" in names
        assert "workspace_status" in names
        assert "write" not in names
        assert "apply_patch" not in names
        assert "bash" not in names

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


def test_runtime_gateway_commits_uncaught_runtime_failure(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.actions import FailedFrame, PromptSubmitted
        from codepilot.runtime.gateway import RuntimeGateway

        class ExplodingModel:
            async def stream(self, _request):
                raise RuntimeError("provider crashed outside structured failure")
                yield  # pragma: no cover

        gateway = RuntimeGateway(model_port=ExplodingModel())
        ref = _open_test_session(gateway, tmp_path)

        frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="hello"))
        ]
        session = _persistent_session(gateway, ref.session_id)
        runs = session.store.load_run_results(limit=1)

        assert isinstance(frames[-1], FailedFrame)
        assert runs
        assert runs[-1]["status"] == "failed"
        assert runs[-1]["stop_reason"] == "internal_error"
        assert session.store.read_meta()["runtime_checkpoint"] is None

    asyncio.run(run_case())


def test_runtime_error_payload_preserves_dict_code_and_message() -> None:
    from codepilot.runtime.gateway import _runtime_error_payload

    payload = _runtime_error_payload(
        {
            "code": "run.max_iterations",
            "message": "Stopped after reaching max_tool_iterations=12",
            "details": {"limit": 12},
        }
    )

    assert payload["code"] == "run.max_iterations"
    assert payload["message"] == "Stopped after reaching max_tool_iterations=12"
    assert payload["details"]["limit"] == 12


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
        from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall
        from codepilot.runtime.actions import (
            ApprovalDecided,
            ApprovalRequiredFrame,
            PromptSubmitted,
            RunFinishedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.contracts import (
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
                self.sources = []

            def catalog(self, current_mode: str = "build"):
                return {"tools": ["shell"]}

            async def execute(self, invocation):
                self.sources.append(invocation.source)
                if invocation.source == "approval_resume":
                    return ToolObservation(
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="success",
                        content=(TextContent(text="created"),),
                        affected_paths=("created.txt",),
                        workspace_changed=True,
                        verification=(
                            RunVerification(
                                tool_call_id=invocation.tool_call_id,
                                tool_name=invocation.name,
                                status="passed",
                                command="approved shell write",
                                exit_code=0,
                                summary="approved write verified",
                            ),
                        ),
                        metadata={"approval_id": "approval1"},
                    )
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
                raise AssertionError("gateway approval resume should use session checkpoint")

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
        original_run_id = approval_frames[-1].approval.run_id
        session = _persistent_session(gateway, ref.session_id)
        runs_after_pause = session.store.load_run_results()
        assert [run["run_id"] for run in runs_after_pause] == [original_run_id]
        assert runs_after_pause[-1]["status"] == "waiting_approval"

        resume_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                ApprovalDecided(approval_id="approval1", decision="approve", reason="ok"),
            )
        ]
        finished = [frame for frame in resume_frames if isinstance(frame, RunFinishedFrame)]

        assert tools.sources == ["agent", "approval_resume"]
        assert finished
        assert finished[-1].record.run_id == original_run_id
        assert finished[-1].record.final_text == "approved done"
        runs_after_resume = session.store.load_run_results()
        assert [run["run_id"] for run in runs_after_resume] == [original_run_id]
        assert runs_after_resume[-1]["status"] == "completed"
        event_ids = [
            event["eventId"]
            for event in session.store.run_store.load_events(original_run_id)
            if isinstance(event.get("eventId"), str)
        ]
        assert len(event_ids) == len(set(event_ids))

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
        from codepilot.tools.contracts import ToolInterruption, ToolObservation

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
            def catalog(self, current_mode: str = "build"):
                return []

            async def execute(self, invocation):
                if invocation.source == "approval_resume":
                    return ToolObservation(
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="success",
                        content=(TextContent(text="decision=approve"),),
                        metadata={"approval_id": "approval1"},
                    )
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
                raise AssertionError("gateway approval resume should use session checkpoint")

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


def test_runtime_gateway_approval_resume_uses_session_checkpoint_when_registry_is_lost(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.ports import LLMCompleted
        from codepilot.protocols import AssistantMessage, RunVerification, TextContent, ToolCall, ToolResultMessage
        from codepilot.runtime.actions import (
            ApprovalDecided,
            ApprovalRequiredFrame,
            PromptSubmitted,
            RunFinishedFrame,
        )
        from codepilot.runtime.gateway import RuntimeGateway
        from codepilot.tools.contracts import (
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
                                    id="call_write",
                                    name="write",
                                    arguments={
                                        "path": "created.txt",
                                        "content": "hello",
                                    },
                                )
                            ]
                        )
                    )
                    return
                last = request.messages[-1]
                assert isinstance(last, ToolResultMessage)
                assert last.tool_call_id == "call_write"
                assert last.approval_id == "approval_lost"
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="approved after restore")])
                )

        class FakeTools:
            def __init__(self) -> None:
                self.sources = []

            def catalog(self, current_mode: str = "build"):
                return {"tools": ["write"]}

            async def execute(self, invocation):
                self.sources.append(invocation.source)
                if invocation.source == "approval_resume":
                    return ToolObservation(
                        tool_call_id=invocation.tool_call_id,
                        name=invocation.name,
                        status="success",
                        content=(TextContent(text="wrote created.txt"),),
                        affected_paths=("created.txt",),
                        workspace_changed=True,
                        verification=(
                            RunVerification(
                                tool_call_id=invocation.tool_call_id,
                                tool_name=invocation.name,
                                status="passed",
                                command="workspace write approved",
                                exit_code=0,
                                summary="approved write verified",
                            ),
                        ),
                        metadata={"approval_id": "approval_lost"},
                    )
                return ToolObservation(
                    tool_call_id=invocation.tool_call_id,
                    name=invocation.name,
                    status="approval_required",
                    interruption=ToolInterruption(
                        approval_id="approval_lost",
                        run_id=invocation.run_id,
                        tool_call_id=invocation.tool_call_id,
                        tool_name=invocation.name,
                        arguments=invocation.arguments,
                        reason="ask mode",
                        risk=ToolRiskView(level="medium"),
                    ),
                    metadata={"approval_id": "approval_lost"},
                )

            async def resume(self, _decision):
                raise AssertionError("checkpoint resume should not require in-memory pending calls")

        tools = FakeTools()
        gateway = RuntimeGateway(model_port=FakeModel(), tool_port=tools)
        ref = _open_test_session(gateway, tmp_path)

        prompt_frames = [
            frame
            async for frame in gateway.dispatch(ref.session_id, PromptSubmitted(text="write file"))
        ]
        assert any(isinstance(frame, ApprovalRequiredFrame) for frame in prompt_frames)
        assert [
            approval.approval_id
            for approval in gateway.describe(ref.session_id).pending_approvals
        ] == ["approval_lost"]

        gateway._approvals.clear()  # noqa: SLF001 - simulate process-local registry loss
        assert [
            approval.approval_id
            for approval in gateway.describe(ref.session_id).pending_approvals
        ] == ["approval_lost"]

        resume_frames = [
            frame
            async for frame in gateway.dispatch(
                ref.session_id,
                ApprovalDecided(approval_id="approval_lost", decision="approve", reason="ok"),
            )
        ]

        assert tools.sources == ["agent", "approval_resume"]
        assert any(isinstance(frame, RunFinishedFrame) for frame in resume_frames)

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
        from codepilot.tools.contracts import ToolInterruption, ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="shell", arguments={})]
                    )
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
                return []

            async def execute(self, invocation):
                if invocation.source == "approval_resume":
                    raise RuntimeError("approval resume adapter failed")
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
                raise AssertionError("gateway approval resume should use session checkpoint")

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
        from codepilot.tools.contracts import ToolInterruption, ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="write", arguments={})]
                    )
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
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
        from codepilot.tools.contracts import ToolInterruption, ToolObservation

        class FakeModel:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(
                        content=[ToolCall(id="call1", name="write", arguments={})]
                    )
                )

        class FakeTools:
            def catalog(self, current_mode: str = "build"):
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
