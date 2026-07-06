from __future__ import annotations

import asyncio


async def _run_real_session_spine(session, text: str, *, run_id: str):
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.loop import run_agent_loop
    from codepilot.llm.adapter import ProviderModelPort
    from codepilot.sessions.contracts import SessionRunIntent
    from codepilot.sessions.controller import _bind_session_runtime

    controller = _bind_session_runtime(session)
    prepared = await controller.prepare_run(
        SessionRunIntent(text=text, run_id=run_id)
    )
    outcome = await run_agent_loop(
        prepared.loop_input,
        AgentLoopPorts(
            model=ProviderModelPort(
                model=session.conversation.model,
                stream_fn=session.stream_fn,
                convert_messages=session.convert_to_llm,
                get_api_key=session.get_api_key,
            ),
            tools=None,
            context=prepared.context_port,
        ),
    )
    record = await controller.commit_run(prepared, outcome)
    if session.conversation.last_run_result is None:
        raise AssertionError("V2 session controller did not commit an AgentRunResult")
    return session.conversation.last_run_result, record


def test_session_controller_does_not_expose_unused_state_queries() -> None:
    from codepilot.sessions.controller import SessionController

    assert not hasattr(SessionController, "messages")
    assert not hasattr(SessionController, "diagnostics")
    assert not hasattr(SessionController, "has_persistent_session")
    assert not hasattr(SessionController, "context_port")
    assert not hasattr(SessionController, "outcome_from_agent_result")
    assert not hasattr(SessionController, "from_options")
    assert not hasattr(SessionController, "from_runtime_session")


def test_session_controller_prepares_and_commits_run_without_exposing_live_session(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome
        from codepilot.protocols import AgentRunCounters, AssistantMessage, Model, TextContent
        from codepilot.sessions.contracts import SessionOptions
        from codepilot.sessions.contracts import SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.prepare import SessionRuntime

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2",
                    name="Session V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                workspace_dir=tmp_path,
                session_id="s1",
                memory_enabled=False,
                task_control_enabled=False,
            )
        )
        controller = _bind_session_runtime(session)

        prepared = await controller.prepare_run(SessionRunIntent(text="  hello  "))
        assert prepared.session_id == "s1"
        assert prepared.loop_input.user_prompt == "hello"
        assert prepared.rollback_baseline is not None

        final = AssistantMessage(content=[TextContent(text="done")])
        record = await controller.commit_run(
            prepared,
            AgentLoopOutcome(
                run_id=prepared.run_id,
                status="completed",
                stop_reason="final_answer",
                new_messages=[final],
                final_message=final,
                counters=AgentRunCounters(model_attempts=1),
            ),
        )

        assert record.run_id == prepared.run_id
        assert record.final_text == "done"
        assert controller.describe().message_count == 2
        assert not hasattr(record, "store")
        assert not hasattr(record, "agent")
        session._close()

    asyncio.run(run_case())


def test_session_controller_applies_command_as_session_intent(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionOptions
        from codepilot.sessions.contracts import SessionCommandIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.prepare import SessionRuntime

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2",
                    name="Session V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                workspace_dir=tmp_path,
                session_id="s1",
                memory_enabled=False,
                task_control_enabled=False,
            )
        )
        controller = _bind_session_runtime(session)
        record = await controller.apply_command(SessionCommandIntent(text="/status"))

        assert record.handled is True
        assert record.output_lines
        assert record.command == "/status"
        session._close()

    asyncio.run(run_case())


def test_session_controller_cannot_be_created_without_runtime_session() -> None:
    import pytest

    from codepilot.sessions.controller import SessionController

    with pytest.raises(ValueError, match="SessionController requires SessionRuntime"):
        SessionController(session_id="s1")


def test_session_controller_drives_real_session_lifecycle(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.sessions.prepare import SessionRuntime
        from codepilot.sessions.contracts import SessionOptions

        hook_calls: list[tuple[str, str, bool]] = []

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
            return stream

        def before_hook(ctx) -> None:
            hook_calls.append(("before", ctx.text, ctx.is_continue))

        def after_hook(ctx) -> None:
            hook_calls.append(("after", ctx.text, ctx.is_continue))

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2",
                    name="Session V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                workspace_dir=tmp_path,
                system_prompt="rules",
                memory_enabled=False,
                before_prompt_hooks=[before_hook],
                after_prompt_hooks=[after_hook],
                stream_fn=fake_stream,
            )
        )

        try:
            assert not hasattr(session, "agent")

            result, record = await _run_real_session_spine(
                session,
                "hello",
                run_id="run_v2_controller",
            )
            record = session._last_session_run_record

            assert result.run_id == "run_v2_controller"
            assert record is not None
            assert record.run_id == "run_v2_controller"
            assert record.status == "completed"
            assert record.final_text == "done"
            assert not hasattr(record, "store")
            assert not hasattr(record, "agent")
            assert hook_calls == [
                ("before", "hello", False),
                ("after", "hello", False),
            ]
            stored_runs = session.store.load_run_results(limit=1)
            assert stored_runs[-1]["run_id"] == "run_v2_controller"
            projection = session.task_recovery.load_projection()
            assert projection is not None
            assert projection["goal"] == "hello"
            assert projection["source_run_id"] == "run_v2_controller"
        finally:
            session._close()

    asyncio.run(run_case())


def test_session_controller_carries_task_control_through_core_and_recovery(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.sessions.prepare import SessionRuntime
        from codepilot.sessions.contracts import SessionOptions

        seen_system_prompts: list[str | None] = []

        async def fake_stream(_model, context, _options):
            seen_system_prompts.append(context.system_prompt)
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="done")]))
            return stream

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2-task",
                    name="Session V2 Task",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                workspace_dir=tmp_path,
                system_prompt="rules",
                memory_enabled=False,
                task_control_enabled=True,
                max_task_replans_per_run=3,
                stream_fn=fake_stream,
            )
        )

        try:
            result, _record = await _run_real_session_spine(
                session,
                "finish the v2 task spine",
                run_id="run_v2_task",
            )
            projection = session.task_recovery.load_projection()

            assert result.task is not None
            assert result.task.goal == "finish the v2 task spine"
            assert result.task.control_signal["mode"] == "build"
            assert result.task.control_signal["phase"] == "finished"
            assert seen_system_prompts
            assert seen_system_prompts[0] is not None
            assert "rules" in seen_system_prompts[0]
            assert "## Current Task" in seen_system_prompts[0]
            assert projection is not None
            assert projection["goal"] == "finish the v2 task spine"
            assert "task_progress" not in projection
            assert all(step["status"] == "completed" for step in projection["steps"])
        finally:
            session._close()

    asyncio.run(run_case())


def test_session_controller_commits_v2_outcome_into_real_session_lifecycle(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome, WorkspaceEffects
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent, ToolResultMessage
        from codepilot.sessions.contracts import SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.prepare import SessionRuntime
        from codepilot.sessions.contracts import SessionOptions

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="unused")]))
            return stream

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2",
                    name="Session V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                workspace_dir=tmp_path,
                system_prompt="rules",
                memory_enabled=False,
                task_control_enabled=False,
                stream_fn=fake_stream,
            )
        )

        try:
            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(
                SessionRunIntent(text="hello", run_id="run_commit_v2")
            )
            final = AssistantMessage(content=[TextContent(text="done")])
            tool = ToolResultMessage(
                tool_call_id="call1",
                tool_name="read",
                content=[TextContent(text="body")],
            )

            record = await controller.commit_run(
                prepared,
                AgentLoopOutcome(
                    run_id=prepared.run_id,
                    status="completed",
                    stop_reason="final_answer",
                    new_messages=[tool, final],
                    final_message=final,
                    workspace_effects=WorkspaceEffects(
                        affected_paths=("README.md",),
                        changed=False,
                    ),
                    events=[{"type": "message_end", "message": final}],
                ),
            )

            assert record.final_text == "done"
            assert session.conversation.last_run_result is not None
            assert session.conversation.last_run_result.run_id == "run_commit_v2"
            assert [type(message).__name__ for message in session.conversation.messages] == [
                "UserMessage",
                "ToolResultMessage",
                "AssistantMessage",
            ]
            stored = session.store.load_session_messages()
            assert len(stored) == 3
            assert session.store.load_run_results(limit=1)[-1]["run_id"] == "run_commit_v2"
        finally:
            session._close()

    asyncio.run(run_case())


def test_session_runtime_subscribers_receive_v2_run_events(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.sessions.contracts import SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.prepare import SessionRuntime
        from codepilot.sessions.contracts import SessionOptions

        async def fake_stream(_model, _context, _options):
            stream = AssistantMessageEventStream()
            stream.end(AssistantMessage(content=[TextContent(text="unused")]))
            return stream

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2",
                    name="Session V2",
                    api="unit-test",
                    provider="unit-test",
                    base_url="",
                    reasoning=False,
                    input=["text"],
                    context_window=4000,
                    max_tokens=500,
                ),
                workspace_dir=tmp_path,
                memory_enabled=False,
                task_control_enabled=False,
                stream_fn=fake_stream,
            )
        )
        events: list[dict] = []
        unsubscribe = session._subscribe(events.append)

        try:
            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(
                SessionRunIntent(text="hello", run_id="run_events_v2")
            )
            final = AssistantMessage(content=[TextContent(text="done")])
            await controller.commit_run(
                prepared,
                AgentLoopOutcome(
                    run_id=prepared.run_id,
                    status="completed",
                    stop_reason="final_answer",
                    new_messages=[final],
                    final_message=final,
                    events=[{"type": "message_end", "runId": prepared.run_id, "message": final}],
                ),
            )

            assert [event["type"] for event in events] == ["message_end"]
        finally:
            unsubscribe()
            session._close()

    asyncio.run(run_case())

