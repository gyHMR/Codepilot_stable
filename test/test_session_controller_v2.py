from __future__ import annotations

import asyncio


def _plan_state(
    plan_id: str,
    *,
    owner_run_id: str = "run_plan",
    status: str = "proposed",
    origin_mode: str = "plan",
    goal: str = "先制定方案",
    summary: str = "阅读当前实现并执行聚焦修改。",
    items: list[dict[str, str]] | None = None,
    explanation: str = "",
) -> dict[str, object]:
    resolved_items = items or [
        {
            "id": "item_1",
            "step": "阅读实现",
            "details": "定位相关代码和约束。",
            "verification": "确认准确的修改点。",
            "status": "pending" if status == "proposed" else "in_progress",
        }
    ]
    completed = status == "completed"
    return {
        "schema_version": 6,
        "plan_id": plan_id,
        "owner_run_id": owner_run_id,
        "status": status,
        "origin_mode": origin_mode,
        "raw_user_request": goal,
        "interpreted_goal": goal,
        "task_understanding": f"用户希望完成：{goal}",
        "current_implementation": "已确认相关实现和测试边界。",
        "target_design": "按现有结构完成聚焦修改。",
        "impact_scope": "影响当前任务相关模块和验证。",
        "risks_and_open_questions": ["暂无阻塞待确认项。"],
        "verification_plan": "运行相关验证。",
        "summary": summary,
        "completion_criteria": ["相关验证通过。"],
        "items": resolved_items,
        "revision": 1,
        "explanation": explanation,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "completed_at": "2026-01-01T00:00:00+00:00" if completed else None,
        "completion_source": "test" if completed else None,
    }


async def _run_real_session_spine(session, text: str, *, run_id: str):
    from codepilot.core.contracts import AgentLoopPorts
    from codepilot.core.runner import run_agent_loop
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
        from codepilot.sessions.runtime import SessionRuntime

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
        session.close()

    asyncio.run(run_case())


def test_session_commit_keeps_plan_approval_checkpoint(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome
        from codepilot.protocols import Model, PlanSummary
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s_plan_wait",
                current_mode="plan",
                memory_enabled=False,
            )
        )
        controller = _bind_session_runtime(session)
        prepared = await controller.prepare_run(SessionRunIntent(text="先给计划"))
        plan = PlanSummary(
            schema_version=6,
            plan_id="plan_wait",
            owner_run_id=prepared.run_id,
            status="proposed",
            origin_mode="plan",
            raw_user_request="先给计划",
            interpreted_goal="形成可执行代码修改方案",
            task_understanding="用户希望先审批方案。",
            current_implementation="已确认相关代码边界。",
            target_design="形成可执行代码修改方案。",
            impact_scope="影响当前任务相关模块。",
            risks_and_open_questions=["暂无阻塞待确认项。"],
            verification_plan="运行相关验证。",
            summary="阅读当前实现并形成执行方案。",
            completion_criteria=["确认修改方案可执行。"],
            items=[
                {
                    "id": "item_1",
                    "step": "阅读实现",
                    "details": "定位相关代码。",
                    "verification": "确认修改点。",
                    "status": "pending",
                }
            ],
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        await controller.commit_run(
            prepared,
            AgentLoopOutcome(
                run_id=prepared.run_id,
                status="waiting_user",
                stop_reason="plan_approval_required",
                plan=plan,
                events=[
                    {
                        "type": "plan_approval_required",
                        "runId": prepared.run_id,
                        "sessionId": session.session_id,
                        "plan": plan.__dict__,
                    }
                ],
            ),
        )

        checkpoint = session.store.read_meta()["runtime_checkpoint"]
        assert checkpoint["phase"] == "plan_approval"
        assert checkpoint["plan_id"] == "plan_wait"
        assert "plan" not in checkpoint
        assert session.pending_plan_approval()["plan_id"] == "plan_wait"
        session.close()

    asyncio.run(run_case())


def test_session_checkpoint_carries_cumulative_run_state_into_resume(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import (
            SessionOptions,
            SessionResumeIntent,
            SessionRunIntent,
        )
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s_run_state",
                memory_enabled=False,
            )
        )
        controller = _bind_session_runtime(session)
        prepared = await controller.prepare_run(
            SessionRunIntent(text="修改文件", run_id="run_stateful")
        )
        session.store.set_checkpoint(
            {
                "phase": "tool_approval",
                "run_id": prepared.run_id,
                "pending_tool_call_ids": ["call_edit"],
                "pending_tool_calls": [
                    {
                        "id": "call_edit",
                        "name": "edit",
                        "arguments": {"path": "app.py"},
                        "approval_id": "approval_edit",
                    }
                ],
            }
        )
        snapshot = {
            "counters": {
                "model_attempts": 2,
                "tool_iterations": 1,
                "tool_calls": 1,
            },
            "workspace_changed": False,
            "affected_paths": [],
            "verification": [],
            "verification_status": "unknown",
            "approval_required": True,
            "seen_tool_call_ids": ["call_edit"],
            "pending_approval_tool_call_ids": ["call_edit"],
        }

        await controller.commit_run(
            prepared,
            AgentLoopOutcome(
                run_id=prepared.run_id,
                status="waiting_approval",
                stop_reason="approval_required",
                run_state=snapshot,
            ),
        )

        checkpoint = session.runtime_checkpoint()
        assert checkpoint is not None
        assert checkpoint["run_state"] == snapshot
        resumed = await controller.prepare_resume(
            SessionResumeIntent(
                approval_id="approval_edit",
                decision="approve",
                run_id=prepared.run_id,
            )
        )
        assert resumed.resume_input is not None
        assert resumed.resume_input.run_state == snapshot
        assert resumed.loop_input.run_state == snapshot
        session.close()

    asyncio.run(run_case())


def test_streamed_plan_event_is_immediately_visible_to_plan_command(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionCommandIntent, SessionOptions
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s_streamed_plan",
                current_mode="plan",
                memory_enabled=False,
            )
        )
        controller = _bind_session_runtime(session)
        plan = _plan_state(
            "plan_streamed",
            owner_run_id="run_streamed",
            goal="优化登录逻辑",
            explanation="等待用户确认",
        )

        session.record_event(
            {
                "type": "plan_proposed",
                "runId": "run_streamed",
                "sessionId": session.session_id,
                "plan": plan,
            }
        )
        record = await controller.apply_command(SessionCommandIntent(text="/plan"))

        assert session.plan_state.current()["plan_id"] == "plan_streamed"
        assert record.handled
        assert any("plan_streamed" in line for line in record.output_lines)
        assert record.data["status"] == "proposed"

        session.record_event(
            {
                "type": "plan_approval_required",
                "runId": "run_streamed",
                "sessionId": session.session_id,
                "turnId": 2,
                "plan": plan,
                "reason": "proposed_plan_waiting_for_user_approval",
            }
        )
        checkpoint = session.store.read_meta()["runtime_checkpoint"]
        assert checkpoint["phase"] == "plan_approval"
        assert checkpoint["plan_id"] == "plan_streamed"
        assert "plan" not in checkpoint
        session.close()

    asyncio.run(run_case())


def test_pending_plan_feedback_forces_plan_mode(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s_plan_feedback",
                current_mode="build",
                memory_enabled=False,
            )
        )
        session.plan_state.save(
            _plan_state("plan_feedback", goal="优化登录")
        )
        session.store.set_checkpoint(
            {
                "phase": "plan_approval",
                "run_id": "run_plan",
                "plan_id": "plan_feedback",
            }
        )

        controller = _bind_session_runtime(session)
        prepared = await controller.prepare_run(SessionRunIntent(text="第二步换成先写测试"))

        assert prepared.loop_input.mode == "plan"
        assert prepared.loop_input.plan_state["plan_id"] == "plan_feedback"
        session.close()

    asyncio.run(run_case())


def test_plan_mode_replanning_uses_fresh_seed_instead_of_active_plan(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s_plan_rework",
                current_mode="build",
                memory_enabled=False,
            )
        )
        try:
            session.plan_state.save(
                _plan_state(
                    "plan_active",
                    owner_run_id="run_old",
                    status="active",
                    origin_mode="build",
                    goal="旧执行计划",
                )
            )
            session.set_current_mode("plan")
            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(SessionRunIntent(text="重新设计方案"))

            archived = session.plan_state.current()
            assert archived is not None
            assert archived["plan_id"] == "plan_active"
            assert archived["status"] == "abandoned"
            assert prepared.loop_input.mode == "plan"
            assert prepared.loop_input.plan_state is None
            assert session.context_plan_state() is None
            assert session.store.read_meta()["active_plan_id"] is None
            assert any(event["type"] == "plan_abandoned" for event in session.store.load_events())
        finally:
            session.close()

    asyncio.run(run_case())


def test_mode_hint_plan_archives_active_plan_before_context(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s_plan_hint_rework",
                current_mode="build",
                memory_enabled=False,
                system_prompt_builder=lambda mode: f"prompt:{mode}",
            )
        )
        try:
            session.plan_state.save(
                _plan_state(
                    "plan_active",
                    owner_run_id="run_old",
                    status="active",
                    origin_mode="build",
                    goal="旧执行计划",
                )
            )
            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(
                SessionRunIntent(text="重新设计方案", mode_hint="plan")
            )

            archived = session.plan_state.current()
            assert archived is not None
            assert archived["plan_id"] == "plan_active"
            assert archived["status"] == "abandoned"
            assert prepared.loop_input.mode == "plan"
            assert prepared.loop_input.context["mode"] == "plan"
            assert prepared.loop_input.context["system_prompt"] == "prompt:build"
            assert prepared.loop_input.plan_state is None
            assert session.context_plan_state_for_mode("plan") is None
            assert session.store.read_meta()["active_plan_id"] is None
            assert session.current_mode == "build"
            assert any(event["type"] == "plan_abandoned" for event in session.store.load_events())
        finally:
            session.close()

    asyncio.run(run_case())


def test_session_controller_applies_command_as_session_intent(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionOptions
        from codepilot.sessions.contracts import SessionCommandIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
            )
        )
        controller = _bind_session_runtime(session)
        record = await controller.apply_command(SessionCommandIntent(text="/status"))

        assert record.handled is True
        assert record.output_lines
        assert record.command == "/status"
        session.close()

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
        from codepilot.sessions.runtime import SessionRuntime
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
            assert result.plan is None
            assert session.plan_state.current() is None
            assert session.store.read_meta()["active_plan_id"] is None
        finally:
            session.close()

    asyncio.run(run_case())


def test_session_controller_uses_run_local_plan_seed_without_persisting_empty_plan(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.sessions.runtime import SessionRuntime
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
                stream_fn=fake_stream,
            )
        )

        try:
            result, _record = await _run_real_session_spine(
                session,
                "finish the v2 task spine",
                run_id="run_v2_task",
            )
            assert result.plan is None
            assert result.signals.verification_status == "unknown"
            assert seen_system_prompts
            assert seen_system_prompts[0] is not None
            assert "rules" in seen_system_prompts[0]
            assert "## Current Task" not in seen_system_prompts[0]
            assert session.plan_state.current() is None
            assert session.store.read_meta()["active_plan_id"] is None
        finally:
            session.close()

    asyncio.run(run_case())


def test_session_continue_reuses_existing_plan_objective(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model, UserMessage
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s1",
                messages=[UserMessage(content="帮我完善修复登录注册功能")],
                memory_enabled=False,
            )
        )
        try:
            session.plan_state.save(
                _plan_state(
                    "plan_existing",
                    owner_run_id="run_seed",
                    status="active",
                    origin_mode="build",
                    goal="帮我完善修复登录注册功能",
                )
            )

            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(
                SessionRunIntent(text="继续", run_id="run_continue")
            )

            state = prepared.loop_input.plan_state
            assert state is not None
            assert state["interpreted_goal"] == "帮我完善修复登录注册功能"
            assert state["plan_id"] == "plan_existing"
            assert prepared.loop_input.user_prompt == "继续"
        finally:
            session.close()

    asyncio.run(run_case())


def test_session_continue_ignores_completed_plan_as_archived_context(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model, UserMessage
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
                session_id="s1",
                messages=[UserMessage(content="帮我完善修复登录注册功能")],
                memory_enabled=False,
            )
        )
        try:
            session.plan_state.save(
                _plan_state(
                    "plan_closed",
                    owner_run_id="run_bad",
                    status="completed",
                    origin_mode="build",
                    goal="帮我完善修复登录注册功能",
                    explanation="已完成上轮计划",
                    items=[
                        {
                            "id": "item_1",
                            "step": "完成当前请求",
                            "details": "完成上轮任务。",
                            "verification": "上轮验证通过。",
                            "status": "completed",
                        }
                    ],
                )
            )

            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(
                SessionRunIntent(text="继续", run_id="run_continue")
            )

            state = prepared.loop_input.plan_state
            assert state is None
            assert session.context_plan_state() is None
            assert session.store.read_meta()["active_plan_id"] is None
        finally:
            session.close()

    asyncio.run(run_case())


def test_mode_build_does_not_approve_proposed_plan_without_plan_command(tmp_path) -> None:
    from codepilot.protocols import Model
    from codepilot.sessions.contracts import SessionOptions
    from codepilot.sessions.runtime import SessionRuntime

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
            session_id="s1",
            memory_enabled=False,
            current_mode="plan",
        )
    )
    try:
        session.plan_state.save(
            _plan_state(
                "plan_proposed",
                explanation="等待用户批准后执行",
            )
        )

        import pytest

        with pytest.raises(ValueError, match="/plan approve"):
            session.set_current_mode("build")

        state = session.plan_state.current()
        assert state is not None
        assert state["status"] == "proposed"
        assert not any(event["type"] == "plan_approved" for event in session.store.load_events())
    finally:
        session.close()


def test_rejected_plan_can_switch_to_build_without_context_plan(tmp_path) -> None:
    from codepilot.protocols import Model
    from codepilot.sessions.contracts import SessionOptions
    from codepilot.sessions.runtime import SessionRuntime

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
            session_id="s_rejected_to_build",
            memory_enabled=False,
            current_mode="plan",
        )
    )
    try:
        session.plan_state.save(_plan_state("plan_rejected"))

        rejected = session.reject_current_plan()
        assert rejected is not None
        assert rejected["status"] == "rejected"
        assert session.set_current_mode("build") == "build"
        assert session.context_plan_state() is None
        assert session.active_plan_state() is None
        assert session.store.read_meta()["active_plan_id"] is None
    finally:
        session.close()


def test_mode_switch_keeps_base_system_prompt_stable(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

        session = SessionRuntime(
            SessionOptions(
                model=Model(
                    id="session-v2-mode",
                    name="Session V2 Mode",
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
                current_mode="build",
                system_prompt_builder=lambda mode: f"rules for {mode}",
            )
        )
        try:
            assert session.conversation.system_prompt == "rules for build"
            assert session.set_current_mode("plan") == "plan"
            assert session.conversation.system_prompt == "rules for build"

            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(SessionRunIntent(text="只制定计划"))

            assert prepared.loop_input.mode == "plan"
            assert prepared.loop_input.context.system_prompt == "rules for build"
        finally:
            session.close()

    asyncio.run(run_case())


def test_plan_commands_approve_reject_and_clear_current_plan(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.protocols import Model
        from codepilot.sessions.contracts import SessionCommandIntent, SessionOptions
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

        def make_session(name: str, *, mode: str = "plan") -> SessionRuntime:
            return SessionRuntime(
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
                    session_id=name,
                    memory_enabled=False,
                    current_mode=mode,  # type: ignore[arg-type]
                )
            )

        def proposed_plan(plan_id: str) -> dict[str, object]:
            return _plan_state(
                plan_id,
                explanation="等待用户确认",
            )

        approve_session = make_session("approve_case")
        reject_session = make_session("reject_case")
        clear_session = make_session("clear_case", mode="build")
        try:
            approve_session.plan_state.save(proposed_plan("plan_approve"))
            approve_controller = _bind_session_runtime(approve_session)
            approve_record = await approve_controller.apply_command(
                SessionCommandIntent(text="/plan approve")
            )
            approve_state = approve_session.plan_state.current()
            assert approve_record.handled
            assert approve_record.data["continuation_kind"] == "plan_approved"
            assert approve_record.data["continuation_run_id"] == "run_plan"
            assert approve_session.current_mode == "build"
            assert approve_controller.current_mode == "build"
            assert approve_state is not None
            assert approve_state["status"] == "active"
            assert approve_session.store.read_meta()["active_plan_id"] == "plan_approve"
            assert any(event["type"] == "plan_approved" for event in approve_session.store.load_events())

            reject_session.plan_state.save(proposed_plan("plan_reject"))
            reject_controller = _bind_session_runtime(reject_session)
            reject_record = await reject_controller.apply_command(
                SessionCommandIntent(text="/plan reject")
            )
            reject_state = reject_session.plan_state.current()
            assert reject_record.handled
            assert reject_session.current_mode == "plan"
            assert reject_state is not None
            assert reject_state["status"] == "rejected"
            assert reject_session.store.read_meta()["active_plan_id"] is None
            assert any(event["type"] == "plan_rejected" for event in reject_session.store.load_events())

            clear_session.plan_state.save(
                _plan_state(
                    "plan_clear",
                    status="active",
                    origin_mode="build",
                )
            )
            clear_controller = _bind_session_runtime(clear_session)
            clear_record = await clear_controller.apply_command(
                SessionCommandIntent(text="/plan clear")
            )
            clear_state = clear_session.plan_state.current()
            assert clear_record.handled
            assert clear_session.current_mode == "build"
            assert clear_state is not None
            assert clear_state["status"] == "abandoned"
            assert clear_session.store.read_meta()["active_plan_id"] is None
            assert clear_session.active_plan_state() is None
            assert any(event["type"] == "plan_abandoned" for event in clear_session.store.load_events())
        finally:
            approve_session.close()
            reject_session.close()
            clear_session.close()

    asyncio.run(run_case())


def test_session_commit_does_not_change_plan_when_failed_outcome_has_no_plan(tmp_path) -> None:
    from codepilot.core.contracts import AgentLoopOutcome
    from codepilot.protocols import AgentRunCounters
    from codepilot.protocols import Model
    from codepilot.sessions.contracts import SessionOptions
    from codepilot.sessions.runtime import SessionRuntime

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
            session_id="s1",
            memory_enabled=False,
        )
    )
    try:
        before = _plan_state(
            "plan_unsaved",
            owner_run_id="run_start",
            status="active",
            origin_mode="build",
            goal="modify register module",
        )
        outcome = AgentLoopOutcome(
            run_id="run_failed",
            status="failed",
            stop_reason="max_iterations",
            counters=AgentRunCounters(model_attempts=1),
            plan=None,
        )

        session._finalize_plan_state(outcome)

        assert before["status"] == "active"
        assert session.plan_state.current() is None
    finally:
        session.close()


def test_session_controller_commits_v2_outcome_into_real_session_lifecycle(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome, WorkspaceEffects
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent, ToolResultMessage
        from codepilot.sessions.contracts import SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime
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
            session.close()

    asyncio.run(run_case())


def test_session_runtime_persists_v2_run_events(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome
        from codepilot.llm.stream import AssistantMessageEventStream
        from codepilot.protocols import AssistantMessage, Model, TextContent
        from codepilot.sessions.contracts import SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime
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
                stream_fn=fake_stream,
            )
        )
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

            assert any(
                event["type"] == "message_end"
                for event in session.store.load_events()
            )
        finally:
            session.close()

    asyncio.run(run_case())


def test_session_runtime_records_streamed_checkpoint_phases(tmp_path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import AgentLoopOutcome
        from codepilot.protocols import (
            AgentRunCounters,
            AssistantMessage,
            Model,
            TextContent,
            ToolCall,
            ToolResultMessage,
        )
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
        from codepilot.sessions.controller import _bind_session_runtime
        from codepilot.sessions.runtime import SessionRuntime

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
            )
        )
        try:
            controller = _bind_session_runtime(session)
            prepared = await controller.prepare_run(
                SessionRunIntent(text="inspect files", run_id="run_checkpoint")
            )
            assistant = AssistantMessage(
                content=[
                    ToolCall(id="call_read_a", name="read", arguments={"path": "a.py"}),
                    ToolCall(id="call_read_b", name="read", arguments={"path": "b.py"}),
                ]
            )
            tool_a = ToolResultMessage(
                tool_call_id="call_read_a",
                tool_name="read",
                content=[TextContent(text="print('ok')")],
            )
            tool_b = ToolResultMessage(
                tool_call_id="call_read_b",
                tool_name="read",
                content=[TextContent(text="print('done')")],
            )
            final = AssistantMessage(content=[TextContent(text="done")])

            session.record_event(
                {
                    "type": "message_end",
                    "eventId": "run_checkpoint:1",
                    "runId": "run_checkpoint",
                    "turnId": 1,
                    "message": assistant,
                }
            )
            assert session.store.read_meta()["runtime_checkpoint"]["phase"] == "tool_approval"

            session.record_event(
                {
                    "type": "message_end",
                    "eventId": "run_checkpoint:2",
                    "runId": "run_checkpoint",
                    "turnId": 1,
                    "message": tool_a,
                }
            )
            checkpoint = session.store.read_meta()["runtime_checkpoint"]
            assert checkpoint["phase"] == "tool_approval"
            assert checkpoint["pending_tool_call_ids"] == ["call_read_b"]

            session.record_event(
                {
                    "type": "message_end",
                    "eventId": "run_checkpoint:3",
                    "runId": "run_checkpoint",
                    "turnId": 1,
                    "message": tool_b,
                }
            )
            assert session.store.read_meta()["runtime_checkpoint"]["phase"] == "tools_completed"

            session.record_event(
                {
                    "type": "message_end",
                    "eventId": "run_checkpoint:4",
                    "runId": "run_checkpoint",
                    "turnId": 2,
                    "message": final,
                }
            )
            assert session.store.read_meta()["runtime_checkpoint"]["phase"] == "final_response"

            await controller.commit_run(
                prepared,
                AgentLoopOutcome(
                    run_id=prepared.run_id,
                    status="completed",
                    stop_reason="final_answer",
                    new_messages=[assistant, tool_a, tool_b, final],
                    final_message=final,
                    counters=AgentRunCounters(model_attempts=2, tool_iterations=1, tool_calls=2),
                    events=[
                        {
                            "type": "message_end",
                            "eventId": "run_checkpoint:1",
                            "runId": "run_checkpoint",
                            "message": assistant,
                        },
                        {
                            "type": "message_end",
                            "eventId": "run_checkpoint:2",
                            "runId": "run_checkpoint",
                            "message": tool_a,
                        },
                        {
                            "type": "message_end",
                            "eventId": "run_checkpoint:3",
                            "runId": "run_checkpoint",
                            "message": tool_b,
                        },
                        {
                            "type": "message_end",
                            "eventId": "run_checkpoint:4",
                            "runId": "run_checkpoint",
                            "message": final,
                        },
                    ],
                ),
            )

            assert session.store.read_meta()["runtime_checkpoint"] is None
            assert len(session.store.load_session_messages()) == 5
        finally:
            session.close()

    asyncio.run(run_case())
