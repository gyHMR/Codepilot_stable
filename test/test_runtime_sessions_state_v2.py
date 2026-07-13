from __future__ import annotations

import asyncio
from pathlib import Path

from codepilot.llm.ports import LLMCompleted
from codepilot.protocols import AssistantMessage, Model, TextContent, ToolCall
from codepilot.runtime import SessionOpenIntent
from codepilot.runtime.actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    CancelledFrame,
    PromptSubmitted,
    RunCancelled,
    RunFinishedFrame,
    RunPausedFrame,
)
from codepilot.runtime.gateway import RuntimeGateway


def _model() -> Model:
    return Model(
        id="runtime-state-v2",
        name="Runtime State V2",
        api="unit-test",
        provider="unit-test",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def _coordinator(gateway: RuntimeGateway, session_id: str):
    controller = gateway._require_session(session_id)  # noqa: SLF001
    coordinator = getattr(controller, "_session", None)
    assert coordinator is not None
    return coordinator


def test_runtime_prompt_commits_terminal_state_through_sessions_v2(tmp_path: Path) -> None:
    async def run_case() -> None:
        class ModelPort:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
            )
        )
        frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="inspect"),
            )
        ]
        finished = next(frame for frame in frames if isinstance(frame, RunFinishedFrame))
        coordinator = _coordinator(gateway, opened.session_id)
        session = coordinator.state_service.get_session(opened.session_id)
        run = coordinator.state_service.get_run(finished.record.run_id)
        messages = coordinator.state_service.load_messages(opened.session_id)
        events = coordinator.state_service.load_events(opened.session_id)

        assert session is not None and session.current_run_id is None
        assert run is not None and run.status == "completed"
        assert run.checkpoint is None
        assert [record.message.role for record in messages] == ["user", "assistant"]
        assert events
        assert all("event_id" in event for event in events)
        event_ids = [str(event["event_id"]) for event in events]
        assert len(event_ids) == len(set(event_ids))
        assert all(
            not {"eventId", "sessionId", "runId"}.intersection(event)
            for event in events
        )
        assert (tmp_path / ".codepilot" / "sessions" / opened.session_id / "session.json").is_file()
        assert (tmp_path / ".codepilot" / "runs" / run.run_id / "run.json").is_file()

    asyncio.run(run_case())


def test_context_projection_event_waits_for_run_boundary(tmp_path: Path) -> None:
    async def run_case() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class ModelPort:
            async def stream(self, _request):
                started.set()
                await release.wait()
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
            )
        )
        run_task = asyncio.create_task(
            _collect_frames(gateway, opened.session_id, PromptSubmitted(text="inspect"))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        coordinator = _coordinator(gateway, opened.session_id)
        before_terminal = coordinator.state_service.load_events(opened.session_id)
        assert not any(event.get("type") == "context_projected" for event in before_terminal)

        release.set()
        await run_task
        after_terminal = coordinator.state_service.load_events(opened.session_id)
        assert any(event.get("type") == "context_projected" for event in after_terminal)

    asyncio.run(run_case())


def test_opening_session_restores_active_plan_and_context_checkpoint(tmp_path: Path) -> None:
    from codepilot.protocols import UserMessage
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
    from codepilot.sessions.contracts import ComponentCheckpoint, SessionOptions, WaitingState
    from codepilot.sessions.service import BeginRunRequest, CommitRunBoundaryRequest

    options = SessionOptions(
        model=_model(),
        workspace_dir=tmp_path,
        session_id="session_restore_components",
        memory_enabled=False,
    )
    original = RuntimeSessionCoordinator(options)
    plan = {
        "schema_version": 6,
        "plan_id": "plan_restore",
        "owner_run_id": "run_restore",
        "status": "proposed",
        "origin_mode": "plan",
        "raw_user_request": "制定方案",
        "interpreted_goal": "执行修改",
        "task_understanding": "先审批再执行。",
        "current_implementation": "已完成代码调查。",
        "target_design": "保持接口并实施修改。",
        "impact_scope": "相关实现和测试。",
        "risks_and_open_questions": ["无"],
        "verification_plan": "运行测试。",
        "summary": "执行聚焦修改。",
        "completion_criteria": ["测试通过"],
        "items": [
            {
                "id": "item_1",
                "step": "实施修改",
                "details": "修改目标代码。",
                "verification": "运行测试。",
                "status": "pending",
            }
        ],
        "revision": 1,
        "explanation": "",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "completed_at": None,
        "completion_source": None,
    }
    begun = original.state_service.begin_run(
        BeginRunRequest(
            session_id=original.session_id,
            run_id="run_restore",
            user_message=UserMessage(content="制定方案"),
        ),
        expected_session_revision=original.session_state.revision,
    )
    started = original.state_service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id="restore-start",
            kind="progress",
            session_id=original.session_id,
            run_id="run_restore",
            expected_run_revision=begun.run.revision,
            expected_session_revision=begun.session.revision,
            phase="model",
            resume_point="before_model",
            core_state={},
        )
    )
    original.state_service.commit_run_boundary(
        CommitRunBoundaryRequest(
            commit_id="restore-waiting",
            kind="waiting",
            session_id=original.session_id,
            run_id="run_restore",
            expected_run_revision=started.run.revision,
            expected_session_revision=started.session.revision,
            phase="model",
            resume_point="after_model",
            core_state={"plan_state": plan},
            waiting=WaitingState(
                kind="plan_confirmation",
                request_id="plan_restore",
            ),
            components=(
                ComponentCheckpoint(
                    owner="context",
                    schema_version=1,
                    state={
                        "compacted_until_message_id": "msg_10",
                        "compact_summary": "saved summary",
                    },
                ),
            ),
        )
    )
    original.close()

    reopened = RuntimeSessionCoordinator(options)

    assert reopened.plan_state.current() == plan
    assert reopened.context_governor.checkpoint_state() == {
        "compacted_until_message_id": "msg_10",
        "compact_summary": "saved summary",
    }


def test_reopened_progress_checkpoint_continues_same_run(tmp_path: Path) -> None:
    async def run_case() -> None:
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionOptions
        from codepilot.sessions.service import BeginRunRequest, CommitRunBoundaryRequest
        from codepilot.protocols import UserMessage

        options = SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id="session_crash_resume",
            memory_enabled=False,
        )
        original = RuntimeSessionCoordinator(options)
        begun = original.state_service.begin_run(
            BeginRunRequest(
                session_id=original.session_id,
                run_id="run_crash_resume",
                user_message=UserMessage(content="same request"),
            ),
            expected_session_revision=original.session_state.revision,
        )
        original.state_service.commit_run_boundary(
            CommitRunBoundaryRequest(
                commit_id="crash-progress",
                kind="progress",
                session_id=original.session_id,
                run_id=begun.run.run_id,
                expected_run_revision=begun.run.revision,
                expected_session_revision=begun.session.revision,
                phase="model",
                resume_point="before_model",
                core_state={},
            )
        )
        original.close()

        class ModelPort:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="resumed")])
                )

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                session_id=options.session_id,
                model=_model(),
                memory_enabled=False,
            )
        )
        frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="same request"),
            )
        ]

        finished = next(frame for frame in frames if isinstance(frame, RunFinishedFrame))
        coordinator = _coordinator(gateway, opened.session_id)
        run = coordinator.state_service.get_run(finished.record.run_id)
        messages = coordinator.state_service.load_messages(opened.session_id)
        assert finished.record.run_id == "run_crash_resume"
        assert run is not None and run.status == "completed"
        assert [record.message.role for record in messages] == ["user", "assistant"]

    asyncio.run(run_case())


def test_tool_approval_resumes_same_v2_run_without_repeating_effect(tmp_path: Path) -> None:
    async def run_case() -> None:
        class ModelPort:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, _request):
                self.calls += 1
                if self.calls == 1:
                    yield LLMCompleted(
                        message=AssistantMessage(
                            content=[
                                ToolCall(
                                    id="call_write",
                                    name="bash",
                                    arguments={"command": "echo ok > approval.txt"},
                                )
                            ],
                            stop_reason="toolUse",
                        )
                    )
                    return
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="written")])
                )

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
                tool_permission_mode="ask",
            )
        )
        paused_frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="write approval file"),
            )
        ]
        approval = next(
            frame.approval
            for frame in paused_frames
            if isinstance(frame, ApprovalRequiredFrame)
        )
        paused = next(frame for frame in paused_frames if isinstance(frame, RunPausedFrame))
        coordinator = _coordinator(gateway, opened.session_id)
        waiting = coordinator.state_service.get_run(paused.record.run_id)

        assert waiting is not None and waiting.status == "waiting"
        assert waiting.checkpoint is not None
        assert waiting.checkpoint.waiting is not None
        assert waiting.checkpoint.waiting.request_id == approval.approval_id

        resumed_frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                ApprovalDecided(
                    approval_id=approval.approval_id,
                    decision="approve",
                ),
            )
        ]
        finished = next(
            frame for frame in resumed_frames if isinstance(frame, RunFinishedFrame)
        )
        run = coordinator.state_service.get_run(paused.record.run_id)

        assert finished.record.run_id == paused.record.run_id
        assert run is not None and run.status == "completed"
        assert run.resume_count == 1
        assert (tmp_path / "approval.txt").read_text(encoding="utf-8").strip() == "ok"

    asyncio.run(run_case())


def test_user_cancel_uses_terminal_cancelled_commit(tmp_path: Path) -> None:
    async def run_case() -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        class ModelPort:
            async def stream(self, _request):
                started.set()
                await release.wait()
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="too late")])
                )

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
            )
        )
        prompt_task = asyncio.create_task(
            _collect_frames(gateway, opened.session_id, PromptSubmitted(text="cancel me"))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        cancel_frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                RunCancelled(reason="user_cancelled"),
            )
        ]
        frames = await prompt_task
        release.set()

        assert any(isinstance(frame, CancelledFrame) for frame in cancel_frames)
        assert any(isinstance(frame, CancelledFrame) for frame in frames)
        coordinator = _coordinator(gateway, opened.session_id)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None and run.status == "cancelled"

    asyncio.run(run_case())


def test_run_deadline_commits_failed_timeout(tmp_path: Path) -> None:
    async def run_case() -> None:
        started = asyncio.Event()

        class ModelPort:
            async def stream(self, _request):
                started.set()
                await asyncio.Event().wait()
                if False:  # pragma: no cover
                    yield LLMCompleted(message=AssistantMessage(content=[]))

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
                run_timeout_seconds=1,
            )
        )
        frames = await _collect_frames(
            gateway,
            opened.session_id,
            PromptSubmitted(text="deadline"),
        )
        assert any(getattr(frame, "kind", None) == "failed" for frame in frames)
        await asyncio.wait_for(started.wait(), timeout=0.1)
        coordinator = _coordinator(gateway, opened.session_id)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None
        assert run.status == "failed"
        assert run.stop_reason == "deadline_exceeded"

    asyncio.run(run_case())


async def _collect_frames(gateway: RuntimeGateway, session_id: str, action) -> list[object]:
    return [frame async for frame in gateway.dispatch(session_id, action)]
