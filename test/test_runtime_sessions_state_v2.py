from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from codepilot.llm.ports import LLMCompleted, LLMFailed, ModelDescriptor
from codepilot.protocols import AssistantMessage, ImageContent, Model, TextContent, ToolCall
from codepilot.runtime import SessionOpenIntent
from codepilot.runtime.actions import (
    ApprovalDecided,
    ApprovalRequiredFrame,
    CancelledFrame,
    FailedFrame,
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
        context_window=32000,
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
        assert any(isinstance(frame, RunFinishedFrame) for frame in frames), frames
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


def test_runtime_prompt_preserves_images_in_model_and_session_messages(tmp_path: Path) -> None:
    async def run_case() -> None:
        observed = None

        class ModelPort:
            async def stream(self, request):
                nonlocal observed
                observed = request.messages[-1]
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="seen")])
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
                PromptSubmitted(
                    text="inspect image",
                    images=["data:image/jpeg;base64,aW1hZ2U="],
                    request_id="request_image",
                ),
            )
        ]

        assert any(isinstance(frame, RunFinishedFrame) for frame in frames)
        assert observed is not None
        assert any(isinstance(block, ImageContent) for block in observed.content)
        coordinator = _coordinator(gateway, opened.session_id)
        stored = coordinator.state_service.load_messages(opened.session_id)[0].message
        assert any(isinstance(block, ImageContent) for block in stored.content)

    asyncio.run(run_case())


def test_prompt_retry_reuses_sessions_run_id_across_runtime_contracts(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.core.state import CoreState
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent

        coordinator = RuntimeSessionCoordinator(
            SessionOptions(
                model=_model(),
                workspace_dir=tmp_path,
                session_id="session_prompt_retry",
                memory_enabled=False,
            )
        )
        intent = SessionRunIntent(
            text="inspect retry",
            request_id="request_prompt_retry",
        )
        model = ModelDescriptor(provider="unit-test", model_id="runtime-state-v2")

        first = await coordinator._prepare_run(  # noqa: SLF001
            intent,
            run_id="run_prompt_retry_original",
            model=model,
        )
        retried = await coordinator._prepare_run(  # noqa: SLF001
            intent,
            run_id="run_prompt_retry_replacement",
            model=model,
        )

        assert first.run_id == "run_prompt_retry_original"
        assert retried.run_id == first.run_id
        assert retried.loop_input.run_id == first.run_id
        assert retried.state_port is not None
        assert retried.state_port.run.run_id == first.run_id
        assert retried.rollback_baseline is not None
        assert retried.rollback_baseline.run_id == first.run_id
        messages = coordinator.state_service.load_messages(coordinator.session_id)
        assert len(messages) == 1

    asyncio.run(run_case())


def test_provider_stream_failure_commits_failed_run_without_success_frame(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        class ModelPort:
            async def stream(self, _request):
                yield LLMFailed(
                    {
                        "code": "llm.provider_unavailable",
                        "message": "provider stream failed",
                        "retryable": False,
                    }
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
                PromptSubmitted(text="fail at provider"),
            )
        ]

        assert len([frame for frame in frames if isinstance(frame, FailedFrame)]) == 1
        assert not any(isinstance(frame, RunFinishedFrame) for frame in frames)
        coordinator = _coordinator(gateway, opened.session_id)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None and run.status == "failed"
        assert run.stop_reason == "model_error"
        assert run.result_ref is not None

    asyncio.run(run_case())


def test_terminal_commit_failure_never_returns_success_frame(tmp_path: Path) -> None:
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
        coordinator = _coordinator(gateway, opened.session_id)
        original_commit = coordinator.state_service.commit_run_boundary

        def fail_terminal_commit(request):
            if request.kind == "terminal":
                raise OSError("terminal store unavailable")
            return original_commit(request)

        coordinator.state_service.commit_run_boundary = fail_terminal_commit  # type: ignore[method-assign]

        frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="finish but fail commit"),
            )
        ]

        assert len([frame for frame in frames if isinstance(frame, FailedFrame)]) == 1
        assert not any(isinstance(frame, RunFinishedFrame) for frame in frames)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None and run.status == "running"
        assert run.checkpoint is not None
        assert run.checkpoint.resume_point == "before_finalization"

    asyncio.run(run_case())


def test_post_commit_effect_failures_cannot_override_completed_run(tmp_path: Path) -> None:
    async def run_case() -> None:
        class ModelPort:
            async def stream(self, _request):
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="done")])
                )

        async def failing_after_prompt(_context) -> None:
            raise RuntimeError("after prompt hook failed")

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
                after_prompt_hooks=(failing_after_prompt,),
            )
        )
        coordinator = _coordinator(gateway, opened.session_id)

        def failing_rollback_metadata(*_args, **_kwargs) -> None:
            raise OSError("rollback metadata unavailable")

        coordinator.state_service.write_rollback_metadata = failing_rollback_metadata  # type: ignore[method-assign]

        frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="finish despite hook failure"),
            )
        ]

        assert len([frame for frame in frames if isinstance(frame, RunFinishedFrame)]) == 1
        assert not any(isinstance(frame, FailedFrame) for frame in frames)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None and run.status == "completed"

    asyncio.run(run_case())


def test_context_projection_report_stays_internal_to_context(tmp_path: Path) -> None:
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
        assert not any(event.get("type") == "context_projected" for event in after_terminal)
        assert coordinator.context_service.latest_report

    asyncio.run(run_case())


def test_opening_session_restores_active_plan_and_context_checkpoint(tmp_path: Path) -> None:
    from codepilot.core.plan import load_plan_state
    from codepilot.core.state import CoreState, TaskState
    from codepilot.protocols import UserMessage
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
    from codepilot.sessions.contracts import ComponentCheckpoint, SessionOptions, WaitingState
    from codepilot.sessions.service import BeginRunRequest, CommitRunBoundaryRequest
    from codepilot.sessions.workspace import capture_workspace_checkpoint

    options = SessionOptions(
        model=_model(),
        workspace_dir=tmp_path,
        session_id="session_restore_components",
        memory_enabled=False,
    )
    original = RuntimeSessionCoordinator(options)
    plan = {
        "schema_version": 7,
        "plan_id": "plan_restore",
        "origin": "plan_mode",
        "status": "proposed",
        "revision": 1,
        "definition": {
            "summary": "执行聚焦修改。",
            "completion_criteria": ["测试通过"],
            "task_understanding": "先审批再执行。",
            "current_implementation": "已完成代码调查。",
            "target_design": "保持接口并实施修改。",
            "impact_scope": "相关实现和测试。",
            "risks_and_open_questions": ["无"],
            "verification_plan": "运行测试。",
            "explanation": "",
        },
        "steps": [
            {
                "step_id": "item_1",
                "step": "实施修改",
                "details": "修改目标代码。",
                "verification": "运行测试。",
                "status": "pending",
                "completion_note": "",
            }
        ],
        "pending_revision": None,
        "close_request": None,
    }
    migrated_plan = load_plan_state(plan)
    assert migrated_plan is not None
    compact_path = Path(
        ".codepilot/runs/run_restore/artifacts/context/compact_restore.json"
    )
    compact_target = tmp_path / compact_path
    compact_target.parent.mkdir(parents=True, exist_ok=True)
    compact_target.write_text(
        json.dumps(
            {
                "snapshot": {
                    "compact_id": "compact_restore",
                    "path": compact_path.as_posix(),
                    "compacted_until_message_id": "msg_10",
                    "source_digest": "sha256:test",
                    "estimated_tokens_before": 100,
                    "estimated_tokens_after": 20,
                },
                "summary": {
                    "original_goal": "执行修改",
                    "user_constraints": [],
                    "decisions": [],
                    "completed_work": [],
                    "files_and_symbols": [],
                    "important_evidence": ["saved summary"],
                    "errors_and_resolutions": [],
                    "verification_state": "",
                    "open_questions": [],
                    "next_actions": [],
                    "source_refs": ["message:msg_10"],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    begun = original.state_service.begin_run(
        BeginRunRequest(
            session_id=original.session_id,
            run_id="run_restore",
            user_message=UserMessage(content="制定方案"),
            workspace=capture_workspace_checkpoint(tmp_path),
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
            workspace=begun.run.checkpoint.workspace,
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
            core_state=CoreState(
                task=TaskState(
                    original_request="制定方案",
                    current_goal="执行修改",
                    plan=migrated_plan,
                )
            ).to_dict(),
            waiting=WaitingState(
                kind="plan_confirmation",
                request_id="plan_restore",
            ),
            components=(
                ComponentCheckpoint(
                    owner="context",
                    schema_version=1,
                    state={
                        "compact_snapshot_ref": compact_path.as_posix(),
                        "compacted_until_message_id": "msg_10",
                    },
                ),
            ),
            workspace=started.run.checkpoint.workspace,
        )
    )
    original.close()

    reopened = RuntimeSessionCoordinator(options)

    assert reopened.current_plan_state() == migrated_plan.to_dict()
    context_checkpoint = reopened.context_service.checkpoint_state()
    assert context_checkpoint["compacted_until_message_id"] == "msg_10"
    snapshot_ref = context_checkpoint["compact_snapshot_ref"]
    assert isinstance(snapshot_ref, str)
    assert (tmp_path / snapshot_ref).is_file()


def test_reopened_progress_checkpoint_continues_same_run(tmp_path: Path) -> None:
    async def run_case() -> None:
        from codepilot.core.state import CoreState
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionOptions
        from codepilot.sessions.service import BeginRunRequest, CommitRunBoundaryRequest
        from codepilot.sessions.workspace import capture_workspace_checkpoint
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
                initial_core_state=CoreState.new("same request").to_dict(),
                workspace=capture_workspace_checkpoint(tmp_path),
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
                core_state=CoreState.new("same request").to_dict(),
                workspace=capture_workspace_checkpoint(tmp_path),
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

        assert any(isinstance(frame, RunFinishedFrame) for frame in frames), frames
        finished = next(frame for frame in frames if isinstance(frame, RunFinishedFrame))
        coordinator = _coordinator(gateway, opened.session_id)
        run = coordinator.state_service.get_run(finished.record.run_id)
        messages = coordinator.state_service.load_messages(opened.session_id)
        assert finished.record.run_id == "run_crash_resume"
        assert run is not None and run.status == "completed"
        assert [record.message.role for record in messages] == ["user", "assistant"]

    asyncio.run(run_case())


def test_reopened_after_model_checkpoint_finishes_without_repeating_model_call(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.protocols import UserMessage
        from codepilot.core.state import CoreState
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionOptions
        from codepilot.sessions.service import BeginRunRequest, CommitRunBoundaryRequest
        from codepilot.sessions.workspace import capture_workspace_checkpoint

        options = SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id="session_after_model_resume",
            memory_enabled=False,
        )
        original = RuntimeSessionCoordinator(options)
        workspace = capture_workspace_checkpoint(tmp_path)
        begun = original.state_service.begin_run(
            BeginRunRequest(
                session_id=original.session_id,
                request_id="request_after_model",
                run_id="run_after_model_resume",
                user_message=UserMessage(content="same request"),
                initial_core_state=CoreState.new("same request").to_dict(),
                workspace=workspace,
            ),
            expected_session_revision=original.session_state.revision,
        )
        original.state_service.commit_run_boundary(
            CommitRunBoundaryRequest(
                commit_id="after-model-progress",
                kind="progress",
                session_id=original.session_id,
                run_id=begun.run.run_id,
                expected_run_revision=begun.run.revision,
                expected_session_revision=begun.session.revision,
                phase="model",
                resume_point="after_model",
                core_state=CoreState.new("same request").to_dict(),
                new_messages=(
                    AssistantMessage(content=[TextContent(text="already complete")]),
                ),
                workspace=workspace,
            )
        )
        original.close()

        class ModelPort:
            calls = 0

            async def stream(self, _request):
                self.calls += 1
                yield LLMCompleted(
                    message=AssistantMessage(content=[TextContent(text="repeated")])
                )

        model = ModelPort()
        gateway = RuntimeGateway(model_port=model)
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
        assert finished.record.run_id == "run_after_model_resume"
        assert finished.record.final_text == "already complete"
        assert model.calls == 0

    asyncio.run(run_case())


def test_reopened_progress_run_restores_durable_rollback_baseline(tmp_path: Path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import CoreBoundary
        from codepilot.core.state import CoreCounters, CoreState, RunFacts
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionContinuationIntent, SessionOptions, SessionRunIntent

        options = SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id="session_rollback_resume",
            memory_enabled=False,
        )
        original = RuntimeSessionCoordinator(options)
        prepared = await original._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="inspect", request_id="request_rollback"),
            run_id="run_rollback_resume",
            model=ModelDescriptor(provider="unit-test", model_id="runtime-state-v2"),
        )
        assert prepared.state_port is not None
        await prepared.state_port.commit(
            CoreBoundary(
                kind="after_model",
                state=CoreState(
                    task=CoreState.new("inspect").task,
                    facts=RunFacts(counters=CoreCounters(model_turns=1)),
                ),
                new_messages=(
                    AssistantMessage(content=[TextContent(text="already complete")]),
                ),
            )
        )
        original.close()

        reopened = RuntimeSessionCoordinator(options)
        resumed = await reopened._prepare_continuation(  # noqa: SLF001
            SessionContinuationIntent(
                kind="automatic_continuation",
                run_id="run_rollback_resume",
            ),
            run_id="run_rollback_resume",
            model=ModelDescriptor(provider="unit-test", model_id="runtime-state-v2"),
        )

        assert resumed.rollback_baseline is not None
        baseline = reopened._rollback_baseline(resumed.rollback_baseline)  # noqa: SLF001
        assert baseline.reason != "missing_rollback_baseline"

    asyncio.run(run_case())


def test_after_tools_checkpoint_clears_tool_intent_but_keeps_rollback_baseline(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import CoreBoundary
        from codepilot.core.state import CoreState
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent

        coordinator = RuntimeSessionCoordinator(
            SessionOptions(
                model=_model(),
                workspace_dir=tmp_path,
                session_id="session_component_merge",
                memory_enabled=False,
            )
        )
        prepared = await coordinator._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="inspect", request_id="request_components"),
            run_id="run_component_merge",
            model=ModelDescriptor(provider="unit-test", model_id="runtime-state-v2"),
        )
        adapter = prepared.state_port
        assert adapter is not None
        tool_state = {"intent": {"tool_calls": [{"id": "call_1"}]}}
        adapter.bind_tool_state(lambda: tool_state)
        state = CoreState.new("inspect")
        await adapter.commit(
            CoreBoundary(
                kind="before_tools",
                state=state,
            )
        )
        tool_state = None
        await adapter.commit(CoreBoundary(kind="after_tools", state=state))

        run = coordinator.state_service.get_run("run_component_merge")
        assert run is not None and run.checkpoint is not None
        assert {component.owner for component in run.checkpoint.components} == {
            "context",
            "rollback",
        }

    asyncio.run(run_case())


def test_target_boundary_collects_component_checkpoints_in_runtime(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import CoreBoundary
        from codepilot.core.events import CoreDomainEvent
        from codepilot.core.state import CoreState, RunFacts, WorkspaceFacts
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import (
            SessionOptions,
            SessionRunIntent,
            WorkspaceCheckpoint,
        )

        coordinator = RuntimeSessionCoordinator(
            SessionOptions(
                model=_model(),
                workspace_dir=tmp_path,
                session_id="session_target_boundary",
                memory_enabled=False,
            )
        )
        prepared = await coordinator._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="inspect", request_id="request_target_boundary"),
            run_id="run_target_boundary",
            model=ModelDescriptor(provider="unit-test", model_id="runtime-state-v2"),
        )
        adapter = prepared.state_port
        assert adapter is not None
        workspace_inputs = []
        adapter.bind_tool_state(lambda: {"pending_attempt": "attempt_1"})
        adapter.context_state = lambda: {"context_cursor": "message_1"}
        adapter.workspace_state = lambda core_state: (
            workspace_inputs.append(core_state)
            or WorkspaceCheckpoint(root=str(tmp_path))
        )
        state = CoreState(
            task=CoreState.new("inspect").task,
            facts=RunFacts(
                workspace=WorkspaceFacts(
                    revision=1,
                    changed=True,
                    affected_paths=("src/app.py",),
                )
            ),
        )

        await adapter.commit(
            CoreBoundary(
                kind="before_tools",
                state=state,
                domain_events=(
                    CoreDomainEvent("workspace_changed", {"path": "src/app.py"}),
                ),
            )
        )

        run = coordinator.state_service.get_run("run_target_boundary")
        assert run is not None and run.checkpoint is not None
        assert run.core_state == state.to_dict()
        assert {component.owner for component in run.checkpoint.components} == {
            "tools",
            "context",
            "rollback",
        }
        assert workspace_inputs == [state.to_dict()]
        events = coordinator.state_service.load_events(
            "session_target_boundary",
            run_id="run_target_boundary",
        )
        domain_event = next(
            event for event in events if event["type"] == "workspace_changed"
        )
        assert domain_event["path"] == "src/app.py"
        assert domain_event["run_id"] == "run_target_boundary"
        assert domain_event["session_id"] == "session_target_boundary"

    asyncio.run(run_case())


def test_target_waiting_boundary_maps_to_sessions_waiting_state(tmp_path: Path) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import CoreBoundary, CoreReason, CoreWait
        from codepilot.core.state import CoreState
        from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator
        from codepilot.sessions.contracts import SessionOptions, SessionRunIntent

        coordinator = RuntimeSessionCoordinator(
            SessionOptions(
                model=_model(),
                workspace_dir=tmp_path,
                session_id="session_target_waiting",
                memory_enabled=False,
            )
        )
        prepared = await coordinator._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="inspect", request_id="request_target_waiting"),
            run_id="run_target_waiting",
            model=ModelDescriptor(provider="unit-test", model_id="runtime-state-v2"),
        )
        adapter = prepared.state_port
        assert adapter is not None
        wait = CoreWait(
            "user_input",
            "question_1",
            CoreReason("task.user_input_required", recoverable=True),
            {"question": "Which file?"},
        )
        state = CoreState.new("inspect")

        await adapter.commit(CoreBoundary(kind="after_model", state=state))
        await adapter.commit(
            CoreBoundary(
                kind="waiting",
                state=state,
                wait=wait,
            )
        )

        run = coordinator.state_service.get_run("run_target_waiting")
        assert run is not None and run.status == "waiting"
        assert run.phase == "model"
        assert run.checkpoint is not None
        assert run.checkpoint.resume_point == "after_model"
        assert run.checkpoint.waiting is not None
        assert run.checkpoint.waiting.kind == "user_input"
        assert run.checkpoint.waiting.request_id == "question_1"

    asyncio.run(run_case())


def test_schema_less_core_payload_is_not_recoverable() -> None:
    from codepilot.core.errors import CoreContractError
    from codepilot.core.state import load_core_state

    with pytest.raises(CoreContractError, match="Unsupported CoreState schema"):
        load_core_state({"counters": {"model_attempts": 1}})


def test_tool_approval_resumes_same_v2_run_without_repeating_effect(
    tmp_path: Path,
    monkeypatch,
) -> None:
    async def run_case() -> None:
        from codepilot.core.contracts import ModelEntry, ToolResultEntry
        from codepilot.runtime import executor as executor_module

        original_run_core = __import__(
            "codepilot.core.driver", fromlist=["run_core"]
        ).run_core
        entries = []

        async def recording_run_core(input_value, ports):
            entries.append(input_value.entry)
            return await original_run_core(input_value, ports)

        monkeypatch.setattr(
            executor_module,
            "run_core",
            recording_run_core,
            raising=False,
        )

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
        assert [type(entry) for entry in entries] == [ModelEntry, ToolResultEntry]

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


def test_dispatch_consumer_cancellation_still_commits_cancelled_terminal_state(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        started = asyncio.Event()
        stopped = asyncio.Event()

        class ModelPort:
            async def stream(self, _request):
                started.set()
                try:
                    await asyncio.Event().wait()
                    if False:  # pragma: no cover
                        yield LLMCompleted(message=AssistantMessage(content=[]))
                finally:
                    stopped.set()

        gateway = RuntimeGateway(model_port=ModelPort())
        opened = gateway.open_session(
            SessionOpenIntent(
                workspace_dir=tmp_path,
                model=_model(),
                memory_enabled=False,
            )
        )
        prompt_task = asyncio.create_task(
            _collect_frames(gateway, opened.session_id, PromptSubmitted(text="cancel stream"))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        coordinator = _coordinator(gateway, opened.session_id)
        original_commit = coordinator.state_service.commit_run_boundary

        def commit(request):
            if request.kind == "terminal":
                assert stopped.is_set()
            return original_commit(request)

        coordinator.state_service.commit_run_boundary = commit  # type: ignore[method-assign]
        prompt_task.cancel()
        result = await asyncio.gather(prompt_task, return_exceptions=True)

        assert isinstance(result[0], asyncio.CancelledError)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None and run.status == "cancelled"

    asyncio.run(run_case())


def test_dispatch_consumer_cancellation_does_not_interrupt_terminal_commit(
    tmp_path: Path,
) -> None:
    async def run_case() -> None:
        commit_started = asyncio.Event()
        allow_commit = asyncio.Event()

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
        session = gateway._sessions.require(opened.session_id)  # noqa: SLF001
        original_commit = session.controller.runs.commit

        async def delayed_commit(prepared, outcome, *, events=()):
            commit_started.set()
            await allow_commit.wait()
            return await original_commit(prepared, outcome, events=events)

        session.controller.runs.commit = delayed_commit  # type: ignore[method-assign]
        prompt_task = asyncio.create_task(
            _collect_frames(gateway, opened.session_id, PromptSubmitted(text="finish"))
        )
        await asyncio.wait_for(commit_started.wait(), timeout=5)

        late_cancel = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                RunCancelled(reason="too_late"),
            )
        ]
        assert isinstance(late_cancel[0], CancelledFrame)
        assert late_cancel[0].cancelled is False

        prompt_task.cancel()
        await asyncio.sleep(0)
        assert gateway._active_runs.is_running(opened.session_id)  # noqa: SLF001
        allow_commit.set()
        result = await asyncio.gather(prompt_task, return_exceptions=True)

        assert isinstance(result[0], asyncio.CancelledError)
        coordinator = _coordinator(gateway, opened.session_id)
        run_id = coordinator.session_state.last_run_id
        run = coordinator.state_service.get_run(run_id) if run_id else None
        assert run is not None and run.status == "completed"
        assert not gateway._active_runs.is_running(opened.session_id)  # noqa: SLF001

    asyncio.run(run_case())


def test_active_run_remains_registered_until_terminal_commit(tmp_path: Path) -> None:
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
        coordinator = _coordinator(gateway, opened.session_id)
        original_commit = coordinator.state_service.commit_run_boundary

        def commit(request):
            if request.kind == "terminal":
                assert gateway._active_runs.is_running(opened.session_id)  # noqa: SLF001
            return original_commit(request)

        coordinator.state_service.commit_run_boundary = commit  # type: ignore[method-assign]
        frames = [
            frame
            async for frame in gateway.dispatch(
                opened.session_id,
                PromptSubmitted(text="finish"),
            )
        ]

        assert any(isinstance(frame, RunFinishedFrame) for frame in frames)
        assert not gateway._active_runs.is_running(opened.session_id)  # noqa: SLF001

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
        result = coordinator.conversation.last_run_result
        assert result is not None and result.error is not None
        assert result.error.code == "runtime.deadline_exceeded"

    asyncio.run(run_case())


async def _collect_frames(gateway: RuntimeGateway, session_id: str, action) -> list[object]:
    return [frame async for frame in gateway.dispatch(session_id, action)]
