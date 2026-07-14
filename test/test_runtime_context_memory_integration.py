from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from codepilot.core.contracts import CoreBoundary, CoreOutcome, CoreReason
from codepilot.llm.ports import (
    LLMCompleted,
    LLMCorrelation,
    LLMFailed,
    LLMRequest,
    LLMStarted,
    LLMTextDelta,
    ModelDescriptor,
)
from codepilot.protocols import AssistantMessage, Model, TextContent, UserMessage
from codepilot.runtime.coordinator import RunCoordinator
from codepilot.runtime.model import RetryingModelPort
from codepilot.sessions.contracts import SessionOptions, SessionRunIntent
from codepilot.sessions.memory import (
    MemoryProposal,
    MemoryProposalReceipt,
)


def _model() -> Model:
    return Model(
        id="unit",
        name="Unit",
        api="unit",
        provider="unit",
        base_url="",
        reasoning=False,
        input=["text"],
        context_window=4000,
        max_tokens=500,
    )


def test_finalization_model_sidecar_is_captured_and_removed_from_visible_answer() -> None:
    sidecar = json.dumps(
        {
            "proposals": [
                {
                    "scope": "project",
                    "type": "project",
                    "key": "project.verification.command",
                    "content": "Use focused pytest commands.",
                }
            ]
        }
    )
    text = (
        "Implementation complete.\n"
        f"<codepilot-memory-proposals>{sidecar}</codepilot-memory-proposals>"
    )

    class _BaseModel:
        async def stream(self, _request):
            yield LLMStarted()
            yield LLMTextDelta(text)
            yield LLMCompleted(
                AssistantMessage(content=[TextContent(text=text)])
            )

    captured: list[tuple[str, tuple[MemoryProposal, ...], str | None]] = []
    port = RetryingModelPort(
        _BaseModel(),
        finalization_sink=lambda run_id, proposals, error: captured.append(
            (run_id, proposals, error)
        ),
    )
    request = LLMRequest(
        model=ModelDescriptor(provider="unit", model_id="unit"),
        messages=(UserMessage(content="finish"),),
        correlation=LLMCorrelation(
            run_id="run_1",
            session_id="session_1",
            purpose="finalization",
        ),
    )

    async def run_case():
        return [event async for event in port.stream(request)]

    events = asyncio.run(run_case())
    completed = next(event for event in events if isinstance(event, LLMCompleted))
    visible = "".join(
        block.text
        for block in completed.message.content
        if isinstance(block, TextContent)
    )

    assert visible == "Implementation complete."
    assert all(
        "codepilot-memory-proposals" not in event.text
        for event in events
        if isinstance(event, LLMTextDelta)
    )
    assert captured == [
        (
            "run_1",
            (
                MemoryProposal(
                    scope="project",
                    type="project",
                    key="project.verification.command",
                    content="Use focused pytest commands.",
                ),
            ),
            None,
        )
    ]


def test_retrying_model_discards_failed_attempt_deltas_and_reports_attempts() -> None:
    class _BaseModel:
        def __init__(self) -> None:
            self.attempt = 0

        async def stream(self, _request):
            self.attempt += 1
            yield LLMStarted()
            if self.attempt == 1:
                yield LLMTextDelta("discarded partial answer")
                yield LLMFailed({"code": "llm.timeout", "retryable": True})
                return
            yield LLMTextDelta("visible final answer")
            yield LLMCompleted(
                AssistantMessage(content=[TextContent(text="visible final answer")])
            )

    port = RetryingModelPort(_BaseModel(), max_retries=1)
    request = LLMRequest(
        model=ModelDescriptor(provider="unit", model_id="unit"),
        messages=(UserMessage(content="finish"),),
        correlation=LLMCorrelation(
            run_id="run_retry",
            session_id="session_retry",
            purpose="reasoning",
        ),
    )

    async def run_case():
        return [event async for event in port.stream(request)]

    events = asyncio.run(run_case())

    assert [
        event.text for event in events if isinstance(event, LLMTextDelta)
    ] == ["visible final answer"]
    completed = next(event for event in events if isinstance(event, LLMCompleted))
    assert completed.attempts == 2


def test_terminal_commit_happens_before_automatic_memory_submission(tmp_path) -> None:
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator

    coordinator = RuntimeSessionCoordinator(
        SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id="session_terminal_memory",
            memory_enabled=True,
        )
    )
    observed_statuses: list[str] = []

    class _Memory:
        def admit_user_prompt(self, *_args, **_kwargs):
            return SimpleNamespace(records=())

        def submit_proposals(self, batch):
            run = coordinator.state_service.get_run(batch.run_id)
            assert run is not None
            observed_statuses.append(run.status)
            return MemoryProposalReceipt()

    coordinator.memory_service = _Memory()  # type: ignore[assignment]

    async def run_case():
        prepared = await coordinator._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="finish the task", request_id="request_1"),
            run_id="run_terminal_memory",
            model=ModelDescriptor(provider="unit", model_id="unit"),
        )
        assert prepared.state_port is not None
        await prepared.state_port.commit(
            CoreBoundary(kind="before_model", state=prepared.loop_input.state)
        )
        proposal = MemoryProposal(
            scope="project",
            type="project",
            key="project.verification.command",
            content="Use focused pytest commands.",
        )
        coordinator.capture_memory_proposals(
            "run_terminal_memory",
            (proposal,),
            None,
        )
        final = AssistantMessage(content=[TextContent(text="done")])
        outcome = CoreOutcome(
            status="completed",
            reason=CoreReason("task.completed"),
            state=prepared.loop_input.state,
            new_messages=(final,),
            final_message=final,
        )
        return await RunCoordinator(coordinator).commit(prepared, outcome)

    try:
        record = asyncio.run(run_case())
    finally:
        coordinator.close()

    assert record.status == "completed"
    assert observed_statuses == ["completed"]


def test_memory_submission_failure_does_not_rollback_completed_run(tmp_path) -> None:
    from codepilot.runtime.session_coordinator import RuntimeSessionCoordinator

    coordinator = RuntimeSessionCoordinator(
        SessionOptions(
            model=_model(),
            workspace_dir=tmp_path,
            session_id="session_memory_failure",
            memory_enabled=True,
        )
    )

    class _FailingMemory:
        def admit_user_prompt(self, *_args, **_kwargs):
            return SimpleNamespace(records=())

        def submit_proposals(self, _batch):
            raise OSError("memory write failed")

    coordinator.memory_service = _FailingMemory()  # type: ignore[assignment]

    async def run_case():
        prepared = await coordinator._prepare_run(  # noqa: SLF001
            SessionRunIntent(text="finish the task", request_id="request_1"),
            run_id="run_memory_failure",
            model=ModelDescriptor(provider="unit", model_id="unit"),
        )
        assert prepared.state_port is not None
        await prepared.state_port.commit(
            CoreBoundary(kind="before_model", state=prepared.loop_input.state)
        )
        coordinator.capture_memory_proposals(
            "run_memory_failure",
            (
                MemoryProposal(
                    scope="project",
                    type="project",
                    key="project.verification.command",
                    content="Use focused pytest commands.",
                ),
            ),
            None,
        )
        final = AssistantMessage(content=[TextContent(text="done")])
        outcome = CoreOutcome(
            status="completed",
            reason=CoreReason("task.completed"),
            state=prepared.loop_input.state,
            new_messages=(final,),
            final_message=final,
        )
        return await RunCoordinator(coordinator).commit(prepared, outcome)

    try:
        record = asyncio.run(run_case())
        run = coordinator.state_service.get_run("run_memory_failure")
        events = coordinator.state_service.load_events(
            "session_memory_failure",
            run_id="run_memory_failure",
        )
    finally:
        coordinator.close()

    assert record.status == "completed"
    assert run is not None and run.status == "completed"
    assert any(event["type"] == "memory_proposal_failed" for event in events)
