from __future__ import annotations

import asyncio
from pathlib import Path

from codepilot.core.contracts import ContextPrepareRequest, CoreContextView
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import Message, UserMessage
from codepilot.sessions.context import ContextBudgetConfig, ContextService
from codepilot.sessions.context.compaction import ContextCompactor
from codepilot.sessions.context.contracts import (
    CompactSummary,
    ContextSummaryRequest,
    ContextSummaryResult,
)


class _Summarizer:
    def __init__(self) -> None:
        self.requests: list[ContextSummaryRequest] = []

    def summarize(self, request: ContextSummaryRequest) -> ContextSummaryResult:
        self.requests.append(request)
        cursor = _message_id(request.messages[-1])
        return ContextSummaryResult(
            summary=CompactSummary(
                original_goal="Refactor context governance.",
                decisions=("Use five layers.",),
                completed_work=("Memory stage completed.",),
                next_actions=("Continue ContextService implementation.",),
                source_refs=tuple(
                    f"message:{_message_id(message)}" for message in request.messages
                ),
            ),
            compacted_until_message_id=cursor,
        )


def _message_id(message: Message) -> str:
    return str(message.metadata["session_message_id"])


def _messages(count: int = 10) -> tuple[Message, ...]:
    return tuple(
        UserMessage(
            content=f"historical request {index} " + ("details " * 80),
            metadata={"session_message_id": f"msg_{index:02d}"},
        )
        for index in range(count)
    )


def test_compactor_uses_auxiliary_summarizer_and_writes_snapshot_artifact(
    tmp_path: Path,
) -> None:
    summarizer = _Summarizer()
    compactor = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=summarizer,
    )

    snapshot = asyncio.run(
        compactor.compact(
            run_id="run_1",
            messages=_messages(),
            original_goal="Refactor context governance.",
        )
    )

    assert summarizer.requests
    assert snapshot is not None
    assert snapshot.path.startswith(".codepilot/runs/run_1/artifacts/context/")
    assert (tmp_path / snapshot.path).is_file()
    assert compactor.current_summary is not None
    assert "Use five layers." in compactor.current_summary.render()
    assert set(compactor.checkpoint_state()) == {
        "compact_snapshot_ref",
        "compacted_until_message_id",
    }


def test_failed_rollup_preserves_the_previous_snapshot(tmp_path: Path) -> None:
    summarizer = _Summarizer()
    compactor = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=summarizer,
    )
    first = asyncio.run(
        compactor.compact(
            run_id="run_1",
            messages=_messages(),
            original_goal="Refactor context governance.",
        )
    )

    class _FailingSummarizer:
        calls = 0

        def summarize(self, _request):
            self.calls += 1
            raise TimeoutError("summary timed out")

    failing = _FailingSummarizer()
    compactor.summarizer = failing
    second = asyncio.run(
        compactor.compact(
            run_id="run_1",
            messages=_messages(16),
            original_goal="Refactor context governance.",
        )
    )

    assert failing.calls == 1
    assert second == first
    assert compactor.checkpoint_state()["compact_snapshot_ref"] == first.path


def test_critical_service_pressure_uses_the_summarizer_before_returning(
    tmp_path: Path,
) -> None:
    summarizer = _Summarizer()
    state = CoreState.new("Refactor context governance")
    request = ContextPrepareRequest(
        session_id="session_1",
        run_id="run_critical",
        purpose="reasoning",
        directive=None,
        messages=_messages(),
        core_view=CoreContextView.from_state(state, "build"),
        model=ModelDescriptor(provider="unit", model_id="unit"),
        tool_catalog=None,
        seed={"system_prompt": "L0 rules."},
    )
    service = ContextService(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=summarizer,
        budget_config=ContextBudgetConfig(
            context_window=1800,
            max_output_tokens=200,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(request))

    assert summarizer.requests
    assert service.checkpoint_state()["compact_snapshot_ref"]
    assert len(prepared.messages) < len(request.messages) + 1
