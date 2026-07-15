from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from codepilot.core.contracts import ContextPrepareRequest, CoreContextView
from codepilot.core.state import CoreState
from codepilot.llm.ports import ModelDescriptor
from codepilot.protocols import (
    AssistantMessage,
    Message,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.sessions.context import (
    ContextBudgetConfig,
    ContextBudgetExceededError,
    ContextService,
)
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


def test_compactor_cursor_ends_after_the_final_result_of_a_parallel_batch(
    tmp_path: Path,
) -> None:
    summarizer = _Summarizer()
    compactor = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=summarizer,
    )
    parallel = _parallel_read_batch("parallel")
    consumer = AssistantMessage(
        content=[TextContent(text="parallel batch consumed")],
        metadata={"session_message_id": "parallel_consumer"},
    )
    messages = (
        *_messages(6),
        *parallel,
        consumer,
        *_messages_with_prefix("tail", 2),
    )

    snapshot = asyncio.run(
        compactor.compact(
            run_id="run_parallel_cursor",
            messages=messages,
            original_goal="Refactor context governance.",
        )
    )

    assert snapshot is not None
    assert snapshot.compacted_until_message_id == "parallel_result_b"
    assert tuple(summarizer.requests[0].messages[-3:]) == parallel


def test_compactor_excludes_the_unconsumed_parallel_batch_from_its_prefix(
    tmp_path: Path,
) -> None:
    summarizer = _Summarizer()
    compactor = ContextCompactor(
        workspace_dir=tmp_path,
        session_id="session_1",
        summarizer=summarizer,
    )
    parallel = _parallel_read_batch("protected")
    messages = (*_messages(6), *parallel, *_messages_with_prefix("tail", 5))

    snapshot = asyncio.run(
        compactor.compact(
            run_id="run_protected_batch",
            messages=messages,
            original_goal="Refactor context governance.",
        )
    )

    assert snapshot is not None
    compacted_ids = {_message_id(message) for message in summarizer.requests[0].messages}
    assert snapshot.compacted_until_message_id == "msg_05"
    assert not compacted_ids.intersection(
        {"protected_assistant", "protected_result_a", "protected_result_b"}
    )


def test_large_unique_read_history_triggers_semantic_compaction_instead_of_projection(
    tmp_path: Path,
) -> None:
    summarizer = _Summarizer()
    messages: list[Message] = []
    for index in range(4):
        call_id = f"read_{index}"
        messages.extend(
            [
                AssistantMessage(
                    content=[
                        ToolCall(
                            id=call_id,
                            name="read",
                            arguments={"path": f"src/file_{index}.py"},
                        )
                    ],
                    metadata={"session_message_id": f"read_assistant_{index}"},
                ),
                ToolResultMessage(
                    tool_call_id=call_id,
                    tool_name="read",
                    content=[TextContent(text=(f"value_{index} = True\n" * 800))],
                    details={
                        "path": f"src/file_{index}.py",
                        "sha256": f"hash-{index}",
                        "offset": 1,
                        "returned_lines": 800,
                    },
                    metadata={"session_message_id": f"read_result_{index}"},
                ),
            ]
        )
    messages.append(
        AssistantMessage(
            content=[TextContent(text="All read batches consumed.")],
            metadata={"session_message_id": "read_consumer"},
        )
    )
    state = CoreState.new("Inspect the repository")
    request = ContextPrepareRequest(
        session_id="session_1",
        run_id="run_projected_reads",
        purpose="reasoning",
        directive=None,
        messages=tuple(messages),
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
            context_window=6_000,
            max_output_tokens=500,
            safety_margin_tokens=0,
        ),
    )

    prepared = asyncio.run(service.prepare(request))

    assert summarizer.requests
    assert service.checkpoint_state()["compact_snapshot_ref"] is not None
    assert service.latest_report["raw_estimate_tokens"] > 5_500
    assert service.latest_report["estimated_tokens_after"] < 5_500
    assert service.latest_report["stage"] in {"critical", "critical+tight"}
    assert prepared.messages


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


def test_critical_pressure_without_a_summarizer_fails_explicitly(
    tmp_path: Path,
) -> None:
    state = CoreState.new("Refactor context governance")
    request = ContextPrepareRequest(
        session_id="session_1",
        run_id="run_missing_summarizer",
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
        summarizer=None,
        budget_config=ContextBudgetConfig(
            context_window=1_800,
            max_output_tokens=200,
            safety_margin_tokens=0,
        ),
    )

    with pytest.raises(
        ContextBudgetExceededError,
        match="context_compaction_failed: Context summarizer is not configured",
    ):
        asyncio.run(service.prepare(request))


def test_latest_unconsumed_batch_is_not_split_when_it_exceeds_the_budget(
    tmp_path: Path,
) -> None:
    summarizer = _Summarizer()
    call_id = "large_read"
    assistant = AssistantMessage(
        content=[ToolCall(id=call_id, name="read", arguments={"path": "large.py"})],
        metadata={"session_message_id": "large_read_assistant"},
    )
    result = ToolResultMessage(
        tool_call_id=call_id,
        tool_name="read",
        content=[TextContent(text="source_line = True\n" * 1_000)],
        details={
            "path": "large.py",
            "sha256": "large-hash",
            "offset": 1,
            "returned_lines": 1_000,
        },
        metadata={"session_message_id": "large_read_result"},
    )
    state = CoreState.new("Inspect the large file")
    request = ContextPrepareRequest(
        session_id="session_1",
        run_id="run_large_batch",
        purpose="reasoning",
        directive=None,
        messages=(*_messages(6), assistant, result),
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
            context_window=2_500,
            max_output_tokens=200,
            safety_margin_tokens=0,
        ),
    )

    with pytest.raises(ContextBudgetExceededError):
        asyncio.run(service.prepare(request))

    assert summarizer.requests
    summarized_ids = {
        _message_id(message) for message in summarizer.requests[0].messages
    }
    assert "large_read_assistant" not in summarized_ids
    assert "large_read_result" not in summarized_ids
    tool_artifacts = (
        tmp_path
        / ".codepilot"
        / "runs"
        / "run_large_batch"
        / "artifacts"
        / "tool_outputs"
    )
    assert not tool_artifacts.exists()


def _messages_with_prefix(prefix: str, count: int) -> tuple[Message, ...]:
    return tuple(
        UserMessage(
            content=f"{prefix} historical request {index} " + ("details " * 80),
            metadata={"session_message_id": f"{prefix}_{index:02d}"},
        )
        for index in range(count)
    )


def _parallel_read_batch(prefix: str) -> tuple[Message, ...]:
    return (
        AssistantMessage(
            content=[
                ToolCall(id=f"{prefix}_a", name="read", arguments={"path": "a.py"}),
                ToolCall(id=f"{prefix}_b", name="read", arguments={"path": "b.py"}),
            ],
            metadata={"session_message_id": f"{prefix}_assistant"},
        ),
        ToolResultMessage(
            tool_call_id=f"{prefix}_a",
            tool_name="read",
            content=[TextContent(text="a")],
            metadata={"session_message_id": f"{prefix}_result_a"},
        ),
        ToolResultMessage(
            tool_call_id=f"{prefix}_b",
            tool_name="read",
            content=[TextContent(text="b")],
            metadata={"session_message_id": f"{prefix}_result_b"},
        ),
    )
