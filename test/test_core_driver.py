from __future__ import annotations

import asyncio

import pytest

from codepilot.core.contracts import (
    CoreBoundary,
    CoreLimits,
    CorePorts,
    CoreRunInput,
    ModelEntry,
    PreparedModelContext,
    ToolResultEntry,
)
from codepilot.core.driver import run_core
from codepilot.core.errors import CoreInvariantError
from codepilot.core.state import CoreState
from codepilot.llm.ports import LLMCompleted, ModelDescriptor
from codepilot.protocols import (
    AssistantMessage,
    TextContent,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from codepilot.tools.contracts import ToolBatchPreparation
from codepilot.tools.registry import ToolCatalogSnapshot
from codepilot.tools.results import ToolResult
from codepilot.tools.security import ApprovalChallenge


class FakeModel:
    def __init__(self, messages, trace=None) -> None:
        self.messages = list(messages)
        self.requests = []
        self.trace = trace if trace is not None else []

    async def stream(self, request):
        self.trace.append("model.call")
        self.requests.append(request)
        yield LLMCompleted(message=self.messages.pop(0))


class FailingAfterFirstModel(FakeModel):
    async def stream(self, request):
        if self.messages:
            async for event in super().stream(request):
                yield event
            return
        raise RuntimeError("model transport disconnected")
        if False:  # pragma: no cover - keeps this an async iterator
            yield None


class FakeContext:
    def __init__(self, trace=None) -> None:
        self.requests = []
        self.trace = trace if trace is not None else []

    async def prepare(self, request):
        self.trace.append("context.prepare")
        self.requests.append(request)
        return PreparedModelContext(
            system_prompt="system",
            messages=request.messages,
            tools=(),
            projection_ref="context:driver",
        )


class FakeBoundary:
    def __init__(self, trace=None, fail_on=None) -> None:
        self.boundaries: list[CoreBoundary] = []
        self.trace = trace if trace is not None else []
        self.fail_on = fail_on

    async def commit(self, boundary):
        self.trace.append(f"boundary.{boundary.kind}")
        self.boundaries.append(boundary)
        if boundary.kind == self.fail_on:
            raise RuntimeError(f"boundary failed: {boundary.kind}")


class FakeTools:
    def __init__(self, *, results=(), preparation_results=(), trace=None) -> None:
        self.results = tuple(results)
        self.preparation_results = tuple(preparation_results)
        self.trace = trace if trace is not None else []
        self.requests = ()
        self.executed = False

    def catalog_snapshot(self, *, mode=None):
        return ToolCatalogSnapshot(f"catalog:{mode}", (), 0)

    def prepare_batch(self, requests):
        self.trace.append("tools.prepare")
        self.requests = tuple(requests)
        if self.preparation_results:
            return ToolBatchPreparation(results=self.preparation_results)
        return ToolBatchPreparation(batch_id="batch_1")

    async def execute_prepared(self, batch_id):
        self.trace.append("tools.execute")
        assert batch_id == "batch_1"
        self.executed = True
        return self.results


class FailingTools(FakeTools):
    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.error = error

    async def execute_prepared(self, batch_id):
        assert batch_id == "batch_1"
        raise self.error

def _input(*, entry=None, messages=None, limits=None, state=None) -> CoreRunInput:
    return CoreRunInput(
        session_id="session_driver",
        run_id="run_driver",
        entry=entry or ModelEntry(),
        messages=messages or (UserMessage(content="inspect"),),
        state=state or CoreState.new("inspect"),
        mode="build",
        model=ModelDescriptor(provider="unit-test", model_id="driver-model"),
        limits=limits or CoreLimits(),
        context_seed={},
    )


def _ports(model, boundary, *, tools=None, context=None, cancellation=None):
    return CorePorts(
        model=model,
        tools=tools,
        context=context or FakeContext(),
        boundary=boundary,
        cancellation=cancellation,
    )


def _final(text="done") -> AssistantMessage:
    return AssistantMessage(content=[TextContent(text=text)])


def _tool_request() -> AssistantMessage:
    return AssistantMessage(
        content=[ToolCall(id="call_1", name="read", arguments={"path": "a.py"})]
    )


def _success_result() -> ToolResult:
    return ToolResult(
        tool_call_id="call_1",
        tool_name="read",
        status="success",
        registration_id="read@1",
    )


def _approval_result() -> ToolResult:
    return ToolResult(
        tool_call_id="call_1",
        tool_name="write",
        status="approval_required",
        approval=ApprovalChallenge(
            approval_id="approval_1",
            request_fingerprint="fingerprint_1",
            run_id="run_driver",
            session_id="session_driver",
            tool_call_id="call_1",
            tool_name="write",
            registration_id="write@1",
            actions=("write",),
            resources=(),
            effects=frozenset({"filesystem_write"}),
            risk="high",
            reason="Writes a file",
            safe_preview={},
        ),
        registration_id="write@1",
    )


def test_driver_commits_exact_model_completion_sequence() -> None:
    boundary = FakeBoundary()

    outcome = asyncio.run(run_core(_input(), _ports(FakeModel([_final()]), boundary)))

    assert outcome.status == "completed"
    assert [item.kind for item in boundary.boundaries] == [
        "before_model",
        "after_model",
        "before_terminal",
    ]
    assert boundary.boundaries[0].new_messages == ()
    assert boundary.boundaries[1].new_messages == (outcome.final_message,)
    assert boundary.boundaries[2].new_messages == ()
    assert outcome.state.facts.counters.model_turns == 1


def test_driver_commits_before_tools_before_any_execution() -> None:
    trace = []
    boundary = FakeBoundary(trace)
    tools = FakeTools(results=(_success_result(),), trace=trace)
    model = FakeModel([_tool_request(), _final()], trace)

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(model, boundary, tools=tools, context=FakeContext(trace)),
        )
    )

    assert outcome.status == "completed"
    assert [item.kind for item in boundary.boundaries] == [
        "before_model",
        "after_model",
        "before_tools",
        "after_tools",
        "before_model",
        "after_model",
        "before_terminal",
    ]
    assert trace.index("tools.prepare") < trace.index("boundary.before_tools")
    assert trace.index("boundary.before_tools") < trace.index("tools.execute")
    after_tools = boundary.boundaries[3]
    assert len(after_tools.new_messages) == 1
    assert isinstance(after_tools.new_messages[0], ToolResultMessage)
    assert outcome.state.facts.counters.tool_calls == 1


def test_context_receives_latest_core_state_after_tool_reduction() -> None:
    boundary = FakeBoundary()
    context = FakeContext()
    tools = FakeTools(
        results=(
            ToolResult(
                tool_call_id="call_1",
                tool_name="read",
                status="success",
                data={
                    "verification": {
                        "status": "passed",
                        "command": "pytest -q",
                    }
                },
                registration_id="read@1",
            ),
        )
    )

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(
                FakeModel([_tool_request(), _final()]),
                boundary,
                tools=tools,
                context=context,
            ),
        )
    )

    assert outcome.state.facts.verification.status == "passed"
    assert context.requests[1].core_view.verification.status == "passed"


def test_driver_returns_waiting_only_after_waiting_boundary_commits() -> None:
    boundary = FakeBoundary()
    tools = FakeTools(preparation_results=(_approval_result(),))

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(FakeModel([_tool_request()]), boundary, tools=tools),
        )
    )

    assert outcome.status == "waiting"
    assert outcome.wait is not None and outcome.wait.kind == "tool_approval"
    assert [item.kind for item in boundary.boundaries] == [
        "before_model",
        "after_model",
        "waiting",
    ]
    assert tools.executed is False


def test_before_tools_boundary_failure_prevents_execution() -> None:
    boundary = FakeBoundary(fail_on="before_tools")
    tools = FakeTools(results=(_success_result(),))

    with pytest.raises(RuntimeError, match="before_tools"):
        asyncio.run(
            run_core(
                _input(),
                _ports(FakeModel([_tool_request()]), boundary, tools=tools),
            )
        )

    assert tools.executed is False


def test_waiting_boundary_failure_is_not_reported_as_waiting() -> None:
    boundary = FakeBoundary(fail_on="waiting")
    tools = FakeTools(preparation_results=(_approval_result(),))

    with pytest.raises(RuntimeError, match="waiting"):
        asyncio.run(
            run_core(
                _input(),
                _ports(FakeModel([_tool_request()]), boundary, tools=tools),
            )
        )


def test_driver_settles_unexecuted_tool_calls_before_budget_wait() -> None:
    boundary = FakeBoundary()

    outcome = asyncio.run(
        run_core(
            _input(limits=CoreLimits(max_model_turns=0)),
            _ports(FakeModel([_tool_request()]), boundary),
        )
    )

    assert outcome.status == "waiting"
    assert [item.kind for item in boundary.boundaries] == [
        "before_model",
        "after_model",
        "after_tools",
        "waiting",
    ]
    tool_messages = [
        message
        for item in boundary.boundaries
        for message in item.new_messages
        if isinstance(message, ToolResultMessage)
    ]
    assert len(tool_messages) == 1
    assert tool_messages[0].tool_call_id == "call_1"
    assert tool_messages[0].status == "interrupted"


def test_tool_execution_cancellation_settles_persisted_calls_before_terminal() -> None:
    boundary = FakeBoundary()

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(
                FakeModel([_tool_request()]),
                boundary,
                tools=FailingTools(asyncio.CancelledError("user_cancelled")),
            ),
        )
    )

    assert outcome.status == "cancelled"
    assert [item.kind for item in boundary.boundaries] == [
        "before_model",
        "after_model",
        "before_tools",
        "after_tools",
        "before_terminal",
    ]
    result = boundary.boundaries[3].new_messages[0]
    assert isinstance(result, ToolResultMessage)
    assert result.tool_call_id == "call_1"
    assert result.status == "interrupted"


def test_tool_execution_exception_settles_persisted_calls_before_failure() -> None:
    boundary = FakeBoundary()

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(
                FakeModel([_tool_request()]),
                boundary,
                tools=FailingTools(RuntimeError("tool transport disconnected")),
            ),
        )
    )

    assert outcome.status == "failed"
    assert outcome.reason.code == "core.execution_error"
    result = boundary.boundaries[3].new_messages[0]
    assert isinstance(result, ToolResultMessage)
    assert result.status == "interrupted"


def test_model_exception_returns_latest_reduced_state() -> None:
    boundary = FakeBoundary()
    tools = FakeTools(
        results=(
            ToolResult(
                tool_call_id="call_1",
                tool_name="read",
                status="success",
                data={
                    "verification": {
                        "status": "passed",
                        "command": "pytest -q",
                    }
                },
                registration_id="read@1",
            ),
        )
    )

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(
                FailingAfterFirstModel([_tool_request()]),
                boundary,
                tools=tools,
            ),
        )
    )

    assert outcome.status == "failed"
    assert outcome.reason.code == "core.execution_error"
    assert outcome.state.facts.verification.status == "passed"


def test_driver_accepts_final_tool_result_entry_without_reexecuting_tool() -> None:
    assistant = _tool_request()
    boundary = FakeBoundary()
    tools = FakeTools()

    outcome = asyncio.run(
        run_core(
            _input(
                entry=ToolResultEntry(results=(_success_result(),)),
                messages=(UserMessage(content="inspect"), assistant),
            ),
            _ports(FakeModel([_final()]), boundary, tools=tools),
        )
    )

    assert outcome.status == "completed"
    assert tools.executed is False
    assert [item.kind for item in boundary.boundaries] == [
        "after_tools",
        "before_model",
        "after_model",
        "before_terminal",
    ]


def test_driver_settles_calls_that_exceed_the_per_turn_limit() -> None:
    boundary = FakeBoundary()
    assistant = AssistantMessage(
        content=[
            ToolCall(id="call_1", name="read", arguments={}),
            ToolCall(id="call_2", name="read", arguments={}),
        ]
    )
    tools = FakeTools()

    outcome = asyncio.run(
        run_core(
            _input(limits=CoreLimits(max_tool_calls_per_turn=1)),
            _ports(FakeModel([assistant]), boundary, tools=tools),
        )
    )

    assert outcome.status == "waiting"
    assert outcome.reason.code == "run.max_tool_calls_per_turn"
    assert tools.executed is False
    assert [item.kind for item in boundary.boundaries] == [
        "before_model",
        "after_model",
        "after_tools",
        "waiting",
    ]
    settled = boundary.boundaries[2].new_messages
    assert [message.tool_call_id for message in settled] == ["call_1", "call_2"]


def test_driver_rejects_results_for_unknown_tool_calls() -> None:
    boundary = FakeBoundary()
    unexpected = ToolResult(
        tool_call_id="unexpected",
        tool_name="read",
        status="success",
        registration_id="read@1",
    )
    tools = FakeTools(results=(unexpected,))

    with pytest.raises(CoreInvariantError, match="unknown ToolCall"):
        asyncio.run(
            run_core(
                _input(),
                _ports(FakeModel([_tool_request()]), boundary, tools=tools),
            )
        )


def test_driver_normalizes_cancellation_at_safe_point() -> None:
    class Cancelled:
        reason = "user_cancelled"

        def raise_if_cancelled(self):
            raise asyncio.CancelledError

    boundary = FakeBoundary()

    outcome = asyncio.run(
        run_core(
            _input(),
            _ports(FakeModel([]), boundary, cancellation=Cancelled()),
        )
    )

    assert outcome.status == "cancelled"
    assert outcome.reason.code == "run.cancelled"
    assert [item.kind for item in boundary.boundaries] == ["before_terminal"]
